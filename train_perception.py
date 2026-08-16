"""Fine-tune YOLO11m for search-and-rescue, then measure it with our own harness.

    python train_perception.py --mode rgb --epochs 50 --seed 42
    python train_perception.py --mode lidar
    python train_perception.py --mode both --epochs 50 --batch 16
    python train_perception.py --selfcheck        # the wiring, without a dataset

This is the "real detector weights" line item Phase 2 deferred. The stubs in
`src/perception/detectors.py` model the *shape* of a detector's behaviour; this
produces the checkpoint `PERCEPTION_MODE=real` loads
(`src/perception/models.py`).

VisDrone is drone-captured aerial imagery, which is the domain that matters — a
model trained on ground-level photographs sees a person from an angle no search
drone ever gets. `data/visdrone/convert_to_yolo.py` has already reduced it to
the two classes a search cares about, `pedestrian` and `people`; everything else
in VisDrone is traffic.

The fixed splits are used, not re-derived
-----------------------------------------
Phase 1 fixed the splits so every later number refers to the same validation
set. This script therefore reads a manifest from `data/splits/` and materialises
the image lists ultralytics wants from it, rather than pointing training at
whatever happens to be in `train/` and `val/`. Without a manifest it falls back
to the converter's own layout and says so — a fallback that announces itself is
fine; a silent one would make two mAP figures quietly incomparable.

Two yardsticks, deliberately
----------------------------
Ultralytics reports mAP50, mAP50-95, precision and recall on its own validation
pass — that is the training framework marking its own homework, and it is worth
printing. Then the same weights are run over the same validation images and
scored by `src/evaluation/harness.py`, which is the project's yardstick and the
one the exit criteria refer to: it adds **recall at a fixed false-alarm rate**,
the operating point an operator actually works at.

One honest gap: the harness also reports geolocation error, and VisDrone has no
telemetry or ground-truth coordinates, so that metric reads `n/a` on this data.
It is not zero and it is not fine — it is unmeasured, and it stays unmeasured
until there is recorded footage with telemetry.

ponytail: single GPU, no DDP, no resume, no early-stopping schedule, no
hyper-parameter sweep. The Phase 9 Optuna study tunes the *pipeline*; tuning the
training recipe is a separate job and a much longer one.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.utils.seed import DEFAULT_SEED, describe, set_global_seed

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "config" / "weights"
SPLITS_DIR = ROOT / "data" / "splits"

MODEL = "yolo11m.pt"          # Medium, per the standardised RGB detector

# Where each sensor's dataset and checkpoint live. The checkpoint paths are the
# ones `src/perception/models.py` loads — they are not free to differ.
DATASETS = {
    "rgb": {
        "dir": ROOT / "data" / "visdrone",
        "weights": WEIGHTS_DIR / "yolo11m_visdrone.pt",
        "names": ("pedestrian", "people"),
        "fallback_yaml": "visdrone_person.yaml",
    },
    "lidar": {
        "dir": ROOT / "data" / "lidar",
        "weights": WEIGHTS_DIR / "yolo11m_lidar.pt",
        "names": ("person",),
        "fallback_yaml": "lidar_person.yaml",
    },
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# -- dataset plumbing ------------------------------------------------------

def find_manifest(sensor, splits_dir=SPLITS_DIR):
    """The newest split manifest for this sensor, or None."""
    if not splits_dir.is_dir():
        return None
    source = DATASETS[sensor]["dir"].name
    manifests = sorted(splits_dir.glob(f"{source}_seed*.json"))
    return manifests[-1] if manifests else None


def images_in(directory):
    return sorted(p for p in Path(directory).rglob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def has_dataset(sensor):
    """Is there anything to train on? Images *and* labels, not just a folder."""
    root = DATASETS[sensor]["dir"]
    return bool(images_in(root)) and any(root.rglob("*.txt"))


def dataset_yaml(sensor, manifest=None, out_dir=None):
    """Build the ultralytics dataset config. Returns (yaml path, description).

    With a manifest, the train and val image lists are written out from it — the
    fixed split, verbatim. Without one, the converter's train/ and val/ layout
    is used and the caller is told, because the two are not interchangeable.
    """
    spec = DATASETS[sensor]
    root = spec["dir"]
    out_dir = Path(out_dir or root)
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(spec["names"]))

    if manifest is not None:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        splits = data["splits"]
        listed = {}
        for split in ("train", "val"):
            paths = [str((ROOT / p).resolve()) for p in splits.get(split, [])]
            if not paths:
                raise ValueError(f"{manifest} has no {split} entries")
            listing = out_dir / f"{sensor}_{split}.txt"
            listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
            listed[split] = listing

        path = out_dir / f"{sensor}_from_split.yaml"
        path.write_text(
            f"# Generated by train_perception.py from {Path(manifest).name}\n"
            f"# seed {data.get('seed')}, {data.get('total')} sample(s)\n"
            f"path: {root.resolve().as_posix()}\n"
            f"train: {listed['train'].resolve().as_posix()}\n"
            f"val: {listed['val'].resolve().as_posix()}\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )
        return path, f"fixed split {Path(manifest).name} (seed {data.get('seed')})"

    fallback = root / spec["fallback_yaml"]
    if fallback.is_file():
        return fallback, f"{fallback.name} — no split manifest, using train/ and val/ as laid out"
    raise FileNotFoundError(
        f"no split manifest in {SPLITS_DIR} and no {fallback}.\n"
        f"  Generate one:  python data/splits/generate_splits.py --dataset "
        f"{root.name} --seed {DEFAULT_SEED}"
    )


def label_path_for(image):
    """YOLO's own convention: .../images/x.jpg -> .../labels/x.txt."""
    parts = list(Path(image).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def ground_truth_for(image, width, height, names):
    """YOLO label file -> harness GroundTruth dicts, in pixels."""
    path = label_path_for(image)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        index, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
        rows.append({
            "frame_id": Path(image).stem,
            "label": names[index] if index < len(names) else str(index),
            "box": [round((cx - w / 2) * width, 2), round((cy - h / 2) * height, 2),
                    round((cx + w / 2) * width, 2), round((cy + h / 2) * height, 2)],
            # No geo: VisDrone has no telemetry. The harness reports geolocation
            # error as n/a rather than pretending to a number.
            "geo": None,
        })
    return rows


# -- training --------------------------------------------------------------

def _train(sensor, args):
    """Shared body: both sensors are the same model on different imagery."""
    from ultralytics import YOLO      # noqa: PLC0415  (training-only dependency)

    spec = DATASETS[sensor]
    manifest = None if args.no_splits else find_manifest(sensor)
    config, source = dataset_yaml(sensor, manifest)

    print(f"\n  {sensor}: {spec['dir']}")
    print(f"  data:    {source}")
    print(f"  model:   {MODEL} -> {spec['weights'].name}")
    print(f"  run:     {args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch}, "
          f"seed {args.seed}\n")

    model = YOLO(MODEL)
    results = model.train(
        data=str(config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        # No `project=`: ultralytics already roots runs at its own runs_dir, and
        # a relative project nests under it — an early pass landed in
        # runs/detect/runs/perception/. `name` alone is enough.
        name=f"yolo11m_{sensor}",
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} is not there")
    spec["weights"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, spec["weights"])

    print(f"\n  {sensor} training complete")
    print(f"    weights   {spec['weights']}")
    _print_ultralytics_metrics(results)
    return spec["weights"], config


def _print_ultralytics_metrics(results):
    """What the training framework makes of its own work."""
    metrics = getattr(results, "results_dict", None) or {}
    print("    ultralytics validation")
    for label, key in (("mAP50", "metrics/mAP50(B)"),
                       ("mAP50-95", "metrics/mAP50-95(B)"),
                       ("precision", "metrics/precision(B)"),
                       ("recall", "metrics/recall(B)")):
        value = metrics.get(key)
        print(f"      {label:<12}{'n/a' if value is None else f'{value:.4f}'}")
    if not metrics:
        print("      (no metrics on the results object — check the run directory)")


def train_rgb(args):
    """Fine-tune YOLO11m on VisDrone, people only."""
    if not has_dataset("rgb"):
        print("\n  RGB dataset not found under data/visdrone — skipping.\n"
              "    bash data/visdrone/download_visdrone.sh\n"
              "    python data/visdrone/convert_to_yolo.py")
        return None, None
    return _train("rgb", args)


def train_lidar(args):
    """Same model on LiDAR range images, when there are any.

    The range images come from `data/lidar/project_to_rangeimg.py`. A projected
    sweep is a single-channel picture of distance, so the same detector
    architecture applies — what changes is what a bright pixel means.
    """
    if not has_dataset("lidar"):
        print("\n  LiDAR dataset not found, skipping.\n"
              "    Range images go in data/lidar/ with YOLO labels alongside:\n"
              "      python data/lidar/project_to_rangeimg.py --input sweep.bin "
              "--output data/lidar/train/images/0001.png")
        return None, None
    return _train("lidar", args)


# -- measuring it with our own harness -------------------------------------

def harness_baseline(sensor, weights, config, conf=0.25, limit=None):
    """Run the trained weights over the validation images and score them here.

    Two artefacts are written and kept: the clues the detector produced, and the
    split they were scored against. Both are plain JSON, so a number in the
    report can be traced back to the box that caused it.
    """
    import yaml                       # noqa: PLC0415  (ships with ultralytics)

    from src.contracts.clue import AgentSource
    from src.guardrails.provenance import TAG_LIDAR, TAG_RGB
    from src.perception.models import RealDetector

    spec = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    val = spec["val"]
    root = Path(spec.get("path", "."))
    listing = Path(val) if Path(val).is_absolute() else root / val
    if listing.suffix == ".txt":
        images = [Path(line) for line in listing.read_text().splitlines() if line.strip()]
    else:
        images = images_in(listing)
    if limit:
        images = images[:limit]
    if not images:
        print("  no validation images to score")
        return None

    names = tuple(spec["names"].values()) if isinstance(spec["names"], dict) else tuple(spec["names"])
    detector = RealDetector(
        weights,
        AgentSource.DRONE_RGB if sensor == "rgb" else AgentSource.DRONE_LIDAR,
        TAG_RGB if sensor == "rgb" else TAG_LIDAR,
        conf=conf,
    )

    clues, truth, frame_ids = [], [], []
    for image in images:
        result = detector.model.predict(str(image), conf=conf, verbose=False)[0]
        height, width = result.orig_shape
        frame_id = image.stem
        frame_ids.append(frame_id)
        rows = [([round(float(v), 2) for v in box.xyxy[0].tolist()],
                 float(box.conf[0]),
                 names[int(box.cls[0])] if int(box.cls[0]) < len(names) else "contact")
                for box in result.boxes]
        clues.extend(detector.to_clues(frame_id, rows, case_id=f"eval-{sensor}"))
        truth.extend(ground_truth_for(image, width, height, names))

    out = WEIGHTS_DIR / f"{sensor}_baseline"
    out.mkdir(parents=True, exist_ok=True)
    clues_path = out / "clues.json"
    split_path = out / "split.json"
    clues_path.write_text(
        json.dumps([json.loads(c.model_dump_json()) for c in clues], indent=2),
        encoding="utf-8")
    split_path.write_text(
        json.dumps({"frame_ids": frame_ids, "ground_truth": truth}, indent=2),
        encoding="utf-8")

    # Run the harness as its own process, with PERCEPTION_MODE=real set, so what
    # runs here is exactly the command a reader can re-run by hand.
    # `--split validation`: the harness names its splits train/validation, while
    # ultralytics and the YOLO layout say val. The manifest is written for one
    # split at a time, so this only names which label the report carries.
    command = [sys.executable, "-m", "src.evaluation.harness",
               "--data", str(split_path), "--split", "validation",
               "--clues", str(clues_path), "--case-id", f"eval-{sensor}"]
    print(f"\n  scoring {len(clues)} clue(s) over {len(images)} frame(s) with the "
          f"Phase 1 harness")
    print("    PERCEPTION_MODE=real " + " ".join(command[1:]))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               env={**os.environ, "PERCEPTION_MODE": "real"})
    print(completed.stdout or completed.stderr)
    print("  geolocation error reads n/a by design: VisDrone carries no telemetry,\n"
          "  so that metric is unmeasured until there is footage with it.")
    return {"clues": clues_path, "split": split_path, "returncode": completed.returncode}


# -- CLI -------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("rgb", "lidar", "both"), default="rgb")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence floor for the harness pass")
    parser.add_argument("--no-splits", action="store_true",
                        help="ignore data/splits/ and use the converter's train/ and val/")
    parser.add_argument("--no-harness", action="store_true",
                        help="skip the Phase 1 scoring pass after training")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    print(f"\n  {describe(set_global_seed(args.seed), args.seed)}")

    trained = {}
    for sensor in (("rgb", "lidar") if args.mode == "both" else (args.mode,)):
        weights, config = (train_rgb if sensor == "rgb" else train_lidar)(args)
        if weights is not None:
            trained[sensor] = (weights, config)

    if not trained:
        print("\n  nothing trained.\n")
        return 1

    if not args.no_harness:
        for sensor, (weights, config) in trained.items():
            harness_baseline(sensor, weights, config, conf=args.conf)

    print("\n  next: PERCEPTION_MODE=real python -m src.perception.agent\n")
    return 0


def selfcheck():
    """Everything that does not need a dataset, a GPU, or seventeen hours."""
    assert set(DATASETS) == {"rgb", "lidar"}
    # The checkpoint paths are the ones the loader reads. If these drift, real
    # mode breaks with a file-not-found that names a path nothing writes.
    from src.perception.models import LIDAR_WEIGHTS, RGB_WEIGHTS

    assert DATASETS["rgb"]["weights"] == RGB_WEIGHTS, DATASETS["rgb"]["weights"]
    assert DATASETS["lidar"]["weights"] == LIDAR_WEIGHTS

    assert label_path_for("data/visdrone/val/images/0001.jpg") == Path(
        "data/visdrone/val/labels/0001.txt")
    # "images" appears twice: only the last one is the directory YOLO means.
    assert label_path_for("/srv/images/set/val/images/a.png") == Path(
        "/srv/images/set/val/labels/a.txt")

    truth = ground_truth_for(Path("nowhere/images/x.jpg"), 100, 100, ("person",))
    assert truth == [], "a missing label file is background, not an error"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images").mkdir()
        (tmp / "labels").mkdir()
        image = tmp / "images" / "frame_0001.jpg"
        image.write_bytes(b"not a real jpeg")
        (tmp / "labels" / "frame_0001.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1\n")

        rows = ground_truth_for(image, 1000, 500, ("pedestrian", "people"))
        assert len(rows) == 2
        assert rows[0]["box"] == [400.0, 150.0, 600.0, 350.0], rows[0]
        assert rows[0]["label"] == "pedestrian" and rows[1]["label"] == "people"
        assert rows[0]["geo"] is None, "VisDrone has no coordinates to claim"
        assert rows[0]["frame_id"] == "frame_0001"

        # A manifest becomes the image lists ultralytics reads, verbatim.
        manifest = tmp / "visdrone_seed42.json"
        manifest.write_text(json.dumps({
            "seed": 42, "total": 2,
            "splits": {"train": ["data/visdrone/train/images/a.jpg"],
                       "val": ["data/visdrone/val/images/b.jpg"],
                       "test": []},
        }))
        config, source = dataset_yaml("rgb", manifest, out_dir=tmp)
        text = config.read_text()
        assert "seed 42" in text and "0: pedestrian" in text and "1: people" in text
        assert "fixed split" in source and "42" in source
        listing = tmp / "rgb_val.txt"
        assert listing.is_file() and listing.read_text().strip().endswith("b.jpg")
        assert (tmp / "rgb_train.txt").is_file()

        # An empty split is refused rather than silently training on nothing.
        empty = tmp / "visdrone_seed7.json"
        empty.write_text(json.dumps({"seed": 7, "splits": {"train": [], "val": []}}))
        try:
            dataset_yaml("rgb", empty, out_dir=tmp)
            raise AssertionError("an empty split must be refused")
        except ValueError as e:
            assert "train" in str(e)

        # No manifest and no fallback yaml: says how to make one.
        try:
            dataset_yaml("lidar", None, out_dir=tmp)
            raise AssertionError("a missing dataset config must be refused")
        except FileNotFoundError as e:
            assert "generate_splits.py" in str(e)

    assert find_manifest("rgb", tmp / "nope") is None, "no splits dir is not a crash"
    assert not has_dataset("lidar"), "there is no LiDAR data in this repository yet"

    print("  ok  the checkpoint paths match what PERCEPTION_MODE=real loads")
    print("  ok  labels resolve from images/ to labels/, last segment only")
    print("  ok  YOLO labels denormalise to pixel boxes the harness can score")
    print("  ok  no geo is claimed for imagery that carries none")
    print("  ok  a split manifest becomes the image lists training reads")
    print("  ok  an empty split and a missing config are refused, with the fix")
    print("  ok  an absent dataset is a skip, not a crash")
    print("\n7 checks passed")
    return 0


if __name__ == "__main__":
    # Required on Windows: the dataloader spawns worker processes, and without
    # this guard each one re-imports the module and starts its own training run.
    sys.exit(main())
