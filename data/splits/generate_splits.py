"""Fixed train/val/test manifests, from a seed.

    python data/splits/generate_splits.py --dataset visdrone --seed 0
    python data/splits/generate_splits.py --src data/lidar --pattern "*.png"
    python data/splits/generate_splits.py --selfcheck

Phase 1 exists so that every later exit test refers to the same numbers. A split
regenerated on the fly is a different validation set every run, and two mAP
figures measured against two different sets are not comparable — which is the
whole point of having a harness. So the split is a *file*: written once, read by
training and evaluation, and carrying the seed that produced it.

Determinism is not "we called random with a seed". The file list is sorted
before shuffling, because a directory listing is in whatever order the
filesystem feels like and would otherwise make the same seed mean two different
splits on two machines. Paths are stored relative to the repository root for the
same reason.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")

# Where each known dataset keeps its samples, and what a sample looks like.
DATASETS = {
    "visdrone": (ROOT / "data" / "visdrone", ("*.jpg", "*.jpeg", "*.png")),
    "lidar": (ROOT / "data" / "lidar", ("*.png", "*.bin", "*.ply", "*.las")),
    "paired": (ROOT / "data" / "paired", ("*.jpg", "*.png", "*.bin")),
}


def collect(source, patterns):
    """Every matching file under `source`, sorted, relative to the repo root."""
    files = set()
    for pattern in patterns:
        files.update(p for p in source.rglob(pattern) if p.is_file())
    return sorted(_relative(p) for p in files)


def _relative(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def split(files, ratios=DEFAULT_RATIOS, seed=0):
    """Shuffle deterministically and cut. Returns {name: [path, ...]}.

    Every file lands in exactly one split — the cut points are computed from the
    running total rather than each ratio separately, so rounding cannot drop a
    sample or put one in two places.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    ordered = sorted(files)             # never trust the caller's order
    random.Random(seed).shuffle(ordered)

    total, cuts, running = len(ordered), [], 0.0
    for ratio in ratios[:-1]:
        running += ratio
        cuts.append(round(total * running))

    out, start = {}, 0
    for name, end in zip(SPLIT_NAMES, cuts + [total]):
        out[name] = ordered[start:end]
        start = end
    return out


def manifest(splits, seed, source, ratios=DEFAULT_RATIOS):
    return {
        "seed": seed,
        "source": source,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {name: len(files) for name, files in splits.items()},
        "total": sum(len(f) for f in splits.values()),
        "splits": splits,
    }


def write(manifest_data, out_dir=HERE, name=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name or f"{Path(manifest_data['source']).name}_seed{manifest_data['seed']}.json")
    path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    return path


def load(path):
    """Read a manifest back. Training and evaluation both come through here."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                        help="a known dataset under data/")
    parser.add_argument("--src", type=Path, default=None,
                        help="any directory, instead of a known dataset")
    parser.add_argument("--pattern", action="append", default=None,
                        help="glob for samples; repeatable (default: images)")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed recorded in the manifest; same seed, same split")
    parser.add_argument("--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if args.src is None and args.dataset is None:
        # Nothing named: use whichever known dataset actually has files in it.
        for name, (source, patterns) in DATASETS.items():
            if source.is_dir() and collect(source, patterns):
                args.dataset = name
                break
    if args.src is not None:
        source = args.src
        patterns = tuple(args.pattern or ("*.jpg", "*.jpeg", "*.png"))
    elif args.dataset is not None:
        source, patterns = DATASETS[args.dataset]
        patterns = tuple(args.pattern or patterns)
    else:
        print("no data found under data/. fetch some first:\n"
              "  bash data/visdrone/download_visdrone.sh\n"
              "  python data/visdrone/convert_to_yolo.py")
        return 1

    files = collect(source, patterns)
    if not files:
        print(f"no files matching {', '.join(patterns)} under {source}")
        return 1

    splits = split(files, tuple(args.ratios), args.seed)
    path = write(manifest(splits, args.seed, _relative(source), tuple(args.ratios)), args.out)

    print(f"  source  {_relative(source)}  ({len(files)} sample(s))")
    for name in SPLIT_NAMES:
        share = len(splits[name]) / len(files)
        print(f"  {name:<6}  {len(splits[name]):>7}  ({share:.1%})")
    print(f"  seed    {args.seed}")
    print(f"  wrote   {path}")
    return 0


def selfcheck():
    """Known answers. No dataset needed, nothing written outside a temp dir."""
    import tempfile

    files = [f"data/visdrone/train/images/{i:05d}.jpg" for i in range(100)]

    a = split(files, seed=0)
    b = split(files, seed=0)
    assert a == b, "the same seed must give the same split"
    assert split(files, seed=1) != a, "a different seed must give a different split"

    # Order in must not change the split: a directory listing is not stable
    # across machines, and the seed would otherwise mean two different things.
    assert split(list(reversed(files)), seed=0) == a

    assert [len(a[n]) for n in SPLIT_NAMES] == [70, 15, 15], [len(a[n]) for n in SPLIT_NAMES]

    everything = [f for name in SPLIT_NAMES for f in a[name]]
    assert len(everything) == len(files), "every sample lands somewhere"
    assert len(set(everything)) == len(files), "and in exactly one split"
    assert set(everything) == set(files), "and nothing was invented"

    # Ratios that do not divide evenly still account for every sample.
    odd = split(files[:7], ratios=(0.7, 0.15, 0.15), seed=3)
    assert sum(len(v) for v in odd.values()) == 7, odd
    assert abs(sum(DEFAULT_RATIOS) - 1.0) < 1e-9

    try:
        split(files, ratios=(0.5, 0.3, 0.3))
        raise AssertionError("ratios that do not sum to 1.0 must be refused")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images").mkdir()
        for i in range(12):
            (tmp / "images" / f"{i:03d}.jpg").write_bytes(b"x")
        (tmp / "images" / "notes.txt").write_text("not a sample")

        found = collect(tmp, ("*.jpg",))
        assert len(found) == 12, found
        assert all(not f.endswith(".txt") for f in found)

        data = manifest(split(found, seed=7), 7, "tmp")
        path = write(data, tmp, name="manifest.json")
        reloaded = load(path)
        assert reloaded["seed"] == 7 and reloaded["total"] == 12
        assert reloaded["splits"] == data["splits"], "a manifest round-trips"
        assert reloaded["counts"]["train"] == len(reloaded["splits"]["train"])

    print("  ok  same seed, same split — and a different seed differs")
    print("  ok  input order cannot change the split")
    print("  ok  every sample lands in exactly one split, nothing invented")
    print("  ok  ratios that do not divide evenly still account for everything")
    print("  ok  ratios that do not sum to 1.0 are refused")
    print("  ok  only matching files are collected")
    print("  ok  a manifest round-trips through JSON")
    print("\n7 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
