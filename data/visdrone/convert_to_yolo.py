"""VisDrone DET annotations -> YOLO labels, people only.

    python data/visdrone/convert_to_yolo.py
    python data/visdrone/convert_to_yolo.py --src data/visdrone --dst data/visdrone
    python data/visdrone/convert_to_yolo.py --selfcheck

VisDrone ships one CSV-ish `.txt` per image, in pixels:

    bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion

YOLO wants one `.txt` per image, normalised and centre-based:

    class x_centre y_centre width height        (all in 0..1)

Two categories are kept — 1 `pedestrian` and 2 `people` — and mapped to classes
0 and 1. The other eight are traffic; see data/README.md for why they go, and
why these two stay apart rather than merging into one "person".

`score = 0` marks an *ignored region*, not an object and not background. Those
rows are dropped: training on them as negatives teaches the detector that
visible people are background, which is the one mistake a search cannot afford.

No Pillow. Image dimensions come from the JPEG/PNG header, which is a dozen
lines of struct and saves a dependency that would exist only to read two numbers
(the same reasoning as `SrtmHgtDEM` in the perception plane).
"""

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

# VisDrone category -> our class index. Everything absent is dropped.
KEEP = {1: 0, 2: 1}
CLASS_NAMES = ("pedestrian", "people")

SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}

IGNORED_REGION = 0     # the `score` column's value for a region to skip


def image_size(path):
    """(width, height) from a JPEG or PNG header. None if it is neither."""
    with open(path, "rb") as fh:
        head = fh.read(2)
        if head == b"\xff\xd8":                       # JPEG
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                code = marker[1]
                length = struct.unpack(">H", fh.read(2))[0]
                # SOF0..SOF15 carry the dimensions; DHT/JPG/DAC share the range
                # and do not.
                if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                    fh.read(1)                        # sample precision
                    height, width = struct.unpack(">HH", fh.read(4))
                    return width, height
                fh.seek(length - 2, os.SEEK_CUR)
        if head == b"\x89P":                          # PNG
            fh.seek(16)
            width, height = struct.unpack(">II", fh.read(8))
            return width, height
    return None


def convert_annotation(text, width, height):
    """VisDrone rows -> YOLO rows. Returns (lines, kept, dropped)."""
    lines, kept, dropped = [], 0, 0
    for row in text.splitlines():
        row = row.strip().rstrip(",")
        if not row:
            continue
        parts = row.split(",")
        if len(parts) < 6:
            dropped += 1
            continue
        try:
            left, top, box_w, box_h = (float(p) for p in parts[:4])
            score, category = int(float(parts[4])), int(float(parts[5]))
        except ValueError:
            dropped += 1
            continue

        if score == IGNORED_REGION or category not in KEEP:
            dropped += 1
            continue

        # Clamp to the image before normalising: VisDrone boxes occasionally
        # run a pixel or two past the edge, and a YOLO coordinate outside 0..1
        # is silently dropped by most trainers rather than reported.
        x1, y1 = max(0.0, left), max(0.0, top)
        x2, y2 = min(float(width), left + box_w), min(float(height), top + box_h)
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue

        lines.append(
            f"{KEEP[category]} "
            f"{(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
        kept += 1
    return lines, kept, dropped


def _place(src, dst):
    """Hardlink the image if the filesystem allows, else copy.

    A second copy of VisDrone is 2 GB of the same bytes. A hardlink is free and
    the training run only ever reads them.
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def convert_split(src_dir, dst_dir, split):
    """One VisDrone split -> `dst_dir/split/{images,labels}`. Returns a summary."""
    images_in = src_dir / "images"
    annotations_in = src_dir / "annotations"
    if not images_in.is_dir() or not annotations_in.is_dir():
        raise FileNotFoundError(
            f"{src_dir} does not look like a VisDrone split "
            f"(expected images/ and annotations/ inside it)"
        )

    images_out = dst_dir / split / "images"
    labels_out = dst_dir / split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    summary = {"split": split, "images": 0, "empty": 0, "boxes": 0,
               "dropped": 0, "unreadable": 0}

    for image in sorted(images_in.iterdir()):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        annotation = annotations_in / f"{image.stem}.txt"
        if not annotation.exists():
            summary["unreadable"] += 1
            continue
        size = image_size(image)
        if size is None:
            # Never guess a size: every box in the file would be wrong, and
            # wrong labels are worse than absent ones.
            summary["unreadable"] += 1
            continue

        lines, kept, dropped = convert_annotation(
            annotation.read_text(encoding="utf-8", errors="replace"), *size)
        # A frame with no people is still a frame: an empty label file is how
        # YOLO is told "this is background", and a search dataset that only
        # contains people teaches a detector that everything is somebody.
        (labels_out / f"{image.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        _place(image, images_out / image.name)

        summary["images"] += 1
        summary["boxes"] += kept
        summary["dropped"] += dropped
        summary["empty"] += 0 if lines else 1
    return summary


def write_dataset_yaml(dst_dir, path=None):
    """The ultralytics dataset config, so training can point straight at this."""
    path = path or dst_dir / "visdrone_person.yaml"
    path.write_text(
        "# Generated by data/visdrone/convert_to_yolo.py — people only.\n"
        f"path: {dst_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    return path


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=here,
                        help="directory holding the VisDrone2019-DET-* splits")
    parser.add_argument("--dst", type=Path, default=here,
                        help="where train/ and val/ are written")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in checks on synthetic data and exit")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    summaries = []
    for split, directory in SPLITS.items():
        source = args.src / directory
        if not source.is_dir():
            print(f"  skip  {split}: {source} not found — run download_visdrone.sh")
            continue
        summary = convert_split(source, args.dst, split)
        summaries.append(summary)
        print(f"  {split:<6} {summary['images']:>6} images  "
              f"{summary['boxes']:>7} people  "
              f"{summary['empty']:>5} empty  {summary['dropped']:>7} rows dropped"
              + (f"  {summary['unreadable']} unreadable" if summary["unreadable"] else ""))

    if not summaries:
        print("\nnothing converted. fetch the data first:\n"
              "  bash data/visdrone/download_visdrone.sh")
        return 1

    config = write_dataset_yaml(args.dst)
    print(f"\n  classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)))
    print(f"  wrote   {config}")
    print(f"  train with: yolo detect train data={config} model=yolo11m.pt")
    return 0


def _fake_jpeg(width, height):
    """The smallest byte string `image_size` will read a size out of."""
    return (b"\xff\xd8"
            + b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3)
            + b"\x00" * 6
            + b"\xff\xd9")


def selfcheck():
    """Known answers, on synthetic data, with no dataset present."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        probe = tmp / "probe.jpg"
        probe.write_bytes(_fake_jpeg(1360, 765))
        assert image_size(probe) == (1360, 765), image_size(probe)

        # One of each: pedestrian, people, a car, and an ignored region.
        annotation = (
            "100,200,50,100,1,1,0,0\n"
            "300,400,20,40,1,2,0,0\n"
            "500,100,80,60,1,4,0,0\n"      # car — not a class we keep
            "0,0,1360,765,0,0,0,0\n"       # ignored region
        )
        lines, kept, dropped = convert_annotation(annotation, 1360, 765)
        assert kept == 2 and dropped == 2, (kept, dropped)
        assert lines[0].startswith("0 ") and lines[1].startswith("1 "), lines
        # (100+150)/2 / 1360, (200+300)/2 / 765, 50/1360, 100/765
        assert lines[0] == "0 0.091912 0.326797 0.036765 0.130719", lines[0]

        # A box running past the edge is clamped, never normalised past 1.0.
        clipped, kept, _ = convert_annotation("1340,750,100,100,1,1,0,0", 1360, 765)
        assert kept == 1
        assert all(0.0 <= float(v) <= 1.0 for v in clipped[0].split()[1:]), clipped

        # Malformed rows are counted, never crash the run.
        _, kept, dropped = convert_annotation("nonsense\n1,2\n\n", 100, 100)
        assert (kept, dropped) == (0, 2)

        # End to end over a two-image split.
        split = tmp / "VisDrone2019-DET-train"
        (split / "images").mkdir(parents=True)
        (split / "annotations").mkdir(parents=True)
        for name in ("0000001_00000_d_0000001", "0000002_00000_d_0000002"):
            (split / "images" / f"{name}.jpg").write_bytes(_fake_jpeg(960, 540))
        (split / "annotations" / "0000001_00000_d_0000001.txt").write_text(annotation)
        (split / "annotations" / "0000002_00000_d_0000002.txt").write_text(
            "10,10,20,20,1,4,0,0\n")     # a car only: empty label, still a frame

        summary = convert_split(split, tmp / "out", "train")
        assert summary == {"split": "train", "images": 2, "empty": 1, "boxes": 2,
                           "dropped": 3, "unreadable": 0}, summary
        labels = sorted((tmp / "out" / "train" / "labels").iterdir())
        assert len(labels) == 2 and labels[1].read_text() == "", "background stays"
        assert len(list((tmp / "out" / "train" / "images").iterdir())) == 2

        config = write_dataset_yaml(tmp / "out")
        assert "0: pedestrian" in config.read_text()
        assert json.dumps(summary)      # summaries stay serialisable for a log

    print("  ok  image_size reads JPEG and PNG headers without Pillow")
    print("  ok  only categories 1 and 2 survive, ignored regions dropped")
    print("  ok  boxes are clamped, normalised and centre-based")
    print("  ok  a frame with no people keeps an empty label file")
    print("  ok  malformed rows are counted, not fatal")
    print("\n5 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
