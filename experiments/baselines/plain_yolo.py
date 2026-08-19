"""Baseline 1: the detector on its own.

    python experiments/baselines/plain_yolo.py
    python experiments/baselines/plain_yolo.py --seeds 42,123,456
    python experiments/baselines/plain_yolo.py --selfcheck

No fusion, no tracker, no agents, no decision chain — the RGB detector's raw
per-frame boxes, scored against the same ground truth everything else in this
repository is scored against. This is the floor the rest of the system has to
justify itself above.

What "raw" means here
---------------------
`tuning/scenario.py` keeps the RGB detector's output *before* weighted box
fusion and *before* BoT-SORT, which is exactly this baseline. Nothing is
re-implemented: the same sortie runs, and this scores the earlier tap. Two
numbers therefore mean what they say when compared with the ablation table —
same frames, same subjects, same seed.

The detector is the stub
------------------------
There are no trained weights in this repository, and the scenario projects
subjects into a frame that does not exist as pixels, so a real checkpoint could
not be handed anything to look at. Every row this writes is labelled
`detector=stub`, and the script **refuses to run under PERCEPTION_MODE=real**
rather than produce a CSV that claims otherwise. Real weights are measured by
`train_perception.py`, which runs them over real validation images.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate, match_detections    # noqa: E402
from src.perception.models import STUB, perception_mode          # noqa: E402
from src.tuning.params import load_params                        # noqa: E402
from src.tuning.scenario import run as run_scenario              # noqa: E402

RESULTS = ROOT / "experiments" / "results"
DEFAULT_SEEDS = (0,)


def detector_precision_recall(detections, ground_truth, iou_threshold=0.5):
    """Precision and recall over every detection, at no confidence threshold.

    Deliberately unthresholded: a detector's precision is a curve, not a number,
    and picking an operating point here would quietly flatter or punish the
    baseline. The harness's `recall_at_far` is the thresholded figure, and it is
    reported alongside.
    """
    matches = match_detections(detections, ground_truth, iou_threshold)
    true_positives = sum(1 for m in matches if m.is_tp)
    return {
        "precision": round(true_positives / len(matches), 4) if matches else 0.0,
        "recall": round(true_positives / len(ground_truth), 4) if ground_truth else 0.0,
        "true_positives": true_positives,
        "false_positives": len(matches) - true_positives,
        "false_negatives": len(ground_truth) - true_positives,
    }


def run(seed=0, frames=6):
    """One sortie, scored at the detector tap."""
    params = load_params()
    result = run_scenario(params, seed=seed, frames=frames)
    _, ground_truth = result.evaluation_inputs()

    from src.evaluation.dataset import Detection
    raw = [Detection.from_clue(clue) for clue in result.raw_detections]

    harness = evaluate(raw, ground_truth, n_frames=result.frames,
                       far_target=params.target_far)
    row = {
        "baseline": "plain_yolo",
        "detector": "stub",
        "seed": seed,
        "frames": result.frames,
        "detections": len(raw),
        "ground_truth": len(ground_truth),
        "mAP": round(harness["mAP"], 4),
        "recall_at_far": round(harness["recall_at_far"], 4),
    }
    row.update(detector_precision_recall(raw, ground_truth))
    return row


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--out", type=Path, default=RESULTS / "baseline_plain_yolo.csv")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if perception_mode() != STUB:
        raise SystemExit(
            "PERCEPTION_MODE=real: this baseline runs the projected-target scenario,\n"
            "  which has no pixels for a real checkpoint to look at. Writing a row\n"
            "  labelled 'real' from a stub would be the one thing worth avoiding.\n"
            "  Measure real weights with:  python train_perception.py --mode rgb"
        )

    seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s]
    rows = [run(seed=seed, frames=args.frames) for seed in seeds]

    print(f"\n  plain YOLO — no fusion, no tracker, no agents, no decision chain")
    print(f"  detector: stub (no trained weights in this repository)\n")
    print(f"  {'seed':<8}{'mAP':>10}{'recall@FAR':>14}{'precision':>12}{'recall':>10}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}")
    for row in rows:
        print(f"  {row['seed']:<8}{row['mAP']:>10.4f}{row['recall_at_far']:>14.4f}"
              f"{row['precision']:>12.4f}{row['recall']:>10.4f}"
              f"{row['true_positives']:>6}{row['false_positives']:>6}"
              f"{row['false_negatives']:>6}")

    print(f"\n  wrote {write_csv(rows, args.out)}\n")
    return 0


def selfcheck():
    """The scoring, on hand-made detections — no sortie needed."""
    from src.evaluation.dataset import Detection, GroundTruth

    truth = [GroundTruth("f1", "person", (10.0, 10.0, 30.0, 60.0)),
             GroundTruth("f1", "person", (100.0, 10.0, 120.0, 60.0)),
             GroundTruth("f2", "person", (10.0, 10.0, 30.0, 60.0))]

    # One exact hit, one miss by position, one frame with nothing found.
    detections = [Detection("f1", "person", (11.0, 11.0, 31.0, 61.0), 0.9),
                  Detection("f1", "person", (400.0, 400.0, 420.0, 450.0), 0.5)]
    scored = detector_precision_recall(detections, truth)
    assert scored["true_positives"] == 1, scored
    assert scored["false_positives"] == 1 and scored["false_negatives"] == 2
    assert scored["precision"] == 0.5, scored
    assert round(scored["recall"], 4) == round(1 / 3, 4), scored

    # Nothing detected is recall 0 and precision 0 — never a division by zero.
    empty = detector_precision_recall([], truth)
    assert empty == {"precision": 0.0, "recall": 0.0, "true_positives": 0,
                     "false_positives": 0, "false_negatives": 3}, empty

    # A perfect run is 1.0 both ways.
    perfect = detector_precision_recall(
        [Detection(gt.frame_id, gt.label, gt.box, 0.99) for gt in truth], truth)
    assert perfect["precision"] == perfect["recall"] == 1.0, perfect

    # Real mode is refused rather than mislabelled.
    previous = os.environ.get("PERCEPTION_MODE")
    os.environ["PERCEPTION_MODE"] = "real"
    try:
        main(["--out", str(RESULTS / "unused.csv")])
        raise AssertionError("real mode must be refused by this baseline")
    except SystemExit as e:
        assert "train_perception.py" in str(e)
    finally:
        if previous is None:
            os.environ.pop("PERCEPTION_MODE", None)
        else:
            os.environ["PERCEPTION_MODE"] = previous

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_csv([{"baseline": "plain_yolo", "detector": "stub", "mAP": 0.5}],
                         Path(tmp) / "sub" / "out.csv")
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "baseline,detector,mAP"
        assert "stub" in text, "every row says which detector produced it"

    print("  ok  precision and recall count TPs, FPs and FNs the way they read")
    print("  ok  an empty detector is 0.0, not a division by zero")
    print("  ok  a perfect run is 1.0 both ways")
    print("  ok  PERCEPTION_MODE=real is refused, not silently run on stubs")
    print("  ok  every row records which detector produced it")
    print("\n5 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
