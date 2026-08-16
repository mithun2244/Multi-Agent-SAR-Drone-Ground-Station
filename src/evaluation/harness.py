"""Phase 1 evaluation harness — the RGB-only baseline run.

Exit criteria (docs/architecture.md, Phase 1):
    The harness reports mAP, recall at a fixed false-alarm rate, and
    geolocation error for a baseline RGB-only run.

Run it:
    python -m src.evaluation.harness
    python -m src.evaluation.harness --split train --far-target 0.05
"""

import argparse
import json

from .dataset import Detection, build_splits, load_clues, load_split, mock_rgb_detector
from .metrics import evaluate

# Calibration knobs. Every one of these is a real tuning decision, not a
# constant: IoU threshold sets how strict a "hit" is, far_target is the operator
# tolerance for false alarms per frame, and seed pins the split so numbers are
# comparable run to run.
DEFAULTS = {
    "n_frames": 120,
    "val_fraction": 0.3,
    "seed": 0,
    "iou_threshold": 0.5,
    "far_target": 0.1,
}


def run_baseline(
    split,
    detector=mock_rgb_detector,
    clues=None,
    iou_threshold=0.5,
    far_target=0.1,
    seed=0,
    case_id=None,
):
    """Score clues against one split.

    `detector` is any callable (split, seed) -> list[ClueContract], so a real
    YOLO11m producer drops straight in. Pass `clues` instead to score whatever
    was captured off the bus.

    `case_id` scopes the run to one search — clues carry the case they belong
    to, and scoring two cases together would fuse unrelated pictures.
    """
    if clues is None:
        clues = detector(split, seed=seed)

    detections = [Detection.from_clue(c) for c in clues]
    skipped = 0
    if case_id is not None:
        kept = [d for d in detections if d.case_id == case_id]
        skipped = len(detections) - len(kept)
        detections = kept

    results = evaluate(
        detections,
        split.ground_truth,
        n_frames=split.n_frames,
        iou_threshold=iou_threshold,
        far_target=far_target,
    )
    results["case_ids"] = sorted({d.case_id for d in detections if d.case_id})
    results["skipped_other_case"] = skipped
    return results


def format_report(results, title):
    geo = results["geolocation"]
    fmt = lambda v: "n/a" if v is None else f"{v:.1f} m"  # noqa: E731
    thr = results["far_threshold"]
    lines = [
        "",
        f"  {title}",
        "  " + "-" * 56,
        f"  frames {results['n_frames']:<6} ground truth {results['n_gt']:<6}"
        f" clues scored {results['n_detections']}",
        f"  case   {', '.join(results.get('case_ids') or ['n/a'])}"
        + (f"   ({results['skipped_other_case']} clue(s) from other cases skipped)"
           if results.get("skipped_other_case") else ""),
        "",
        f"  mAP @ IoU {results['iou_threshold']:.2f}        {results['mAP']:.4f}",
    ]
    for label, ap in sorted(results["ap_per_class"].items()):
        lines.append(f"    AP [{label}]{'':<12}{ap:.4f}")
    lines += [
        f"  Recall @ {results['far_target']:.2f} FP/frame  {results['recall_at_far']:.4f}"
        f"   (conf >= {'n/a' if thr is None else f'{thr:.3f}'},"
        f" achieved {results['far_achieved']:.3f} FP/frame)",
        "",
        f"  Geolocation error (n={geo['n']})",
        f"    mean {fmt(geo['mean_m'])}   median {fmt(geo['median_m'])}"
        f"   p90 {fmt(geo['p90_m'])}   max {fmt(geo['max_m'])}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="validation", choices=("train", "validation"))
    p.add_argument("--data", help="JSON split file; omitted means the mock dataset")
    p.add_argument("--clues", help="JSON array of ClueContract; omitted means the mock detector")
    p.add_argument("--case-id", help="score only clues belonging to this case")
    p.add_argument("--n-frames", type=int, default=DEFAULTS["n_frames"])
    p.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--iou-threshold", type=float, default=DEFAULTS["iou_threshold"])
    p.add_argument("--far-target", type=float, default=DEFAULTS["far_target"],
                   help="tolerated false alarms per frame")
    p.add_argument("--json", action="store_true", help="emit raw results as JSON")
    args = p.parse_args(argv)

    if args.data:
        split = load_split(args.data, args.split)
        source = args.data
    else:
        split = build_splits(args.n_frames, args.val_fraction, args.seed)[args.split]
        source = f"mock dataset (seed={args.seed})"

    results = run_baseline(
        split,
        clues=load_clues(args.clues) if args.clues else None,
        iou_threshold=args.iou_threshold,
        far_target=args.far_target,
        seed=args.seed,
        case_id=args.case_id,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_report(results, f"RGB-only baseline — {split.name} split — {source}"))
    return results


if __name__ == "__main__":
    main()
