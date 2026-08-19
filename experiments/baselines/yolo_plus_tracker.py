"""Baseline 2: detector plus BoT-SORT, and nothing else.

    python experiments/baselines/yolo_plus_tracker.py
    python experiments/baselines/yolo_plus_tracker.py --seeds 42,123,456
    python experiments/baselines/yolo_plus_tracker.py --selfcheck

One sensor, one tracker: no weighted box fusion, no agents, no decision chain.
The step up from `plain_yolo.py` is *identity* — boxes become tracks that
persist across frames — so this is scored on tracking metrics as well as
detection ones.

Single sensor is enforced, not assumed: `ABLATION_DISABLE_SENSORS=lidar` drops
the second feed, and the pipeline is already sensor-agnostic, so with one feed
weighted box fusion does not run at all. That is the honest way to say "no
fusion" — not a flag that skips a merge, but an airframe that has nothing to
merge.

MOTA and IDF1
-------------
Both are computed here rather than pulled from a library, because the whole
scoring path in this repository is stdlib and a tracking-metrics package would
be the first dependency that exists only for a table.

    MOTA = 1 - (FN + FP + IDSW) / |GT|
    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)

MOTA can go negative — that is not a bug, it is what "more mistakes than there
were people" looks like, and clamping it at zero would hide the case worth
seeing.

ponytail: identity assignment for IDF1 is greedy, not the Hungarian matching the
IDF1 paper specifies. With the two or three subjects a search scenario holds
they agree; with a crowd they can differ. Same ceiling and the same upgrade path
as `_associate` in `perception/tracking.py` — `scipy.optimize.linear_sum_assignment`.
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

from src.evaluation.metrics import evaluate                      # noqa: E402
from src.geometry import iou                                     # noqa: E402
from src.perception.models import STUB, perception_mode          # noqa: E402
from src.tuning.params import load_params                        # noqa: E402
from src.utils.ablation import DISABLE_SENSORS                   # noqa: E402

RESULTS = ROOT / "experiments" / "results"
DEFAULT_SEEDS = (0,)
IOU_THRESHOLD = 0.5


def match_frame(tracks, truths, iou_threshold=IOU_THRESHOLD):
    """Greedy IoU match within one frame. Returns (pairs, unmatched counts)."""
    candidates = []
    for ti, (track_id, box) in enumerate(tracks):
        for gi, (subject_id, gt_box) in enumerate(truths):
            overlap = iou(box, gt_box)
            if overlap >= iou_threshold:
                candidates.append((-overlap, ti, gi))

    candidates.sort()
    pairs, used_tracks, used_truths = [], set(), set()
    for _, ti, gi in candidates:
        if ti in used_tracks or gi in used_truths:
            continue
        used_tracks.add(ti)
        used_truths.add(gi)
        pairs.append((truths[gi][0], tracks[ti][0]))
    return pairs, len(tracks) - len(pairs), len(truths) - len(pairs)


def tracking_metrics(frames, iou_threshold=IOU_THRESHOLD):
    """MOTA, IDF1 and their parts. `frames` is [(tracks, truths)] in order.

    `tracks` is [(track_id, box)] and `truths` is [(subject_id, box)].
    """
    false_positives = false_negatives = switches = 0
    total_truth = total_tracks = 0
    last_seen = {}            # subject_id -> the track id that held it last
    co_occurrence = {}        # (subject_id, track_id) -> frames matched

    for tracks, truths in frames:
        total_truth += len(truths)
        total_tracks += len(tracks)
        pairs, unmatched_tracks, unmatched_truths = match_frame(tracks, truths, iou_threshold)
        false_positives += unmatched_tracks
        false_negatives += unmatched_truths

        for subject_id, track_id in pairs:
            key = (subject_id, track_id)
            co_occurrence[key] = co_occurrence.get(key, 0) + 1
            # An identity switch is a subject that *was* being tracked coming
            # back under a different id — not a subject appearing for the first
            # time, which is simply a track starting.
            if last_seen.get(subject_id) not in (None, track_id):
                switches += 1
            last_seen[subject_id] = track_id

    mota = (1.0 - (false_negatives + false_positives + switches) / total_truth
            if total_truth else None)

    # IDF1: give each subject the track it shared most frames with, best pair
    # first, and no track to two subjects. See the ponytail note above.
    id_true_positives, claimed_subjects, claimed_tracks = 0, set(), set()
    for (subject_id, track_id), count in sorted(co_occurrence.items(),
                                                key=lambda kv: (-kv[1], kv[0])):
        if subject_id in claimed_subjects or track_id in claimed_tracks:
            continue
        claimed_subjects.add(subject_id)
        claimed_tracks.add(track_id)
        id_true_positives += count

    id_false_positives = total_tracks - id_true_positives
    id_false_negatives = total_truth - id_true_positives
    denominator = 2 * id_true_positives + id_false_positives + id_false_negatives
    idf1 = (2 * id_true_positives / denominator) if denominator else None

    return {
        "MOTA": None if mota is None else round(mota, 4),
        "IDF1": None if idf1 is None else round(idf1, 4),
        "id_switches": switches,
        "track_false_positives": false_positives,
        "track_false_negatives": false_negatives,
        "gt_boxes": total_truth,
        "track_boxes": total_tracks,
    }


def run(seed=0, frames=6):
    """One single-sensor sortie: detector, tracker, and the scoring."""
    from src.tuning.scenario import run as run_scenario

    # Set around the sortie, then put back. `keep_feeds` reads this when the
    # feeds are assembled, so it has to be set here — but leaving it set would
    # silently make every later call in this process single-sensor too, which is
    # the kind of leak that turns one baseline into every result.
    previous = os.environ.get(DISABLE_SENSORS)
    os.environ[DISABLE_SENSORS] = "lidar"
    try:
        params = load_params()
        result = run_scenario(params, seed=seed, frames=frames)
    finally:
        if previous is None:
            os.environ.pop(DISABLE_SENSORS, None)
        else:
            os.environ[DISABLE_SENSORS] = previous
    detections, ground_truth = result.evaluation_inputs()

    harness = evaluate(detections, ground_truth, n_frames=result.frames,
                       far_target=params.target_far)

    # Tracked boxes and truth, per frame, in the order they were flown.
    by_frame = {}
    for clue in result.detections:
        by_frame.setdefault(clue.frame_id, ([], []))[0].append(
            (clue.agent_metadata.get("track_id"), tuple(clue.spatial_context.bounding_box)))
    for frame_id, subject in result.truth_boxes:
        by_frame.setdefault(frame_id, ([], []))[1].append(
            (subject.subject_id, tuple(subject.bounding_box)))

    row = {
        "baseline": "yolo_plus_tracker",
        "detector": "stub",
        "seed": seed,
        "frames": result.frames,
        "mAP": round(harness["mAP"], 4),
        "recall_at_far": round(harness["recall_at_far"], 4),
        "tracks": len({t for tracks, _ in by_frame.values() for t, _ in tracks}),
    }
    row.update(tracking_metrics([by_frame[f] for f in sorted(by_frame)]))
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
    parser.add_argument("--out", type=Path,
                        default=RESULTS / "baseline_yolo_plus_tracker.csv")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if perception_mode() != STUB:
        raise SystemExit(
            "PERCEPTION_MODE=real: this baseline runs the projected-target scenario,\n"
            "  which has no pixels for a real checkpoint to look at.\n"
            "  Measure real weights with:  python train_perception.py --mode rgb"
        )

    seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s]
    rows = [run(seed=seed, frames=args.frames) for seed in seeds]

    print("\n  YOLO + BoT-SORT — one sensor, so no fusion; no agents, no decision chain")
    print("  detector: stub (no trained weights in this repository)\n")
    print(f"  {'seed':<8}{'mAP':>10}{'MOTA':>10}{'IDF1':>10}{'IDSW':>7}"
          f"{'FP':>6}{'FN':>6}{'tracks':>8}")
    for row in rows:
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"      # noqa: E731
        print(f"  {row['seed']:<8}{row['mAP']:>10.4f}{fmt(row['MOTA']):>10}"
              f"{fmt(row['IDF1']):>10}{row['id_switches']:>7}"
              f"{row['track_false_positives']:>6}{row['track_false_negatives']:>6}"
              f"{row['tracks']:>8}")

    print(f"\n  wrote {write_csv(rows, args.out)}\n")
    return 0


def selfcheck():
    """MOTA and IDF1 on hand-built frames, where the answer is arithmetic."""
    box_a, box_b = (10.0, 10.0, 30.0, 60.0), (100.0, 10.0, 120.0, 60.0)

    # Perfect tracking: one subject, one track, three frames.
    perfect = tracking_metrics([([("1", box_a)], [("s1", box_a)])] * 3)
    assert perfect["MOTA"] == 1.0 and perfect["IDF1"] == 1.0, perfect
    assert perfect["id_switches"] == 0

    # A missed frame: 1 FN out of 3 -> MOTA 2/3, and IDF1 penalises the gap.
    missed = tracking_metrics([([("1", box_a)], [("s1", box_a)]),
                               ([], [("s1", box_a)]),
                               ([("1", box_a)], [("s1", box_a)])])
    assert round(missed["MOTA"], 4) == round(2 / 3, 4), missed
    assert missed["track_false_negatives"] == 1 and missed["IDF1"] == 0.8, missed

    # A phantom track: 1 FP, no FN.
    phantom = tracking_metrics([([("1", box_a), ("9", box_b)], [("s1", box_a)])])
    assert phantom["track_false_positives"] == 1 and phantom["MOTA"] == 0.0, phantom

    # An identity switch: same subject, new track id in frame two.
    switched = tracking_metrics([([("1", box_a)], [("s1", box_a)]),
                                 ([("2", box_a)], [("s1", box_a)])])
    assert switched["id_switches"] == 1, switched
    assert switched["MOTA"] == 0.5, "one switch over two ground-truth boxes"
    # IDF1 keeps only the larger half of a split identity, so it drops below 1.0
    # even though every frame was matched — which is the point of having it.
    assert switched["IDF1"] == 0.5, switched

    # A subject appearing for the first time is not a switch.
    appearing = tracking_metrics([([], []), ([("1", box_a)], [("s1", box_a)])])
    assert appearing["id_switches"] == 0, appearing

    # MOTA goes negative when the mistakes outnumber the people, and is not
    # clamped: that is the case worth seeing.
    awful = tracking_metrics([([("1", box_b), ("2", box_b)], [("s1", box_a)])])
    assert awful["MOTA"] < 0, awful

    # Two subjects, two tracks, no crossing: both identities survive.
    two = tracking_metrics([([("1", box_a), ("2", box_b)],
                             [("s1", box_a), ("s2", box_b)])] * 2)
    assert two["MOTA"] == 1.0 and two["IDF1"] == 1.0, two

    # Nothing at all is not a score.
    assert tracking_metrics([])["MOTA"] is None

    # A box that overlaps too little is not a match.
    apart = tracking_metrics([([("1", (200.0, 200.0, 220.0, 250.0))], [("s1", box_a)])])
    assert apart["track_false_positives"] == 1 and apart["track_false_negatives"] == 1

    print("  ok  perfect tracking is MOTA 1.0 and IDF1 1.0")
    print("  ok  a missed frame and a phantom track each cost what they should")
    print("  ok  an identity switch is counted, and a first appearance is not")
    print("  ok  IDF1 falls when one subject is split across two track ids")
    print("  ok  MOTA is allowed to go negative rather than being clamped")
    print("  ok  an empty run scores nothing instead of zero")
    print("\n6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
