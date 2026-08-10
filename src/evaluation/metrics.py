"""Detection and geolocation metrics for the Phase 1 baseline harness.

Pure stdlib. The three metrics here are the ones every later phase's exit test
refers back to (docs/architecture.md, Phase 1):

  * mean average precision (mAP)
  * recall at a fixed false-alarm rate
  * geolocation error in metres

All three are derived from a single greedy matching pass, so a detection is
counted as a true positive by exactly the same rule everywhere.
"""

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from statistics import mean, median

from ..geometry import iou  # noqa: F401  (re-exported: metrics is where scoring code looks for it)

# Mean Earth radius (IUGG), metres.
EARTH_RADIUS_M = 6371008.8


@dataclass
class Match:
    """One detection, resolved against ground truth."""

    confidence: float
    label: str
    is_tp: bool
    geo_error_m: float | None = None


def haversine_m(p, q):
    """Great-circle distance between two (lat, lon) points, in metres."""
    lat1, lon1, lat2, lon2 = map(radians, (p[0], p[1], q[0], q[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(min(1.0, h)))


def match_detections(detections, ground_truth, iou_threshold=0.5):
    """Greedy match, highest confidence first. Each GT can be claimed once.

    Returns matches in descending confidence order — the order every curve
    below walks.
    """
    by_frame = {}
    for gt in ground_truth:
        by_frame.setdefault((gt.frame_id, gt.label), []).append(gt)

    claimed = set()
    matches = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        key = (det.frame_id, det.label)
        best_idx, best_iou = None, 0.0
        for idx, gt in enumerate(by_frame.get(key, ())):
            if (key, idx) in claimed:
                continue
            score = iou(det.box, gt.box)
            if score > best_iou:
                best_idx, best_iou = idx, score

        if best_idx is not None and best_iou >= iou_threshold:
            claimed.add((key, best_idx))
            gt = by_frame[key][best_idx]
            error = None
            if det.geo is not None and gt.geo is not None:
                error = haversine_m(det.geo, gt.geo)
            matches.append(Match(det.confidence, det.label, True, error))
        else:
            matches.append(Match(det.confidence, det.label, False))
    return matches


def average_precision(matches, n_gt):
    """All-point interpolated AP (VOC2010+/COCO convention).

    `matches` must already be in descending confidence order. Returns None when
    there is no ground truth to score against.
    """
    if n_gt == 0:
        return None
    recalls, precisions = [0.0], [1.0]
    tp = fp = 0
    for m in matches:
        tp, fp = tp + m.is_tp, fp + (not m.is_tp)
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))

    # Precision envelope: max precision at any recall >= this one.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum(
        (recalls[i + 1] - recalls[i]) * precisions[i + 1]
        for i in range(len(recalls) - 1)
    )


def mean_average_precision(matches, gt_counts):
    """mAP over classes that actually have ground truth."""
    per_class = {}
    for label, n_gt in gt_counts.items():
        ap = average_precision([m for m in matches if m.label == label], n_gt)
        if ap is not None:
            per_class[label] = ap
    return (mean(per_class.values()) if per_class else 0.0), per_class


def recall_at_far(matches, n_gt, n_frames, far_target):
    """Best recall reachable while false alarms per frame stay <= far_target.

    False-alarm rate is false positives per frame — the operationally useful
    form for search: how much does an operator get flagged per frame of video.
    FAR is monotonic as the confidence threshold drops, so the walk stops at
    the first crossing.
    """
    if n_gt == 0 or n_frames == 0:
        return 0.0, None, 0.0

    tp = fp = 0
    best_recall, threshold, far_at_best = 0.0, None, 0.0
    for m in matches:
        tp, fp = tp + m.is_tp, fp + (not m.is_tp)
        far = fp / n_frames
        if far > far_target:
            break
        if tp / n_gt >= best_recall:
            best_recall, threshold, far_at_best = tp / n_gt, m.confidence, far
    return best_recall, threshold, far_at_best


def geolocation_error(matches):
    """Distribution of geolocation error over true positives carrying a fix."""
    errors = sorted(m.geo_error_m for m in matches if m.is_tp and m.geo_error_m is not None)
    if not errors:
        return {"n": 0, "mean_m": None, "median_m": None, "p90_m": None, "max_m": None}
    # ponytail: nearest-rank p90, fine at this n; interpolate if n gets small.
    p90 = errors[min(len(errors) - 1, int(0.9 * len(errors)))]
    return {
        "n": len(errors),
        "mean_m": mean(errors),
        "median_m": median(errors),
        "p90_m": p90,
        "max_m": errors[-1],
    }


def evaluate(detections, ground_truth, n_frames, iou_threshold=0.5, far_target=0.1):
    """Run all three Phase 1 metrics over one split."""
    matches = match_detections(detections, ground_truth, iou_threshold)
    gt_counts = {}
    for gt in ground_truth:
        gt_counts[gt.label] = gt_counts.get(gt.label, 0) + 1

    m_ap, per_class = mean_average_precision(matches, gt_counts)
    recall, threshold, far = recall_at_far(matches, len(ground_truth), n_frames, far_target)
    return {
        "n_frames": n_frames,
        "n_gt": len(ground_truth),
        "n_detections": len(detections),
        "iou_threshold": iou_threshold,
        "mAP": m_ap,
        "ap_per_class": per_class,
        "far_target": far_target,
        "recall_at_far": recall,
        "far_threshold": threshold,
        "far_achieved": far,
        "geolocation": geolocation_error(matches),
    }
