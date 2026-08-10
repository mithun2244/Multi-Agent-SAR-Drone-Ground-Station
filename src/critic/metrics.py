"""Metrics the critic scores a picture with. Pure functions, no state.

Four questions, each with its own number:

  * did we find them?          matching + recall
  * where did we say they were? geolocation residuals in metres
  * did we look there first?    NDCG and Kendall's tau over the ranking
  * how much did we cry wolf?   a rank-weighted false-positive penalty

Ranking is scored separately from detection on purpose. A system that finds
every subject but buries the real one under three phantoms has good recall and
is useless to a team with two hours of daylight left — the ranking metrics are
what notice that.
"""

import math

from ..geometry import ground_distance_m, iou

DEFAULT_MATCH_RADIUS_M = 100.0


def match_targets(targets, subjects, match_radius_m=DEFAULT_MATCH_RADIUS_M):
    """Pair ranked targets with the subjects they actually correspond to.

    Greedy from the top of the ranking down, nearest first, one subject per
    target. Working down the ranking rather than by distance is deliberate: it
    scores the picture as an operator would work it, top entry first.

    Returns (matches, unmatched_targets, unmatched_subjects) where matches are
    (rank, target, subject, distance_m) with rank 1-based.
    """
    matches = []
    remaining = list(subjects)
    unmatched_targets = []

    for rank, target in enumerate(targets, 1):
        if not getattr(target, "located", False):
            # A target nobody could place cannot be matched by distance. It is
            # not a false positive either — just unscorable on position.
            unmatched_targets.append((rank, target))
            continue

        best, best_distance = None, match_radius_m
        for subject in remaining:
            distance = ground_distance_m(target.position, subject.position)
            if distance <= best_distance:
                best, best_distance = subject, distance

        if best is None:
            unmatched_targets.append((rank, target))
        else:
            remaining.remove(best)
            matches.append((rank, target, best, round(best_distance, 3)))

    return matches, unmatched_targets, tuple(remaining)


def geolocation_residuals(matches):
    """Distance in metres between where we said and where they were."""
    distances = sorted(distance for _, _, _, distance in matches)
    if not distances:
        return {"n": 0, "mean_m": None, "median_m": None, "p90_m": None, "max_m": None}
    return {
        "n": len(distances),
        "mean_m": round(sum(distances) / len(distances), 3),
        "median_m": distances[len(distances) // 2],
        "p90_m": distances[min(len(distances) - 1, int(0.9 * len(distances)))],
        "max_m": distances[-1],
    }


def detection_iou(matches):
    """Mean box IoU over matched pairs that both carry a pixel extent.

    None when no pair has boxes on both sides — a picture built from geolocated
    tracks alone has no pixels to overlap, and reporting 0.0 there would read as
    "we got the box wrong" rather than "there was no box".
    """
    scores = [
        iou(target.bounding_box, subject.bounding_box)
        for _, target, subject, _ in matches
        if getattr(target, "bounding_box", None) and subject.bounding_box
    ]
    return round(sum(scores) / len(scores), 6) if scores else None


def relevance_vector(targets, matches):
    """Graded relevance per ranked target: the subject's priority, or 0.

    This is the ideal the ranking metrics are scored against — a phantom is
    worth nothing, and a high-priority subject is worth more than a low one.
    """
    by_rank = {rank: subject.priority for rank, _, subject, _ in matches}
    return [by_rank.get(rank, 0.0) for rank in range(1, len(targets) + 1)]


def dcg(relevances, k=None):
    scored = relevances[:k] if k else relevances
    return sum(rel / math.log2(position + 1) for position, rel in enumerate(scored, 1))


def ndcg(relevances, k=None):
    """How close the ranking is to the best possible ordering of the same items.

    1.0 means every real subject sits above every phantom. Returns None when
    nothing is relevant — there is no ordering to be right or wrong about.
    """
    ideal = dcg(sorted(relevances, reverse=True), k)
    if ideal <= 0.0:
        return None
    return round(dcg(relevances, k) / ideal, 6)


def kendall_tau(a, b):
    """Kendall's tau-b between two rankings, in [-1, 1].

    Tau-b rather than tau-a because relevance ties are the normal case here:
    every phantom shares a relevance of zero, and tau-a would count those pairs
    as disagreements and drag a good ranking down for no reason.
    """
    n = len(a)
    if n != len(b):
        raise ValueError(f"sequences differ in length: {n} vs {len(b)}")
    if n < 2:
        return None

    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            product = da * db
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                ties_a += da == 0
                ties_b += db == 0

    denominator = math.sqrt((concordant + discordant + ties_a)
                            * (concordant + discordant + ties_b))
    if denominator == 0.0:
        return None
    return round((concordant - discordant) / denominator, 6)


def ranking_tau(relevances):
    """Agreement between the order we published and the order we should have.

    Position is negated so that rank 1 is the *highest* value, matching the
    direction relevance runs in.
    """
    if len(relevances) < 2:
        return None
    return kendall_tau([-rank for rank in range(1, len(relevances) + 1)], relevances)


def false_positive_penalty(targets, unmatched_targets):
    """Cost of phantoms, weighted by how high up they were put.

    A phantom at rank 1 sends a team to the wrong place; one at rank 9 wastes a
    glance. Both are false positives and a plain count would call them equal, so
    the penalty uses the same log discount the ranking metrics do.
    """
    if not targets:
        return 0.0
    total = sum(1.0 / math.log2(rank + 1) for rank in range(1, len(targets) + 1))
    charged = sum(1.0 / math.log2(rank + 1) for rank, _ in unmatched_targets)
    return round(charged / total, 6) if total else 0.0
