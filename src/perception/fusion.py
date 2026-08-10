"""Weighted Box Fusion — merge RGB and LiDAR detections into confirmed targets.

Phase 2 (docs/architecture.md): YOLO11n runs on the RGB frame to classify
targets, a second detector runs on the LiDAR range image for depth and range,
and their detections are combined with Weighted Box Fusion so each confirmed
target carries both an RGB class and a measured LiDAR range.

Follows Solovyev, Wang & Gabruseva (2021), "Weighted Boxes Fusion", with one
deliberate change noted at `_support` below.

Unlike NMS, WBF does not discard the losing box: every clustered box
contributes to the fused coordinates in proportion to its confidence, so two
sensors that each see the target slightly off produce a better box than either
alone. The parents survive in `parent_clue_ids`.
"""

import uuid
from dataclasses import dataclass

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..geometry import iou
from ..guardrails.provenance import TAG_WBF

# Provenance for anything this module mints. Must be on the Phase 7 allow-list.
FUSION_PROVENANCE = TAG_WBF


@dataclass
class _Entry:
    """One detection entering fusion, with its detector's weight applied."""

    clue: ClueContract
    model: int
    box: tuple
    score: float  # confidence_score * detector weight


def weighted_box_fusion(
    clue_groups,
    weights=None,
    iou_threshold=0.55,
    score_threshold=0.0,
):
    """Fuse detections from several detectors into one clue per target.

    `clue_groups` is one list of ClueContract per detector — e.g.
    `[rgb_clues, lidar_clues]`. `weights` says how much each detector is
    trusted; the Phase 9 tuner searches these along with `iou_threshold`.

    Boxes fuse only within the same `(frame_id, class_label)`: two sensors
    looking at different frames, or calling the target different things, do not
    corroborate each other. Returns a new list of clues, each carrying its
    parents in `parent_clue_ids`. Inputs are never mutated.
    """
    groups = [list(g) for g in clue_groups]
    if not groups:
        return []
    if weights is None:
        weights = [1.0] * len(groups)
    if len(weights) != len(groups):
        raise ValueError(f"got {len(weights)} weights for {len(groups)} detector groups")
    if any(w <= 0 for w in weights):
        raise ValueError(f"detector weights must be positive, got {list(weights)}")

    n_models, weight_sum = len(groups), sum(weights)

    buckets = {}
    for model, (group, weight) in enumerate(zip(groups, weights)):
        for clue in group:
            box = clue.detection_box()  # trust boundary: unlocatable clues are rejected
            if clue.confidence_score < score_threshold:
                continue
            buckets.setdefault((clue.frame_id, clue.class_label), []).append(
                _Entry(clue, model, box, clue.confidence_score * weight)
            )

    fused = []
    for (frame_id, label), entries in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        entries.sort(key=lambda e: -e.score)
        clusters, cluster_boxes = [], []
        for entry in entries:
            # Match against the cluster's current fused box, not its seed box,
            # so a cluster drifts toward consensus as members join.
            best_idx, best_iou = -1, iou_threshold
            for i, box in enumerate(cluster_boxes):
                overlap = iou(box, entry.box)
                if overlap > best_iou:
                    best_idx, best_iou = i, overlap
            if best_idx < 0:
                clusters.append([entry])
                cluster_boxes.append(entry.box)
            else:
                clusters[best_idx].append(entry)
                cluster_boxes[best_idx] = _weighted_box(clusters[best_idx])

        for cluster in clusters:
            fused.append(_fuse_cluster(cluster, frame_id, label, n_models, weight_sum, iou_threshold))
    return fused


def _weighted_mean(values, weights):
    total = sum(weights)
    if total <= 0.0:  # every box scored 0.0 — fall back to an unweighted mean
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total


def _weighted_box(cluster):
    """Fused coordinates: each corner averaged in proportion to confidence."""
    scores = [e.score for e in cluster]
    return tuple(_weighted_mean([e.box[k] for e in cluster], scores) for k in range(4))


def _support(cluster):
    """How many distinct detectors back this cluster.

    The reference implementation counts *boxes*, which lets one detector firing
    twice on the same target look like corroboration. Counting distinct
    detectors is what the architecture actually claims: "a target that shows up
    in one but not the other is suspect". Two RGB boxes are one opinion.
    """
    return len({e.model for e in cluster})


def _fuse_cluster(cluster, frame_id, label, n_models, weight_sum, iou_threshold):
    cases = {e.clue.case_id for e in cluster}
    if len(cases) > 1:
        raise ValueError(f"refusing to fuse clues from different cases: {sorted(cases)}")

    box = _weighted_box(cluster)
    support = _support(cluster)
    # Paper's rescale: a cluster only one detector saw is worth proportionally
    # less than one both agree on.
    raw = sum(e.score for e in cluster) / len(cluster)
    score = min(1.0, max(0.0, raw * min(n_models, support) / weight_sum))

    dominant = max(cluster, key=lambda e: e.score)
    parents = sorted(e.clue.clue_id for e in cluster)
    sources = sorted({e.clue.source_agent.value for e in cluster})

    located = [
        e for e in cluster
        if e.clue.spatial_context
        and e.clue.spatial_context.latitude is not None
        and e.clue.spatial_context.longitude is not None
    ]
    lat = lon = alt = None
    if located:
        # ponytail: confidence-weighted mean of the parent fixes. Phase 2's real
        # geolocation should instead prefer the LiDAR-measured range over an
        # RGB fix inferred from terrain height — swap this when that lands.
        w = [e.score for e in located]
        lat = _weighted_mean([e.clue.spatial_context.latitude for e in located], w)
        lon = _weighted_mean([e.clue.spatial_context.longitude for e in located], w)
        with_alt = [e for e in located if e.clue.spatial_context.altitude_m is not None]
        if with_alt:
            alt = _weighted_mean(
                [e.clue.spatial_context.altitude_m for e in with_alt],
                [e.score for e in with_alt],
            )

    # Carry the measured range through. Architecture Phase 2: a confirmed target
    # carries "both an RGB class and a measured LiDAR range" — if fusion dropped
    # it, geolocation would be left inferring range from terrain for a target the
    # LiDAR had already ranged.
    metadata = {
        "wbf_support": support,
        "wbf_sources": sources,
        "wbf_cluster_size": len(cluster),
        "wbf_iou_threshold": iou_threshold,
    }
    ranged = [e for e in cluster if e.clue.agent_metadata.get("range_m") is not None]
    if ranged:
        best = max(ranged, key=lambda e: e.score)
        metadata["range_m"] = best.clue.agent_metadata["range_m"]
        metadata["range_source"] = best.clue.source_agent.value

    return ClueContract(
        # Deterministic in the parents: fusing the same detections twice yields
        # the same clue_id instead of a duplicate on the blackboard.
        clue_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "sar:wbf:" + "|".join(parents))),
        case_id=cases.pop(),
        parent_clue_ids=parents,
        timestamp=dominant.clue.timestamp,
        # A merged clue is its own kind of observation, not the RGB or LiDAR one
        # that happened to score highest. Its parents are in parent_clue_ids.
        source_agent=AgentSource.PERCEPTION_FUSION,
        confidence_score=score,
        finding_summary=(
            f"{label or 'contact'} confirmed by {support} of {n_models} sensors "
            f"({', '.join(sources)})"
        ),
        spatial_context=SpatialContext(
            latitude=lat,
            longitude=lon,
            altitude_m=alt,
            bounding_box=[round(v, 4) for v in box],
        ),
        frame_id=frame_id,
        class_label=label,
        provenance_tag=FUSION_PROVENANCE,
        agent_metadata=metadata,
    )


if __name__ == "__main__":  # a worked example, for eyeballing a merge
    from .detectors import Target, lidar_stub, yolo11n_stub

    targets = [Target((100, 100, 160, 240), (46.8182, 8.2275)),
               Target((600, 300, 650, 420), (46.8190, 8.2280))]
    rgb, lidar = yolo11n_stub(seed=3), lidar_stub(seed=3)
    rgb_clues = rgb.detect("frame_0001", targets)
    lidar_clues = lidar.detect("frame_0001", targets)

    print(f"\n  RGB   emitted {len(rgb_clues)}:")
    for c in rgb_clues:
        print(f"    {c.confidence_score:.3f}  {c.spatial_context.bounding_box}")
    print(f"  LiDAR emitted {len(lidar_clues)}:")
    for c in lidar_clues:
        print(f"    {c.confidence_score:.3f}  {c.spatial_context.bounding_box}")

    print("\n  fused:")
    for c in weighted_box_fusion([rgb_clues, lidar_clues]):
        print(f"    {c.confidence_score:.3f}  {c.spatial_context.bounding_box}"
              f"  support={c.agent_metadata['wbf_support']}  parents={len(c.parent_clue_ids)}")
    print()
