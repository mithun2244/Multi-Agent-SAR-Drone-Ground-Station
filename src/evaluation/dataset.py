"""Dataset loading, clue ingestion, and the Phase 1 placeholder data.

Real paired RGB/LiDAR search-and-rescue frames do not exist yet (Phase 1 in
docs/architecture.md is where they get settled), so this module provides:

  * a JSON loader for ground-truth annotations,
  * a loader for clues in Phase 0 `ClueContract` form,
  * a deterministic mock dataset and mock RGB detector,
  * a deterministic train/validation split.

Producers emit `ClueContract` (src/contracts/clue.py). `Detection` below is a
projection of that contract onto the handful of fields scoring needs — built
only by `Detection.from_clue`, so the bus schema stays the single source of
truth and the metrics module never has to know about it.

Ground truth is *not* a clue: it is annotation data, not something an agent
observed and published, so it keeps its own shape.
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..contracts.clue import AgentSource, ClueContract, SpatialContext

# Degrees of latitude per metre. Longitude is scaled by cos(lat) at use site.
_DEG_PER_M = 1.0 / 111_320.0

# Deterministic clock for mock clues.
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class GroundTruth:
    frame_id: str
    label: str
    box: tuple  # (x1, y1, x2, y2) in pixels
    geo: tuple | None = None  # (lat, lon)


@dataclass(frozen=True)
class Detection:
    """Scoring projection of one `ClueContract`."""

    frame_id: str
    label: str
    box: tuple
    confidence: float
    geo: tuple | None = None
    clue_id: str | None = None
    case_id: str | None = None

    @classmethod
    def from_clue(cls, clue):
        """Project a ClueContract — or its dict/JSON form, as it arrives off the
        bus — onto the fields the evaluator scores.

        This is a trust boundary: a clue that cannot be scored is rejected loudly
        rather than silently counted as a miss, which would quietly deflate
        recall and hide a broken producer.

        `frame_id` and `class_label` are optional on the contract so non-frame
        agents (Weather, Health) can omit them, but a detection clue without a
        frame cannot be matched — optional on the bus, required here.
        """
        if not isinstance(clue, ClueContract):
            clue = ClueContract.model_validate(clue)

        box = clue.detection_box()
        spatial = clue.spatial_context
        geo = None
        if spatial.latitude is not None and spatial.longitude is not None:
            geo = (spatial.latitude, spatial.longitude)

        return cls(
            frame_id=str(clue.frame_id),
            label=str(clue.class_label or clue.source_agent.value),
            box=box,
            confidence=clue.confidence_score,
            geo=geo,
            clue_id=clue.clue_id,
            case_id=clue.case_id,
        )


@dataclass
class Split:
    name: str
    frame_ids: list
    ground_truth: list = field(default_factory=list)

    @property
    def n_frames(self):
        return len(self.frame_ids)


def load_split(path, name):
    """Load ground-truth annotations for a split from JSON.

    Expected shape:
        {"frame_ids": [...],
         "ground_truth": [{"frame_id":..., "label":..., "box":[x1,y1,x2,y2],
                           "geo": [lat, lon]}, ...]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Split(
        name=name,
        frame_ids=list(data["frame_ids"]),
        ground_truth=[
            GroundTruth(
                frame_id=g["frame_id"],
                label=g["label"],
                box=tuple(g["box"]),
                geo=tuple(g["geo"]) if g.get("geo") else None,
            )
            for g in data.get("ground_truth", [])
        ],
    )


def load_clues(path):
    """Load a JSON array of clues as they would arrive off the bus."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ClueContract.model_validate(c) for c in data]


def mock_dataset(n_frames=120, seed=0, origin=(46.8182, 8.2275)):
    """Deterministic stand-in dataset. Default origin is central Switzerland."""
    rng = random.Random(seed)
    frame_ids, truth = [], []
    for i in range(n_frames):
        frame_id = f"frame_{i:04d}"
        frame_ids.append(frame_id)
        for _ in range(rng.choices([0, 1, 2], weights=[0.35, 0.5, 0.15])[0]):
            x1, y1 = rng.uniform(0, 1180), rng.uniform(0, 620)
            w, h = rng.uniform(28, 90), rng.uniform(40, 130)
            lat = origin[0] + rng.uniform(-0.01, 0.01)
            lon = origin[1] + rng.uniform(-0.01, 0.01)
            truth.append(
                GroundTruth(frame_id, "person", (x1, y1, x1 + w, y1 + h), (lat, lon))
            )
    return frame_ids, truth


def split_frames(frame_ids, val_fraction=0.3, seed=0):
    """Fixed train/validation split — same seed always gives the same split."""
    ids = sorted(frame_ids)
    random.Random(seed).shuffle(ids)
    cut = round(len(ids) * (1.0 - val_fraction))
    return ids[:cut], ids[cut:]


def build_splits(n_frames=120, val_fraction=0.3, seed=0):
    """Mock dataset, split into train and validation."""
    frame_ids, truth = mock_dataset(n_frames=n_frames, seed=seed)
    train_ids, val_ids = split_frames(frame_ids, val_fraction, seed)
    splits = {}
    for name, ids in (("train", train_ids), ("validation", val_ids)):
        wanted = set(ids)
        splits[name] = Split(
            name=name,
            frame_ids=sorted(wanted),
            ground_truth=[g for g in truth if g.frame_id in wanted],
        )
    return splits


def mock_rgb_detector(
    split,
    seed=0,
    case_id="case-0000",
    recall=0.78,
    fp_per_frame=0.25,
    geo_sigma_m=18.0,
):
    """Placeholder RGB-only detector: the Phase 1 baseline to beat.

    Stands in for YOLO11n on the RGB frame and emits `ClueContract` exactly as a
    real producer would. It misses some targets, invents some false alarms, and
    its geolocation is noisy because an RGB-only fix infers range from terrain
    instead of measuring it — the gap Phase 2 closes by adding LiDAR range.

    Swap this for a real detector by passing any callable with this signature to
    `harness.run_baseline`; anything emitting valid clues will score.
    """
    rng = random.Random(seed + 1)
    # Separate stream for identifiers: minting a clue_id must not perturb the
    # detection sampling, or adding a contract field would silently shift the
    # data distribution and move every baseline number.
    id_rng = random.Random(seed + 1001)
    frame_index = {fid: i for i, fid in enumerate(sorted(split.frame_ids))}
    clues = []

    def emit(frame_id, box, confidence, geo, summary):
        lat, lon = geo if geo else (None, None)
        clues.append(
            ClueContract(
                clue_id=str(uuid.UUID(int=id_rng.getrandbits(128), version=4)),
                case_id=case_id,
                timestamp=_EPOCH + timedelta(seconds=frame_index.get(frame_id, 0)),
                source_agent=AgentSource.DRONE_RGB,
                confidence_score=confidence,
                finding_summary=summary,
                spatial_context=SpatialContext(
                    latitude=lat, longitude=lon, bounding_box=[round(v, 2) for v in box]
                ),
                frame_id=frame_id,
                class_label="person",
                provenance_tag="onboard:rgb-camera",
            )
        )

    for gt in split.ground_truth:
        if rng.random() > recall:
            continue  # missed target
        x1, y1, x2, y2 = gt.box
        jitter = (x2 - x1) * 0.08
        box = tuple(v + rng.uniform(-jitter, jitter) for v in (x1, y1, x2, y2))
        # Range inferred from terrain, so the fix drifts by tens of metres.
        north, east = rng.gauss(0.0, geo_sigma_m), rng.gauss(0.0, geo_sigma_m)
        lat = gt.geo[0] + north * _DEG_PER_M
        lon = gt.geo[1] + east * _DEG_PER_M / max(0.1, abs(_cos_deg(gt.geo[0])))
        confidence = rng.uniform(0.45, 0.99)
        emit(gt.frame_id, box, confidence, (lat, lon),
             f"Possible person detected in RGB frame (conf {confidence:.2f})")

    for frame_id in split.frame_ids:
        for _ in range(_poisson_ish(rng, fp_per_frame)):
            x1, y1 = rng.uniform(0, 1180), rng.uniform(0, 620)
            w, h = rng.uniform(28, 90), rng.uniform(40, 130)
            confidence = rng.uniform(0.05, 0.65)  # false alarms skew low-confidence
            emit(frame_id, (x1, y1, x1 + w, y1 + h), confidence, None,
                 f"Low-confidence RGB contact, no range fix (conf {confidence:.2f})")

    return clues


def _cos_deg(degrees):
    from math import cos, radians

    return cos(radians(degrees))


def _poisson_ish(rng, rate):
    """Small-count draw. ponytail: Knuth sampling, exact enough below rate ~10."""
    from math import exp

    limit, k, p = exp(-rate), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
