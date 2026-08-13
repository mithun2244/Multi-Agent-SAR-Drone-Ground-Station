"""Detector stubs for Phase 2, standing in for the real models.

`YOLO11m` on the RGB frame and the LiDAR range-image detector are not wired up
yet. These deterministic stubs emit valid `ClueContract` instances with the
right *characteristics* so the fusion logic downstream can be built and tested
against something honest:

  * RGB classifies well but its box is looser and its fix is inferred from
    terrain height, so it drifts.
  * LiDAR measures range directly, so its fix is tight, but it detects a
    shape rather than a class and is less certain what it found.

That asymmetry is the whole reason Weighted Box Fusion is worth doing, so the
stubs model it rather than emitting identical boxes.

Replace `StubDetector.detect` with a real model call. Nothing downstream cares,
as long as it returns valid clues.
"""

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from .geolocation import offset_enu

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Target:
    """A real thing in the world, for the stubs to detect or miss.

    `range_m` is the true slant range from the drone. Supply it and the LiDAR
    stub measures it (with noise) instead of inventing one, which is what makes
    a synthetic frame geometrically consistent: the box, the world position, and
    the range then all describe the same target.
    """

    box: tuple  # (x1, y1, x2, y2) in pixels
    geo: tuple | None = None  # (lat, lon)
    label: str = "person"
    range_m: float | None = None


class StubDetector:
    """Deterministic fake detector.

    Every knob here is a property of the sensor being simulated, not a magic
    number: `recall` is how often it sees a real target, `box_jitter` how
    sloppy its box is as a fraction of target width, `geo_sigma_m` how far its
    fix drifts, `fp_per_frame` how often it invents a contact.
    """

    def __init__(
        self,
        source_agent,
        provenance_tag,
        seed=0,
        recall=0.85,
        box_jitter=0.06,
        conf_range=(0.45, 0.99),
        geo_sigma_m=18.0,
        fp_per_frame=0.0,
        label=None,
        emits_range=False,
        range_sigma_m=0.5,
        device_id=None,
    ):
        self.source_agent = source_agent
        self.provenance_tag = provenance_tag
        self.recall = recall
        self.box_jitter = box_jitter
        self.conf_range = conf_range
        self.geo_sigma_m = geo_sigma_m
        self.fp_per_frame = fp_per_frame
        self.label = label
        self.emits_range = emits_range
        self.range_sigma_m = range_sigma_m
        # Hardware identity of the sensor. The provenance guard checks it
        # against the approved fleet before anything it says reaches fusion.
        self.device_id = device_id
        self._rng = random.Random(seed)
        # Identifiers draw from their own stream so minting a clue_id cannot
        # perturb the detection sampling.
        self._id_rng = random.Random(seed + 1001)
        self._frames = {}

    def detect(self, frame_id, targets, case_id="case-0000"):
        """Detect over one frame. Returns the clues this sensor would publish."""
        clues = []
        for target in targets:
            if self._rng.random() > self.recall:
                continue  # missed
            clues.append(self._clue(frame_id, target, case_id))

        for _ in range(self._n_false_alarms()):
            x1, y1 = self._rng.uniform(0, 1180), self._rng.uniform(0, 620)
            w, h = self._rng.uniform(28, 90), self._rng.uniform(40, 130)
            clues.append(
                self._clue(frame_id, Target((x1, y1, x1 + w, y1 + h), None), case_id, spurious=True)
            )
        return clues

    def _clue(self, frame_id, target, case_id, spurious=False):
        x1, y1, x2, y2 = target.box
        jitter = (x2 - x1) * self.box_jitter
        box = [round(v + self._rng.uniform(-jitter, jitter), 2) for v in (x1, y1, x2, y2)]

        lo, hi = self.conf_range
        confidence = self._rng.uniform(0.05, 0.55) if spurious else self._rng.uniform(lo, hi)

        lat = lon = None
        if target.geo is not None:
            north, east = (self._rng.gauss(0.0, self.geo_sigma_m) for _ in range(2))
            lat, lon = offset_enu(target.geo[0], target.geo[1], east, north)

        metadata = {"device_id": self.device_id} if self.device_id else {}
        if self.emits_range and not spurious:
            # A measured range, which is exactly what the RGB path lacks. When
            # the target states its true range, measure that; otherwise this is
            # an unanchored stand-in and the number is only plausible, not real.
            if target.range_m is not None:
                metadata["range_m"] = round(
                    max(0.5, target.range_m + self._rng.gauss(0.0, self.range_sigma_m)), 2
                )
            else:
                metadata["range_m"] = round(self._rng.uniform(20.0, 140.0), 1)

        return ClueContract(
            clue_id=str(uuid.UUID(int=self._id_rng.getrandbits(128), version=4)),
            case_id=case_id,
            timestamp=_EPOCH + timedelta(seconds=self._frame_index(frame_id)),
            source_agent=self.source_agent,
            confidence_score=round(confidence, 4),
            finding_summary=(
                f"{'Low-confidence contact' if spurious else 'Possible ' + (self.label or 'contact')}"
                f" on {self.source_agent.value}"
            ),
            spatial_context=SpatialContext(latitude=lat, longitude=lon, bounding_box=box),
            frame_id=frame_id,
            class_label=self.label,
            provenance_tag=self.provenance_tag,
            agent_metadata=metadata,
        )

    def _frame_index(self, frame_id):
        return self._frames.setdefault(frame_id, len(self._frames))

    def _n_false_alarms(self):
        """Knuth sampling. ponytail: exact enough below rate ~10."""
        from math import exp

        if self.fp_per_frame <= 0.0:
            return 0
        limit, k, p = exp(-self.fp_per_frame), 0, 1.0
        while True:
            p *= self._rng.random()
            if p <= limit:
                return k
            k += 1


def yolo11m_stub(seed=0, **kwargs):
    """RGB detector: confident about *what*, looser about *where*.

    Stands in for YOLO11m (Medium), the operational RGB detector.

    These knobs are a plausible shape for a medium model, not a measurement:
    nobody has run the real weights on this data yet. Writing a figure in here
    that looked measured would put an invented accuracy number inside the thing
    built to measure accuracy. Pass `recall=` to explore the trade; settle it
    with the Phase 1 harness and real weights.
    """
    defaults = dict(
        recall=0.85,
        box_jitter=0.06,
        conf_range=(0.55, 0.99),
        geo_sigma_m=18.0,
        fp_per_frame=0.15,
        label="person",
        emits_range=False,
    )
    return StubDetector(AgentSource.DRONE_RGB, "onboard:rgb-camera", seed=seed, **{**defaults, **kwargs})


def lidar_stub(seed=0, **kwargs):
    """LiDAR range-image detector: tight box and a measured fix, less certain
    what it is looking at."""
    defaults = dict(
        recall=0.72,
        box_jitter=0.03,
        conf_range=(0.40, 0.85),
        geo_sigma_m=4.0,
        fp_per_frame=0.10,
        label="person",
        emits_range=True,
    )
    return StubDetector(AgentSource.DRONE_LIDAR, "onboard:lidar", seed=seed, **{**defaults, **kwargs})
