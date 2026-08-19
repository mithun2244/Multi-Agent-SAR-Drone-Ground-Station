"""A repeatable search, run end to end under one configuration.

The objective needs to answer "would this configuration have found them?", which
means running the real pipeline — detectors, weighted box fusion, tracking,
geolocation, the bus, coordinator fusion — and handing the resulting picture to
the critic. This module is that run, with every tunable threaded through.

Deliberately synthetic
----------------------
Subjects are projected into the frame from known world positions, so ground
truth is exact and a fold is reproducible from a seed. That makes the search
honest about the *plumbing* — thresholds, trust weights, ranking weights — and
says nothing about how a real YOLO11m behaves on real footage. A configuration
tuned here is a starting point for a real tuning run against real recordings,
not a substitute for one.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..agents.weather import Assessment, Conditions, WeatherAgent
from ..bus import FakeRedisStreams, RedisBus, stream_for
from ..coordinator.blackboard import Blackboard
from ..coordinator.fusion import CoordinatorFusion
from ..critic.outcomes import CaseOutcome, Subject
from ..perception.agent import DetectionAgent
from ..perception.detectors import Target as StubTarget
from ..perception.detectors import lidar_stub, yolo11m_stub
from ..perception.geolocation import (
    Camera,
    Telemetry,
    geolocate,
    slant_range_m,
    world_to_pixel,
)
from ..perception.terrain import GridDEM
from ..perception.tracking import BoTSORT
from ..utils.ablation import keep_feeds
from .params import TunedParams

DRONE = (46.8182, 8.2275)
# Flown higher and on a longer lens than the demo. Both matter: at 120 m AGL a
# 40-degree frame covers about 90 m of ground, so three people spread across it
# land under 25 m apart and coordinator fusion correctly merges them into one
# target. That is real behaviour, but it makes the scenario a test of the merge
# radius rather than of the detection thresholds. At ~290 m AGL the frame spans
# nearly 200 m and the subjects are properly separate.
CAMERA = Camera(fx=2400.0, fy=2400.0, cx=640.0, cy=360.0)
TELEMETRY = Telemetry(DRONE[0], DRONE[1], altitude_m=1750.0, yaw_deg=15.0, pitch_deg=55.0)
FRAME_SIZE = (1280, 720)
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Everything is anchored by where it appears in the *frame*, then ray-cast onto
# the terrain to get its world position. Choosing lat/lon by hand puts most of a
# search area outside a 55-degree-down field of view, and a subject the camera
# cannot see is not a detection problem.
#
# Three people out there. The casualty is the one that has to come out first,
# and is also the hardest to see — which is the whole difficulty.
SUBJECT_PIXELS = (
    ("hiker-1", (200.0, 640.0), 1.0, "clear"),
    ("casualty", (430.0, 470.0), 3.0, "faint"),
    ("hiker-3", (120.0, 400.0), 1.0, "clear"),
)

# Things that look like people and are not: a boulder with jacket-coloured
# lichen, a pack left on a col. Visible to RGB and invisible to LiDAR, because
# they have no human shape to range — exactly the cross-check the architecture
# relies on ("a target that shows up in one but not the other is suspect").
# Whether the tuner discovers that is the point of the exercise.
DECOY_PIXELS = (
    ("decoy-boulder", (1120.0, 620.0)),
    ("decoy-pack", (1180.0, 380.0)),
)


def search_area_dem():
    return GridDEM.from_function(
        lambda lat, lon: 1450.0 + (lat - DRONE[0]) * 20_000.0,
        lat_min=46.79, lon_min=8.20, lat_step=0.0005, lon_step=0.0005,
        n_lat=120, n_lon=120,
    )


@dataclass
class RunResult:
    """Everything the critic and the Phase 1 harness need to score one run."""

    picture: object
    outcome: CaseOutcome
    # What the perception plane published, not what the raw sensors saw: mAP is
    # only interesting if it responds to the parameters being tuned, and WBF and
    # the tracker sit between the two.
    detections: list = field(default_factory=list)
    raw_detections: list = field(default_factory=list)
    truth_boxes: list = field(default_factory=list)  # ground-truth boxes per frame
    frames: int = 0

    @property
    def targets(self):
        return len(self.picture.targets)

    def evaluation_inputs(self):
        """Detections and ground truth in the Phase 1 harness's own terms.

        Lets the same run be scored two ways: by the critic on the operational
        picture, and by the Phase 1 evaluator on raw detections for mAP.
        """
        from ..evaluation.dataset import Detection, GroundTruth
        detections = [Detection.from_clue(clue) for clue in self.detections]
        ground_truth = [
            GroundTruth(frame_id, subject.label, tuple(subject.bounding_box),
                        (subject.latitude, subject.longitude))
            for frame_id, subject in self.truth_boxes
        ]
        return detections, ground_truth


def place(dem, pixel):
    """Where on the ground a frame pixel actually falls, and its box there.

    Ray-cast to the terrain, then project a standing person back — so the box is
    consistent with the world position rather than guessed alongside it.
    """
    fix = geolocate(pixel, CAMERA, TELEMETRY, dem=dem)
    if fix is None:
        raise ValueError(f"pixel {pixel} does not strike the terrain")

    feet = world_to_pixel(fix.latitude, fix.longitude, fix.elevation_m, CAMERA, TELEMETRY)
    head = world_to_pixel(fix.latitude, fix.longitude, fix.elevation_m + 1.7,
                          CAMERA, TELEMETRY)
    half_width = abs(feet[1] - head[1]) * 0.35
    box = (feet[0] - half_width, head[1], feet[0] + half_width, feet[1])
    if not _in_frame(box):
        raise ValueError(f"pixel {pixel} projects a box outside the frame: {box}")
    return box, fix, slant_range_m(TELEMETRY, fix.latitude, fix.longitude, fix.elevation_m)


def _in_frame(box, size=FRAME_SIZE):
    x1, y1, x2, y2 = box
    return 0 <= x1 and 0 <= y1 and x2 <= size[0] and y2 <= size[1]


def _stub_targets(dem):
    """Place subjects and decoys on the ground under the camera."""
    clear, faint, decoys, truth = [], [], [], []
    for subject_id, pixel, priority, visibility in SUBJECT_PIXELS:
        box, fix, distance = place(dem, pixel)
        target = StubTarget(box=box, geo=(fix.latitude, fix.longitude), range_m=distance)
        (faint if visibility == "faint" else clear).append(target)
        truth.append(Subject(subject_id=subject_id, latitude=fix.latitude,
                             longitude=fix.longitude, elevation_m=fix.elevation_m,
                             priority=priority, bounding_box=box, label="person"))

    for _, pixel in DECOY_PIXELS:
        box, fix, _ = place(dem, pixel)
        # No range_m: the LiDAR never sees these, so nothing measures them.
        decoys.append(StubTarget(box=box, geo=(fix.latitude, fix.longitude)))

    return clear, faint, decoys, truth


def _weather_clue(case_id, position, temperature_c, wind_kmh, window, risk):
    agent = WeatherAgent(RedisBus(FakeRedisStreams()), case_id)
    conditions = Conditions(temperature_c=temperature_c, wind_kmh=wind_kmh,
                            precipitation_mm=0.0, humidity_pct=60.0, observed_at=EPOCH)
    verdict = Assessment(apparent_c=temperature_c, wet=False,
                         hypothermia_risk=risk, survival_window_hours=window)
    return agent.build_clue(position[0], position[1], conditions, verdict)


def run(params=None, seed=0, frames=6, case_id="tune-0000", missing_hours=2.0):
    """Fly one sortie under `params` and return everything needed to score it.

    Five things are out there and only three are people. The casualty is the
    faintest of the three and the most urgent, so no single threshold sorts this
    out — that tension is what makes the search worth running.
    """
    params = params or TunedParams()
    dem = search_area_dem()
    clear, faint, decoys, truth = _stub_targets(dem)

    bus = RedisBus(FakeRedisStreams())
    blackboard = Blackboard()
    case = blackboard.open_case(case_id, opened_at=EPOCH,
                                missing_since=EPOCH - timedelta(hours=missing_hours))

    detection_agent = DetectionAgent(
        bus, case.case_id, CAMERA, dem=dem,
        tracker=BoTSORT(high_thresh=params.track_high_thresh,
                        new_track_thresh=params.track_new_thresh,
                        min_hits=params.track_min_hits),
        fusion_weights=params.wbf_weights,
        fusion_iou_threshold=params.wbf_iou_threshold,
    )
    # Separate detector instances per population: the clear subjects, the faint
    # casualty, and the decoys that only RGB can see.
    rgb_clear = yolo11m_stub(seed=seed, recall=0.95, conf_range=(0.65, 0.95),
                             fp_per_frame=0.15)
    lidar_clear = lidar_stub(seed=seed, recall=0.88, conf_range=(0.55, 0.88),
                             fp_per_frame=0.10)
    rgb_faint = yolo11m_stub(seed=seed + 100, recall=0.72, conf_range=(0.40, 0.68),
                             fp_per_frame=0.0)
    lidar_faint = lidar_stub(seed=seed + 100, recall=0.60, conf_range=(0.35, 0.60),
                             fp_per_frame=0.0)
    rgb_decoy = yolo11m_stub(seed=seed + 200, recall=0.92, conf_range=(0.55, 0.85),
                             fp_per_frame=0.0)

    detections, raw_detections, truth_boxes = [], [], []
    for index in range(frames):
        frame_id = f"frame_{index:04d}"
        rgb_clues = (rgb_clear.detect(frame_id, clear, case.case_id)
                     + rgb_faint.detect(frame_id, faint, case.case_id)
                     + rgb_decoy.detect(frame_id, decoys, case.case_id))
        lidar_clues = (lidar_clear.detect(frame_id, clear, case.case_id)
                       + lidar_faint.detect(frame_id, faint, case.case_id))
        raw_detections.extend(rgb_clues)
        detections.extend(detection_agent.process_frame(
            frame_id, keep_feeds({"rgb": rgb_clues, "lidar": lidar_clues}), TELEMETRY))
        truth_boxes.extend((frame_id, subject) for subject in truth)

    # A cold front over the casualty only, so urgency has something to sort by.
    casualty = next(s for s in truth if s.subject_id == "casualty")
    sheltered = next(s for s in truth if s.subject_id == "hiker-1")
    bus.publish(_weather_clue(case.case_id, (casualty.latitude, casualty.longitude),
                              1.0, 15.0, 6, True))
    bus.publish(_weather_clue(case.case_id, (sheltered.latitude, sheltered.longitude),
                              9.0, 4.0, 24, False))

    fusion = CoordinatorFusion(
        bus, blackboard, params=params, weather_radius_m=40.0,
        clock=lambda: EPOCH,
    )
    fusion.ingest(case.case_id)
    picture = fusion.picture(case.case_id)

    return RunResult(
        picture=picture,
        outcome=CaseOutcome(case_id=case.case_id, subjects=tuple(truth), resolved_at=EPOCH),
        detections=detections,
        raw_detections=raw_detections,
        truth_boxes=truth_boxes,
        frames=frames,
    )


def main(argv=None):
    """Fly one scenario and report what it produced.

    `run(seed=...)` already seeds every stub detector explicitly — that is what
    makes a fold reproducible. `--seed` sets that *and* the global generators,
    so the whole process is pinned, not just the parts this module owns.

        python -m src.tuning.scenario --seed 42
        python -m src.tuning.scenario --seed 42 --frames 8
    """
    import argparse

    from ..utils.seed import DEFAULT_SEED, describe, set_global_seed

    parser = argparse.ArgumentParser(description="One repeatable tuning scenario")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args(argv)

    print(f"  {describe(set_global_seed(args.seed), args.seed)}")
    result = run(seed=args.seed, frames=args.frames)

    print(f"  {args.frames} frame(s), {len(result.truth_boxes)} ground-truth box(es), "
          f"{len(result.raw_detections)} raw detection(s)")
    print(f"  {len(result.picture.targets)} target(s) in the picture, "
          f"{len(result.outcome.subjects)} subject(s) actually out there")
    for rank, target in enumerate(result.picture.targets, 1):
        where = (f"{target.latitude:.5f}, {target.longitude:.5f}"
                 if target.located else "NOT LOCATED")
        print(f"    {rank}  {target.target_id:<4} conf {target.confidence:.3f}  "
              f"urg {target.urgency:.2f}  {where}")
    print(f"\n  re-run with the same --seed for an identical sortie\n")
    return result


if __name__ == "__main__":
    main()
