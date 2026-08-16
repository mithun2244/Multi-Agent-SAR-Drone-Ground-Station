"""Known-answer checks for Weighted Box Fusion and the detector stubs.

    python -m src.perception.test_perception
"""

import itertools
import math
import os
import struct
import tempfile
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..evaluation.dataset import Detection
from ..evaluation.metrics import haversine_m
from .detectors import Target, lidar_stub, yolo11m_stub
from .fusion import FUSION_PROVENANCE, weighted_box_fusion
from ..bus import CLUE_FIELD, FakeRedisStreams, RedisBus, stream_for
from .agent import TRACK_PROVENANCE, DetectionAgent
from .geolocation import (
    Camera,
    Fix,
    RangeEstimate,
    RangeSource,
    Telemetry,
    geolocate,
    geolocate_clue,
    ground_distance_m,
    intersect_dem,
    offset_enu,
    pixel_ray,
    project,
    select_range,
    slant_range_m,
    world_to_pixel,
)
from .terrain import (
    ConstantDEM,
    GeoTiffDEM,
    GridDEM,
    SrtmHgtDEM,
    load_dem,
    open_dem,
    parse_hgt_name,
)
from .tracking import Affine, BoTSORT, TrackState

_ids = itertools.count(1)


def _clue(box, conf, source=AgentSource.DRONE_RGB, frame="frame_0001", label="person",
          case="case-0000", geo=None, clue_id=None):
    lat, lon = geo if geo else (None, None)
    return ClueContract(
        clue_id=clue_id or f"clue-{next(_ids):03d}",
        case_id=case,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_agent=source,
        confidence_score=conf,
        finding_summary="test clue",
        spatial_context=SpatialContext(latitude=lat, longitude=lon, bounding_box=list(box)),
        frame_id=frame,
        class_label=label,
        provenance_tag="test",
    )


def _lidar(box, conf, **kw):
    return _clue(box, conf, source=AgentSource.DRONE_LIDAR, **kw)


def test_two_sensors_merge_into_one_target():
    rgb = _clue((0, 0, 10, 10), 0.9, clue_id="rgb-1")
    lid = _lidar((2, 0, 12, 10), 0.3, clue_id="lidar-1")
    fused = weighted_box_fusion([[rgb], [lid]])

    assert len(fused) == 1, "one physical target must yield one fused clue"
    f = fused[0]
    # Coordinates are confidence-weighted: (0.9*0 + 0.3*2)/1.2 = 0.5
    assert f.spatial_context.bounding_box == [0.5, 0.0, 10.5, 10.0]
    # Score: mean(0.9, 0.3) * min(2 models, 2 supporting) / 2 total weight
    assert abs(f.confidence_score - 0.6) < 1e-9
    assert f.agent_metadata["wbf_support"] == 2


def test_lineage_is_preserved():
    rgb = _clue((0, 0, 10, 10), 0.9, clue_id="rgb-1")
    lid = _lidar((1, 1, 11, 11), 0.8, clue_id="lidar-1")
    f = weighted_box_fusion([[rgb], [lid]])[0]

    assert f.parent_clue_ids == ["lidar-1", "rgb-1"], "both parents, sorted"
    assert f.clue_id not in f.parent_clue_ids
    assert f.provenance_tag == FUSION_PROVENANCE
    assert sorted(f.agent_metadata["wbf_sources"]) == ["DRONE_LIDAR", "DRONE_RGB"]


def test_single_sensor_support_is_discounted():
    """A target only one sensor saw is worth less than one both agree on."""
    lonely = weighted_box_fusion([[_clue((0, 0, 10, 10), 0.9)], []])
    assert len(lonely) == 1
    assert abs(lonely[0].confidence_score - 0.45) < 1e-9  # 0.9 * min(2,1)/2
    assert lonely[0].agent_metadata["wbf_support"] == 1
    assert len(lonely[0].parent_clue_ids) == 1

    # With only one detector configured there is nothing to corroborate against,
    # so the score must not be penalised.
    solo = weighted_box_fusion([[_clue((0, 0, 10, 10), 0.9)]])
    assert abs(solo[0].confidence_score - 0.9) < 1e-9


def test_distant_boxes_stay_separate():
    a = _clue((0, 0, 10, 10), 0.9)
    b = _lidar((500, 500, 510, 510), 0.8)
    fused = weighted_box_fusion([[a], [b]])
    assert len(fused) == 2, "non-overlapping detections are different targets"
    assert all(f.agent_metadata["wbf_support"] == 1 for f in fused)


def test_frames_and_labels_never_cross():
    same_box = (0, 0, 10, 10)
    assert len(weighted_box_fusion([
        [_clue(same_box, 0.9, frame="frame_0001")],
        [_lidar(same_box, 0.9, frame="frame_0002")],
    ])) == 2, "different frames must not corroborate"

    assert len(weighted_box_fusion([
        [_clue(same_box, 0.9, label="person")],
        [_lidar(same_box, 0.9, label="backpack")],
    ])) == 2, "different classes must not corroborate"


def test_different_cases_refuse_to_fuse():
    try:
        weighted_box_fusion([
            [_clue((0, 0, 10, 10), 0.9, case="case-A")],
            [_lidar((0, 0, 10, 10), 0.9, case="case-B")],
        ])
        raise AssertionError("clues from two different searches must not merge")
    except ValueError as e:
        assert "different cases" in str(e)


def test_detector_weights_shift_the_result():
    rgb = _clue((0, 0, 10, 10), 0.9)
    lid = _lidar((2, 0, 12, 10), 0.3)
    trusted_rgb = weighted_box_fusion([[rgb], [lid]], weights=[2.0, 1.0])[0]
    # (1.8*0 + 0.3*2)/2.1 = 0.2857..., pulled toward the trusted detector
    assert abs(trusted_rgb.spatial_context.bounding_box[0] - 0.2857) < 1e-3
    assert abs(trusted_rgb.confidence_score - 0.7) < 1e-9

    for bad in ([1.0], [1.0, 0.0], [1.0, -1.0]):
        try:
            weighted_box_fusion([[rgb], [lid]], weights=bad)
            raise AssertionError(f"weights {bad} should be rejected")
        except ValueError:
            pass


def test_iou_threshold_controls_merging():
    a = _clue((0, 0, 10, 10), 0.9)
    b = _lidar((5, 0, 15, 10), 0.9)  # IoU = 5/15 = 0.333
    assert len(weighted_box_fusion([[a], [b]], iou_threshold=0.3)) == 1
    assert len(weighted_box_fusion([[a], [b]], iou_threshold=0.5)) == 2


def test_score_threshold_drops_weak_detections():
    strong, weak = _clue((0, 0, 10, 10), 0.9), _lidar((300, 300, 310, 310), 0.05)
    assert len(weighted_box_fusion([[strong], [weak]], score_threshold=0.1)) == 1


def test_geo_is_confidence_weighted_over_parents():
    rgb = _clue((0, 0, 10, 10), 0.9, geo=(46.0, 8.0))
    lid = _lidar((1, 0, 11, 10), 0.3, geo=(47.0, 9.0))
    f = weighted_box_fusion([[rgb], [lid]])[0]
    assert abs(f.spatial_context.latitude - (0.9 * 46.0 + 0.3 * 47.0) / 1.2) < 1e-9
    assert abs(f.spatial_context.longitude - (0.9 * 8.0 + 0.3 * 9.0) / 1.2) < 1e-9

    # A parent with no fix must not drag the fused position toward zero.
    f2 = weighted_box_fusion([[_clue((0, 0, 10, 10), 0.9, geo=(46.0, 8.0))],
                              [_lidar((1, 0, 11, 10), 0.3)]])[0]
    assert f2.spatial_context.latitude == 46.0


def test_fusion_is_deterministic_and_pure():
    rgb, lid = _clue((0, 0, 10, 10), 0.9, clue_id="r"), _lidar((1, 1, 11, 11), 0.8, clue_id="l")
    before = [rgb.model_dump(), lid.model_dump()]

    a = weighted_box_fusion([[rgb], [lid]])[0]
    b = weighted_box_fusion([[rgb], [lid]])[0]
    assert a.clue_id == b.clue_id, "same parents must mint the same fused clue_id"
    assert a.model_dump() == b.model_dump()
    assert [rgb.model_dump(), lid.model_dump()] == before, "inputs must not be mutated"


def test_unlocatable_clues_are_rejected():
    weather = ClueContract(
        clue_id="w-1", case_id="case-0000",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_agent=AgentSource.WEATHER_API, confidence_score=0.9,
        finding_summary="Hypothermia risk high", provenance_tag="api:meteoswiss",
    )
    try:
        weighted_box_fusion([[weather], []])
        raise AssertionError("a clue with no box cannot be fused")
    except ValueError as e:
        assert "bounding_box" in str(e)


def test_empty_inputs_are_harmless():
    assert weighted_box_fusion([]) == []
    assert weighted_box_fusion([[], []]) == []


def test_fused_clue_is_consumable_downstream():
    """Fusion output must satisfy the same contract its inputs did."""
    rgb = _clue((0, 0, 10, 10), 0.9, geo=(46.8, 8.2))
    lid = _lidar((1, 1, 11, 11), 0.8, geo=(46.8, 8.2))
    f = weighted_box_fusion([[rgb], [lid]])[0]

    assert f.detection_box() == tuple(f.spatial_context.bounding_box)
    d = Detection.from_clue(f)  # the Phase 1 evaluator accepts it unchanged
    assert d.frame_id == "frame_0001" and d.label == "person"
    assert 0.0 <= f.confidence_score <= 1.0
    assert ClueContract.model_validate(f.model_dump()).parent_clue_ids == f.parent_clue_ids


def test_stubs_emit_valid_clues():
    targets = [Target((100, 100, 160, 240), (46.8182, 8.2275))]
    rgb = yolo11m_stub(seed=1).detect("frame_0001", targets)
    lidar = lidar_stub(seed=1).detect("frame_0001", targets)

    assert all(isinstance(c, ClueContract) for c in rgb + lidar)
    assert all(c.detection_box() for c in rgb + lidar)
    assert all(c.source_agent is AgentSource.DRONE_RGB for c in rgb)
    assert all(c.source_agent is AgentSource.DRONE_LIDAR for c in lidar)
    assert all(c.parent_clue_ids is None for c in rgb + lidar), "raw detections have no parents"
    # Only LiDAR carries a measured range — the reason it is worth fusing in.
    assert any("range_m" in c.agent_metadata for c in lidar)
    assert not any("range_m" in c.agent_metadata for c in rgb)


def test_stub_output_fuses_end_to_end():
    targets = [Target((100, 100, 160, 240), (46.8182, 8.2275)),
               Target((600, 300, 650, 420), (46.8190, 8.2280))]
    rgb = yolo11m_stub(seed=3, recall=1.0, fp_per_frame=0.0)
    lidar = lidar_stub(seed=3, recall=1.0, fp_per_frame=0.0)
    fused = weighted_box_fusion([rgb.detect("frame_0001", targets),
                                 lidar.detect("frame_0001", targets)])

    assert len(fused) == 2, f"two targets, both seen by both sensors: {len(fused)}"
    assert all(f.agent_metadata["wbf_support"] == 2 for f in fused)
    assert all(len(f.parent_clue_ids) == 2 for f in fused)
    assert all(0.0 <= f.confidence_score <= 1.0 for f in fused)


# --------------------------------------------------------------------------
# Fusion: source and range propagation
# --------------------------------------------------------------------------

def test_fused_clue_reports_fusion_as_its_source():
    f = weighted_box_fusion([[_clue((0, 0, 10, 10), 0.9)], [_lidar((1, 1, 11, 11), 0.8)]])[0]
    assert f.source_agent is AgentSource.PERCEPTION_FUSION
    assert sorted(f.agent_metadata["wbf_sources"]) == ["DRONE_LIDAR", "DRONE_RGB"]


def test_measured_range_survives_fusion():
    """A confirmed target must carry the LiDAR range, or geolocation is left
    inferring a range the sensor already measured."""
    rgb = _clue((0, 0, 10, 10), 0.9)
    lid = _lidar((1, 1, 11, 11), 0.8)
    lid = lid.model_copy(update={"agent_metadata": {"range_m": 87.5}})

    f = weighted_box_fusion([[rgb], [lid]])[0]
    assert f.agent_metadata["range_m"] == 87.5
    assert f.agent_metadata["range_source"] == "DRONE_LIDAR"
    # No parent measured anything -> no range claimed.
    assert "range_m" not in weighted_box_fusion([[rgb], []])[0].agent_metadata


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

def _track_frames(tracker, boxes_per_frame, conf=0.9, motion=None):
    out = []
    for i, boxes in enumerate(boxes_per_frame):
        clues = [_clue(b, conf, frame=f"frame_{i:04d}") for b in boxes]
        out.append(tracker.update(clues, frame_id=f"frame_{i:04d}", camera_motion=motion))
    return out


def test_track_confirms_after_min_hits():
    tracker = BoTSORT(min_hits=3)
    box = (100, 100, 140, 200)
    frames = _track_frames(tracker, [[box], [box], [box]])
    assert frames[0] == [] and frames[1] == [], "a single sighting is not a target"
    assert len(frames[2]) == 1
    assert frames[2][0].state is TrackState.CONFIRMED
    assert frames[2][0].hits == 3


def test_same_target_keeps_one_id_across_frames():
    tracker = BoTSORT(min_hits=2)
    boxes = [[(100 + 8 * i, 100, 140 + 8 * i, 200)] for i in range(6)]
    confirmed = _track_frames(tracker, boxes)
    ids = {t.track_id for frame in confirmed[1:] for t in frame}
    assert ids == {1}, f"a single moving target must not spawn new ids: {ids}"
    assert len(tracker.tracks) == 1


def test_two_targets_do_not_swap_identity():
    tracker = BoTSORT(min_hits=2)
    left = [(100 + 5 * i, 100, 140 + 5 * i, 200) for i in range(6)]
    right = [(600 - 5 * i, 300, 640 - 5 * i, 400) for i in range(6)]
    _track_frames(tracker, [[left[i], right[i]] for i in range(6)])

    assert len(tracker.tracks) == 2
    by_id = {t.track_id: t.box for t in tracker.tracks}
    assert min(by_id) == 1 and max(by_id) == 2
    # The track that started on the left is still on the left.
    assert by_id[1][0] < by_id[2][0]


def test_camera_motion_compensation_keeps_the_track():
    """The drone pans; the target has not moved. Without compensation that
    looks like the target teleporting, and the track is lost."""
    box = (100, 100, 140, 200)
    shifted = (50, 100, 90, 200)  # 50 px left: zero overlap with the original
    pan = Affine.from_camera_delta(dx_px=-50.0)

    naive = BoTSORT(min_hits=2)
    _track_frames(naive, [[box], [box], [box]])
    naive.update([_clue(shifted, 0.9)], frame_id="frame_0003")
    assert len({t.track_id for t in naive.tracks}) == 2, "uncompensated pan should break the track"

    compensated = BoTSORT(min_hits=2)
    _track_frames(compensated, [[box], [box], [box]])
    live = compensated.update([_clue(shifted, 0.9)], frame_id="frame_0003", camera_motion=pan)
    assert len(compensated.tracks) == 1, "compensated pan must keep one track"
    assert live and live[0].track_id == 1


def test_low_confidence_detection_rescues_a_track():
    """ByteTrack stage two: a target dimmed by canopy keeps its track."""
    tracker = BoTSORT(min_hits=2, high_thresh=0.5, new_track_thresh=0.6)
    box = (100, 100, 140, 200)
    _track_frames(tracker, [[box], [box]])

    tracker.update([_clue(box, 0.2)], frame_id="frame_0002")  # below high_thresh
    assert len(tracker.tracks) == 1, "a weak detection must not spawn a second track"
    assert tracker.tracks[0].track_id == 1
    assert tracker.tracks[0].age == 0, "the weak detection should have updated the track"


def test_weak_detections_never_start_a_track():
    tracker = BoTSORT(new_track_thresh=0.6)
    tracker.update([_clue((100, 100, 140, 200), 0.55)], frame_id="frame_0000")
    assert tracker.tracks == []


def test_lost_track_is_dropped_after_max_age():
    tracker = BoTSORT(min_hits=2, max_age=3)
    box = (100, 100, 140, 200)
    _track_frames(tracker, [[box], [box]])
    assert tracker.tracks[0].state is TrackState.CONFIRMED

    for _ in range(3):
        tracker.update([], frame_id="gap")
        assert tracker.tracks[0].state is TrackState.LOST
    tracker.update([], frame_id="gap")
    assert tracker.tracks == [], "a track missing past max_age must be removed"


def test_unconfirmed_track_dies_on_first_miss():
    tracker = BoTSORT(min_hits=3)
    tracker.update([_clue((100, 100, 140, 200), 0.9)], frame_id="frame_0000")
    assert len(tracker.tracks) == 1 and tracker.tracks[0].state is TrackState.TENTATIVE
    tracker.update([], frame_id="frame_0001")
    assert tracker.tracks == [], "one-frame noise must not linger"


def test_track_predicts_through_a_gap():
    tracker = BoTSORT(min_hits=2, max_age=10)
    _track_frames(tracker, [[(100 + 10 * i, 100, 140 + 10 * i, 200)] for i in range(5)])
    before = tracker.tracks[0].box[0]
    tracker.update([], frame_id="gap")
    after = tracker.tracks[0].box[0]
    assert after > before, "a constant-velocity track must coast through a missed frame"


def test_track_carries_lineage_and_range():
    tracker = BoTSORT(min_hits=2)
    box = (100, 100, 140, 200)
    ranged = _clue(box, 0.9, clue_id="fused-1").model_copy(
        update={"agent_metadata": {"range_m": 64.0}}
    )
    tracker.update([ranged], frame_id="frame_0000")
    tracker.update([_clue(box, 0.9, clue_id="fused-2")], frame_id="frame_0001")

    track = tracker.tracks[0]
    assert track.clue_ids == ["fused-1", "fused-2"]
    assert track.range_m == 64.0, "last measured range must persist across frames"
    assert track.first_frame_id == "frame_0000" and track.last_frame_id == "frame_0001"


def test_classes_never_associate_across_labels():
    tracker = BoTSORT(min_hits=2, max_age=10)
    box = (100, 100, 140, 200)
    tracker.update([_clue(box, 0.9, label="person")], frame_id="frame_0000")
    tracker.update([_clue(box, 0.9, label="person")], frame_id="frame_0001")
    assert tracker.tracks[0].state is TrackState.CONFIRMED

    # Same pixels, different class: perfect IoU must still not associate.
    tracker.update([_clue(box, 0.9, label="backpack")], frame_id="frame_0002")
    assert len(tracker.tracks) == 2, "a backpack must not inherit a person's track"
    assert {t.class_label for t in tracker.tracks} == {"person", "backpack"}


def test_affine_transforms_boxes():
    assert Affine().is_identity
    assert Affine().apply_box((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert Affine.from_camera_delta(dx_px=10, dy_px=-5).apply_box((0, 0, 10, 10)) == (10, -5, 20, 5)

    # A quarter turn about the origin maps the unit box onto the other axis.
    turned = Affine.from_camera_delta(rotation_rad=math.pi / 2).apply_box((0, 0, 10, 10))
    assert all(abs(a - b) < 1e-9 for a, b in zip(turned, (-10, 0, 0, 10))), turned
    assert abs(Affine.from_camera_delta(scale=2.0).scale - 2.0) < 1e-9


# --------------------------------------------------------------------------
# Geolocation
# --------------------------------------------------------------------------

_CAM = Camera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
_CENTRE = (640.0, 360.0)
_DRONE = (46.8182, 8.2275)


def _telemetry(**kw):
    base = dict(latitude=_DRONE[0], longitude=_DRONE[1], altitude_m=100.0,
                yaw_deg=0.0, pitch_deg=90.0)
    return Telemetry(**{**base, **kw})


def test_nadir_view_places_target_under_the_drone():
    fix = geolocate(_CENTRE, _CAM, _telemetry(), dem=ConstantDEM(0.0))
    assert abs(fix.range_m - 100.0) < 1e-3, "straight down, range is the altitude"
    assert ground_distance_m((fix.latitude, fix.longitude), _DRONE) < 0.01
    assert abs(fix.elevation_m - 0.0) < 1e-3
    assert fix.range_source is RangeSource.INFERRED_TERRAIN


def test_forty_five_degrees_puts_the_target_one_altitude_away():
    fix = geolocate(_CENTRE, _CAM, _telemetry(pitch_deg=45.0), dem=ConstantDEM(0.0))
    assert abs(fix.range_m - 100.0 * math.sqrt(2)) < 1e-3
    assert abs(ground_distance_m((fix.latitude, fix.longitude), _DRONE) - 100.0) < 0.01
    assert fix.latitude > _DRONE[0], "yaw 0 looks north"
    assert abs(fix.longitude - _DRONE[1]) < 1e-9


def test_yaw_rotates_the_fix_around_the_drone():
    east = geolocate(_CENTRE, _CAM, _telemetry(pitch_deg=45.0, yaw_deg=90.0))
    assert abs(ground_distance_m((east.latitude, east.longitude), _DRONE) - 100.0) < 0.01
    assert east.longitude > _DRONE[1]
    # A geodesic heading due east curves very slightly toward the equator —
    # parallels are not geodesics, only meridians are. Under a metre at 100 m,
    # but not zero, and a flat-earth offset would wrongly report exactly zero.
    assert 0.0 < abs(east.latitude - _DRONE[0]) < 1e-5

    south = geolocate(_CENTRE, _CAM, _telemetry(pitch_deg=45.0, yaw_deg=180.0))
    assert south.latitude < _DRONE[0]
    assert abs(south.longitude - _DRONE[1]) < 1e-12, "due south follows a meridian exactly"


def test_geodesy_is_ellipsoidal_not_flat_earth():
    """A flat-earth degree offset skews with latitude; a geodesic does not."""
    for latitude in (0.0, 46.8182, 78.0):
        origin = (latitude, 8.2275)
        north = offset_enu(*origin, east_m=0.0, north_m=1000.0)
        east = offset_enu(*origin, east_m=1000.0, north_m=0.0)
        assert abs(ground_distance_m(origin, north) - 1000.0) < 0.01
        assert abs(ground_distance_m(origin, east) - 1000.0) < 0.01

    # The flat-earth shortcut this replaced: 1 degree of latitude is not a
    # constant 111_320 m, so it drifts measurably even over one degree.
    flat = 1.0 / 111_320.0
    high = (78.0, 8.2275)
    geodesic = offset_enu(*high, east_m=0.0, north_m=10_000.0)
    naive = (high[0] + 10_000.0 * flat, high[1])
    assert ground_distance_m(geodesic, naive) > 5.0, "flat-earth error must be visible here"


def test_project_round_trips_through_the_ray():
    telemetry = _telemetry(pitch_deg=35.0, yaw_deg=210.0)
    ray = pixel_ray((700.0, 400.0), _CAM, telemetry)
    latitude, longitude, elevation = project(
        telemetry.latitude, telemetry.longitude, telemetry.altitude_m, ray, 250.0
    )
    horizontal = ground_distance_m((latitude, longitude), _DRONE)
    assert abs(horizontal - 250.0 * math.hypot(ray[0], ray[1])) < 0.02
    assert abs((elevation - telemetry.altitude_m) - 250.0 * ray[2]) < 1e-9


def test_measured_range_always_overrides_inferred():
    """The core asymmetry: a measurement wins even when the inference claims to
    be far more precise."""
    measured = RangeEstimate(50.0, RangeSource.MEASURED_LIDAR, sigma_m=25.0)
    inferred = RangeEstimate(100.0, RangeSource.INFERRED_TERRAIN, sigma_m=0.01)
    assert select_range([inferred, measured]) is measured
    assert select_range([measured, inferred]) is measured

    fix = geolocate(_CENTRE, _CAM, _telemetry(), measured_ranges=[measured])
    assert fix.range_m == 50.0 and fix.is_measured
    assert abs(fix.elevation_m - 50.0) < 1e-6, "measured range must set the elevation"


def test_measured_range_works_where_terrain_inference_cannot():
    """Looking at the horizon, no ground intersection exists — but a measured
    range needs no terrain model at all."""
    horizon = _telemetry(pitch_deg=0.0)
    assert geolocate(_CENTRE, _CAM, horizon) is None
    fix = geolocate(_CENTRE, _CAM, horizon,
                    measured_ranges=[RangeEstimate(80.0, RangeSource.MEASURED_LIDAR)])
    assert fix is not None and fix.range_m == 80.0


def test_range_selection_edge_cases():
    a = RangeEstimate(40.0, RangeSource.MEASURED_LIDAR, sigma_m=5.0)
    b = RangeEstimate(41.0, RangeSource.MEASURED_LIDAR, sigma_m=1.0)
    assert select_range([a, b]) is b, "ties within a class break on uncertainty"

    unknown = RangeEstimate(60.0, RangeSource.INFERRED_TERRAIN)
    stated = RangeEstimate(61.0, RangeSource.INFERRED_TERRAIN, sigma_m=9.0)
    assert select_range([unknown, stated]) is stated, "an unstated sigma is not a good sigma"

    for bad in (0.0, -1.0):
        try:
            RangeEstimate(bad, RangeSource.MEASURED_LIDAR)
            raise AssertionError(f"range {bad} must be rejected")
        except ValueError:
            pass
    try:
        select_range([])
        raise AssertionError("empty candidates must raise")
    except ValueError:
        pass


def test_terrain_inference_refuses_impossible_geometry():
    looking_down = _telemetry(pitch_deg=45.0)
    ray = pixel_ray(_CENTRE, _CAM, looking_down)
    assert intersect_dem(ray, looking_down, ConstantDEM(0.0)) is not None
    # Drone below the terrain it is supposedly looking down at.
    assert intersect_dem(ray, looking_down, ConstantDEM(500.0)) is None
    # Ray pointing at the sky over flat ground.
    up = _telemetry(pitch_deg=-30.0)
    assert intersect_dem(pixel_ray(_CENTRE, _CAM, up), up, ConstantDEM(0.0)) is None


def test_terrain_elevation_shifts_the_fix():
    low = geolocate(_CENTRE, _CAM, _telemetry(), dem=ConstantDEM(0.0))
    high = geolocate(_CENTRE, _CAM, _telemetry(), dem=ConstantDEM(40.0))
    assert abs(high.range_m - 60.0) < 1e-3, "higher ground is closer to the drone"
    assert high.range_m < low.range_m


def test_geolocate_clue_prefers_the_measured_range():
    box = (100.0, 100.0, 140.0, 200.0)
    telemetry = _telemetry(pitch_deg=60.0)
    plain = _clue(box, 0.9)
    ranged = plain.model_copy(update={"agent_metadata": {"range_m": 55.0}})

    inferred_fix = geolocate_clue(plain, _CAM, telemetry, dem=ConstantDEM(0.0))
    measured_fix = geolocate_clue(ranged, _CAM, telemetry, dem=ConstantDEM(0.0))
    assert inferred_fix.range_source is RangeSource.INFERRED_TERRAIN
    assert measured_fix.range_source is RangeSource.MEASURED_LIDAR
    assert measured_fix.range_m == 55.0
    assert measured_fix.sigma_m == 0.5


def test_geolocate_clue_uses_the_ground_contact_pixel():
    """Bottom-centre, not centre: a standing person is located at their feet."""
    box = (100.0, 100.0, 140.0, 200.0)
    telemetry = _telemetry(pitch_deg=60.0)
    fix = geolocate_clue(_clue(box, 0.9), _CAM, telemetry, dem=ConstantDEM(0.0))

    feet = geolocate((120.0, 200.0), _CAM, telemetry, dem=ConstantDEM(0.0))
    centre = geolocate((120.0, 150.0), _CAM, telemetry, dem=ConstantDEM(0.0))
    assert (fix.latitude, fix.longitude) == (feet.latitude, feet.longitude)
    assert (fix.latitude, fix.longitude) != (centre.latitude, centre.longitude)


# --------------------------------------------------------------------------
# DEM
# --------------------------------------------------------------------------

def _slope_dem(rise_per_degree_north=0.0, base=0.0, **kw):
    """DEM tile around the drone, rising toward the north.

    Elevation is anchored at the drone's own latitude so the drone starts above
    ground; a tile referenced to its south edge would bury it.
    """
    return GridDEM.from_function(
        lambda lat, lon: base + (lat - _DRONE[0]) * rise_per_degree_north,
        lat_min=46.80, lon_min=8.20, lat_step=0.0005, lon_step=0.0005,
        n_lat=80, n_lon=80, **kw,
    )


def test_grid_dem_interpolates_and_clamps():
    dem = GridDEM([[0.0, 10.0], [20.0, 30.0]], lat_min=46.0, lon_min=8.0,
                  lat_step=1.0, lon_step=1.0)
    assert dem.elevation(46.0, 8.0) == 0.0      # corner
    assert dem.elevation(47.0, 9.0) == 30.0     # opposite corner
    assert abs(dem.elevation(46.5, 8.5) - 15.0) < 1e-9   # bilinear centre
    assert abs(dem.elevation(46.0, 8.5) - 5.0) < 1e-9    # along an edge
    # Outside the tile clamps rather than extrapolating away.
    assert dem.elevation(40.0, 0.0) == 0.0
    assert dem.elevation(90.0, 90.0) == 30.0
    assert dem.covers(46.5, 8.5) and not dem.covers(50.0, 8.5)

    for bad in ([], [[]], [[1.0, 2.0], [3.0]]):
        try:
            GridDEM(bad, 46.0, 8.0, 1.0, 1.0)
            raise AssertionError(f"malformed grid {bad} must be rejected")
        except ValueError:
            pass


def test_dem_slope_moves_the_fix_off_the_flat_answer():
    """The whole point of a DEM: on a slope the flat-ground answer is wrong."""
    telemetry = _telemetry(pitch_deg=30.0, altitude_m=200.0)  # shallow look north
    flat = geolocate(_CENTRE, _CAM, telemetry, dem=ConstantDEM(0.0))
    rising = geolocate(_CENTRE, _CAM, telemetry, dem=_slope_dem(rise_per_degree_north=20_000.0))

    assert rising.range_m < flat.range_m, "ground rising toward the target is nearer"
    displacement = ground_distance_m(
        (flat.latitude, flat.longitude), (rising.latitude, rising.longitude)
    )
    assert displacement > 25.0, f"flat-ground assumption is off by {displacement:.1f} m"


def test_ray_can_strike_rising_ground_above_the_horizon():
    """A flat-ground model can never report this; ray marching can."""
    telemetry = _telemetry(pitch_deg=-2.0, altitude_m=50.0)  # looking slightly up
    assert geolocate(_CENTRE, _CAM, telemetry, dem=ConstantDEM(0.0)) is None

    hillside = _slope_dem(rise_per_degree_north=60_000.0)
    fix = geolocate(_CENTRE, _CAM, telemetry, dem=hillside)
    assert fix is not None, "an upward ray must still hit a hill in front of it"
    assert fix.elevation_m > telemetry.altitude_m, "the hit is above the drone"


def test_dem_uncertainty_propagates_into_the_fix():
    steep = geolocate(_CENTRE, _CAM, _telemetry(pitch_deg=90.0), dem=ConstantDEM(0.0))
    shallow = geolocate(_CENTRE, _CAM, _telemetry(pitch_deg=8.0), dem=ConstantDEM(0.0))
    assert shallow.sigma_m > steep.sigma_m, "a grazing look magnifies terrain error"
    assert steep.sigma_m >= ConstantDEM().vertical_uncertainty_m

    # A measured range is unaffected by how good or bad the DEM is.
    measured = geolocate(_CENTRE, _CAM, _telemetry(pitch_deg=8.0),
                         measured_ranges=[RangeEstimate(40.0, RangeSource.MEASURED_LIDAR, 0.5)],
                         dem=ConstantDEM(0.0, vertical_uncertainty_m=500.0))
    assert measured.sigma_m == 0.5


# --------------------------------------------------------------------------
# Real raster tiles
# --------------------------------------------------------------------------

def _write_hgt(directory, name="N46E008.hgt", side=1201, height=lambda r, c: 1000 + r):
    """A real SRTM tile: raw big-endian int16, north-west corner in the name."""
    values = [height(r, c) for r in range(side) for c in range(side)]
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(struct.pack(f">{len(values)}h", *values))
    return path


def test_srtm_tile_names_give_the_south_west_corner():
    assert parse_hgt_name("N46E008.hgt") == (46.0, 8.0)
    assert parse_hgt_name("S34W071.hgt") == (-34.0, -71.0)
    assert parse_hgt_name("n46e008.hgt") == (46.0, 8.0), "case must not matter"
    for bad in ("hello.hgt", "X46E008.hgt", "N46Q008.hgt"):
        try:
            parse_hgt_name(bad)
            raise AssertionError(f"{bad} must not parse")
        except ValueError:
            pass


def test_a_real_srtm_tile_reads_without_any_dependency():
    """`.hgt` is fully specified — a square of big-endian int16 with the corner
    in the filename — so it needs no parser and no third-party package."""
    directory = tempfile.mkdtemp()
    dem = SrtmHgtDEM(_write_hgt(directory))

    assert dem.rows == dem.cols == 1201
    assert abs(dem.lat_min - 46.0) < 1e-9 and abs(dem.lat_max - 47.0) < 1e-9
    assert abs(dem.lon_min - 8.0) < 1e-9 and abs(dem.lon_max - 9.0) < 1e-9
    # Row 0 is the northern edge, so height rises going south in this fixture.
    assert dem.elevation(47.0, 8.0) == 1000.0
    assert dem.elevation(46.0, 8.0) == 1000.0 + 1200
    # Bilinear between two rows.
    half_step = (3.0 / 3600.0) / 2.0
    assert abs(dem.elevation(47.0 - half_step, 8.5) - 1000.5) < 1e-6
    # Three arcseconds at 46 degrees north.
    assert 60.0 < dem.resolution_m < 95.0
    assert dem.covers(46.5, 8.5) and not dem.covers(48.0, 8.5)


def test_a_tile_of_the_wrong_size_is_refused():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "N46E008.hgt")
    with open(path, "wb") as handle:
        handle.write(b"\x00" * 4096)
    try:
        SrtmHgtDEM(path)
        raise AssertionError("a non-square byte count must be refused")
    except ValueError as e:
        assert "SRTM tile" in str(e)


def test_srtm_voids_are_not_treated_as_elevations():
    """SRTM leaves holes in steep shadowed terrain. Averaging one in would put
    a target 32 km underground."""
    directory = tempfile.mkdtemp()
    hole = lambda r, c: -32768 if (r, c) == (600, 600) else 1500   # noqa: E731
    dem = SrtmHgtDEM(_write_hgt(directory, side=1201, height=hole))

    step = 3.0 / 3600.0
    # Exactly on the void there is nothing real to weight, so there is no
    # answer. Filling from a neighbour here would be a silent interpolation
    # across the hole, which is a processing decision, not a measurement.
    assert dem.elevation(47.0 - 600 * step, 8.0 + 600 * step) is None
    # Half a post away, the three real corners carry it.
    assert dem.elevation(47.0 - 600.5 * step, 8.0 + 600.5 * step) == 1500.0
    assert dem.elevation(46.3, 8.3) == 1500.0, "far from the hole, nothing changes"

    everywhere = SrtmHgtDEM(_write_hgt(directory, "N10E010.hgt", 1201, lambda r, c: -32768))
    assert everywhere.elevation(10.5, 10.5) is None, "all void means no answer, not zero"


def test_ray_marching_swaps_onto_a_real_tile_unchanged():
    """The point of the interface: geolocation does not know or care which DEM
    it is marching against."""
    directory = tempfile.mkdtemp()
    dem = SrtmHgtDEM(_write_hgt(directory, height=lambda r, c: 1200))
    telemetry = Telemetry(46.5, 8.5, altitude_m=1500.0, yaw_deg=0.0, pitch_deg=90.0)

    fix = geolocate(_CENTRE, _CAM, telemetry, dem=dem)
    assert fix is not None and fix.range_source is RangeSource.INFERRED_TERRAIN
    assert abs(fix.range_m - 300.0) < 1.0, "1500 m above 1200 m ground, looking straight down"
    assert abs(fix.elevation_m - 1200.0) < 1.0
    assert fix.sigma_m >= dem.vertical_uncertainty_m


def test_marching_into_a_void_declines_rather_than_guesses():
    directory = tempfile.mkdtemp()
    dem = SrtmHgtDEM(_write_hgt(directory, "N20E020.hgt", 1201, lambda r, c: -32768))
    telemetry = Telemetry(20.5, 20.5, altitude_m=1500.0, yaw_deg=0.0, pitch_deg=90.0)
    assert geolocate(_CENTRE, _CAM, telemetry, dem=dem) is None, (
        "unknown terrain must produce no fix, never an assumed one")


def test_geotiff_needs_rasterio_and_says_so():
    """Rather than shipping a half-correct TIFF parser that fails on somebody's
    tile in the field with a silently wrong altitude."""
    try:
        import rasterio  # noqa: F401
        return  # installed here; the import path is exercised for real
    except ImportError:
        pass

    try:
        GeoTiffDEM("nonexistent.tif")
        raise AssertionError("must not silently succeed without rasterio")
    except ImportError as e:
        assert "pip install rasterio" in str(e)
        assert "SrtmHgtDEM" in str(e), "and point at the dependency-free option"


def test_open_dem_dispatches_on_extension():
    directory = tempfile.mkdtemp()
    assert isinstance(open_dem(_write_hgt(directory)), SrtmHgtDEM)

    grid = GridDEM([[0.0, 1.0], [2.0, 3.0]], 46.0, 8.0, 1.0, 1.0)
    json_path = os.path.join(directory, "tile.json")
    grid.to_json(json_path)
    assert isinstance(open_dem(json_path), GridDEM)

    try:
        open_dem("terrain.xyz")
        raise AssertionError("an unknown extension must raise")
    except ValueError as e:
        assert "no DEM reader" in str(e)


def test_every_dem_source_offers_the_same_interface():
    """Ray-marching depends on exactly three members; a source missing one
    would fail only when a search was already under way."""
    directory = tempfile.mkdtemp()
    sources = [
        ConstantDEM(1200.0),
        GridDEM([[1.0, 2.0], [3.0, 4.0]], 46.0, 8.0, 0.1, 0.1),
        SrtmHgtDEM(_write_hgt(directory, "N30E030.hgt")),
    ]
    for dem in sources:
        assert callable(dem.elevation)
        assert isinstance(dem.vertical_uncertainty_m, float)
        assert dem.resolution_m > 0
        assert repr(dem)


def test_dem_json_round_trip(tmp=None):
    import tempfile, os
    dem = _slope_dem(rise_per_degree_north=1000.0, base=400.0)
    path = os.path.join(tempfile.mkdtemp(), "dem.json")
    dem.to_json(path)
    loaded = load_dem(path)
    assert loaded.n_lat == dem.n_lat and loaded.n_lon == dem.n_lon
    for lat, lon in ((46.81, 8.21), (46.8182, 8.2275), (46.83, 8.23)):
        assert abs(loaded.elevation(lat, lon) - dem.elevation(lat, lon)) < 1e-9


def test_pixel_ray_is_a_unit_vector():
    for pitch in (0.0, 30.0, 90.0):
        for pixel in (_CENTRE, (0.0, 0.0), (1280.0, 720.0)):
            ray = pixel_ray(pixel, _CAM, _telemetry(pitch_deg=pitch, roll_deg=15.0))
            assert abs(math.sqrt(sum(c * c for c in ray)) - 1.0) < 1e-12


def test_fused_track_geolocates_end_to_end():
    """The Phase 2 chain: two sensors -> fusion -> tracker -> a position."""
    targets = [Target((600, 300, 650, 420), (46.8190, 8.2280))]
    rgb = yolo11m_stub(seed=5, recall=1.0, fp_per_frame=0.0)
    lidar = lidar_stub(seed=5, recall=1.0, fp_per_frame=0.0)
    tracker = BoTSORT(min_hits=2)

    confirmed = []
    for i in range(4):
        frame = f"frame_{i:04d}"
        fused = weighted_box_fusion([rgb.detect(frame, targets), lidar.detect(frame, targets)])
        assert len(fused) == 1
        confirmed = tracker.update(fused, frame_id=frame)

    assert len(confirmed) == 1
    track = confirmed[0]
    assert track.state is TrackState.CONFIRMED and track.range_m is not None

    fix = geolocate(
        ((track.box[0] + track.box[2]) / 2.0, track.box[3]),
        _CAM,
        _telemetry(pitch_deg=70.0),
        measured_ranges=[RangeEstimate(track.range_m, RangeSource.MEASURED_LIDAR, 0.5)],
    )
    assert isinstance(fix, Fix) and fix.is_measured
    assert fix.range_m == track.range_m


def test_camera_model_round_trips_world_to_pixel_to_world():
    """world -> pixel -> geolocate must return the point it started from.

    This pins the camera model, the rotation conventions and the geodesy against
    each other. A sign error anywhere breaks it.
    """
    telemetry = _telemetry(pitch_deg=55.0, yaw_deg=15.0, altitude_m=1570.0, roll_deg=4.0)
    for latitude, longitude, elevation in (
        (46.8190, 8.2280, 1460.0),
        (46.8186, 8.2265, 1455.0),
        (46.8175, 8.2290, 1490.0),
    ):
        pixel = world_to_pixel(latitude, longitude, elevation, _CAM, telemetry)
        assert pixel is not None
        fix = geolocate(pixel, _CAM, telemetry, measured_ranges=[
            RangeEstimate(slant_range_m(telemetry, latitude, longitude, elevation),
                          RangeSource.MEASURED_LIDAR, 0.5)
        ])
        assert ground_distance_m((fix.latitude, fix.longitude), (latitude, longitude)) < 0.01
        assert abs(fix.elevation_m - elevation) < 0.01


def test_dem_intersection_round_trips_without_any_measured_range():
    """The RGB-only path: a point standing on the terrain must be recovered from
    its pixel using the DEM alone. This is what validates the ray march."""
    telemetry = _telemetry(pitch_deg=50.0, yaw_deg=0.0, altitude_m=1600.0)
    dem = _slope_dem(rise_per_degree_north=20_000.0, base=1450.0)

    for latitude, longitude in ((46.8195, 8.2275), (46.8200, 8.2290)):
        elevation = dem.elevation(latitude, longitude)
        pixel = world_to_pixel(latitude, longitude, elevation, _CAM, telemetry)
        assert pixel is not None
        fix = geolocate(pixel, _CAM, telemetry, dem=dem)
        assert fix is not None and fix.range_source is RangeSource.INFERRED_TERRAIN
        error = ground_distance_m((fix.latitude, fix.longitude), (latitude, longitude))
        assert error < 0.05, f"DEM intersection off by {error:.3f} m"


def test_world_to_pixel_rejects_points_behind_the_camera():
    looking_north = _telemetry(pitch_deg=0.0, yaw_deg=0.0)
    behind = offset_enu(*_DRONE, east_m=0.0, north_m=-500.0)
    assert world_to_pixel(behind[0], behind[1], 100.0, _CAM, looking_north) is None


# --------------------------------------------------------------------------
# Bus + detection agent
# --------------------------------------------------------------------------

def _bus(**kwargs):
    """RedisBus over a mocked connection — the bus code under test is real."""
    return RedisBus(FakeRedisStreams(**kwargs))


def test_bus_round_trips_clues_through_serialization():
    bus = _bus()
    clue = _clue((0, 0, 10, 10), 0.9, case="case-0000")
    entry_id = bus.publish(clue)

    assert bus.streams() == ["clues:case-0000"], "one stream per case"
    assert bus.length("clues:case-0000") == 1
    (read_id, read_clue), = bus.read("clues:case-0000")
    assert read_id == entry_id
    assert read_clue.model_dump() == clue.model_dump(), "must survive the wire unchanged"

    try:
        bus.publish({"clue_id": "not-a-contract"})
        raise AssertionError("only ClueContract may be published")
    except TypeError:
        pass


def test_bus_reads_resume_from_the_last_id():
    bus = _bus()
    ids = [bus.publish(_clue((0, 0, 10, 10), 0.5)) for _ in range(5)]
    stream = stream_for("case-0000")

    assert len(bus.read(stream)) == 5
    assert [i for i, _ in bus.read(stream, last_id=ids[2])] == ids[3:]
    assert bus.read(stream, last_id=ids[-1]) == [], "XREAD returns strictly newer entries"
    assert len(bus.read(stream, count=2)) == 2
    assert bus.read("clues:never-existed") == []


def test_bus_keeps_cases_apart():
    bus = _bus()
    bus.publish(_clue((0, 0, 10, 10), 0.9, case="case-A"))
    bus.publish(_clue((0, 0, 10, 10), 0.9, case="case-B"))
    assert bus.length("clues:case-A") == 1 and bus.length("clues:case-B") == 1
    (_, only), = bus.read("clues:case-A")
    assert only.case_id == "case-A"
    assert bus.streams() == ["clues:case-A", "clues:case-B"]


def test_bus_survives_a_bytes_client():
    """redis-py returns bytes unless decode_responses is set. Code that only
    ever sees str works in tests and breaks against a live server."""
    bus = _bus(decode_responses=False)
    clue = _clue((0, 0, 10, 10), 0.9, case="case-0000")
    entry_id = bus.publish(clue)

    assert isinstance(entry_id, str), "entry ids must be decoded on the way out"
    (read_id, read_clue), = bus.read(stream_for("case-0000"))
    assert read_id == entry_id
    assert read_clue.model_dump() == clue.model_dump()
    assert bus.streams() == ["clues:case-0000"]


def test_bus_entry_ids_are_redis_shaped_and_monotonic():
    bus = _bus()
    ids = [bus.publish(_clue((0, 0, 10, 10), 0.5)) for _ in range(20)]
    parsed = [tuple(int(p) for p in i.split("-")) for i in ids]
    assert all(len(i.split("-")) == 2 for i in ids), "Redis ids are <ms>-<seq>"
    assert parsed == sorted(parsed) and len(set(parsed)) == len(parsed)


def test_bus_publishes_one_json_field_per_entry():
    """The wire format is what a real consumer will see."""
    client = FakeRedisStreams()
    RedisBus(client).publish(_clue((0, 0, 10, 10), 0.9, case="case-0000"))
    (_, fields), = client.streams["clues:case-0000"]
    assert list(fields) == [CLUE_FIELD]
    assert ClueContract.model_validate_json(fields[CLUE_FIELD]).case_id == "case-0000"


def test_mocked_connection_matches_the_real_redis_client():
    """The mock is only worth anything if it accepts what redis-py accepts.

    No live server here, so this is the guard against drift: every call
    `RedisBus` makes must bind against the real `redis.Redis` signature *and*
    against the fake's. A renamed kwarg fails here instead of in flight.
    """
    import inspect

    import redis

    calls = {
        "xadd": (("clues:case-0000", {CLUE_FIELD: "{}"}), {"maxlen": 100, "approximate": True}),
        "xread": (({"clues:case-0000": "0-0"},), {"count": 2}),
        "xlen": (("clues:case-0000",), {}),
        "scan_iter": ((), {"match": "clues:*"}),
        "ping": ((), {}),
    }
    fake = FakeRedisStreams()
    for name, (args, kwargs) in calls.items():
        real_method = getattr(redis.Redis, name, None)
        assert real_method is not None, f"redis-py has no {name}"
        # `self` is unbound on the class, hence the leading None.
        inspect.signature(real_method).bind(None, *args, **kwargs)
        inspect.signature(getattr(fake, name)).bind(*args, **kwargs)

    assert callable(getattr(redis.Redis, "from_url", None))


def test_bus_maxlen_trims_the_stream():
    bus = RedisBus(FakeRedisStreams(), maxlen=3)
    for _ in range(10):
        bus.publish(_clue((0, 0, 10, 10), 0.5))
    stream = stream_for("case-0000")
    assert bus.length(stream) == 3, "a long flight must not grow the stream without bound"
    assert len(bus.read(stream)) == 3


def _run_agent(frames=4, dem=None, telemetry=None, case_id="case-0000", targets=None,
               use_rgb=True, use_lidar=True, tracker=None):
    bus = _bus()
    agent = DetectionAgent(
        bus, case_id, _CAM, dem=dem if dem is not None else ConstantDEM(0.0),
        tracker=tracker or BoTSORT(min_hits=2),
    )
    targets = targets or [Target((600, 300, 650, 420), (46.8190, 8.2280))]
    rgb = yolo11m_stub(seed=7, recall=1.0, fp_per_frame=0.0)
    lidar = lidar_stub(seed=7, recall=1.0, fp_per_frame=0.0)

    published = []
    for i in range(frames):
        frame = f"frame_{i:04d}"
        sensors = [rgb.detect(frame, targets, case_id)] if use_rgb else []
        if use_lidar:
            sensors.append(lidar.detect(frame, targets, case_id))
        published += agent.process_frame(
            frame, sensors, telemetry or _telemetry(pitch_deg=70.0, altitude_m=120.0)
        )
    return bus, agent, published


def test_confirmed_tracks_land_on_the_bus():
    bus, agent, published = _run_agent(frames=4)
    stream = stream_for("case-0000")

    assert agent.published == len(published) > 0
    assert bus.length(stream) == agent.published
    # min_hits=2, so frame 0 publishes nothing and frames 1..3 each publish once.
    assert len(published) == 3

    for _, clue in bus.read(stream):
        assert clue.source_agent is AgentSource.PERCEPTION_FUSION
        assert clue.provenance_tag == TRACK_PROVENANCE
        assert clue.case_id == "case-0000"
        assert clue.agent_metadata["track_state"] == "CONFIRMED"
        assert clue.spatial_context.bounding_box is not None


def test_published_clues_are_properly_geolocated():
    bus, agent, _ = _run_agent(frames=4)
    (_, clue), = bus.read(stream_for("case-0000"), count=1)

    assert agent.without_fix == 0
    assert clue.agent_metadata["geolocation"] == "located"
    assert clue.agent_metadata["range_source"] == "MEASURED_LIDAR", "LiDAR measured it"
    assert clue.spatial_context.latitude is not None
    # The fix must land near the real target, not at the drone or at (0, 0).
    error = ground_distance_m(
        (clue.spatial_context.latitude, clue.spatial_context.longitude), (46.8190, 8.2280)
    )
    assert error < 400.0, f"fix is {error:.0f} m from the target"


def test_lineage_survives_all_the_way_to_the_bus():
    """RGB + LiDAR -> fused -> track -> published clue, parents intact."""
    bus, _, _ = _run_agent(frames=4)
    (_, clue), = bus.read(stream_for("case-0000"), count=1)
    assert clue.parent_clue_ids, "a published track must name the clues behind it"
    assert all(isinstance(p, str) for p in clue.parent_clue_ids)
    assert clue.clue_id not in clue.parent_clue_ids


def test_unlocatable_track_publishes_without_inventing_a_position():
    """No DEM intersection and no measured range: the detection is still
    reported, the position is not guessed. Placeholder coordinates would send a
    team to the wrong place.

    RGB only — with LiDAR present a measured range geolocates this fine, which
    is the whole point of preferring it.
    """
    horizon = _telemetry(pitch_deg=-5.0, altitude_m=120.0)  # looking at the sky
    bus, agent, published = _run_agent(
        frames=4, telemetry=horizon, dem=ConstantDEM(0.0), use_lidar=False
    )

    located = [c for c in published if c.agent_metadata["geolocation"] == "located"]
    unlocated = [c for c in published if c.agent_metadata["geolocation"] == "no_fix"]
    assert unlocated and not located
    assert agent.without_fix == len(unlocated)

    for clue in unlocated:
        assert clue.spatial_context.latitude is None
        assert clue.spatial_context.longitude is None
        assert clue.spatial_context.bounding_box is not None, "the sighting is not lost"
        assert "position unavailable" in clue.finding_summary

    # Same geometry, but LiDAR measured the range: now it locates.
    _, with_lidar, lidar_published = _run_agent(
        frames=4, telemetry=horizon, dem=ConstantDEM(0.0), use_lidar=True
    )
    assert with_lidar.without_fix == 0
    assert all(c.agent_metadata["range_source"] == "MEASURED_LIDAR" for c in lidar_published)


def test_agent_clue_ids_are_stable_per_track_and_frame():
    bus_a, _, first = _run_agent(frames=3)
    bus_b, _, second = _run_agent(frames=3)
    assert [c.clue_id for c in first] == [c.clue_id for c in second]
    assert len({c.clue_id for c in first}) == len(first), "one clue per track per frame"


def test_agent_refuses_clues_from_another_case():
    bus = _bus()
    agent = DetectionAgent(bus, "case-mine", _CAM, dem=ConstantDEM(0.0))
    try:
        agent.process_frame(
            "frame_0000",
            [[_clue((0, 0, 10, 10), 0.9, case="case-theirs")], []],
            _telemetry(),
        )
        raise AssertionError("clues from another case must not be published")
    except ValueError as e:
        assert "case-theirs" in str(e)
    assert bus.length(stream_for("case-mine")) == 0


def test_rgb_only_airframe_never_fuses_and_says_so():
    """A DJI with one camera fitted: YOLO's boxes reach the tracker untouched."""
    bus, agent, published = _run_agent(frames=4, use_lidar=False)

    assert agent.published == len(published) == 3
    for clue in published:
        assert clue.agent_metadata["sensors"] == ["DRONE_RGB"]
        assert clue.agent_metadata["range_source"] == "INFERRED_TERRAIN", "no LiDAR to measure it"
        assert clue.spatial_context.bounding_box is not None
        # Still the tracked product of perception, whatever fed it — the tag and
        # the agent are what the provenance allow-list binds together.
        assert clue.source_agent is AgentSource.PERCEPTION_FUSION
        assert clue.provenance_tag == TRACK_PROVENANCE
        assert "DRONE_RGB" in clue.finding_summary


def test_lidar_only_airframe_keeps_its_measured_range():
    """And needs its spawn threshold retuned, which is the point of pinning it.

    BoT-SORT's shipped `new_track_thresh` of 0.6 is a *fused-stream* number. The
    LiDAR detector is deliberately the less certain of the two — it sees a shape,
    not a class — so alone it rarely clears 0.6 and confirms nothing at all. The
    sensor-agnostic pipeline routes single-sensor boxes correctly; it cannot make
    a threshold calibrated for two sensors right for one.
    """
    _, _, unretuned = _run_agent(frames=4, use_rgb=False)
    assert unretuned == [], "a LiDAR-only airframe confirms nothing at the fused threshold"

    bus, agent, published = _run_agent(
        frames=4, use_rgb=False, tracker=BoTSORT(min_hits=2, new_track_thresh=0.45),
    )
    assert agent.published == len(published) == 3
    assert agent.without_fix == 0
    for clue in published:
        assert clue.agent_metadata["sensors"] == ["DRONE_LIDAR"]
        assert clue.agent_metadata["range_source"] == "MEASURED_LIDAR"
        assert clue.spatial_context.latitude is not None


def test_both_feeds_still_fuse_and_name_both_sensors():
    _, _, published = _run_agent(frames=4)
    for clue in published:
        assert clue.agent_metadata["sensors"] == ["DRONE_LIDAR", "DRONE_RGB"]


def test_a_quiet_feed_is_not_an_absent_one():
    """The corroboration penalty must survive the sensor-agnostic refactor.

    A LiDAR detection that the fitted RGB camera did not back is worth less than
    the same detection on an airframe with no camera to back it. Dropping empty
    groups would quietly delete that distinction and make every lone sighting
    look as good as a corroborated one.
    """
    targets = [Target((600, 300, 650, 420), (46.8190, 8.2280))]
    lidar_clues = lidar_stub(seed=7, recall=1.0, fp_per_frame=0.0).detect(
        "frame_0000", targets, "case-0000")
    telemetry = _telemetry(pitch_deg=70.0, altitude_m=120.0)

    def publish(feeds):
        # Thresholds dropped so both runs spawn a track: the penalty under test
        # is large enough to push the penalised detection under the default
        # new_track_thresh, and then there would be nothing to compare.
        agent = DetectionAgent(_bus(), "case-0000", _CAM, dem=ConstantDEM(0.0),
                               tracker=BoTSORT(min_hits=1, high_thresh=0.1,
                                               new_track_thresh=0.2))
        (clue,) = agent.process_frame("frame_0000", feeds, telemetry)
        return clue

    quiet_camera = publish([[], lidar_clues])   # RGB fitted, saw nothing
    no_camera = publish([lidar_clues])          # no RGB fitted at all

    assert quiet_camera.confidence_score < no_camera.confidence_score
    assert quiet_camera.agent_metadata["sensors"] == ["DRONE_LIDAR"]
    assert no_camera.agent_metadata["sensors"] == ["DRONE_LIDAR"]


def test_capture_runs_only_the_sensors_that_delivered():
    """No feed, no model call. A LiDAR model on an RGB-only drone is pure cost."""
    calls = []

    def counted(name, detector):
        def detect(frame_id, feed, case_id):
            calls.append(name)
            return detector.detect(frame_id, feed, case_id)
        return detect

    agent = DetectionAgent(
        _bus(), "case-0000", _CAM, dem=ConstantDEM(0.0), tracker=BoTSORT(min_hits=1),
        detectors={"rgb": counted("rgb", yolo11m_stub(seed=7, recall=1.0, fp_per_frame=0.0)),
                   "lidar": counted("lidar", lidar_stub(seed=7, recall=1.0, fp_per_frame=0.0))},
    )
    targets = [Target((600, 300, 650, 420), (46.8190, 8.2280))]
    published = agent.capture("frame_0000", {"rgb": targets},
                              _telemetry(pitch_deg=70.0, altitude_m=120.0))

    assert calls == ["rgb"], f"only the fitted feed may run a model, got {calls}"
    assert published and published[0].agent_metadata["sensors"] == ["DRONE_RGB"]

    published = agent.capture("frame_0001", {"rgb": targets, "lidar": targets},
                              _telemetry(pitch_deg=70.0, altitude_m=120.0))
    assert calls == ["rgb", "rgb", "lidar"], "both feeds present, both models run"
    assert published[0].agent_metadata["sensors"] == ["DRONE_LIDAR", "DRONE_RGB"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
