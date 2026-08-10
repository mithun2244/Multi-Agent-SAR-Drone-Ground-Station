"""Known-answer checks for the Phase 1 harness.

    python -m src.evaluation.test_evaluation
"""

import json
import uuid
from datetime import datetime, timezone

from pydantic import ValidationError

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from .dataset import Detection, GroundTruth, Split, build_splits, mock_rgb_detector
from .harness import format_report, run_baseline
from .metrics import (
    average_precision,
    evaluate,
    haversine_m,
    iou,
    match_detections,
    recall_at_far,
)


def _gt(frame, box, geo=(46.8, 8.2)):
    return GroundTruth(frame, "person", box, geo)


def _det(frame, box, conf, geo=(46.8, 8.2)):
    return Detection(frame, "person", box, conf, geo)


def test_iou():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert iou((0, 0, 2, 2), (2, 2, 4, 4)) == 0.0  # touching, not overlapping
    assert iou((0, 0, 2, 2), (5, 5, 6, 6)) == 0.0
    # half-overlap: intersection 2, union 6
    assert abs(iou((0, 0, 2, 2), (1, 0, 3, 2)) - 2 / 6) < 1e-9
    assert iou((0, 0, 0, 0), (0, 0, 2, 2)) == 0.0  # degenerate box, no divide-by-zero


def test_haversine():
    # One degree of latitude at the equator.
    assert abs(haversine_m((0.0, 0.0), (1.0, 0.0)) - 111_194.9) < 1.0
    assert haversine_m((46.8, 8.2), (46.8, 8.2)) == 0.0


def test_each_gt_claimed_once():
    gts = [_gt("f0", (0, 0, 10, 10))]
    dets = [_det("f0", (0, 0, 10, 10), 0.9), _det("f0", (0, 0, 10, 10), 0.8)]
    matches = match_detections(dets, gts)
    assert [m.is_tp for m in matches] == [True, False], "one GT must not score twice"


def test_frames_are_isolated():
    matches = match_detections([_det("f1", (0, 0, 10, 10), 0.9)], [_gt("f0", (0, 0, 10, 10))])
    assert matches[0].is_tp is False, "a detection must not match another frame's GT"


def test_average_precision_known_value():
    # 2 GT; detections in confidence order: TP, FP, TP -> AP = 0.5 + 0.5*(2/3)
    matches = match_detections(
        [
            _det("f0", (0, 0, 10, 10), 0.9),
            _det("f0", (500, 500, 510, 510), 0.8),
            _det("f1", (0, 0, 10, 10), 0.7),
        ],
        [_gt("f0", (0, 0, 10, 10)), _gt("f1", (0, 0, 10, 10))],
    )
    assert [m.is_tp for m in matches] == [True, False, True]
    assert abs(average_precision(matches, 2) - 0.8333333) < 1e-6
    assert average_precision(matches, 0) is None  # nothing to score against


def test_recall_at_far_respects_budget():
    matches = match_detections(
        [
            _det("f0", (0, 0, 10, 10), 0.9),
            _det("f0", (500, 500, 510, 510), 0.8),
            _det("f1", (0, 0, 10, 10), 0.7),
        ],
        [_gt("f0", (0, 0, 10, 10)), _gt("f1", (0, 0, 10, 10))],
    )
    # 10 frames, 0.1 FP/frame => 1 false alarm allowed => both TPs reachable.
    recall, thr, far = recall_at_far(matches, n_gt=2, n_frames=10, far_target=0.1)
    assert recall == 1.0 and thr == 0.7 and abs(far - 0.1) < 1e-9
    # Zero tolerance stops before the false alarm.
    assert recall_at_far(matches, 2, 10, 0.0)[0] == 0.5
    assert recall_at_far(matches, 0, 10, 0.1) == (0.0, None, 0.0)


def test_perfect_detector_is_perfect():
    gts = [_gt(f"f{i}", (0, 0, 10, 10)) for i in range(5)]
    dets = [_det(g.frame_id, g.box, 1.0, g.geo) for g in gts]
    r = evaluate(dets, gts, n_frames=5, far_target=0.0)
    assert r["mAP"] == 1.0
    assert r["recall_at_far"] == 1.0
    assert r["geolocation"]["mean_m"] == 0.0 and r["geolocation"]["n"] == 5


def test_no_detections_scores_zero_not_crash():
    r = evaluate([], [_gt("f0", (0, 0, 10, 10))], n_frames=1)
    assert r["mAP"] == 0.0 and r["recall_at_far"] == 0.0
    assert r["geolocation"]["mean_m"] is None


def test_splits_are_fixed_and_disjoint():
    a = build_splits(60, 0.3, seed=0)
    b = build_splits(60, 0.3, seed=0)
    assert a["train"].frame_ids == b["train"].frame_ids, "same seed must give same split"
    assert not set(a["train"].frame_ids) & set(a["validation"].frame_ids)
    assert a["train"].n_frames + a["validation"].n_frames == 60
    assert a["validation"].n_frames == 18  # 30% of 60
    # Ground truth follows its frames.
    val_frames = set(a["validation"].frame_ids)
    assert all(g.frame_id in val_frames for g in a["validation"].ground_truth)


def test_baseline_runs_and_reports():
    split = build_splits(120, 0.3, seed=0)["validation"]
    r = run_baseline(split, seed=0)
    assert 0.0 <= r["mAP"] <= 1.0
    assert 0.0 <= r["recall_at_far"] <= 1.0
    assert r["far_achieved"] <= r["far_target"] + 1e-9
    # Mock detector has ~78% recall and ~18 m geo sigma; sanity-bracket both.
    assert 0.3 < r["mAP"] < 1.0, r["mAP"]
    assert 5.0 < r["geolocation"]["median_m"] < 60.0, r["geolocation"]
    assert "mAP" in format_report(r, "check")

    # Deterministic: same seed, same numbers.
    assert run_baseline(split, seed=0)["mAP"] == r["mAP"]


def test_empty_split_does_not_crash():
    empty = Split("empty", [], [])
    r = run_baseline(empty)
    assert r["mAP"] == 0.0 and r["n_detections"] == 0


def _clue(frame="f0", conf=0.9, box=(0, 0, 10, 10), geo=(46.8, 8.2), case="case-0000",
          label="person"):
    lat, lon = geo if geo else (None, None)
    return ClueContract(
        clue_id=str(uuid.uuid4()),
        case_id=case,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_agent=AgentSource.DRONE_RGB,
        confidence_score=conf,
        finding_summary="Possible person detected in RGB frame",
        spatial_context=SpatialContext(
            latitude=lat, longitude=lon, bounding_box=list(box) if box else None
        ),
        frame_id=frame,
        class_label=label,
        provenance_tag="onboard:rgb-camera",
    )


def test_contract_enforces_confidence_range():
    for bad in (-0.1, 1.1):
        try:
            _clue(conf=bad)
        except ValidationError:
            continue
        raise AssertionError(f"confidence_score={bad} should not validate")
    assert _clue(conf=0.0).confidence_score == 0.0  # bounds are inclusive
    assert _clue(conf=1.0).confidence_score == 1.0


def test_contract_requires_case_and_provenance():
    for missing in ("case_id", "provenance_tag"):
        fields = _clue().model_dump()
        fields.pop(missing)
        try:
            ClueContract(**fields)
        except ValidationError:
            continue
        raise AssertionError(f"{missing} must be required")


def test_detection_projects_clue_fields():
    d = Detection.from_clue(_clue(frame="f7", conf=0.42, box=(1, 2, 3, 4), geo=(46.8, 8.2)))
    assert d.frame_id == "f7" and d.label == "person"
    assert d.box == (1, 2, 3, 4) and d.confidence == 0.42
    assert d.geo == (46.8, 8.2) and d.case_id == "case-0000"


def test_detection_accepts_bus_json():
    """A clue arriving as JSON off the bus scores identically to the object."""
    clue = _clue(frame="f3")
    from_obj = Detection.from_clue(clue)
    from_json = Detection.from_clue(json.loads(clue.model_dump_json()))
    assert from_obj == from_json


def test_unscorable_clues_are_rejected_loudly():
    # No bounding box: nothing to match against.
    try:
        Detection.from_clue(_clue(box=None))
        raise AssertionError("clue without bounding_box must be rejected")
    except ValueError as e:
        assert "bounding_box" in str(e)
    # No frame identity: cannot tell which frame it belongs to.
    try:
        Detection.from_clue(_clue(frame=None))
        raise AssertionError("clue without frame_id must be rejected")
    except ValueError as e:
        assert "frame_id" in str(e)
    # Wrong arity box.
    try:
        Detection.from_clue(_clue(box=(0, 0, 10)))
        raise AssertionError("3-element bounding_box must be rejected")
    except ValueError as e:
        assert "bounding_box" in str(e)


def test_non_detection_agents_omit_frame_fields():
    """Weather (Phase 4) observes no frame and no class. Omitting both must
    validate cleanly — that is why they are Optional rather than required."""
    clue = ClueContract(
        clue_id=str(uuid.uuid4()),
        case_id="case-0000",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_agent=AgentSource.WEATHER_API,
        confidence_score=0.9,
        finding_summary="Hypothermia risk high; survival window ~6h",
        provenance_tag="api:meteoswiss",
    )
    assert clue.frame_id is None and clue.class_label is None
    # ...but it is not a scorable detection, and says so.
    try:
        Detection.from_clue(clue)
        raise AssertionError("a weather clue is not scorable as a detection")
    except ValueError as e:
        assert "bounding_box" in str(e)


def test_label_falls_back_to_source_agent():
    assert Detection.from_clue(_clue(label=None)).label == "DRONE_RGB"


def test_clue_without_geo_still_scores():
    d = Detection.from_clue(_clue(geo=None))
    assert d.geo is None and d.box == (0, 0, 10, 10)


def test_mock_detector_emits_valid_clues():
    split = build_splits(60, 0.3, seed=0)["validation"]
    clues = mock_rgb_detector(split, seed=0, case_id="case-0042")
    assert clues and all(isinstance(c, ClueContract) for c in clues)
    assert all(c.case_id == "case-0042" for c in clues)
    assert all(c.provenance_tag == "onboard:rgb-camera" for c in clues)
    assert all(c.source_agent is AgentSource.DRONE_RGB for c in clues)
    assert len({c.clue_id for c in clues}) == len(clues), "clue_id must be unique"
    assert all(c.frame_id and c.class_label == "person" for c in clues)
    # Every emitted clue must survive the scoring projection.
    assert all(Detection.from_clue(c).frame_id in set(split.frame_ids) for c in clues)


def test_mock_detector_is_calibrated():
    """The mock must actually emit at its configured recall and false-alarm rate,
    otherwise the baseline numbers describe a detector nobody configured."""
    split = build_splits(120, 0.3, seed=0)["validation"]
    n_gt, n_frames, seeds = len(split.ground_truth), split.n_frames, range(40)
    recalls, fars = [], []
    for s in seeds:
        dets = [Detection.from_clue(c) for c in mock_rgb_detector(split, seed=s)]
        recalls.append(sum(d.geo is not None for d in dets) / n_gt)
        fars.append(sum(d.geo is None for d in dets) / n_frames)
    assert abs(sum(recalls) / len(recalls) - 0.78) < 0.05, sum(recalls) / len(recalls)
    assert abs(sum(fars) / len(fars) - 0.25) < 0.06, sum(fars) / len(fars)


def test_clue_ids_do_not_perturb_sampling():
    """Identifier minting draws from its own stream, so contract changes cannot
    move the data distribution."""
    split = build_splits(60, 0.3, seed=0)["validation"]
    a = [Detection.from_clue(c) for c in mock_rgb_detector(split, seed=0)]
    b = [Detection.from_clue(c) for c in mock_rgb_detector(split, seed=0, case_id="other")]
    assert [(d.frame_id, d.box, d.confidence) for d in a] == [
        (d.frame_id, d.box, d.confidence) for d in b
    ]


def test_case_scoping_excludes_other_cases():
    split = build_splits(60, 0.3, seed=0)["validation"]
    mine = mock_rgb_detector(split, seed=0, case_id="case-mine")
    theirs = mock_rgb_detector(split, seed=1, case_id="case-theirs")

    scoped = run_baseline(split, clues=mine + theirs, case_id="case-mine")
    assert scoped["case_ids"] == ["case-mine"]
    assert scoped["skipped_other_case"] == len(theirs)
    # Scoping must give the same answer as never having seen the other case.
    assert scoped["mAP"] == run_baseline(split, clues=mine)["mAP"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
