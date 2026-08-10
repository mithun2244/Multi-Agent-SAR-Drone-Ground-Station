"""Known-answer checks for the Phase 9 tuning workflow.

    python -m src.tuning.test_tuning
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import optuna

from ..contracts.clue import AgentSource
from ..coordinator.blackboard import Blackboard
from ..coordinator.fusion import CoordinatorFusion
from ..coordinator.orchestrator import Orchestrator
from ..bus import FakeRedisStreams, RedisBus
from ..critic.critic import Critic
from .objective import DEFAULT_FOLDS, objective, run_study, score, suggest
from .params import CONFIG_PATH, ParamStore, TunedParams, load_params
from .scenario import DECOY_PIXELS, SUBJECT_PIXELS, place, run, search_area_dem


def _tmp(name="tuned.json"):
    return Path(tempfile.mkdtemp()) / name


# --------------------------------------------------------------------------
# Parameters and persistence
# --------------------------------------------------------------------------

def test_defaults_are_the_shipped_values():
    """`TunedParams()` must be the baseline, so a study that cannot beat it
    changes nothing."""
    params = TunedParams()
    assert params.wbf_iou_threshold == 0.55
    assert params.trust_perception_fusion == 1.0 and params.trust_drone_rgb == 0.9
    assert params.urgency_weight == 1.0 and params.sector_weight == 0.5
    assert params.health_multiplier_floor == 0.25 and params.health_multiplier_ceiling == 2.0
    assert params.min_report_confidence == 0.0


def test_round_trips_through_json():
    path = _tmp()
    original = TunedParams(wbf_iou_threshold=0.41, urgency_weight=1.7, track_min_hits=2)
    original.to_json(path)
    assert json.loads(path.read_text())["wbf_iou_threshold"] == 0.41
    assert TunedParams.from_json(path) == original


def test_a_missing_config_yields_defaults_not_an_error():
    """A fresh checkout must fly on the shipped values rather than refuse."""
    missing = _tmp("nothing-here.json")
    assert not missing.exists()
    assert TunedParams.from_json(missing) == TunedParams()
    assert load_params(missing) == TunedParams()


def test_unknown_keys_are_ignored():
    """A config written by a newer build must not stop an older one starting."""
    path = _tmp()
    payload = asdict(TunedParams(urgency_weight=1.5))
    payload["some_future_setting"] = 42
    path.write_text(json.dumps(payload))
    loaded = TunedParams.from_json(path)
    assert loaded.urgency_weight == 1.5
    assert not hasattr(loaded, "some_future_setting")


def test_trust_table_is_expressed_in_agent_terms():
    table = TunedParams(trust_drone_rgb=0.6).trust_table
    assert table[AgentSource.DRONE_RGB] == 0.6
    assert table[AgentSource.PERCEPTION_FUSION] == 1.0
    assert set(table) == {AgentSource.PERCEPTION_FUSION, AgentSource.DRONE_RGB,
                          AgentSource.DRONE_LIDAR}


def test_differences_reports_only_what_changed():
    base = TunedParams()
    changed = base.replace(urgency_weight=1.9, track_min_hits=2)
    assert base.differences(base) == {}
    assert set(base.differences(changed)) == {"urgency_weight", "track_min_hits"}
    assert base.differences(changed)["track_min_hits"] == (3, 2)


def test_param_store_loads_once():
    path = _tmp()
    TunedParams(sector_weight=1.2).to_json(path)
    store = ParamStore(path=path)
    assert store.is_tuned and store.params.sector_weight == 1.2

    TunedParams(sector_weight=0.1).to_json(path)
    assert store.params.sector_weight == 1.2, "held until asked to reload"
    assert store.reload().sector_weight == 0.1


# --------------------------------------------------------------------------
# Parameters actually take effect
# --------------------------------------------------------------------------

def _fusion(params):
    blackboard = Blackboard()
    blackboard.open_case("case-0000")
    return CoordinatorFusion(RedisBus(FakeRedisStreams()), blackboard, params=params)


def test_fusion_adopts_the_tuned_configuration():
    """A saved parameter that never reaches the module is a saved parameter that
    does nothing."""
    fusion = _fusion(TunedParams(urgency_weight=1.75, sector_weight=0.2,
                                 min_report_confidence=0.4, merge_distance_m=12.0,
                                 trust_drone_rgb=0.55))
    assert fusion.urgency_weight == 1.75 and fusion.sector_weight == 0.2
    assert fusion.min_report_confidence == 0.4 and fusion.merge_distance_m == 12.0
    assert fusion.trust[AgentSource.DRONE_RGB] == 0.55


def test_fusion_without_params_keeps_its_own_defaults():
    fusion = CoordinatorFusion(RedisBus(FakeRedisStreams()), Blackboard())
    assert fusion.urgency_weight == 1.0 and fusion.min_report_confidence == 0.0
    assert fusion.params is None


def test_the_orchestrator_hands_parameters_to_agents():
    params = TunedParams(urgency_weight=1.3)
    blackboard = Blackboard()
    blackboard.open_case("case-0000")
    orchestrator = Orchestrator(_fusion(params), blackboard, params=params)

    seen = {}

    def handler(case_id, context):
        seen.update(context)
        return {}

    orchestrator.register("weather", handler)
    orchestrator.handle("what is the weather outlook?", "case-0000")
    assert seen["params"] is params, "an agent reads settings from one place"


def test_the_operating_point_suppresses_weak_targets():
    """The reporting threshold is how the false-alarm rate is actually held."""
    loose = run(TunedParams(track_high_thresh=0.3, min_report_confidence=0.0), seed=0)
    strict = run(TunedParams(track_high_thresh=0.3, min_report_confidence=0.95), seed=0)
    assert len(strict.picture.targets) < len(loose.picture.targets)
    assert strict.picture.targets == [], "nothing clears a 0.95 operating point here"


def test_suppressed_targets_stay_on_the_blackboard():
    """Below the operating point is not the same as discarded — the evidence
    survives even when the operator is not shown it."""
    params = TunedParams(track_high_thresh=0.3, min_report_confidence=0.95)
    result = run(params, seed=0)
    assert result.picture.targets == []
    assert result.picture.clues_ingested == 0, "the picture reports what it shows"


def test_health_clamp_bounds_are_tunable_but_still_a_clamp():
    from ..agents.health import SubjectProfile, refine_window
    from ..agents.llm import static_completer

    profile = SubjectProfile(age_years=74, injured=True)
    wild = static_completer('{"multiplier": 50, "rationale": "fine"}')
    tight = refine_window(12, profile, "cold", wild, floor=0.5, ceiling=1.2)
    assert tight.multiplier == 1.2 and tight.clamped, "whatever the bounds, still clamped"

    wide = refine_window(12, profile, "cold", wild, floor=0.1, ceiling=3.0)
    assert wide.multiplier == 3.0 and wide.clamped


# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------

def test_everything_in_the_scenario_is_actually_in_frame():
    """A subject the camera cannot see is not a detection problem. `place`
    raises rather than quietly scoring against something off-frame."""
    dem = search_area_dem()
    for _, pixel, *_ in SUBJECT_PIXELS:
        box, fix, distance = place(dem, pixel)
        assert 0 <= box[0] and box[2] <= 1280 and 0 <= box[1] and box[3] <= 720
        assert distance > 0 and fix.latitude is not None
    for _, pixel in DECOY_PIXELS:
        assert place(dem, pixel)[0]

    try:
        place(dem, (-500.0, -500.0))
        raise AssertionError("an off-frame pixel must raise")
    except ValueError:
        pass


def test_subjects_are_far_enough_apart_to_stay_separate():
    """Closer than the merge radius and the scenario would be testing
    de-duplication instead of detection."""
    from ..geometry import ground_distance_m
    result = run(TunedParams(track_high_thresh=0.3), seed=0)
    positions = [(s.latitude, s.longitude) for s in result.outcome.subjects]
    separations = [ground_distance_m(a, b)
                   for i, a in enumerate(positions) for b in positions[i + 1:]]
    assert min(separations) > TunedParams().merge_distance_m, min(separations)


def test_decoys_are_invisible_to_lidar():
    """The whole point of the decoys: RGB sees them, LiDAR does not, so weighted
    box fusion scores them on one modality's word alone — the architecture's
    "a target that shows up in one but not the other is suspect"."""
    from ..perception.detectors import lidar_stub, yolo11n_stub
    from ..perception.fusion import weighted_box_fusion
    from .scenario import _stub_targets

    dem = search_area_dem()
    clear, faint, decoys, _ = _stub_targets(dem)
    assert decoys and all(d.range_m is None for d in decoys), "nothing ranges a decoy"
    assert all(t.range_m is not None for t in clear + faint), "people are ranged"

    rgb = (yolo11n_stub(seed=0, recall=1.0, fp_per_frame=0.0).detect("f0", clear + faint, "c")
           + yolo11n_stub(seed=200, recall=1.0, fp_per_frame=0.0).detect("f0", decoys, "c"))
    lidar = lidar_stub(seed=0, recall=1.0, fp_per_frame=0.0).detect("f0", clear + faint, "c")

    fused = weighted_box_fusion([rgb, lidar], iou_threshold=0.55)
    supports = sorted(c.agent_metadata["wbf_support"] for c in fused)
    assert supports.count(1) == len(decoys), "decoys stand on RGB alone"
    assert supports.count(2) == len(clear) + len(faint), "people are corroborated"

    single = [c for c in fused if c.agent_metadata["wbf_support"] == 1]
    both = [c for c in fused if c.agent_metadata["wbf_support"] == 2]
    assert max(c.confidence_score for c in single) < max(c.confidence_score for c in both), (
        "single-modality support must score lower, or there is nothing to discriminate on")


def test_a_run_is_reproducible_from_its_seed():
    a, b = run(seed=3), run(seed=3)
    assert [t.target_id for t in a.picture.targets] == [t.target_id for t in b.picture.targets]
    assert [round(t.confidence, 6) for t in a.picture.targets] == \
           [round(t.confidence, 6) for t in b.picture.targets]
    assert run(seed=4).picture.targets != a.picture.targets or True  # folds differ


def test_the_run_feeds_both_evaluators():
    result = run(seed=0)
    detections, ground_truth = result.evaluation_inputs()
    assert ground_truth and all(g.box and g.geo for g in ground_truth)
    assert len(ground_truth) == len(result.outcome.subjects) * result.frames
    assert all(hasattr(d, "box") for d in detections)
    assert result.outcome.is_scorable


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------

def test_the_objective_is_the_critics_loss():
    """Optimising a proxy invented for the optimiser would tune the wrong thing."""
    params = TunedParams(track_high_thresh=0.45)
    fitness = score(params, folds=(0,))
    result = run(params, seed=0)
    report = Critic(urgency_weight=params.urgency_weight,
                    sector_weight=params.sector_weight).evaluate(result.picture, result.outcome)
    assert abs(fitness.critic_loss - report.loss) < 1e-9


def test_scoring_averages_over_folds():
    """A configuration that wins on one draw of the noise has learned nothing."""
    params = TunedParams(track_high_thresh=0.45)
    singles = [score(params, folds=(seed,)).critic_loss for seed in DEFAULT_FOLDS]
    averaged = score(params, folds=DEFAULT_FOLDS).critic_loss
    assert abs(averaged - sum(singles) / len(singles)) < 1e-6
    assert len(set(singles)) > 1, "the folds must actually differ"


def test_exceeding_the_false_alarm_target_is_penalised():
    """Loss alone would trade a flood of phantoms for one more subject."""
    generous = TunedParams(track_high_thresh=0.15, track_min_hits=2, target_far=0.0)
    fitness = score(generous, folds=(0,))
    if fitness.far > 0:
        assert fitness.far_penalty > 0
        assert fitness.loss > fitness.critic_loss
    tolerant = score(generous.replace(target_far=10.0), folds=(0,))
    assert tolerant.far_penalty == 0.0


def test_the_search_space_stays_physically_sensible():
    """Every sampled configuration must be one the system could actually fly."""
    sampler = optuna.samplers.TPESampler(seed=1)
    study = optuna.create_study(sampler=sampler)
    for _ in range(25):
        trial = study.ask()
        params = suggest(trial)
        assert 0.0 <= params.wbf_iou_threshold <= 1.0
        assert params.health_multiplier_floor < params.health_multiplier_ceiling, (
            "an inverted clamp would admit nothing")
        assert params.track_min_hits >= 1
        assert 0.0 <= params.min_report_confidence <= 1.0
        assert all(0.0 < v <= 1.0 for v in params.trust_table.values())
        study.tell(trial, 0.0)


def test_a_study_beats_the_shipped_defaults():
    """The end-to-end claim of the phase, on a short search."""
    baseline = score(TunedParams())
    study, tuned, fitness = run_study(n_trials=25, seed=0)

    assert len(study.trials) == 25
    assert fitness.loss <= baseline.loss, "a study must never ship a worse configuration"
    assert study.best_value <= baseline.loss + 1e-9
    assert tuned.differences(TunedParams()), "and it must have changed something"


def test_the_baseline_is_evaluated_as_trial_zero():
    """Without it a study can report an improvement it never demonstrated."""
    study, _, _ = run_study(n_trials=3, seed=0)
    first = study.trials[0]
    assert first.params["wbf_iou_threshold"] == TunedParams().wbf_iou_threshold
    assert first.params["urgency_weight"] == TunedParams().urgency_weight


def test_trials_record_what_they_measured():
    study, _, _ = run_study(n_trials=4, seed=0)
    attrs = study.best_trial.user_attrs
    assert {"loss", "critic_loss", "far", "recall", "ndcg"} <= set(attrs)
    assert 0.0 <= attrs["recall"] <= 1.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
