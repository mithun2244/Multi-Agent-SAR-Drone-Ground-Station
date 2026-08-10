"""Known-answer checks for the Phase 8 critic.

    python -m src.critic.test_critic
"""

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ..bus import FakeRedisStreams, RedisBus
from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..coordinator.blackboard import Blackboard, Target
from ..coordinator.fusion import CoordinatorFusion
from ..evaluation.dataset import build_splits
from ..geometry import ground_distance_m
from ..guardrails.provenance import TAG_HEALTH, TAG_TRACK, TAG_WEATHER
from ..perception.geolocation import offset_enu
from .critic import SIGNAL_AGENTS, AgentScore, Critic, CriticReport
from .metrics import (
    detection_iou,
    false_positive_penalty,
    geolocation_residuals,
    kendall_tau,
    match_targets,
    ndcg,
    ranking_tau,
    relevance_vector,
)
from .outcomes import CaseOutcome, OutcomeLog, Subject

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SITE = (46.8182, 8.2275)


def _target(target_id="T1", geo=_SITE, confidence=0.8, urgency=0.0, sector=0.0,
            weather_urgency=None, hazard_urgency=0.0, baseline_urgency=None,
            box=None, priority=None):
    urgency = urgency
    weather = urgency if weather_urgency is None else weather_urgency
    baseline = urgency if baseline_urgency is None else baseline_urgency
    return Target(
        target_id=target_id,
        case_id="case-0000",
        class_label="person",
        latitude=geo[0] if geo else None,
        longitude=geo[1] if geo else None,
        bounding_box=box,
        confidence=confidence,
        observations=3,
        urgency=urgency,
        weather_urgency=weather,
        hazard_urgency=hazard_urgency,
        baseline_urgency=baseline,
        sector_probability=sector,
        priority=(confidence * (1.0 + urgency + 0.5 * sector)
                  if priority is None else priority),
    )


def _picture(targets, case_id="case-0000"):
    """A detached picture, exactly what fusion hands out."""
    from ..coordinator.fusion import Picture
    ranked = sorted(targets, key=lambda t: (-t.priority, t.target_id))
    return Picture(case_id=case_id, targets=ranked)


def _outcome(*subjects, case_id="case-0000"):
    return CaseOutcome(case_id=case_id, subjects=tuple(subjects), resolved_at=_EPOCH)


def _subject(subject_id="S1", geo=_SITE, priority=1.0, found=True, box=None):
    return Subject(subject_id=subject_id, latitude=geo[0] if geo else None,
                   longitude=geo[1] if geo else None, priority=priority,
                   found=found, bounding_box=box)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def test_a_target_on_the_subject_matches():
    near = offset_enu(*_SITE, east_m=20.0, north_m=0.0)
    matches, unmatched, missed = match_targets([_target(geo=near)], [_subject()])
    assert len(matches) == 1 and unmatched == [] and missed == ()
    rank, _, subject, distance = matches[0]
    assert rank == 1 and subject.subject_id == "S1" and 19.0 < distance < 21.0


def test_a_target_in_the_wrong_place_is_a_false_positive():
    far = offset_enu(*_SITE, east_m=800.0, north_m=0.0)
    matches, unmatched, missed = match_targets([_target(geo=far)], [_subject()])
    assert matches == [] and len(unmatched) == 1 and len(missed) == 1


def test_each_subject_matches_at_most_once():
    """Two targets on one person is one find and one phantom."""
    matches, unmatched, missed = match_targets(
        [_target("T1"), _target("T2", geo=offset_enu(*_SITE, east_m=10.0, north_m=0.0))],
        [_subject()],
    )
    assert len(matches) == 1 and len(unmatched) == 1 and missed == ()


def test_matching_works_down_the_ranking_as_an_operator_would():
    near = offset_enu(*_SITE, east_m=5.0, north_m=0.0)
    nearer = _SITE
    top = _target("T1", geo=near, confidence=0.9)
    second = _target("T2", geo=nearer, confidence=0.5)
    matches, _, _ = match_targets([top, second], [_subject()])
    assert matches[0][1].target_id == "T1", "the top of the ranking claims the subject first"


def test_an_unlocated_target_is_not_a_false_positive():
    """Seen but unplaceable is not the same as wrong."""
    matches, unmatched, missed = match_targets([_target(geo=None)], [_subject()])
    assert matches == [] and len(unmatched) == 1 and len(missed) == 1
    assert not unmatched[0][1].located


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_geolocation_residuals():
    subjects = [_subject("S1"), _subject("S2", geo=offset_enu(*_SITE, east_m=400.0, north_m=0.0))]
    targets = [
        _target("T1", geo=offset_enu(*_SITE, east_m=10.0, north_m=0.0)),
        _target("T2", geo=offset_enu(*_SITE, east_m=430.0, north_m=0.0)),
    ]
    matches, _, _ = match_targets(targets, subjects)
    residuals = geolocation_residuals(matches)
    assert residuals["n"] == 2
    assert 9.0 < residuals["median_m"] < 31.0
    assert geolocation_residuals([])["mean_m"] is None


def test_detection_iou_is_none_without_boxes():
    matches, _, _ = match_targets([_target()], [_subject()])
    assert detection_iou(matches) is None, "no pixels means no IoU, not a score of zero"

    boxed_matches, _, _ = match_targets(
        [_target(box=(0.0, 0.0, 10.0, 10.0))], [_subject(box=(0.0, 0.0, 10.0, 10.0))])
    assert detection_iou(boxed_matches) == 1.0

    half, _, _ = match_targets(
        [_target(box=(0.0, 0.0, 10.0, 10.0))], [_subject(box=(5.0, 0.0, 15.0, 10.0))])
    assert abs(detection_iou(half) - 1 / 3) < 1e-6


def test_ndcg_rewards_putting_the_real_one_first():
    assert ndcg([1.0, 0.0, 0.0]) == 1.0
    assert ndcg([0.0, 0.0, 1.0]) < ndcg([0.0, 1.0, 0.0]) < 1.0
    assert ndcg([0.0, 0.0, 0.0]) is None, "nothing relevant means no ordering to judge"
    assert ndcg([2.0, 1.0]) == 1.0


def test_kendall_tau_handles_the_ties_phantoms_create():
    assert kendall_tau([3, 2, 1], [3, 2, 1]) == 1.0
    assert kendall_tau([3, 2, 1], [1, 2, 3]) == -1.0
    assert kendall_tau([1, 1, 1], [1, 2, 3]) is None, "no variation, no correlation"
    assert kendall_tau([1], [1]) is None
    try:
        kendall_tau([1, 2], [1])
        raise AssertionError("mismatched lengths must raise")
    except ValueError:
        pass


def test_ranking_tau_agrees_with_ndcg_about_direction():
    best, worst = ranking_tau([1.0, 0.0, 0.0]), ranking_tau([0.0, 0.0, 1.0])
    # Tau-b cannot reach 1.0 here: positions are all distinct while the two
    # phantoms tie at relevance 0, and that tie is a real limit on agreement.
    assert 0.8 < best < 1.0 and -1.0 < worst < -0.8
    assert abs(best + worst) < 1e-9, "symmetric about a reversed ranking"
    assert ranking_tau([1.0, 0.0]) == 1.0, "no ties, so a perfect ranking scores 1"
    assert ranking_tau([1.0]) is None


def test_false_positives_cost_more_the_higher_they_rank():
    targets = [_target(f"T{i}") for i in range(1, 4)]
    top = false_positive_penalty(targets, [(1, targets[0])])
    bottom = false_positive_penalty(targets, [(3, targets[2])])
    assert top > bottom, "a phantom at rank 1 sends a team to the wrong place"
    assert false_positive_penalty(targets, []) == 0.0
    assert abs(false_positive_penalty(targets, list(enumerate(targets, 1))) - 1.0) < 1e-9


def test_relevance_uses_the_subjects_own_priority():
    subjects = [_subject("S1", priority=3.0)]
    matches, _, _ = match_targets([_target("T1"), _target("T2", geo=(46.9, 8.4))], subjects)
    assert relevance_vector([_target("T1"), _target("T2")], matches) == [3.0, 0.0]


# --------------------------------------------------------------------------
# The critic
# --------------------------------------------------------------------------

def test_a_perfect_picture_scores_a_near_zero_loss():
    picture = _picture([_target("T1", confidence=0.9)])
    report = Critic().evaluate(picture, _outcome(_subject()))

    assert report.scorable and report.recall == 1.0
    assert report.matched == 1 and report.missed == 0 and report.false_positives == 0
    assert report.ndcg == 1.0 and report.false_positive_penalty == 0.0
    assert report.loss is not None and report.loss < 0.01
    assert report.errors == ()
    assert "Critic report" in report.render()


def test_a_missed_subject_dominates_the_loss():
    """Missing somebody must cost more than any other failure."""
    found = Critic().evaluate(_picture([_target("T1")]), _outcome(_subject()))
    nothing = Critic().evaluate(_picture([]), _outcome(_subject()))

    assert nothing.recall == 0.0 and nothing.missed == 1
    assert nothing.loss > found.loss
    assert nothing.loss_terms["miss"] == 2.0, "weighted twice, by default"
    assert any(e.startswith("MISSED_SUBJECT") for e in nothing.errors)


def test_a_phantom_ranked_first_is_penalised_and_labelled():
    wrong_place = offset_enu(*_SITE, east_m=900.0, north_m=0.0)
    picture = _picture([_target("T1", geo=wrong_place, confidence=0.95),
                        _target("T2", geo=_SITE, confidence=0.5)])
    report = Critic().evaluate(picture, _outcome(_subject()))

    assert report.matched == 1 and report.false_positives == 1
    assert report.ndcg is not None and report.ndcg < 1.0
    assert report.kendall_tau is not None and report.kendall_tau < 0
    assert report.false_positive_penalty > 0.0
    assert any(e.startswith("FALSE_POSITIVE") for e in report.errors)
    assert any(e.startswith("RANKING_ERROR") for e in report.errors)


def test_ranking_errors_name_the_right_cause():
    """Two real subjects in the wrong priority order is not "below a phantom" —
    saying so would send someone hunting a false positive that does not exist."""
    high = _subject("S-casualty", geo=_SITE, priority=5.0)
    low = _subject("S-walker", geo=offset_enu(*_SITE, east_m=600.0, north_m=0.0))
    picture = _picture([_target("T1", geo=low.position, confidence=0.9),
                        _target("T2", geo=high.position, confidence=0.4)])

    report = Critic().evaluate(picture, _outcome(high, low))
    assert report.matched == 2 and report.false_positives == 0
    assert report.ndcg < 0.8
    (error,) = [e for e in report.errors if e.startswith("RANKING_ERROR")]
    assert "out of priority order" in error and "phantom" not in error


def test_geolocation_drift_is_labelled():
    drifted = offset_enu(*_SITE, east_m=80.0, north_m=0.0)
    report = Critic().evaluate(_picture([_target("T1", geo=drifted)]), _outcome(_subject()))
    assert report.matched == 1
    assert 79.0 < report.geolocation["median_m"] < 81.0
    assert report.loss_terms["geolocation"] > 0.5
    assert any(e.startswith("GEOLOCATION_DRIFT") for e in report.errors)


def test_an_unresolved_case_is_not_scored():
    """A search that never found anyone cannot say whether the picture was right;
    scoring it would reward giving up."""
    report = Critic().evaluate(_picture([_target("T1")]),
                               _outcome(_subject(found=False, geo=None)))
    assert not report.scorable and report.loss is None
    assert "never located anyone" in report.reason
    assert "not scorable" in report.render()

    assert not Critic().evaluate(_picture([_target("T1")]), None).scorable


def test_the_loss_weights_are_tunable():
    picture = _picture([_target("T1", geo=offset_enu(*_SITE, east_m=90.0, north_m=0.0))])
    outcome = _outcome(_subject())
    strict = Critic(loss_weights={"geolocation": 10.0}).evaluate(picture, outcome)
    lax = Critic(loss_weights={"geolocation": 0.0}).evaluate(picture, outcome)
    assert strict.loss > lax.loss
    assert lax.loss_terms["geolocation"] == 0.0


def test_the_match_radius_is_tunable():
    drifted = offset_enu(*_SITE, east_m=150.0, north_m=0.0)
    picture, outcome = _picture([_target("T1", geo=drifted)]), _outcome(_subject())
    assert Critic(match_radius_m=100.0).evaluate(picture, outcome).matched == 0
    assert Critic(match_radius_m=200.0).evaluate(picture, outcome).matched == 1


# --------------------------------------------------------------------------
# Per-agent counterfactuals
# --------------------------------------------------------------------------

def test_urgency_that_promoted_the_right_target_scores_as_helping():
    """Exactly the architecture's question: did the Health agent's multiplier
    actually get the right target looked at first?"""
    wrong_place = offset_enu(*_SITE, east_m=900.0, north_m=0.0)
    # The real subject is the weaker detection, lifted to the top by urgency.
    real = _target("T1", geo=_SITE, confidence=0.5, urgency=0.9, baseline_urgency=0.1)
    phantom = _target("T2", geo=wrong_place, confidence=0.8)
    picture = _picture([real, phantom])
    assert picture.targets[0].target_id == "T1", "urgency put the real one on top"

    report = Critic().evaluate(picture, _outcome(_subject()))
    scores = {s.signal: s for s in report.agent_scores}

    assert report.ndcg == 1.0
    assert scores["urgency"].verdict == "helped"
    assert scores["urgency"].ndcg_without < scores["urgency"].ndcg_with
    assert set(scores["urgency"].agents) == {"weather", "health", "scene"}

    health = scores["health_refinement"]
    assert health.verdict == "helped", "the refinement, not just the raw window"
    assert health.agents == ("health",)


def test_a_signal_that_hurt_is_reported_as_hurting():
    """Urgency that promoted a phantom must score negative, not be excused."""
    wrong_place = offset_enu(*_SITE, east_m=900.0, north_m=0.0)
    phantom = _target("T1", geo=wrong_place, confidence=0.5, urgency=0.95)
    real = _target("T2", geo=_SITE, confidence=0.8)
    picture = _picture([phantom, real])
    assert picture.targets[0].target_id == "T1"

    report = Critic().evaluate(picture, _outcome(_subject()))
    scores = {s.signal: s for s in report.agent_scores}
    assert scores["urgency"].verdict == "hurt"
    assert scores["urgency"].delta < 0


def test_a_signal_nobody_emitted_is_unscorable():
    picture = _picture([_target("T1", confidence=0.9), _target("T2", geo=(46.9, 8.4))])
    report = Critic().evaluate(picture, _outcome(_subject()))
    scores = {s.signal: s for s in report.agent_scores}
    assert scores["sector_prior"].verdict == "unscorable", "the path model never ran"
    assert scores["sector_prior"].active_targets == 0
    assert scores["hazard_urgency"].verdict == "unscorable"


def test_the_sector_prior_is_scored_separately():
    wrong_place = offset_enu(*_SITE, east_m=900.0, north_m=0.0)
    real = _target("T1", geo=_SITE, confidence=0.6, sector=1.0)
    phantom = _target("T2", geo=wrong_place, confidence=0.8)
    report = Critic().evaluate(_picture([real, phantom]), _outcome(_subject()))
    scores = {s.signal: s for s in report.agent_scores}
    assert scores["sector_prior"].verdict == "helped"
    assert scores["sector_prior"].agents == ("path",)


def test_hazard_and_weather_urgency_are_attributed_separately():
    wrong_place = offset_enu(*_SITE, east_m=900.0, north_m=0.0)
    real = _target("T1", geo=_SITE, confidence=0.5, urgency=0.9,
                   weather_urgency=0.0, hazard_urgency=0.9, baseline_urgency=0.9)
    phantom = _target("T2", geo=wrong_place, confidence=0.8)
    report = Critic().evaluate(_picture([real, phantom]), _outcome(_subject()))
    scores = {s.signal: s for s in report.agent_scores}

    assert scores["hazard_urgency"].verdict == "helped", "the scene agent earned this"
    assert scores["weather_urgency"].verdict == "unscorable", "weather said nothing"
    assert scores["health_refinement"].verdict == "unscorable", "and neither did health"


def test_every_signal_maps_to_named_agents():
    report = Critic().evaluate(_picture([_target("T1")]), _outcome(_subject()))
    assert {s.signal for s in report.agent_scores} == set(SIGNAL_AGENTS)
    assert all(isinstance(s, AgentScore) and s.agents for s in report.agent_scores)


# --------------------------------------------------------------------------
# Data safety
# --------------------------------------------------------------------------

def test_evaluation_never_mutates_the_picture():
    """The critic runs while teams are on the hill. A scoring pass that nudged a
    ranking would change the thing it claims to measure."""
    targets = [_target("T1", confidence=0.5, urgency=0.9),
               _target("T2", geo=(46.9, 8.4), confidence=0.8, sector=1.0)]
    picture = _picture(targets)
    before = [(t.target_id, t.priority, t.urgency, t.sector_probability, t.confidence)
              for t in picture.targets]
    order = [t.target_id for t in picture.targets]

    Critic().evaluate(picture, _outcome(_subject()))

    after = [(t.target_id, t.priority, t.urgency, t.sector_probability, t.confidence)
             for t in picture.targets]
    assert after == before, "target fields must be untouched"
    assert [t.target_id for t in picture.targets] == order, "and so must the order"


def test_evaluation_never_touches_the_blackboard():
    bus = RedisBus(FakeRedisStreams())
    blackboard = Blackboard()
    blackboard.open_case("case-0000", opened_at=_EPOCH)
    fusion = CoordinatorFusion(bus, blackboard, clock=lambda: _EPOCH)
    bus.publish(ClueContract(
        clue_id="c1", case_id="case-0000", timestamp=_EPOCH,
        source_agent=AgentSource.PERCEPTION_FUSION, confidence_score=0.9,
        finding_summary="person", provenance_tag=TAG_TRACK,
        spatial_context=SpatialContext(latitude=_SITE[0], longitude=_SITE[1],
                                       bounding_box=[0.0, 0.0, 10.0, 10.0]),
        frame_id="frame_0001", class_label="person",
        agent_metadata={"track_id": "1", "track_state": "CONFIRMED"},
    ))
    picture = fusion.refresh("case-0000")
    live = blackboard.targets("case-0000")[0]
    snapshot = (live.priority, live.urgency, live.confidence, live.observations)

    Critic().evaluate(picture, _outcome(_subject()))

    live_after = blackboard.targets("case-0000")[0]
    assert (live_after.priority, live_after.urgency, live_after.confidence,
            live_after.observations) == snapshot
    assert live_after.priority == 0.0, "live targets never carry a computed priority"


def test_a_second_pass_scores_the_same():
    """No hidden state carried between evaluations."""
    picture = _picture([_target("T1", confidence=0.6, urgency=0.5)])
    critic = Critic()
    outcome = _outcome(_subject())
    first = critic.evaluate(picture, outcome)
    second = critic.evaluate(picture, outcome)
    assert (first.loss, first.ndcg, first.recall) == (second.loss, second.ndcg, second.recall)
    assert len(critic.history) == 2


# --------------------------------------------------------------------------
# Logging for the tuner
# --------------------------------------------------------------------------

def test_reports_serialise_for_a_tuning_run():
    critic = Critic()
    report = critic.evaluate(_picture([_target("T1", urgency=0.4)]), _outcome(_subject()))
    payload = critic.as_dict(report)

    assert payload["case_id"] == "case-0000"
    assert isinstance(payload["loss"], float)
    assert set(payload["loss_terms"]) == {"miss", "false_positive", "geolocation", "ranking"}
    assert isinstance(payload["agent_scores"], list) and payload["agent_scores"]
    assert isinstance(payload["generated_at"], str)
    import json
    assert json.loads(json.dumps(payload))["case_id"] == "case-0000"


def test_the_critic_tracks_a_campaign():
    critic = Critic()
    critic.evaluate(_picture([_target("T1")]), _outcome(_subject()))
    critic.evaluate(_picture([]), _outcome(_subject()))
    critic.evaluate(_picture([_target("T1")]), None)

    summary = critic.summary()
    assert summary["passes"] == 3 and summary["scored"] == 2
    assert 0.0 < summary["mean_loss"] < 2.0
    assert summary["mean_recall"] == 0.5
    assert Critic().summary()["mean_loss"] is None


# --------------------------------------------------------------------------
# Phase 1 dataset link
# --------------------------------------------------------------------------

def test_ground_truth_comes_from_the_phase_one_dataset():
    """The critic can be exercised against a labelled split before a single real
    case has resolved."""
    split = build_splits(60, 0.3, seed=0)["validation"]
    outcome = CaseOutcome.from_split("case-0000", split)

    assert outcome.is_scorable and outcome.subjects
    assert len(outcome.found_subjects) == len(outcome.subjects)
    assert all(s.located and s.bounding_box and s.frame_id for s in outcome.subjects)
    assert "validation" in outcome.notes


def test_the_critic_scores_a_picture_built_from_dataset_truth():
    split = build_splits(60, 0.3, seed=0)["validation"]
    outcome = CaseOutcome.from_split("case-0000", split)
    subjects = outcome.found_subjects[:3]

    # A picture that found two of three, one of them 30 m out.
    targets = [
        _target("T1", geo=subjects[0].position, confidence=0.9),
        _target("T2", geo=offset_enu(*subjects[1].position, east_m=30.0, north_m=0.0),
                confidence=0.7),
    ]
    report = Critic().evaluate(_picture(targets), _outcome(*subjects))

    assert report.subjects == 3 and report.matched == 2 and report.missed == 1
    assert abs(report.recall - 2 / 3) < 1e-6
    assert report.geolocation["max_m"] > 25.0
    assert report.loss > 0.0


def test_the_outcome_log_keeps_cases_apart():
    log = OutcomeLog()
    log.record(_outcome(_subject(), case_id="case-A"))
    log.record(_outcome(_subject(found=False, geo=None), case_id="case-B"))

    assert len(log) == 2
    assert log.get("case-A").is_scorable and not log.get("case-B").is_scorable
    assert [o.case_id for o in log.scorable()] == ["case-A"]
    assert log.get("case-missing") is None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
