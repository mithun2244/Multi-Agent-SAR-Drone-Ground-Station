"""Known-answer checks for the ablation switches.

    python -m src.utils.test_ablation

Every check does the same two things: run the component with the switch on and
with it off, and assert the *observable difference* — not that a flag was read.
A test that only asserted `enabled(...) is False` would pass just as happily if
the flag were wired to nothing, which is exactly the failure worth catching.

The environment is manipulated through `_env`, which always restores what was
there. A test that leaks `ABLATION_WBF=off` into the process would silently
ablate every suite that runs after it.
"""

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from ..bus import FakeRedisStreams, RedisBus
from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..coordinator.blackboard import Blackboard
from ..coordinator.decision import (
    CommanderBrief,
    FlatReport,
    decide,
    flatten,
)
from ..coordinator.fusion import CoordinatorFusion
from ..coordinator.orchestrator import Orchestrator
from ..coordinator.router import ALL_AGENTS, RouteScenario, ScenarioRouter
from ..guardrails.provenance import TAG_LIDAR, TAG_RGB
from ..perception.fusion import pass_through, weighted_box_fusion
from ..perception.tracking import Affine, BoTSORT
from .ablation import (
    CMC,
    DECISION,
    DISABLE_AGENTS,
    WBF,
    AblationError,
    active,
    describe,
    disabled_agents,
    enabled,
)

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SITE = (46.8182, 8.2275)


@contextmanager
def _env(**values):
    """Set ablation variables for one block, then put the environment back."""
    previous = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _detection(source, tag, box, conf, frame="frame_0001", case="case-0000", seconds=0,
               track=None, range_m=None):
    metadata = {"device_id": "AA:BB:CC:DD:EE:01"}
    if track is not None:
        metadata["track_id"] = track
    if range_m is not None:
        metadata["range_m"] = range_m
    return ClueContract(
        clue_id=f"{source.value}-{frame}-{box[0]}-{conf}",
        case_id=case,
        timestamp=_EPOCH + timedelta(seconds=seconds),
        source_agent=source,
        confidence_score=conf,
        finding_summary="test detection",
        spatial_context=SpatialContext(latitude=None, longitude=None, bounding_box=list(box)),
        frame_id=frame,
        class_label="person",
        provenance_tag=tag,
        agent_metadata=metadata,
    )


def _rgb(box, conf, **kwargs):
    return _detection(AgentSource.DRONE_RGB, TAG_RGB, box, conf, **kwargs)


def _lidar(box, conf, **kwargs):
    return _detection(AgentSource.DRONE_LIDAR, TAG_LIDAR, box, conf, **kwargs)


# --------------------------------------------------------------------------
# The switch itself
# --------------------------------------------------------------------------

def test_switches_default_to_on():
    """An unset environment is the shipped system. Nothing else is acceptable:
    a default-off switch would ablate every run that forgot to set it."""
    for name in (WBF, CMC, DECISION):
        assert enabled(name, env={}) is True, name
        assert enabled(name, env={name: ""}) is True, "blank is unset, not off"
    assert disabled_agents(env={}) == frozenset()
    assert active(env={}) == () and "none" in describe(env={})


def test_switches_read_the_spellings_people_actually_use():
    for off in ("off", "OFF", " off ", "0", "false", "no", "disabled"):
        assert enabled(WBF, env={WBF: off}) is False, off
    for on in ("on", "1", "true", "yes", "enabled"):
        assert enabled(WBF, env={WBF: on}) is True, on


def test_an_unrecognised_switch_value_is_refused():
    """Silently treating `ABLATION_WBF=maybe` as on would mean a run that was
    meant to be ablated quietly was not, and a number reported for the wrong
    system."""
    for bad in ("maybe", "2", "onn"):
        try:
            enabled(WBF, env={WBF: bad})
            raise AssertionError(f"{bad!r} must not be accepted")
        except AblationError as e:
            assert WBF in str(e)


def test_an_explicit_argument_beats_the_environment():
    assert enabled(WBF, True, env={WBF: "off"}) is True
    assert enabled(WBF, False, env={WBF: "on"}) is False
    assert disabled_agents("weather", env={DISABLE_AGENTS: "path"}) == {"weather"}


def test_what_is_ablated_is_always_stated():
    """An ablated run that looks like a normal one is how a wrong number ends up
    in a report."""
    env = {WBF: "off", DISABLE_AGENTS: "weather,path"}
    assert set(active(env=env)) == {WBF, f"{DISABLE_AGENTS}=path,weather"}
    line = describe(env=env)
    assert "ABLATED" in line and WBF in line and "weather" in line


# --------------------------------------------------------------------------
# 1. Weighted Box Fusion
# --------------------------------------------------------------------------

def test_wbf_on_merges_two_sensors_into_one_box():
    rgb = _rgb((100.0, 100.0, 140.0, 200.0), 0.9)
    lidar = _lidar((104.0, 102.0, 144.0, 204.0), 0.6)

    fused = weighted_box_fusion([[rgb], [lidar]])
    assert len(fused) == 1, "one target, one clue"
    box = fused[0].spatial_context.bounding_box
    assert box != rgb.spatial_context.bounding_box, "the box moved toward the pair"
    assert box[0] > 100.0 and box[0] < 104.0, box
    assert len(fused[0].parent_clue_ids) == 2, "both parents survive"


def test_wbf_off_passes_the_best_detector_through_untouched():
    """The ablation: no merge, no corroboration, one sensor's boxes verbatim."""
    rgb = _rgb((100.0, 100.0, 140.0, 200.0), 0.9)
    lidar = _lidar((104.0, 102.0, 144.0, 204.0), 0.6)

    with _env(**{WBF: "off"}):
        out = weighted_box_fusion([[rgb], [lidar]])

    assert out == [rgb], "the RGB feed held the best detection and went through as-is"
    assert out[0].spatial_context.bounding_box == [100.0, 100.0, 140.0, 200.0]
    assert out[0].parent_clue_ids is None, "nothing was fused, so nothing has parents"
    assert out[0].source_agent is AgentSource.DRONE_RGB, "still a sensor clue, not a fusion clue"


def test_wbf_off_can_be_asked_for_without_touching_the_environment():
    rgb, lidar = _rgb((0.0, 0.0, 10.0, 20.0), 0.4), _lidar((0.0, 0.0, 10.0, 20.0), 0.8)
    assert weighted_box_fusion([[rgb], [lidar]], fuse=False) == [lidar], (
        "the LiDAR feed was the confident one this time"
    )
    assert len(weighted_box_fusion([[rgb], [lidar]], fuse=True)) == 1


def test_wbf_off_discards_the_other_sensor_which_is_the_cost():
    """Stated as a check because it is the thing being measured: with the merge
    off, the LiDAR range never reaches the tracker on an RGB-won frame."""
    rgb = _rgb((100.0, 100.0, 140.0, 200.0), 0.95)
    lidar = _lidar((101.0, 101.0, 141.0, 201.0), 0.5, range_m=61.4)

    with _env(**{WBF: "off"}):
        out = weighted_box_fusion([[rgb], [lidar]])
    assert "range_m" not in out[0].agent_metadata, "the measured range went with the loser"

    fused = weighted_box_fusion([[rgb], [lidar]])
    assert fused[0].agent_metadata.get("range_m") == 61.4, "the merge keeps it"


def test_pass_through_prefers_a_feed_with_anything_in_it():
    rgb = _rgb((0.0, 0.0, 10.0, 20.0), 0.3)
    assert pass_through([[], [rgb]]) == [rgb], "an empty feed loses to any detection"
    assert pass_through([[], []]) == [], "and all-empty is empty, not an error"


# --------------------------------------------------------------------------
# 2. Camera-motion compensation
# --------------------------------------------------------------------------

def _track_through_pan(cmc_env):
    """Same pan, same detections, tracker built under `cmc_env`. Returns the
    track boxes after the camera moved but before the next detection lands."""
    with _env(**{CMC: cmc_env}):
        tracker = BoTSORT(min_hits=1, new_track_thresh=0.3)
    box = (100.0, 100.0, 140.0, 200.0)
    tracker.update([_rgb(box, 0.9, frame="frame_0000", track="1")], frame_id="frame_0000")
    assert tracker.tracks, "a track exists to move"

    # The camera pans 40 px right between frames.
    tracker.update([], frame_id="frame_0001", camera_motion=Affine(tx=40.0, ty=0.0))
    return tracker, tracker.tracks[0].box


def test_cmc_on_moves_tracks_with_the_camera():
    tracker, box = _track_through_pan("on")
    assert tracker.cmc is True
    assert box[0] > 130.0, f"the track followed the pan, got {box}"


def test_cmc_off_leaves_the_track_where_it_was():
    """The tracker still runs — predicts, associates, ages — it is just no
    longer told the camera moved, which on a drone is most of the motion."""
    tracker, box = _track_through_pan("off")
    assert tracker.cmc is False
    assert box[0] < 110.0, f"the track ignored the pan, got {box}"

    on_tracker, on_box = _track_through_pan("on")
    assert on_box[0] - box[0] > 30.0, "and the difference is the whole compensation"
    assert len(tracker.tracks) == len(on_tracker.tracks), "same tracker, one signal fewer"


def test_cmc_off_still_tracks_a_static_camera_identically():
    """The ablation must remove the compensation and nothing else: with no
    camera motion to compensate for, both configurations agree exactly."""
    boxes = []
    for setting in ("on", "off"):
        with _env(**{CMC: setting}):
            tracker = BoTSORT(min_hits=2, new_track_thresh=0.3)
        for i in range(4):
            tracker.update([_rgb((100.0 + i, 100.0, 140.0 + i, 200.0), 0.9,
                                 frame=f"frame_{i:04d}", track="1", seconds=i)],
                           frame_id=f"frame_{i:04d}")
        boxes.append([round(v, 6) for v in tracker.tracks[0].box])
    assert boxes[0] == boxes[1], boxes


def test_cmc_can_be_switched_without_the_environment():
    assert BoTSORT(cmc=False).cmc is False
    assert BoTSORT(cmc=True).cmc is True
    assert BoTSORT().cmc is True, "unset means the shipped tracker"


# --------------------------------------------------------------------------
# 3. Disabling agents
# --------------------------------------------------------------------------

def test_agents_are_dropped_from_every_route():
    with _env(**{DISABLE_AGENTS: "weather,path"}):
        router = ScenarioRouter()
        assert router.disabled == {"weather", "path"}
        for trigger in (RouteScenario.PERCEPTION_EVENT, "full briefing",
                        "what now?", "what is the weather outlook?"):
            agents = router.route(trigger).agents
            assert "weather" not in agents and "path" not in agents, trigger

    # Even the route whose entire purpose was the disabled agent comes back
    # empty rather than quietly running it.
    with _env(**{DISABLE_AGENTS: "weather"}):
        assert ScenarioRouter().route("what is the weather outlook?").agents == ()


def test_a_disabled_agent_is_named_in_the_reason():
    with _env(**{DISABLE_AGENTS: "scene"}):
        route = ScenarioRouter().route("full briefing")
    assert "ablated: scene" in route.reason
    assert "scene" in route.agents_skipped


def test_disabling_nothing_is_the_shipped_router():
    with _env(**{DISABLE_AGENTS: ""}):
        assert ScenarioRouter().route("full briefing").agents == ALL_AGENTS


def test_disabling_an_agent_nobody_has_is_refused():
    """A typo would otherwise ablate nothing and report as though it had."""
    try:
        ScenarioRouter(disabled="wheather")
        raise AssertionError("an unknown agent name must be refused")
    except ValueError as e:
        assert "wheather" in str(e) and "detection" in str(e)


def test_a_disabled_agent_never_runs():
    """Through the dispatcher, end to end: the handler is registered and still
    is not called."""
    bus = RedisBus(FakeRedisStreams())
    blackboard = Blackboard()
    blackboard.open_case("case-0000", opened_at=_EPOCH)
    fusion = CoordinatorFusion(bus, blackboard, clock=lambda: _EPOCH)

    with _env(**{DISABLE_AGENTS: "weather,scene,health,path"}):
        orchestrator = Orchestrator(fusion, blackboard)
        calls = []
        for name in ALL_AGENTS:
            orchestrator.register(name, lambda case_id, ctx, n=name: calls.append(n))
        dispatch = orchestrator.handle("full briefing", "case-0000")

    assert calls == ["detection"], calls
    assert dispatch.agents == ("detection",)
    assert set(dispatch.agents_skipped) == {"weather", "scene", "health", "path"}


# --------------------------------------------------------------------------
# 4. The decision chain
# --------------------------------------------------------------------------

def _picture(bus, blackboard, fusion, conf=0.8, window=None):
    bus.publish(ClueContract(
        clue_id=f"track-{conf}",
        case_id="case-0000",
        timestamp=_EPOCH,
        source_agent=AgentSource.PERCEPTION_FUSION,
        confidence_score=conf,
        finding_summary="confirmed track",
        spatial_context=SpatialContext(latitude=_SITE[0], longitude=_SITE[1],
                                       bounding_box=[0.0, 0.0, 10.0, 20.0]),
        frame_id="frame_0001",
        class_label="person",
        provenance_tag="perception:track",
        agent_metadata={"track_id": "1"},
    ))
    if window is not None:
        bus.publish(ClueContract(
            clue_id=f"weather-{window}",
            case_id="case-0000",
            timestamp=_EPOCH,
            source_agent=AgentSource.WEATHER_API,
            confidence_score=0.9,
            finding_summary="conditions",
            spatial_context=SpatialContext(latitude=_SITE[0], longitude=_SITE[1]),
            provenance_tag="api:open-meteo",
            agent_metadata={"hypothermia_risk": True, "survival_window_hours": window},
        ))
    return fusion.refresh("case-0000")


def _wired():
    bus = RedisBus(FakeRedisStreams())
    blackboard = Blackboard()
    blackboard.open_case("case-0000", opened_at=_EPOCH)
    return bus, blackboard, CoordinatorFusion(bus, blackboard, clock=lambda: _EPOCH)


def test_decision_on_produces_a_reasoned_brief():
    picture = _picture(*_wired(), window=4)
    brief = decide(picture)
    assert isinstance(brief, CommanderBrief)
    assert brief.risk.score >= 6 and brief.recommendation.action
    assert brief.facts.targets == 1 and brief.consistent


def test_decision_off_hands_the_picture_straight_to_the_commander():
    """The ablation: fusion's ranking, with nothing reasoned over it."""
    picture = _picture(*_wired(), window=4)

    with _env(**{DECISION: "off"}):
        report = decide(picture)

    assert isinstance(report, FlatReport) and not isinstance(report, CommanderBrief)
    assert not hasattr(report, "risk"), "no risk score exists to be misread as zero"
    assert not hasattr(report, "recommendation")
    # The picture itself is intact — the ranking is what a flat report *is*.
    assert len(report.targets) == len(picture.targets) == 1
    assert report.targets[0].target_id == picture.targets[0].target_id
    assert report.survival_window_hours == 4 and report.hypothermia_risk
    assert "ABLATED" in report.render() and "no risk score" in report.render()


def test_the_commander_is_called_either_way():
    picture = _picture(*_wired())
    delivered = []
    decide(picture, commander=delivered.append)
    with _env(**{DECISION: "off"}):
        decide(picture, commander=delivered.append)
    assert len(delivered) == 2
    assert isinstance(delivered[0], CommanderBrief) and isinstance(delivered[1], FlatReport)
    # One line each, whichever shape a caller got.
    assert all(report.summary() for report in delivered)
    assert "risk" in delivered[0].summary() and "ablated" in delivered[1].summary()


def test_decision_off_reads_the_picture_and_changes_nothing():
    bus, blackboard, fusion = _wired()
    picture = _picture(bus, blackboard, fusion, window=4)
    before = [(t.target_id, t.priority) for t in picture.targets]

    with _env(**{DECISION: "off"}):
        decide(picture)

    after = fusion.picture("case-0000")
    assert [(t.target_id, t.priority) for t in after.targets] == before
    assert blackboard.targets("case-0000")[0].confidence == 0.8


def test_the_chain_can_be_bypassed_without_the_environment():
    picture = _picture(*_wired())
    assert isinstance(decide(picture, chain=False), FlatReport)
    assert isinstance(decide(picture, chain=True), CommanderBrief)
    assert isinstance(flatten(picture), FlatReport)


# --------------------------------------------------------------------------
# Everything back where it was
# --------------------------------------------------------------------------

def test_no_check_leaked_a_switch_into_the_process():
    """The suites that run after this one must see the shipped system."""
    assert active() == (), f"left set: {active()}"
    assert enabled(WBF) and enabled(CMC) and enabled(DECISION)
    assert disabled_agents() == frozenset()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    # The leak check runs last whatever it is called, since it is the only one
    # that asserts about the state every other check has been mutating.
    tests.sort(key=lambda t: t.__name__ == "test_no_check_leaked_a_switch_into_the_process")
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
