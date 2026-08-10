"""Known-answer checks for the Phase 6 guardrails.

    python -m src.guardrails.test_guardrails
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract
from .audit import PROVENANCE_REJECTED, AuditLog
from .cache import ResponseCache
from .contradiction import (
    COUNT_MISMATCH,
    DENIAL,
    ELEVATION_MISMATCH,
    Finding,
    PerceptionFacts,
    check,
)
from .parsers import DISCARDED, OK, REPAIRED, extract_json, parse_reply, parse_text, repair_json
from .provenance import (
    AGENT_MISMATCH,
    MISSING_IDENTITY,
    MISSING_TAG,
    TAG_HEALTH,
    TAG_HISTORY,
    TAG_INTERVIEW,
    TAG_LIDAR,
    TAG_PATH,
    TAG_RGB,
    TAG_SCENE,
    TAG_TRACK,
    TAG_WBF,
    TAG_WEATHER,
    UNKNOWN_TAG,
    UNTRUSTED_IDENTITY,
    ProvenanceRegistry,
)
from .schemas import HealthReply, InterviewReply, SceneReply, TextReply
from .tamper import (
    MALFORMED_IMAGE,
    FrameLedger,
    detection_divergence,
    digest_of,
    sniff_mime,
)

_GOOD_SCENE = json.dumps({
    "description": "Figure prone on scree beside a stream.",
    "terrain": "loose scree",
    "visibility": "low cloud",
    "hazards": ["fast water"],
    "subject_state": "not moving",
})


# --------------------------------------------------------------------------
# Parsing: the happy path
# --------------------------------------------------------------------------

def test_clean_json_parses_first_time():
    result = parse_reply(_GOOD_SCENE, SceneReply)
    assert result.outcome == OK and result.attempts == 1 and result.ok
    assert result.value.description.startswith("Figure prone")
    assert result.value.hazards == ["fast water"]
    assert result.error is None


def test_fenced_and_wrapped_json_still_parses():
    for wrapper in (
        "```json\n{body}\n```",
        "```\n{body}\n```",
        "Here is the JSON you asked for:\n{body}\nHope that helps!",
    ):
        result = parse_reply(wrapper.format(body=_GOOD_SCENE), SceneReply)
        assert result.ok, wrapper
        assert result.value.description.startswith("Figure prone")


# --------------------------------------------------------------------------
# Parsing: repair
# --------------------------------------------------------------------------

def test_repair_fixes_syntax_models_actually_get_wrong():
    broken = [
        '{"description": "A figure on scree", "hazards": ["water",],}',   # trailing commas
        '{"description": "A figure on scree", "subject_state": None}',    # python literals
        "{'description': 'A figure on scree'}",                           # single quotes
    ]
    for text in broken:
        result = parse_reply(text, SceneReply)
        assert result.outcome == REPAIRED, f"{text} -> {result.outcome} {result.error}"
        assert result.attempts == 2
        assert result.value.description == "A figure on scree"


def test_repair_never_invents_a_missing_field():
    """A description the model never wrote must not appear because a schema
    wanted one."""
    result = parse_reply('{"terrain": "scree", "hazards": []}', SceneReply)
    assert result.discarded and result.value is None
    assert result.attempts == 2, "two strikes, then stop"
    assert "description" in (result.error or "")


def test_two_failures_discard_and_return_the_default():
    fallback = SceneReply(description="unavailable")
    result = parse_reply("the weather is nice today", SceneReply, default=fallback)
    assert result.outcome == DISCARDED and result.attempts == 2
    assert result.value is fallback, "the caller's safe default, not a guess"
    assert result.raw == "the weather is nice today", "the raw reply is kept for audit"


def test_extract_and_repair_are_safe_on_junk():
    assert extract_json("") == ""
    assert extract_json("no braces here") == "no braces here"
    assert repair_json("") == ""
    assert parse_reply(None, SceneReply).discarded
    assert parse_reply("", SceneReply).discarded


# --------------------------------------------------------------------------
# Schemas: the coercions models actually need
# --------------------------------------------------------------------------

def test_scene_schema_coerces_a_lone_hazard_string():
    single = SceneReply.model_validate({"description": "x", "hazards": "fast water"})
    assert single.hazards == ["fast water"]
    listed = SceneReply.model_validate({"description": "x", "hazards": "water, rock"})
    assert listed.hazards == ["water", "rock"]
    assert SceneReply.model_validate({"description": "x", "hazards": None}).hazards == []


def test_schemas_treat_placeholder_words_as_unknown():
    reply = InterviewReply.model_validate({
        "time_last_seen": "N/A", "clothing": "  ", "direction_of_travel": "unknown"})
    assert reply.time_last_seen is None and reply.clothing is None
    assert reply.direction_of_travel is None


def test_interview_confidence_is_clamped_not_rejected():
    assert InterviewReply.model_validate({"confidence": 5}).confidence == 1.0
    assert InterviewReply.model_validate({"confidence": -2}).confidence == 0.0
    assert InterviewReply.model_validate({"confidence": "wet"}).confidence == 0.5
    assert InterviewReply.model_validate({"confidence": "0.7"}).confidence == 0.7


def test_health_schema_keeps_wild_multipliers_for_the_agent_to_clamp():
    """Rejecting them here would throw away the rationale with the number and
    lose the audit trail of what the model said."""
    reply = HealthReply.model_validate({"multiplier": 50, "rationale": "will be fine"})
    assert reply.multiplier == 50.0 and reply.rationale == "will be fine"
    assert HealthReply.model_validate({"multiplier": "0.6"}).multiplier == 0.6


def test_text_replies_are_bounded_at_both_ends():
    assert parse_text("A perfectly reasonable field briefing.", TextReply).ok
    assert parse_text("", TextReply).discarded, "an empty briefing is a failure"
    assert parse_text("ok", TextReply).discarded, "so is a two-character one"
    assert parse_text("x" * 5000, TextReply).discarded, "and so is a runaway"


def test_text_parsing_tidies_chat_scaffolding():
    result = parse_text("Here is the briefing:\nWork sector one first, then two.", TextReply)
    assert result.ok and result.value.text.startswith("Work sector one")
    assert parse_text("```\nWork sector one first, then two.\n```", TextReply).value.text == (
        "Work sector one first, then two.")


def test_text_parsing_never_eats_content_before_a_colon():
    """A general 'strip anything before a colon' rule would swallow this."""
    briefing = "Sector 1: work the drainage first, then the scree above it."
    assert parse_text(briefing, TextReply).value.text == briefing


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _counting_completer(reply="answer"):
    calls = []

    def complete(prompt, image=None, mime_type="image/jpeg"):
        calls.append((prompt, image))
        return f"{reply}-{len(calls)}"

    complete.calls = calls
    return complete


def test_repeat_prompts_do_not_hit_the_api_twice():
    cache = ResponseCache()
    inner = _counting_completer()
    cached = cache.wrap(inner)

    assert cached("what sectors?") == "answer-1"
    assert cached("what sectors?") == "answer-1", "same prompt, same answer, no call"
    assert len(inner.calls) == 1
    assert cache.hits == 1 and cache.misses == 1 and cache.calls_saved == 1

    assert cached("something else") == "answer-2"
    assert len(inner.calls) == 2 and cache.misses == 2


def test_images_are_part_of_the_key():
    """Two frames sharing a prompt are not the same question."""
    cache = ResponseCache()
    inner = _counting_completer()
    cached = cache.wrap(inner)

    cached("describe", image=b"frame-one")
    cached("describe", image=b"frame-two")
    assert len(inner.calls) == 2, "different images must not share a cache entry"
    cached("describe", image=b"frame-one")
    assert len(inner.calls) == 2 and cache.hits == 1


def test_entries_expire():
    now = {"t": 0.0}
    cache = ResponseCache(ttl_seconds=60, clock=lambda: now["t"])
    inner = _counting_completer()
    cached = cache.wrap(inner)

    cached("conditions?")
    now["t"] = 30.0
    cached("conditions?")
    assert len(inner.calls) == 1, "still fresh"

    now["t"] = 61.0
    cached("conditions?")
    assert len(inner.calls) == 2, "a stale safety-relevant answer must not be served"
    assert cache.expirations == 1


def test_least_recently_used_is_evicted():
    cache = ResponseCache(maxsize=2)
    cached = cache.wrap(_counting_completer())
    cached("a")
    cached("b")
    cached("a")          # 'a' is now the most recent
    cached("c")          # evicts 'b'
    assert cache.size == 2 and cache.evictions == 1
    assert cache.get(cache.key("b")) is None
    assert cache.get(cache.key("a")) is not None


def test_failures_are_never_cached():
    """A dead endpoint is a fact about one moment, not an answer to remember."""
    state = {"fail": True}

    def flaky(prompt, image=None, mime_type="image/jpeg"):
        if state["fail"]:
            raise RuntimeError("503")
        return "recovered"

    cached = ResponseCache().wrap(flaky)
    try:
        cached("query")
        raise AssertionError("the failure must propagate")
    except RuntimeError:
        pass

    state["fail"] = False
    assert cached("query") == "recovered", "the agent must not stay degraded"


def test_cache_can_be_cleared_and_invalidated():
    cache = ResponseCache()
    inner = _counting_completer()
    cached = cache.wrap(inner)

    cached("a")
    cached("b")
    assert cache.invalidate("a") is True
    assert cache.invalidate("a") is False
    cached("a")
    assert len(inner.calls) == 3

    cache.clear()
    assert cache.size == 0
    cached("b")
    assert len(inner.calls) == 4


def test_cached_falsy_values_are_still_hits():
    cache = ResponseCache()
    calls = []

    def empty(prompt, image=None, mime_type="image/jpeg"):
        calls.append(prompt)
        return ""

    cached = cache.wrap(empty)
    cached("q")
    cached("q")
    assert len(calls) == 1, "an empty string is a cached answer, not a miss"


# --------------------------------------------------------------------------
# Contradiction guard
# --------------------------------------------------------------------------

@dataclass
class _Target:
    located: bool = True
    elevation_m: float | None = 1200.0
    class_label: str = "person"
    scene_description: str | None = None


def _facts(n=1, elevation=1200.0):
    return PerceptionFacts.from_targets([_Target(elevation_m=elevation) for _ in range(n)])


def test_denial_of_a_confirmed_detection_is_overridden():
    """The example that matters: perception has a person at 1200 m and the
    model says nobody is there."""
    for denial in (
        "No target was found in the search area.",
        "Nothing was detected during this sortie.",
        "The sector is clear.",
        "No sign of the subject.",
        "Teams were unable to locate anyone.",
        "The search found no one.",
    ):
        result = check(denial, _facts(1))
        assert result.overridden, denial
        assert result.severe and result.findings[0].rule == DENIAL
        assert result.original == denial, "the raw claim is kept for audit"
        assert "withheld" in result.text and "1 confirmed person" in result.text


def test_denial_is_fine_when_nothing_has_been_found():
    result = check("No target was found in the search area.", _facts(0))
    assert result.ok and not result.overridden
    assert result.text == "No target was found in the search area."


def test_override_can_be_turned_off_for_flag_only_operation():
    result = check("No target was found.", _facts(1), override=False)
    assert result.findings and result.severe
    assert not result.overridden and result.text == "No target was found."


def test_count_mismatch_is_flagged_not_suppressed():
    result = check("Two targets are confirmed on the ridge.", _facts(1))
    assert not result.overridden, "a loose count is not worth binning a whole briefing"
    assert [f.rule for f in result.findings] == [COUNT_MISMATCH]
    assert result.text == "Two targets are confirmed on the ridge."
    assert check("1 person located.", _facts(1)).ok


def test_count_rule_ignores_talk_about_historical_cases():
    """History insights legitimately count cases and hikers."""
    for benign in (
        "3 comparable case(s) retrieved from the archive.",
        "Two hikers benighted on scree followed the watercourse downhill.",
        "Subjects were found between 200 m and 1400 m from the last known point.",
    ):
        assert check(benign, _facts(1)).ok, benign


def test_elevation_mismatch_is_flagged():
    result = check("The subject is at an altitude of 1900 m.", _facts(1, elevation=1200.0))
    assert [f.rule for f in result.findings] == [ELEVATION_MISMATCH]
    assert not result.overridden
    assert check("The subject is at an altitude of 1250 m.", _facts(1, elevation=1200.0)).ok, (
        "inside tolerance"
    )


def test_elevation_rule_ignores_plain_distances():
    """'1400 m from the last known point' is a distance, not an altitude."""
    assert check("Found 1400 m from the last known point.", _facts(1, 1200.0)).ok
    assert check("Sector 1 lies 1450 m out on a bearing of 177 degrees.", _facts(1, 1200.0)).ok


def test_guard_is_quiet_on_ordinary_prose():
    facts = _facts(2, elevation=1200.0)
    good = ("Both confirmed tracks sit on open scree below the ridge; work the "
            "drainage first and approach from the south.")
    result = check(good, facts)
    assert result.ok and not result.overridden and result.text == good


def test_facts_summarise_what_perception_holds():
    facts = PerceptionFacts.from_targets([
        _Target(elevation_m=1200.0), _Target(elevation_m=1460.0, located=False)])
    assert facts.confirmed_targets == 2 and facts.located_targets == 1
    summary = facts.summarise()
    assert "2 confirmed person" in summary and "1200-1460 m" in summary
    assert PerceptionFacts().summarise() == "Perception holds no confirmed targets."


def test_empty_text_is_never_a_contradiction():
    assert check("", _facts(1)).ok
    assert check(None, _facts(1)).ok


# --------------------------------------------------------------------------
# Provenance registry
# --------------------------------------------------------------------------

_DRONE_MAC = "AA:BB:CC:DD:EE:01"
_ROGUE_MAC = "00:11:22:33:44:FF"
_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


def _clue(tag=TAG_RGB, agent=AgentSource.DRONE_RGB, metadata=None, case="case-0000"):
    return ClueContract(
        clue_id=f"clue-{tag}-{agent.value}",
        case_id=case,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_agent=agent,
        confidence_score=0.9,
        finding_summary="test clue",
        provenance_tag=tag,
        agent_metadata=metadata or {},
    )


def _fleet(**kwargs):
    return ProvenanceRegistry(devices={_DRONE_MAC}, endpoints={_ENDPOINT},
                              operators={"op-17"}, **kwargs)


def test_a_registered_source_passes():
    verdict = _fleet().verify(_clue(metadata={"device_id": _DRONE_MAC}))
    assert verdict.allowed and bool(verdict) and verdict.reason is None
    assert verdict.rule.tag == TAG_RGB


def test_every_agent_tag_in_use_is_registered():
    """A tag an agent emits but the registry has never heard of would have every
    clue from that agent silently rejected in the field."""
    registry = ProvenanceRegistry()
    for tag in (TAG_RGB, TAG_LIDAR, TAG_WBF, TAG_TRACK, TAG_WEATHER, TAG_PATH,
                TAG_SCENE, TAG_HEALTH, TAG_HISTORY, TAG_INTERVIEW):
        assert tag in registry.tags, tag
    assert len(registry.tags) == 10


def test_an_unknown_tag_is_rejected():
    verdict = ProvenanceRegistry().verify(_clue(tag="onboard:someone-elses-drone"))
    assert not verdict and verdict.reason == UNKNOWN_TAG
    assert "not a registered origin" in verdict.detail


def test_a_missing_tag_is_rejected():
    clue = _clue()
    spoofed = clue.model_copy(update={"provenance_tag": "   "})
    verdict = ProvenanceRegistry().verify(spoofed)
    assert not verdict and verdict.reason == MISSING_TAG


def test_a_borrowed_tag_is_rejected():
    """The obvious forgery: reuse a legitimate tag from the wrong agent."""
    verdict = ProvenanceRegistry().verify(
        _clue(tag=TAG_WEATHER, agent=AgentSource.PERCEPTION_FUSION))
    assert not verdict and verdict.reason == AGENT_MISMATCH
    assert "may not use" in verdict.detail and "WEATHER_API" in verdict.detail


def test_tag_and_agent_binding_needs_no_configuration():
    """The anti-spoofing layer works out of the box; nothing has to be set up
    before it starts refusing forged clues."""
    bare = ProvenanceRegistry()
    assert bare.unconfigured == ("device", "endpoint", "operator")
    assert not bare.verify(_clue(tag=TAG_PATH, agent=AgentSource.DRONE_RGB))
    assert bare.verify(_clue(metadata={}))  # identity unchecked when unconfigured


def test_an_unapproved_drone_is_rejected():
    verdict = _fleet().verify(_clue(metadata={"device_id": _ROGUE_MAC}))
    assert not verdict and verdict.reason == UNTRUSTED_IDENTITY
    assert _ROGUE_MAC in verdict.detail


def test_a_drone_that_will_not_identify_itself_is_rejected():
    for metadata in ({}, {"device_id": None}, {"device_id": ""}):
        verdict = _fleet().verify(_clue(metadata=metadata))
        assert not verdict and verdict.reason == MISSING_IDENTITY, metadata


def test_an_unauthorised_api_endpoint_is_rejected():
    registry = _fleet()
    good = _clue(TAG_WEATHER, AgentSource.WEATHER_API, {"endpoint": _ENDPOINT})
    bad = _clue(TAG_WEATHER, AgentSource.WEATHER_API,
                {"endpoint": "http://weather.attacker.example/forecast"})
    assert registry.verify(good)
    assert registry.verify(bad).reason == UNTRUSTED_IDENTITY


def test_an_unverified_operator_is_rejected():
    registry = _fleet()
    assert registry.verify(_clue(TAG_INTERVIEW, AgentSource.INTERVIEW_LLM,
                                 {"operator_id": "op-17"}))
    assert registry.verify(_clue(TAG_INTERVIEW, AgentSource.INTERVIEW_LLM,
                                 {"operator_id": "op-99"})).reason == UNTRUSTED_IDENTITY


def test_internal_computation_needs_no_hardware_identity():
    """Fusion and the model agents have nothing external to authenticate."""
    registry = _fleet()
    for tag, agent in ((TAG_WBF, AgentSource.PERCEPTION_FUSION),
                       (TAG_PATH, AgentSource.PATH_MODEL),
                       (TAG_SCENE, AgentSource.SCENE_VLM),
                       (TAG_HEALTH, AgentSource.HEALTH_LLM),
                       (TAG_HISTORY, AgentSource.HISTORY_RAG)):
        assert registry.verify(_clue(tag, agent)), tag


def test_an_empty_credential_set_trusts_nothing():
    """A legitimate way to ground a fleet: configured, but with no members."""
    grounded = ProvenanceRegistry(devices=set())
    assert grounded.unconfigured == ("endpoint", "operator")
    assert not grounded.verify(_clue(metadata={"device_id": _DRONE_MAC}))


def test_unconfigured_reports_exactly_what_is_unchecked():
    assert ProvenanceRegistry().unconfigured == ("device", "endpoint", "operator")
    assert _fleet().unconfigured == (), "a fully configured registry has nothing left off"
    assert ProvenanceRegistry(devices={_DRONE_MAC}).unconfigured == ("endpoint", "operator")


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

def test_a_rejection_is_recorded_as_a_security_event():
    log = AuditLog()
    clue = _clue(tag="onboard:someone-elses-drone")
    verdict = ProvenanceRegistry().verify(clue)
    event = log.reject_clue(clue, verdict, case_id="case-0000")

    assert event.kind == PROVENANCE_REJECTED and event.reason == UNKNOWN_TAG
    assert event.clue_id == clue.clue_id and event.case_id == "case-0000"
    assert event.source_agent == "DRONE_RGB"
    assert event.provenance_tag == "onboard:someone-elses-drone"
    assert "DRONE_RGB" in event.describe() and "not a registered origin" in event.describe()
    assert log.total == 1 and log.counts == {UNKNOWN_TAG: 1}


def test_the_log_separates_cases():
    log = AuditLog()
    registry = ProvenanceRegistry()
    for case in ("case-A", "case-A", "case-B"):
        clue = _clue(tag="bogus", case=case)
        log.reject_clue(clue, registry.verify(clue), case_id=case)
    assert len(log.for_case("case-A")) == 2 and len(log.for_case("case-B")) == 1
    assert log.total == 3


def test_counters_survive_a_flood_even_when_detail_does_not():
    """A flood must not exhaust memory, and must not hide that it happened."""
    log = AuditLog(maxlen=10)
    registry = ProvenanceRegistry()
    for i in range(100):
        clue = _clue(tag=f"bogus-{i}")
        log.reject_clue(clue, registry.verify(clue))

    assert log.total == 100, "the count of attempts is never lost"
    assert log.counts[UNKNOWN_TAG] == 100
    assert len(log.events) == 10 and log.dropped == 90
    assert "90 detail(s) dropped" in repr(log)


def test_the_log_can_be_cleared():
    log = AuditLog()
    clue = _clue(tag="bogus")
    log.reject_clue(clue, ProvenanceRegistry().verify(clue))
    log.clear()
    assert log.total == 0 and log.events == [] and log.counts == {}


# --------------------------------------------------------------------------
# Frame ledger
# --------------------------------------------------------------------------

_JPEG = b"\xff\xd8\xff\xe0" + b"frame payload" * 8


def test_ledger_records_and_verifies_a_frame():
    ledger = FrameLedger()
    record = ledger.record("frame_0001", _JPEG, device_id="AA:BB")
    assert record.digest == digest_of(_JPEG)
    assert record.size_bytes == len(_JPEG) and record.mime_type == "image/jpeg"
    assert ledger.verify("frame_0001", _JPEG)
    assert ledger.verified == 1 and ledger.rejected == 0


def test_header_sniffing_knows_the_formats_a_camera_produces():
    assert sniff_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert sniff_mime(b"II*\x00rest") == "image/tiff"
    assert sniff_mime(b"not an image") is None


def test_header_check_can_be_relaxed_for_synthetic_frames():
    """Stub sorties have no real JPEGs. The relaxation is explicit, because a
    check that must be bypassed in testing is a check nobody trusts."""
    strict, relaxed = FrameLedger(), FrameLedger(strict_header=False)
    for ledger in (strict, relaxed):
        ledger.record("f", b"<synthetic frame>")
    assert strict.verify("f", b"<synthetic frame>").reason == MALFORMED_IMAGE
    assert relaxed.verify("f", b"<synthetic frame>")


def test_divergence_is_symmetric_difference_over_union():
    a = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    assert detection_divergence(a, a) == 0.0
    assert detection_divergence(a, a[:1]) == 0.5, "one of two lost"
    assert detection_divergence(a, []) == 1.0
    assert detection_divergence([], []) == 0.0
    # Small shifts inside tolerance are the same detection.
    nudged = [(2.0, 2.0, 12.0, 12.0), (101.0, 101.0, 111.0, 111.0)]
    assert detection_divergence(a, nudged) == 0.0
    # Tighter than the smaller nudge (1 px), so neither box matches.
    assert detection_divergence(a, nudged, tolerance_px=0.5) == 1.0
    # Between the two nudges (2 px and 1 px): one matches, one does not.
    assert detection_divergence(a, nudged, tolerance_px=1.0) > 0.5


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
