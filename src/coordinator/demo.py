"""Phase 3 and 4 exit criteria, end to end.

    operator query -> orchestrator dispatches -> fusion emits a ranked picture
    two sources fused into one picture, weather visibly moving the ranking

Runs the real agents behind the orchestrator: stub sensors -> weighted box
fusion -> BoT-SORT -> geolocation -> Redis Streams -> coordinator fusion -> the
picture, with the Weather agent as the second source.

Weather is fetched from Open-Meteo only when asked; by default the conditions
are fixed so the demo is deterministic and works offline.

    python -m src.coordinator.demo
    python -m src.coordinator.demo --live-weather
    REDIS_URL=redis://localhost:6379/0 python -m src.coordinator.demo
"""
from dotenv import load_dotenv
load_dotenv()
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..agents.health import HealthAgent, SubjectProfile
from ..agents.history import HistoryAgent
from ..agents.interview import InterviewAgent
from ..agents.llm import FAST_LLM_MODEL, NimClient, static_completer
from ..agents.path import PathAgent, PathModel
from ..agents.scene import SceneAgent
from ..agents.weather import Conditions, WeatherAgent, open_meteo_fetch
from ..bus import FakeRedisStreams, RedisBus
from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.audit import AuditLog
from ..guardrails.cache import ResponseCache
from ..guardrails.provenance import OPEN_METEO_ENDPOINT, TAG_TRACK, ProvenanceRegistry
from ..tuning.params import CONFIG_PATH, load_params
from ..perception.agent import DetectionAgent
from ..perception.detectors import Target, lidar_stub, yolo11m_stub
from ..perception.geolocation import (
    Camera,
    Telemetry,
    ground_distance_m,
    slant_range_m,
    world_to_pixel,
)
from ..perception.terrain import GridDEM
from ..perception.tracking import BoTSORT
from .blackboard import Blackboard
from .fusion import CoordinatorFusion
from .orchestrator import Orchestrator, Scenario

DRONE = (46.8182, 8.2275)
SUBJECTS = ((46.8190, 8.2280), (46.8186, 8.2265))

# The approved fleet, endpoints and operators for this search. Anything on the
# bus not matching these is refused at the blackboard and logged.
DRONE_ID = "AA:BB:CC:DD:EE:01"
OPERATOR_ID = "op-17"


def _search_area_dem():
    """An alpine slope rising ~180 m per km toward the north."""
    return GridDEM.from_function(
        lambda lat, lon: 1450.0 + (lat - DRONE[0]) * 20_000.0,
        lat_min=46.79, lon_min=8.20, lat_step=0.0005, lon_step=0.0005,
        n_lat=120, n_lon=120,
    )


def build_detection_agent(bus, case_id):
    """The Phase 2 agent, wrapped as something the orchestrator can dispatch."""
    dem = _search_area_dem()
    camera = Camera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    telemetry = Telemetry(DRONE[0], DRONE[1], altitude_m=1570.0, yaw_deg=15.0, pitch_deg=55.0)

    targets = []
    for lat, lon in SUBJECTS:
        ground = dem.elevation(lat, lon)
        feet = world_to_pixel(lat, lon, ground, camera, telemetry)
        head = world_to_pixel(lat, lon, ground + 1.7, camera, telemetry)
        half_width = abs(feet[1] - head[1]) * 0.35
        targets.append(Target(
            box=(feet[0] - half_width, head[1], feet[0] + half_width, feet[1]),
            geo=(lat, lon),
            range_m=slant_range_m(telemetry, lat, lon, ground),
        ))

    agent = DetectionAgent(bus, case_id, camera, dem=dem, tracker=BoTSORT(min_hits=3),
                           device_id=DRONE_ID)
    rgb = yolo11m_stub(seed=11, recall=0.95, device_id=DRONE_ID)
    lidar = lidar_stub(seed=11, recall=0.8, device_id=DRONE_ID)
    sortie = {"frame": 0}

    def handler(case_id, context):
        """One dispatch flies a short sortie and publishes what it confirms."""
        published = 0
        for _ in range(context.get("frames", 4)):
            name = f"frame_{sortie['frame']:04d}"
            sortie["frame"] += 1
            published += len(agent.process_frame(
                name,
                [rgb.detect(name, targets, case_id), lidar.detect(name, targets, case_id)],
                telemetry,
            ))
        return {"frames": context.get("frames", 4), "clues_published": published}

    return handler


EXPOSED, SHELTERED = SUBJECTS[1], SUBJECTS[0]

# Last seen high on the ridge, 1.4 km uphill of where the drone is now looking.
# Terrain here falls away to the south, so the Path model's downhill drift is
# what should carry the simulated walkers down onto the search area.
POINT_LAST_SEEN = (SUBJECTS[0][0] + 0.0126, SUBJECTS[0][1] - 0.0004)


def _micro_climate_fetch():
    """Fixed conditions that differ by where you stand.

    An exposed ridge in wind versus a sheltered gully a hundred metres away.
    Real forecast points are far coarser than this — the split is exaggerated so
    the per-sector behaviour is visible in one screen.
    """
    ridge = Conditions(temperature_c=1.0, wind_kmh=15.0, precipitation_mm=0.0, humidity_pct=70.0)
    gully = Conditions(temperature_c=9.0, wind_kmh=4.0, precipitation_mm=0.0, humidity_pct=55.0)

    def fetch(latitude, longitude):
        return ridge if ground_distance_m((latitude, longitude), EXPOSED) < 60.0 else gully

    return fetch


def build_weather_agent(bus, case_id, live=False):
    """The Weather agent, wrapped for the orchestrator.

    Reads conditions at each subject's position, so the picture knows which
    sector is dangerous rather than treating the whole search area as one.
    """
    agent = WeatherAgent(
        bus, case_id, fetch=open_meteo_fetch if live else _micro_climate_fetch()
    )

    def handler(case_id, context):
        readings = {}
        for position in context.get("weather_at", SUBJECTS):
            clue = agent.observe(*position)
            readings[f"{position[0]:.4f},{position[1]:.4f}"] = (
                f"{clue.agent_metadata['apparent_c']:.1f} C, "
                f"{'RISK' if clue.agent_metadata['hypothermia_risk'] else 'no risk'}, "
                f"window {clue.agent_metadata['survival_window_hours']} h"
            )
        return readings

    return handler


STUB_SCENE = json.dumps({
    "description": "Figure prone on open scree beside a meltwater channel.",
    "terrain": "loose scree, no path",
    "visibility": "low cloud, failing light",
    "hazards": ["fast water", "loose rock"],
    "subject_state": "not moving",
})


def build_path_agent(bus, case_id, dem, complete, pls, elapsed_hours):
    """The Monte-Carlo sector model, with the LLM writing prose over its output."""
    agent = PathAgent(bus, case_id, dem=dem, model=PathModel(walkers=800, top_k=4),
                      complete=complete, seed=4)

    def handler(case_id, context):
        clue = agent.project(pls, context.get("elapsed_hours", elapsed_hours),
                             context=context.get("conditions"))
        sectors = clue.agent_metadata["sectors"]
        return {
            "sectors": len(sectors),
            "best": f"{sectors[0]['probability']:.0%} at {sectors[0]['distance_m']:.0f} m"
                    if sectors else "none",
            "briefing_source": clue.agent_metadata["briefing_source"],
        }

    return handler, agent


def build_scene_agent(bus, case_id, describe, stream, registry=None):
    """The VLM, gated on confirmed detections.

    Keeps its own cursor on the stream: fusion and this agent are independent
    consumers of the same clues, which is exactly what Redis Streams is for.

    It applies the provenance allow-list itself. Fusion guards the blackboard,
    but a spoofed clue reaching *this* consumer would spend an API call and
    describe a frame from a drone we do not trust — every bus consumer has to
    check, not just the one writing to the blackboard.
    """
    agent = SceneAgent(bus, case_id, describe=describe,
                       # Stand-in frame bytes. Real sorties supply captured
                       # frames; the agent refuses to describe anything it is
                       # not actually handed.
                       image_loader=lambda frame_id: b"<stub frame bytes>")
    cursor = {"id": "0-0"}

    def handler(case_id, context):
        entries = bus.read(stream, last_id=cursor["id"])
        if entries:
            cursor["id"] = entries[-1][0]
        published = agent.process(_trusted(entries, registry))
        return {
            "described": len(published),
            "api_calls": agent.api_calls,
            "skipped_unconfirmed": agent.skipped_unconfirmed,
        }

    return handler, agent


WITNESS_STATEMENT = (
    "I passed him on the ridge path about half seven this morning. Red shell jacket, "
    "jeans, no rucksack that I saw. He carried on north towards the saddle. He looked "
    "tired and he was limping a bit."
)

STUB_INTERVIEW = json.dumps({
    "time_last_seen": "about 07:30",
    "clothing": "red shell jacket and jeans",
    "direction_of_travel": "north towards the saddle",
    "confidence": 0.8,
})

STUB_HEALTH = json.dumps({
    "multiplier": 0.6,
    "rationale": "reported limping and wearing jeans, which soak and hold cold",
})


def build_interview_agent(bus, case_id, complete):
    """NER over the witness statement. Runs on the small fast model."""
    agent = InterviewAgent(bus, case_id, complete=complete, operator_id=OPERATOR_ID)
    done = {"ran": False}

    def handler(case_id, context):
        if done["ran"]:
            return {"skipped": "statement already processed"}
        done["ran"] = True
        clue = agent.interview(context.get("statement", WITNESS_STATEMENT), witness="hill walker")
        if clue is None:
            return {"extracted": 0}
        return {k: clue.agent_metadata[k]
                for k in ("time_last_seen", "clothing", "direction_of_travel")}

    return handler, agent


def _trusted(entries, registry):
    """Clues from registered origins only. Every bus consumer must filter."""
    if registry is None:
        return [clue for _, clue in entries]
    return [clue for _, clue in entries if registry.verify(clue)]


def build_health_agent(bus, case_id, complete, stream, profile, registry=None, **bounds):
    """Refines whatever survival windows the Weather agent just published."""
    agent = HealthAgent(bus, case_id, complete=complete, profile=profile, **bounds)
    cursor = {"id": "0-0"}

    def handler(case_id, context):
        entries = bus.read(stream, last_id=cursor["id"])
        if entries:
            cursor["id"] = entries[-1][0]
        published = agent.assess_all(_trusted(entries, registry))
        if not published:
            return {"refined": 0}
        return {
            "refined": len(published),
            "windows": [f"{c.agent_metadata['baseline_window_hours']} h -> "
                        f"{c.agent_metadata['survival_window_hours']} h" for c in published],
            "source": published[0].agent_metadata["window_source"],
        }

    return handler, agent


def build_history_agent(bus, case_id, complete):
    """Retrieves comparable past incidents and synthesises one insight."""
    agent = HistoryAgent(bus, case_id, complete=complete)
    done = {"ran": False}

    def handler(case_id, context):
        if done["ran"]:
            return {"skipped": "archive already queried"}
        done["ran"] = True
        query = context.get(
            "history_query",
            "adult hiker lost on an exposed north ridge in cloud, autumn, scree and gullies",
        )
        clue = agent.recall(query, position=DRONE)
        if clue is None:
            return {"retrieved": 0}
        return {
            "retrieved": len(clue.agent_metadata["retrieved"]),
            "cases": [r["case_id"] for r in clue.agent_metadata["retrieved"]],
            "blocked_by_allow_list": clue.agent_metadata["blocked_by_allow_list"],
        }

    return handler, agent


@dataclass
class Wiring:
    """Everything a demo needs after the case is stood up."""

    orchestrator: object
    fusion: object
    case: object
    bus: object
    registry: object = None
    audit: object = None
    cache: object = None
    dem: object = None
    backend: str = ""
    client: object = None
    agents: dict = field(default_factory=dict)


def build_case(live_weather=False):
    """Stand up a fully wired case: bus, guards, agents, orchestrator.

    Shared by this demo and the critic's, so both exercise the same system
    rather than two subtly different ones.
    """
    url = os.environ.get("REDIS_URL")
    if url:
        bus, backend = RedisBus.from_url(url), url
        bus.client.ping()
    else:
        bus, backend = RedisBus(FakeRedisStreams()), "mocked connection (set REDIS_URL for a server)"

    now = datetime.now(timezone.utc)
    blackboard = Blackboard()
    case = blackboard.open_case(
        # CASE_ID joins a case another process is already publishing to — the
        # mock drone feed, say. Unset mints a fresh one, as before.
        case_id=os.environ.get("CASE_ID") or None,
        sector="north ridge",
        reported_missing="2 hikers",
        missing_since=now - timedelta(hours=2),
    )
    # Tight radius so each reading covers only its own sector; see
    # `_micro_climate_fetch` on why that is exaggerated for the demo.
    registry = ProvenanceRegistry(
        devices={DRONE_ID}, endpoints={OPEN_METEO_ENDPOINT}, operators={OPERATOR_ID},
    )
    audit = AuditLog()
    # Tuned parameters if a study has been run, the shipped defaults otherwise.
    # Loaded once here and handed to fusion and the orchestrator, so nothing
    # downstream reads its own copy off disk.
    params = load_params()
    fusion = CoordinatorFusion(bus, blackboard, weather_radius_m=60.0,
                               provenance=registry, audit=audit, params=params)
    orchestrator = Orchestrator(fusion, blackboard, params=params)

    dem = _search_area_dem()
    stream = f"clues:{case.case_id}"
    client = NimClient.from_env()
    fast = client.using(model=FAST_LLM_MODEL) if client else None

    # Live client when NVIDIA_API_KEY is set; fixed replies otherwise, so the
    # demo is deterministic and never depends on a quota.
    #
    # One shared cache across every agent. Repeated operator queries re-ask the
    # same questions — the archive query and the field briefing do not change
    # between two presses of the button — and the NIM free tier is rate limited.
    # A 10-minute TTL keeps weather-derived text from going stale in a search.
    cache = ResponseCache(maxsize=256, ttl_seconds=600)

    def wrap(completer):
        return None if completer is None else cache.wrap(completer)

    complete = wrap(client.complete if client else None)
    describe = wrap(client.complete if client else static_completer(STUB_SCENE))
    interview_complete = wrap(fast.complete if fast else static_completer(STUB_INTERVIEW))
    health_complete = wrap(client.complete if client else static_completer(STUB_HEALTH))
    # No canned reply for history: a fixed string would cite cases the archive
    # did not actually return. Offline it falls back to a summary built from the
    # real retrieval, which is the honest thing to show.
    history_complete = wrap(client.complete) if client else None

    profile = SubjectProfile(age_years=68, clothing="red shell jacket and jeans",
                             injured=True, fitness="average")
    health_bounds = dict(multiplier_floor=params.health_multiplier_floor,
                         multiplier_ceiling=params.health_multiplier_ceiling)

    path_handler, _ = build_path_agent(bus, case.case_id, dem, complete, POINT_LAST_SEEN, 2.0)
    scene_handler, scene = build_scene_agent(bus, case.case_id, describe, stream, registry)
    interview_handler, interview = build_interview_agent(bus, case.case_id, interview_complete)
    health_handler, health = build_health_agent(bus, case.case_id, health_complete, stream,
                                                profile, registry, **health_bounds)
    history_handler, history = build_history_agent(bus, case.case_id, history_complete)

    orchestrator.register("interview", interview_handler)
    orchestrator.register("detection", build_detection_agent(bus, case.case_id))
    orchestrator.register("weather", build_weather_agent(bus, case.case_id, live=live_weather))
    orchestrator.register("health", health_handler)
    orchestrator.register("history", history_handler)
    orchestrator.register("path", path_handler)
    orchestrator.register("scene", scene_handler)

    return Wiring(
        orchestrator=orchestrator, fusion=fusion, case=case, bus=bus,
        registry=registry, audit=audit, cache=cache, dem=dem,
        backend=backend, client=client,
        agents={"scene": scene, "health": health, "history": history,
                "interview": interview},
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    live_weather = "--live-weather" in argv

    wiring = build_case(live_weather=live_weather)
    orchestrator, fusion, case, bus = (wiring.orchestrator, wiring.fusion,
                                       wiring.case, wiring.bus)
    registry, audit, cache, dem = (wiring.registry, wiring.audit,
                                   wiring.cache, wiring.dem)
    backend, client = wiring.backend, wiring.client
    scene, health = wiring.agents["scene"], wiring.agents["health"]
    history, interview = wiring.agents["history"], wiring.agents["interview"]

    nim_line = (f"live — {client.model} / {client.vision_model}" if client
                else "stubbed (set NVIDIA_API_KEY for live NIM calls)")
    weather_line = "Open-Meteo (live)" if live_weather else "fixed conditions (offline)"

    print(f"\n  redis:   {backend}")
    print(f"  weather: {weather_line}")
    print(f"  nim:     {nim_line}")
    print(f"  trust:   {len(registry.tags)} registered origin(s); "
          f"unconfigured checks: {list(registry.unconfigured) or 'none'}")
    print(f"  params:  {'tuned from ' + str(CONFIG_PATH) if CONFIG_PATH.exists() else 'shipped defaults (no tuning run yet)'}")
    print(f"  case:    {case.case_id}  missing 2 h, sector {case.metadata['sector']!r}")
    print(f"  last seen at {POINT_LAST_SEEN[0]:.5f}, {POINT_LAST_SEEN[1]:.5f} "
          f"({dem.elevation(*POINT_LAST_SEEN):.0f} m, uphill of the search area)")
    print(f"  agents:  {orchestrator.agents}")

    triggers = (
        (Scenario.DRONE_AIRBORNE, "sortie launched"),
        ("What is the weather outlook up there?", "targeted weather question"),
        ("What happened in similar past cases?", "targeted archive question"),
        ("What is the weather outlook up there?", "the same question again"),
        ("Give me a full briefing", "everything"),
    )

    for trigger, note in triggers:
        label = trigger.value if isinstance(trigger, Scenario) else f"{trigger!r}"
        print(f"\n  >>> {label}  ({note})")
        dispatch = orchestrator.handle(trigger, case.case_id, frames=4)
        skipped = dispatch.agents_skipped
        print(f"      routed {dispatch.scenario.value}: {list(dispatch.agents)}"
              f"  [{dispatch.route.reason}]")
        if skipped:
            print(f"      skipped {len(skipped)} agent(s): {list(skipped)}")
        for agent, result in dispatch.results.items():
            print(f"      {agent:<10} -> {result}")

    # An attacker puts a clue on the bus: a real-looking detection from a drone
    # that is not ours, claiming a target in the wrong valley.
    print("\n  >>> injecting a spoofed detection from an unapproved airframe")
    bus.publish(ClueContract(
        clue_id="spoofed-0001",
        case_id=case.case_id,
        timestamp=datetime.now(timezone.utc),
        source_agent=AgentSource.PERCEPTION_FUSION,
        confidence_score=0.99,
        finding_summary="Subject located, stand down other sectors",
        spatial_context=SpatialContext(latitude=46.90, longitude=8.40,
                                       bounding_box=[0.0, 0.0, 40.0, 90.0]),
        frame_id="frame_9999",
        class_label="person",
        provenance_tag=TAG_TRACK,
        agent_metadata={"track_id": "99", "track_state": "CONFIRMED",
                        "device_id": "00:11:22:33:44:FF"},
    ))

    print(orchestrator.handle("full briefing", case.case_id, frames=0).picture.render())

    picture = fusion.picture(case.case_id)
    print(f"  {fusion.blackboard}")
    print(f"  bus: {bus.length(f'clues:{case.case_id}')} clues on the stream")
    print(f"  located {len(picture.located)}, not located {len(picture.unlocated)}")
    for target in picture.targets:
        print(f"    {target.target_id}: {target.observations} observation(s), "
              f"tracks {sorted(target.track_ids)}, {len(target.clue_ids)} clue(s) of lineage, "
              f"range via {target.range_source}")

    print(f"\n  scene VLM: {scene.api_calls} call(s) for {len(scene.described)} frame(s); "
          f"{scene.skipped_unconfirmed} unconfirmed detection(s) skipped, "
          f"{scene.skipped_already_seen} repeat frame(s) skipped")
    print(f"  health:    {health.published} window(s) refined, {health.skipped} clue(s) skipped")
    print(f"  history:   {history.published} insight(s), "
          f"{history.archive.blocked} record(s) blocked by the provenance allow-list")
    print(f"  interview: {interview.published} statement(s), "
          f"{interview.injection_flags} flagged as suspicious")
    print(f"  cache:     {cache.hits} hit(s), {cache.misses} miss(es) — "
          f"{cache.calls_saved} API call(s) not made")

    guarded = fusion.picture(case.case_id).guard_findings
    print(f"  guardrail: {len(guarded)} contradiction(s) with confirmed perception")

    sector_clues = [c for _, c in bus.read(f"clues:{case.case_id}")
                    if c.source_agent.value == "PATH_MODEL"]
    if sector_clues:
        latest = sector_clues[-1].agent_metadata
        print(f"\n  field briefing ({latest['briefing_source']}):")
        for line in _wrap(latest["briefing"], 76):
            print(f"    {line}")
    print()


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
