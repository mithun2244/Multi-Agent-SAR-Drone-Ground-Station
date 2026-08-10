"""The Path agent — where is the subject likely to be by now?

Architecture, Phase 5: "Path — Monte-Carlo core; the LLM only writes the field
briefing over computed sectors."

That division is the whole design. Sectors come from simulation over real
terrain; the model only turns them into prose a team leader can read out. Asking
an LLM *where to search* would be inventing coordinates, which is the one thing
this system never does. If the model is unreachable the sectors are still valid
and a deterministic summary goes out in place of the briefing, clearly labelled.

The walk model
--------------
Lost people do not diffuse like gas. They hold a heading for a while, they rest,
and they go downhill far more often than up — drainages and slopes are where
searches concentrate for a reason. So each walker is a *correlated* random walk
with a downhill drift read off the DEM, not a Brownian one, and steep ground is
refused rather than crossed.

ponytail: the mobility numbers (speed, rest fraction, downhill bias) are plain
defaults, not fitted to lost-person statistics. They are the knobs Phase 9 tunes
and Phase 8's critic retunes against found locations. The *structure* is right;
the constants are a starting point.
"""

import math
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..perception.geolocation import ground_distance_m, offset_enu
from ..guardrails.parsers import parse_text
from ..guardrails.provenance import TAG_PATH
from ..guardrails.schemas import TextReply
from .llm import DEFAULT_LLM_MODEL, LLMUnavailable

PATH_PROVENANCE = TAG_PATH
BRIEFING_MODEL_TAG = DEFAULT_LLM_MODEL


@dataclass(frozen=True)
class Sector:
    """One patch of ground, with the share of simulated walkers that ended in it."""

    rank: int
    latitude: float
    longitude: float
    radius_m: float
    probability: float
    samples: int
    bearing_deg: float          # from the point last seen
    distance_m: float           # from the point last seen


@dataclass(frozen=True)
class PathModel:
    """Simulation settings. Every field is a tuning knob, not a constant."""

    speed_kmh: float = 1.2          # slow, tired, off-trail walking
    rest_fraction: float = 0.35     # share of steps spent stationary
    downhill_bias: float = 0.6      # 0 = no preference, 1 = always downhill
    heading_sigma_deg: float = 35.0 # per-step wander around the held heading
    max_slope_deg: float = 35.0     # ground too steep to cross
    step_minutes: float = 15.0
    cell_m: float = 150.0           # sector grid resolution
    walkers: int = 1500
    top_k: int = 5


def simulate_sectors(pls, elapsed_hours, dem=None, model=None, seed=0):
    """Monte-Carlo the subject's position and bin the results into sectors.

    `pls` is the point last seen, (lat, lon). Returns sectors best-first; an
    empty list when there is no elapsed time to simulate over.
    """
    model = model or PathModel()
    if elapsed_hours <= 0 or model.walkers <= 0:
        return []

    rng = random.Random(seed)
    steps = max(1, int(round(elapsed_hours * 60.0 / model.step_minutes)))
    step_m = model.speed_kmh * 1000.0 * (model.step_minutes / 60.0)

    endpoints = []
    for _ in range(model.walkers):
        endpoints.append(_walk(pls, steps, step_m, dem, model, rng))

    return _to_sectors(pls, endpoints, model)


def _walk(pls, steps, step_m, dem, model, rng):
    """One walker, returning its (east, north) offset from the PLS in metres."""
    east = north = 0.0
    heading = rng.uniform(0.0, 360.0)

    for _ in range(steps):
        if rng.random() < model.rest_fraction:
            continue  # resting, sheltering, or milling in place

        heading = (heading + rng.gauss(0.0, model.heading_sigma_deg)) % 360.0
        if dem is not None and model.downhill_bias > 0.0:
            downhill = _downhill_heading(pls, east, north, dem)
            if downhill is not None:
                heading = _blend_headings(heading, downhill, model.downhill_bias)

        radians = math.radians(heading)
        step_east = step_m * math.sin(radians)
        step_north = step_m * math.cos(radians)

        if dem is not None and _slope_deg(pls, east, north, step_east, step_north, dem) > model.max_slope_deg:
            continue  # too steep to cross; the walker stays put this step

        east, north = east + step_east, north + step_north
    return (east, north)


def _elevation(pls, east, north, dem):
    latitude, longitude = offset_enu(pls[0], pls[1], east, north)
    return dem.elevation(latitude, longitude)


def _downhill_heading(pls, east, north, dem, probe_m=40.0):
    """Compass heading of steepest descent, or None on flat ground."""
    here = _elevation(pls, east, north, dem)
    d_east = _elevation(pls, east + probe_m, north, dem) - here
    d_north = _elevation(pls, east, north + probe_m, dem) - here
    if abs(d_east) < 1e-9 and abs(d_north) < 1e-9:
        return None
    return math.degrees(math.atan2(-d_east, -d_north)) % 360.0


def _blend_headings(heading, target, weight):
    """Rotate `heading` toward `target` by `weight`, the short way round."""
    delta = ((target - heading + 180.0) % 360.0) - 180.0
    return (heading + weight * delta) % 360.0


def _slope_deg(pls, east, north, step_east, step_north, dem):
    rise = abs(_elevation(pls, east + step_east, north + step_north, dem)
               - _elevation(pls, east, north, dem))
    run = math.hypot(step_east, step_north)
    return math.degrees(math.atan2(rise, run)) if run > 0 else 0.0


def _to_sectors(pls, endpoints, model):
    """Bin walker endpoints onto a grid and rank the cells."""
    cells = {}
    for east, north in endpoints:
        key = (math.floor(east / model.cell_m), math.floor(north / model.cell_m))
        cells.setdefault(key, []).append((east, north))

    ranked = sorted(cells.values(), key=len, reverse=True)[:model.top_k]
    total = float(len(endpoints))

    sectors = []
    for rank, members in enumerate(ranked, 1):
        mean_east = sum(e for e, _ in members) / len(members)
        mean_north = sum(n for _, n in members) / len(members)
        spread = math.sqrt(
            sum((e - mean_east) ** 2 + (n - mean_north) ** 2 for e, n in members) / len(members)
        )
        latitude, longitude = offset_enu(pls[0], pls[1], mean_east, mean_north)
        sectors.append(Sector(
            rank=rank,
            latitude=latitude,
            longitude=longitude,
            radius_m=round(max(spread, model.cell_m / 2.0), 1),
            probability=round(len(members) / total, 4),
            samples=len(members),
            bearing_deg=round(math.degrees(math.atan2(mean_east, mean_north)) % 360.0, 1),
            distance_m=round(ground_distance_m(pls, (latitude, longitude)), 1),
        ))
    return sectors


def build_briefing(sectors, pls, elapsed_hours, complete=None, context=None):
    """Prose over the computed sectors. Returns (text, source).

    The model never chooses where to search — the sectors are already fixed
    before this runs, and they are handed over as facts. When the model is
    unreachable the sectors still stand and the fallback says exactly what it
    is, so nobody mistakes a generated summary for a written one.
    """
    if not sectors:
        return ("No sectors: no elapsed time to simulate over.", "computed-fallback")

    if complete is not None:
        try:
            reply = complete(_briefing_prompt(sectors, pls, elapsed_hours, context))
        except LLMUnavailable:
            reply = None
        if reply is not None:
            # An empty or runaway briefing is a failure dressed as an answer.
            result = parse_text(reply, TextReply)
            if result.ok:
                return (result.value.text, BRIEFING_MODEL_TAG)
        # fall through to the deterministic summary

    return (_fallback_briefing(sectors, elapsed_hours), "computed-fallback")


def _briefing_prompt(sectors, pls, elapsed_hours, context):
    lines = [
        "You are writing a short field briefing for a mountain search-and-rescue team.",
        f"The subject was last seen at {pls[0]:.5f}, {pls[1]:.5f}, about "
        f"{elapsed_hours:.1f} hours ago.",
        "",
        "A Monte-Carlo terrain simulation has ALREADY produced these search sectors.",
        "Do not invent, move, or re-rank them. Describe them and say how to work them.",
        "",
    ]
    for s in sectors:
        lines.append(
            f"  Sector {s.rank}: {s.probability:.0%} of simulated outcomes, "
            f"{s.distance_m:.0f} m from the last known point on a bearing of "
            f"{s.bearing_deg:.0f} degrees, radius about {s.radius_m:.0f} m "
            f"({s.latitude:.5f}, {s.longitude:.5f})."
        )
    if context:
        lines += ["", f"Conditions: {context}"]
    lines += [
        "",
        "Write at most 120 words. Plain language, no bullet points, no preamble.",
        "State the priority order and one practical note per sector.",
    ]
    return "\n".join(lines)


def _fallback_briefing(sectors, elapsed_hours):
    top = sectors[0]
    parts = [
        f"Automated summary (no language model available). After {elapsed_hours:.1f} h, "
        f"simulation puts {top.probability:.0%} of outcomes in sector 1, {top.distance_m:.0f} m "
        f"out on a bearing of {top.bearing_deg:.0f} degrees, radius {top.radius_m:.0f} m."
    ]
    if len(sectors) > 1:
        rest = ", ".join(
            f"sector {s.rank} ({s.probability:.0%}, {s.distance_m:.0f} m at {s.bearing_deg:.0f} deg)"
            for s in sectors[1:]
        )
        parts.append(f"Then {rest}.")
    return " ".join(parts)


class PathAgent:
    """Runs the simulation and publishes the sectors with a briefing."""

    def __init__(self, bus, case_id, dem=None, model=None, complete=None, stream=None, seed=0):
        self.bus = bus
        self.case_id = case_id
        self.dem = dem
        self.model = model or PathModel()
        self.complete = complete
        self.stream = stream
        self.seed = seed
        self.published = 0

    def project(self, pls, elapsed_hours, context=None, at=None):
        """Simulate, brief, publish. Returns the clue that went on the bus."""
        sectors = simulate_sectors(pls, elapsed_hours, self.dem, self.model, self.seed)
        briefing, briefing_source = build_briefing(
            sectors, pls, elapsed_hours, self.complete, context
        )
        clue = self.build_clue(pls, elapsed_hours, sectors, briefing, briefing_source, at)
        self.bus.publish(clue, stream=self.stream)
        self.published += 1
        return clue

    def build_clue(self, pls, elapsed_hours, sectors, briefing, briefing_source, at=None):
        observed_at = at or datetime.now(timezone.utc)
        top = sectors[0] if sectors else None
        return ClueContract(
            clue_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"sar:path:{self.case_id}:{pls[0]:.5f}:{pls[1]:.5f}:{elapsed_hours:.2f}",
            )),
            case_id=self.case_id,
            timestamp=observed_at,
            source_agent=AgentSource.PATH_MODEL,
            # How much of the simulated mass the best sector holds. A flat
            # spread over many sectors is a weak prediction and says so.
            confidence_score=round(min(1.0, top.probability * 3.0), 4) if top else 0.0,
            finding_summary=(
                f"{len(sectors)} search sector(s) from {self.model.walkers} simulated tracks over "
                f"{elapsed_hours:.1f} h; best holds {top.probability:.0%} at {top.distance_m:.0f} m "
                f"bearing {top.bearing_deg:.0f} deg"
                if top else "No sectors could be simulated"
            ),
            # The best sector's centre. Sectors are areas, so this is a handle,
            # not a fix — the full geometry is in agent_metadata.
            spatial_context=SpatialContext(
                latitude=top.latitude if top else None,
                longitude=top.longitude if top else None,
            ),
            provenance_tag=PATH_PROVENANCE,
            agent_metadata={
                "sectors": [asdict(s) for s in sectors],
                "point_last_seen": list(pls),
                "elapsed_hours": round(elapsed_hours, 3),
                "walkers": self.model.walkers,
                "briefing": briefing,
                "briefing_source": briefing_source,
                "terrain_aware": self.dem is not None,
                "path_model": asdict(self.model),
            },
        )
