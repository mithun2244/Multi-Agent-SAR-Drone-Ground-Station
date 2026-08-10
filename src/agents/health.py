"""The Health agent — refine the survival window for *this* subject.

Architecture, Phase 5: "Health — LLM; needs weather and subject profile already
flowing."

The Weather agent's window comes from a coarse band table over apparent
temperature. It knows nothing about who is out there. A fit adult in a shell
jacket and a seventy-year-old in a cotton shirt do not have the same number of
hours, and this agent is where that difference enters the picture.

Guardrails
----------
This is a life-safety number, so the model does not get to set it. It returns a
*multiplier* on the computed baseline, clamped to [0.25, 2.0], and anything
unparseable or out of range falls back to 1.0 — the weather baseline, unchanged.
The worst an unreliable model can do is scale a number by four in either
direction, and it can never invent one.

That asymmetry is deliberate: an LLM inventing "36 hours" for a hypothermic
subject would read as authoritative and could stand a search down. A multiplier
on an already-defensible number cannot.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.parsers import parse_reply
from ..guardrails.provenance import TAG_HEALTH
from ..guardrails.schemas import HealthReply
from .llm import DEFAULT_LLM_MODEL, LLMUnavailable

HEALTH_PROVENANCE = TAG_HEALTH
HEALTH_MODEL_TAG = DEFAULT_LLM_MODEL

MULTIPLIER_FLOOR = 0.25
MULTIPLIER_CEILING = 2.0
MIN_WINDOW_HOURS = 1  # a window of zero hours would be a statement nobody can act on


@dataclass(frozen=True)
class SubjectProfile:
    """What is known about the person, usually from the Interview agent."""

    age_years: int | None = None
    clothing: str | None = None
    injured: bool | None = None
    fitness: str | None = None            # "poor" | "average" | "good"
    medical_conditions: tuple = ()
    notes: str | None = None

    @property
    def is_empty(self):
        return not any([
            self.age_years, self.clothing, self.injured is not None,
            self.fitness, self.medical_conditions, self.notes,
        ])

    def describe(self):
        parts = []
        if self.age_years is not None:
            parts.append(f"age {self.age_years}")
        if self.fitness:
            parts.append(f"{self.fitness} fitness")
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        if self.injured is not None:
            parts.append("reported injured" if self.injured else "no reported injury")
        if self.medical_conditions:
            parts.append("conditions: " + ", ".join(self.medical_conditions))
        if self.notes:
            parts.append(self.notes)
        return "; ".join(parts) or "no subject details known"


@dataclass
class Refinement:
    hours: int
    baseline_hours: int
    multiplier: float
    rationale: str
    source: str
    clamped: bool = False


def refine_window(baseline_hours, profile, conditions, complete=None,
                  floor=MULTIPLIER_FLOOR, ceiling=MULTIPLIER_CEILING):
    """Scale a computed survival window to the subject. Never replaces it.

    `floor` and `ceiling` are the guardrail, and they are tunable — but they are
    a guardrail either way: whatever the search picks, the model returns a
    multiplier inside them and never a window of its own.
    """
    profile = profile or SubjectProfile()
    if baseline_hours is None:
        return None

    if complete is None or profile.is_empty:
        # Nothing to reason about, or nothing to reason with. The baseline is
        # already defensible; leave it alone rather than inventing precision.
        return Refinement(
            hours=int(baseline_hours),
            baseline_hours=int(baseline_hours),
            multiplier=1.0,
            rationale=("no subject details available" if profile.is_empty
                       else "no language model available"),
            source="computed-fallback",
        )

    try:
        reply = complete(_prompt(baseline_hours, profile, conditions))
    except LLMUnavailable:
        return Refinement(int(baseline_hours), int(baseline_hours), 1.0,
                          "language model unavailable", "computed-fallback")

    result = parse_reply(reply, HealthReply)
    if result.discarded:
        return Refinement(int(baseline_hours), int(baseline_hours), 1.0,
                          f"model gave no usable multiplier: {reply[:120]}", "computed-fallback")

    multiplier = result.value.multiplier
    clamped = not (floor <= multiplier <= ceiling)
    multiplier = min(ceiling, max(floor, multiplier))
    hours = max(MIN_WINDOW_HOURS, int(round(baseline_hours * multiplier)))
    return Refinement(
        hours=hours,
        baseline_hours=int(baseline_hours),
        multiplier=round(multiplier, 3),
        rationale=result.value.rationale or "no rationale given",
        source=HEALTH_MODEL_TAG,
        clamped=clamped,
    )


def _prompt(baseline_hours, profile, conditions):
    return f"""You are advising a mountain rescue medical officer.

A survival window of {baseline_hours} hours has ALREADY been computed from the
weather alone, using apparent temperature and wetness. Your job is only to scale
it for this specific subject. Do not produce a window of your own.

Subject: {profile.describe()}
Conditions: {conditions or "see computed window"}

Reply with JSON only:
{{"multiplier": <number between {MULTIPLIER_FLOOR} and {MULTIPLIER_CEILING}>,
  "rationale": "one sentence"}}

A multiplier below 1 means this subject has less time than an average adult;
above 1 means more. Use 1.0 if the details do not clearly justify a change."""


class HealthAgent:
    """Consumes weather clues and republishes a subject-specific window."""

    def __init__(self, bus, case_id, complete=None, profile=None, stream=None,
                 multiplier_floor=MULTIPLIER_FLOOR, multiplier_ceiling=MULTIPLIER_CEILING):
        self.bus = bus
        self.case_id = case_id
        self.complete = complete
        self.profile = profile or SubjectProfile()
        self.stream = stream
        self.multiplier_floor = multiplier_floor
        self.multiplier_ceiling = multiplier_ceiling
        self.published = 0
        self.skipped = 0

    def assess(self, weather_clue, at=None):
        """Refine one weather clue's window. Returns the clue, or None.

        Only weather clues that actually carry a window are refinable; anything
        else is counted and skipped rather than guessed at.
        """
        if weather_clue.source_agent is not AgentSource.WEATHER_API:
            self.skipped += 1
            return None
        baseline = weather_clue.agent_metadata.get("survival_window_hours")
        if baseline is None:
            self.skipped += 1
            return None

        refinement = refine_window(
            baseline, self.profile, weather_clue.finding_summary, self.complete,
            floor=self.multiplier_floor, ceiling=self.multiplier_ceiling,
        )
        clue = self.build_clue(weather_clue, refinement, at)
        self.bus.publish(clue, stream=self.stream)
        self.published += 1
        return clue

    def assess_all(self, clues, at=None):
        """Refine every weather clue in a batch. Returns what was published."""
        return [c for c in (self.assess(clue, at) for clue in clues) if c is not None]

    def build_clue(self, weather_clue, refinement, at=None):
        spatial = weather_clue.spatial_context
        risk = bool(weather_clue.agent_metadata.get("hypothermia_risk", False))
        observed_at = at or datetime.now(timezone.utc)
        return ClueContract(
            clue_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"sar:health:{self.case_id}:{weather_clue.clue_id}"
            )),
            case_id=self.case_id,
            parent_clue_ids=[weather_clue.clue_id],
            timestamp=observed_at,
            source_agent=AgentSource.HEALTH_LLM,
            # How much this refinement is worth: a scaled baseline is worth more
            # than an unscaled one only when a real profile drove the scaling.
            confidence_score=0.8 if refinement.source == HEALTH_MODEL_TAG else 0.6,
            finding_summary=(
                f"Survival window {refinement.hours} h for this subject "
                f"({refinement.baseline_hours} h baseline x {refinement.multiplier}): "
                f"{refinement.rationale}"
            ),
            spatial_context=SpatialContext(
                latitude=spatial.latitude if spatial else None,
                longitude=spatial.longitude if spatial else None,
            ),
            provenance_tag=HEALTH_PROVENANCE,
            agent_metadata={
                "hypothermia_risk": risk,
                "survival_window_hours": refinement.hours,
                "baseline_window_hours": refinement.baseline_hours,
                "multiplier": refinement.multiplier,
                "multiplier_clamped": refinement.clamped,
                "rationale": refinement.rationale,
                "window_source": refinement.source,
                "subject_profile": self.profile.describe(),
            },
        )
