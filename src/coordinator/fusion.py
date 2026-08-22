"""Coordinator fusion — accumulate clues into one ranked operational picture.

Phase 3 (docs/architecture.md): "Fusion reads clues off the blackboard, removes
duplicate tracks, and keeps a current picture. There is nothing to corroborate
yet with a single source. The point of the phase is to get the accumulation and
confidence-weighting logic in place so it is ready once the second source
arrives."

Not to be confused with `perception/fusion.py`: that one merges *boxes* from two
sensors on one frame. This one merges *clues* from any producer across the whole
case, and it is the thing an operator actually looks at.

Why repeated sightings do not raise confidence
----------------------------------------------
Evidence from one source is correlated with itself. Ten frames of the same
camera seeing the same person is one opinion observed ten times, not ten
independent confirmations, and combining them as if independent is the classic
way to manufacture false certainty — in a search, that means an operator
trusting a phantom. So confidence is the *best* observation per source,
combined across *distinct* sources with a trust-weighted noisy-OR.

With one source that reduces to `trust x best confidence`, which is exactly
right for Phase 3. When Weather arrives in Phase 4 the same expression starts
doing real corroboration work with no change here.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..bus import stream_for
from ..contracts.clue import AgentSource
from ..guardrails.audit import PROVENANCE_REJECTED, AuditLog
from ..guardrails.contradiction import PerceptionFacts, check
from ..guardrails.provenance import ProvenanceRegistry
from ..perception.geolocation import ground_distance_m
from .blackboard import Environment, SearchSector, Target

# Sources that describe a *thing found*. These become targets.
DETECTION_SOURCES = frozenset({
    AgentSource.PERCEPTION_FUSION,
    AgentSource.DRONE_RGB,
    AgentSource.DRONE_LIDAR,
})

# Sources that describe the *situation*. These become case context, not targets.
# Health refines the same window Weather computed, so it lands in the same place.
CONTEXT_SOURCES = frozenset({AgentSource.WEATHER_API, AgentSource.HEALTH_LLM})

# Sources that inform the operator without asserting anything about a target.
# History summarises past cases; Interview reports what a witness said. Neither
# may create a target or move a confidence — witness text in particular can
# never trigger a state change, which is enforced structurally here.
ADVISORY_SOURCES = frozenset({AgentSource.HISTORY_RAG, AgentSource.INTERVIEW_LLM})

# Sources that predict *where to look*. Also case context: a prediction says
# where to search, never that anyone is there.
PREDICTION_SOURCES = frozenset({AgentSource.PATH_MODEL})

# Sources that describe a target we already believe in. These annotate; they
# never add confidence. Scene only runs *because* detection fired, so its
# agreement is guaranteed and worth nothing as corroboration — folding it into
# the noisy-OR would be counting one opinion twice.
ANNOTATION_SOURCES = frozenset({AgentSource.SCENE_VLM})

# How much each source's word is worth. Phase 8's critic retunes this table.
DEFAULT_TRUST = {
    AgentSource.PERCEPTION_FUSION: 1.0,
    AgentSource.DRONE_RGB: 0.9,
    AgentSource.DRONE_LIDAR: 0.9,
}

# Forecasts are coarse; a reading applies to a wide area around its point.
WEATHER_RADIUS_M = 5_000.0

# Urgency saturates as the remaining window closes inside this horizon.
URGENCY_HORIZON_H = 24.0

# How much urgency each visible hazard adds, and the ceiling it can reach.
#
# The ceiling sits in the lower half of the scale, deliberately clear of what a
# closing survival window reaches (0.9+). A hazard is something a model *saw* in
# a picture; a window is computed from a measured temperature and a time. Letting
# two hazards in a description rival a real one — which is what a 0.9 ceiling did
# — put a sheltered subject ahead of a hypothermic one, and the critic caught it
# as a ranking error. Same principle as `select_range`: an inference does not get
# to outrank a measurement.
HAZARD_URGENCY_STEP = 0.3
HAZARD_URGENCY_CAP = 0.45

_MAX_LINEAGE = 32


@dataclass
class Picture:
    """The current operational picture: targets, best first."""

    case_id: str
    targets: list = field(default_factory=list)
    clues_ingested: int = 0
    clues_ignored: int = 0
    generated_at: datetime | None = None
    environment: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    guard_findings: list = field(default_factory=list)
    security_events: list = field(default_factory=list)

    @property
    def hypothermia_risk(self):
        return any(e.hypothermia_risk for e in self.environment)

    @property
    def survival_window_hours(self):
        """The tightest window anywhere in the search area."""
        windows = [e.survival_window_hours for e in self.environment
                   if e.survival_window_hours is not None]
        return min(windows) if windows else None

    @property
    def located(self):
        return [t for t in self.targets if t.located]

    @property
    def unlocated(self):
        return [t for t in self.targets if not t.located]

    def render(self):
        lines = [
            "",
            f"  Operational picture — {self.case_id}",
            "  " + "-" * 84,
            f"  {len(self.targets)} target(s) from {self.clues_ingested} clue(s)"
            + (f", {self.clues_ignored} ignored" if self.clues_ignored else ""),
        ]
        if self.environment:
            window = self.survival_window_hours
            sources = {e.window_source for e in self.environment if e.window_source}
            lines.append(
                f"  conditions: {'HYPOTHERMIA RISK' if self.hypothermia_risk else 'no cold risk'}"
                + (f", survival window {window} h" if window is not None else "")
                + (f" (via {', '.join(sorted(sources))})" if sources else "")
                + f"  {len(self.environment)} reading(s)"
            )
        if self.profile:
            known = {k: v for k, v in self.profile.items()
                     if k not in ("untrusted", "injection_suspected") and v}
            if known:
                lines.append("  subject:    " + "; ".join(f"{k.replace('_', ' ')} {v}"
                                                          for k, v in sorted(known.items()))
                             + ("  [witness statement, untrusted]" if self.profile.get("untrusted")
                                else ""))
        if self.sectors:
            best = max(self.sectors, key=lambda s: s.probability)
            lines.append(
                f"  sectors:    {len(self.sectors)} predicted, best holds "
                f"{best.probability:.0%} ({best.latitude:.5f}, {best.longitude:.5f}, "
                f"r {best.radius_m:.0f} m)"
            )
        lines.append("")

        if not self.targets:
            # No early return: a blocked injection with nothing found is exactly
            # the case an operator most needs to see, and it belongs below.
            lines.append("  (nothing detected yet)")
        else:
            lines.append(
                f"  {'#':<4}{'prio':<8}{'conf':<8}{'urg':<7}{'sect':<7}{'obs':<6}"
                f"{'position':<28}{'label':<10}tracks"
            )
            for rank, t in enumerate(self.targets, 1):
                where = f"{t.latitude:.6f}, {t.longitude:.6f}" if t.located else "NOT LOCATED"
                sector = f"S{t.sector_rank}" if t.sector_rank else "-"
                lines.append(
                    f"  {rank:<4}{t.priority:<8.3f}{t.confidence:<8.3f}{t.urgency:<7.2f}"
                    f"{sector:<7}{t.observations:<6}{where:<28}"
                    f"{(t.class_label or '-'):<10}{','.join(sorted(t.track_ids)) or '-'}"
                )

        described = [t for t in self.targets if t.scene_description]
        if described:
            lines.append("")
            for t in described:
                lines.append(f"  {t.target_id}: {t.scene_description}")
                if t.scene_hazards:
                    lines.append(f"      hazards: {', '.join(t.scene_hazards)}")
                if t.immediate_risks:
                    lines.append(f"      risks: {', '.join(t.immediate_risks)}")
                if t.access_difficulty:
                    lines.append(f"      access: {t.access_difficulty}")

        if self.insights:
            lines.append("")
            for note in self.insights:
                cited = ", ".join(r["case_id"] for r in note.get("retrieved", []))
                lines.append(f"  history: {note['insight']}")
                if cited:
                    lines.append(f"      cases: {cited}")

        if self.guard_findings:
            lines.append("")
            lines.append(f"  GUARDRAIL: {len(self.guard_findings)} contradiction(s) with "
                         f"confirmed perception")
            for finding in self.guard_findings:
                mark = "OVERRIDDEN" if finding.severe else "flagged"
                lines.append(f"      [{mark}] {finding.rule}: {finding.detail}")

        if self.security_events:
            lines.append("")
            lines.append(f"  SECURITY: {len(self.security_events)} event(s) at the guards")
            for event in self.security_events[-5:]:
                mark = "REJECTED" if event.kind == PROVENANCE_REJECTED else "FLAGGED"
                lines.append(f"      [{mark}] {event.describe()}")
        lines.append("")
        return "\n".join(lines)


class CoordinatorFusion:
    """Reads clues off the bus, accumulates them, keeps one picture per case."""

    def __init__(
        self,
        bus,
        blackboard,
        trust=None,
        merge_distance_m=25.0,
        detection_sources=DETECTION_SOURCES,
        context_sources=CONTEXT_SOURCES,
        prediction_sources=PREDICTION_SOURCES,
        annotation_sources=ANNOTATION_SOURCES,
        advisory_sources=ADVISORY_SOURCES,
        weather_radius_m=WEATHER_RADIUS_M,
        urgency_horizon_h=URGENCY_HORIZON_H,
        urgency_weight=1.0,
        sector_weight=0.5,
        guard_contradictions=True,
        provenance=None,
        audit=None,
        min_report_confidence=0.0,
        params=None,
        clock=None,
    ):
        self.bus = bus
        self.blackboard = blackboard
        self.trust = dict(DEFAULT_TRUST if trust is None else trust)
        self.merge_distance_m = merge_distance_m
        self.detection_sources = frozenset(detection_sources)
        self.context_sources = frozenset(context_sources)
        self.prediction_sources = frozenset(prediction_sources)
        self.annotation_sources = frozenset(annotation_sources)
        self.advisory_sources = frozenset(advisory_sources)
        self.weather_radius_m = weather_radius_m
        self.urgency_horizon_h = urgency_horizon_h
        self.urgency_weight = urgency_weight
        self.sector_weight = sector_weight
        self.min_report_confidence = min_report_confidence
        # A tuned configuration overrides the hand-picked defaults above. Passed
        # in rather than read from disk here, so a test or a tuning trial can
        # hand over a configuration without touching the filesystem.
        if params is not None:
            self.trust = dict(params.trust_table)
            self.urgency_weight = params.urgency_weight
            self.sector_weight = params.sector_weight
            self.min_report_confidence = params.min_report_confidence
            self.merge_distance_m = params.merge_distance_m
        self.params = params
        self.guard_contradictions = guard_contradictions
        # On by default. Tag-and-agent binding needs no configuration, so there
        # is no version of this system that should run without it.
        self.provenance = provenance if provenance is not None else ProvenanceRegistry()
        self.audit = audit if audit is not None else AuditLog()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.ignored = {}
        self.rejected = {}

    def ingest(self, case_id, count=None):
        """Absorb every clue published since the last call. Returns how many."""
        case = self.blackboard.case(case_id)
        entries = self.bus.read(stream_for(case_id), last_id=self.blackboard.cursor(case_id),
                                count=count)
        absorbed = 0
        for entry_id, clue in entries:
            self.blackboard.set_cursor(case_id, entry_id)
            if clue.case_id != case.case_id:
                # The stream is per case, so this means a producer wrote to the
                # wrong one. Never quietly fold it into another search.
                raise ValueError(f"clue {clue.clue_id} is for {clue.case_id}, not {case_id}")

            # The trust boundary. Everything past this line is treated as
            # something the system said about itself, so nothing unverified
            # gets past it — and a rejection is evidence, not litter.
            verdict = self.provenance.verify(clue)
            if not verdict:
                self.audit.reject_clue(clue, verdict, case_id=case_id)
                self.rejected[verdict.reason] = self.rejected.get(verdict.reason, 0) + 1
                continue
            if clue.source_agent in self.detection_sources:
                self._absorb(case_id, clue)
            elif clue.source_agent in self.context_sources:
                self._absorb_context(case_id, clue)
            elif clue.source_agent in self.prediction_sources:
                self._absorb_prediction(case_id, clue)
            elif clue.source_agent in self.annotation_sources:
                self._absorb_annotation(case_id, clue)
            elif clue.source_agent in self.advisory_sources:
                self._absorb_advisory(case_id, clue)
            else:
                key = clue.source_agent.value
                self.ignored[key] = self.ignored.get(key, 0) + 1
                continue
            absorbed += 1
        return absorbed

    def picture(self, case_id, now=None):
        """Rank the accumulated targets, most urgent-and-believed first.

        Ranking is by *priority*, which is confidence scaled by urgency —
        deliberately two numbers rather than one. A cold night does not make a
        detection more likely to be real, so weather must never touch
        confidence; it changes which believed target you go to first. Folding
        urgency into confidence would be the same error as letting repeat
        sightings inflate belief.

        Location is not a ranking input either: a target that was seen but could
        not be geolocated must not sink below a weaker located one, or the
        picture would quietly bury the hardest sightings.
        """
        now = now or self.clock()
        environment = self.blackboard.environment(case_id)
        sectors = self.blackboard.sectors(case_id)
        case = self.blackboard.case(case_id)
        best_probability = max((s.probability for s in sectors), default=0.0)

        targets = []
        for live in self.blackboard.targets(case_id):
            if live.confidence < self.min_report_confidence:
                # Below the operating point. It stays on the blackboard — the
                # evidence is not discarded — but it does not reach an operator,
                # which is how the false-alarm rate is actually held down.
                continue
            target = live.snapshot()
            (target.urgency, target.weather_urgency,
             target.hazard_urgency, target.baseline_urgency) = self._urgency(
                target, environment, case, now)
            prior, rank = self._sector_prior(target, sectors, best_probability)
            target.sector_probability, target.sector_rank = prior, rank
            target.priority = round(
                target.confidence
                * (1.0 + self.urgency_weight * target.urgency + self.sector_weight * prior),
                6,
            )
            targets.append(target)

        targets.sort(key=lambda t: (-t.priority, -t.confidence, -t.observations, t.target_id))

        # Guard advisory prose against what the sensors hold. Run here, not at
        # absorb time, for two reasons: the facts are complete only once every
        # clue is in, and the blackboard keeps the model's raw words for audit
        # while the operator sees the guarded version.
        insights = self.blackboard.insights(case_id)
        findings = []
        if self.guard_contradictions:
            facts = PerceptionFacts.from_targets(targets)
            insights, notes = self._guard_insights(insights, facts)
            findings.extend(notes)
            findings.extend(self._guard_descriptions(targets, facts))

        return Picture(
            case_id=case_id,
            targets=targets,
            clues_ingested=sum(t.observations for t in targets),
            clues_ignored=sum(self.ignored.values()),
            generated_at=max((t.last_seen for t in targets), default=None),
            environment=environment,
            sectors=sectors,
            insights=insights,
            profile=self.blackboard.profile(case_id),
            guard_findings=findings,
            security_events=self.audit.for_case(case_id),
        )

    def _guard_insights(self, insights, facts):
        guarded, findings = [], []
        for note in insights:
            result = check(note.get("insight", ""), facts)
            entry = dict(note)
            if result.findings:
                entry["insight"] = result.text
                entry["guard_overridden"] = result.overridden
                if result.overridden:
                    entry["withheld"] = result.original
                findings.extend(result.findings)
            guarded.append(entry)
        return guarded, findings

    def _guard_descriptions(self, targets, facts):
        findings = []
        for target in targets:
            if not target.scene_description:
                continue
            result = check(target.scene_description, facts)
            if result.findings:
                target.scene_description = result.text
                findings.extend(result.findings)
        return findings

    def refresh(self, case_id):
        """Ingest whatever is new, then hand back the current picture."""
        self.ingest(case_id)
        return self.picture(case_id)

    # -- accumulation ------------------------------------------------------

    def _absorb(self, case_id, clue):
        target = self._match(case_id, clue) or self.blackboard.add_target(
            Target(target_id=self.blackboard.next_target_id(case_id), case_id=case_id)
        )
        self._update(target, clue)

    def _match(self, case_id, clue):
        """Find the target this clue belongs to, if any.

        Two ways a clue is a repeat of something already known:

        1. same track id — the tracker already decided it is one target;
        2. close enough on the ground to a target of the same class — a track
           that was lost and re-acquired comes back with a *new* track id, and
           without this the picture would show one person twice.
        """
        track_id = _track_id(clue)
        candidates = self.blackboard.targets(case_id)

        if track_id is not None:
            for target in candidates:
                if track_id in target.track_ids:
                    return target

        position = _position(clue)
        if position is None:
            return None

        nearest, best = None, self.merge_distance_m
        for target in candidates:
            if not target.located or target.class_label != clue.class_label:
                continue
            distance = ground_distance_m(position, target.position)
            if distance <= best:
                nearest, best = target, distance
        return nearest

    def _update(self, target, clue):
        source = clue.source_agent.value
        target.best_by_source[source] = max(
            target.best_by_source.get(source, 0.0), clue.confidence_score
        )
        target.observations += 1
        target.class_label = target.class_label or clue.class_label
        target.clue_ids = (target.clue_ids + [clue.clue_id])[-_MAX_LINEAGE:]

        track_id = _track_id(clue)
        if track_id is not None:
            target.track_ids.add(track_id)

        position = _position(clue)
        if position is not None:
            # Latest located observation wins: the target may actually be
            # moving, and an average across a walk is a place nobody is.
            target.latitude, target.longitude = position
            target.elevation_m = clue.spatial_context.altitude_m
            target.range_source = clue.agent_metadata.get("range_source")
        if clue.spatial_context is not None and clue.spatial_context.bounding_box:
            target.bounding_box = tuple(clue.spatial_context.bounding_box)

        if target.first_seen is None or clue.timestamp < target.first_seen:
            target.first_seen = clue.timestamp
        if target.last_seen is None or clue.timestamp > target.last_seen:
            target.last_seen = clue.timestamp

        target.confidence = self._combine(target)

    def _absorb_context(self, case_id, clue):
        """Conditions are not a target — they are the situation a target is in.

        Weather sets the baseline window; Health republishes the same sector
        with a subject-specific one. Both land here, and the later clue wins, so
        a refinement supersedes the band table exactly as intended.
        """
        position = _position(clue)
        if position is None:
            # Conditions with no location cannot be tied to a sector.
            key = f"{clue.source_agent.value}:no_position"
            self.ignored[key] = self.ignored.get(key, 0) + 1
            return
        metadata = clue.agent_metadata
        self.blackboard.put_environment(case_id, Environment(
            latitude=position[0],
            longitude=position[1],
            hypothermia_risk=bool(metadata.get("hypothermia_risk", False)),
            survival_window_hours=metadata.get("survival_window_hours"),
            apparent_c=metadata.get("apparent_c"),
            observed_at=clue.timestamp,
            clue_id=clue.clue_id,
            window_source=metadata.get("window_source") or clue.source_agent.value,
            baseline_window_hours=metadata.get("baseline_window_hours"),
        ))

    def _absorb_advisory(self, case_id, clue):
        """Notes for the operator. Never a target, never a confidence."""
        metadata = clue.agent_metadata
        if clue.source_agent is AgentSource.HISTORY_RAG:
            self.blackboard.put_insight(case_id, {
                "insight": metadata.get("insight") or clue.finding_summary,
                "source": metadata.get("insight_source"),
                "retrieved": metadata.get("retrieved") or [],
                "clue_id": clue.clue_id,
            })
            return

        # Interview. Structurally incapable of placing anyone: the fields below
        # are the only thing taken, and none of them is a position.
        self.blackboard.update_profile(case_id, {
            "time_last_seen": metadata.get("time_last_seen"),
            "clothing": metadata.get("clothing"),
            "direction_of_travel": metadata.get("direction_of_travel"),
            "witness": metadata.get("witness"),
            "untrusted": True,
            "injection_suspected": bool(metadata.get("injection_suspected")),
        })

    def _absorb_prediction(self, case_id, clue):
        """Path sectors: where to look, not who is there."""
        sectors = []
        for raw in clue.agent_metadata.get("sectors") or ():
            try:
                sectors.append(SearchSector(
                    rank=int(raw["rank"]),
                    latitude=float(raw["latitude"]),
                    longitude=float(raw["longitude"]),
                    radius_m=float(raw["radius_m"]),
                    probability=float(raw["probability"]),
                ))
            except (KeyError, TypeError, ValueError):
                self.ignored["PATH_MODEL:malformed_sector"] = self.ignored.get(
                    "PATH_MODEL:malformed_sector", 0) + 1
        self.blackboard.put_sectors(case_id, sectors)

    def _absorb_annotation(self, case_id, clue):
        """Scene description: attach it to the target it is about.

        Never creates a target. A description with nothing to attach to means
        the detection it came from was merged away or never absorbed, and
        inventing a target from prose would be inventing a sighting.
        """
        target = self._match(case_id, clue)
        if target is None:
            self.ignored["SCENE_VLM:no_target"] = self.ignored.get("SCENE_VLM:no_target", 0) + 1
            return
        metadata = clue.agent_metadata
        target.scene_description = metadata.get("description") or clue.finding_summary
        target.scene_hazards = [str(h) for h in (metadata.get("hazards") or [])]
        target.subject_state = metadata.get("subject_state") or metadata.get("person_state")
        target.scene_environment = metadata.get("environment")
        target.immediate_risks = [str(r) for r in (metadata.get("immediate_risks") or [])]
        target.access_difficulty = metadata.get("access_difficulty")
        target.clue_ids = (target.clue_ids + [clue.clue_id])[-_MAX_LINEAGE:]
        # Deliberately no confidence update: see ANNOTATION_SOURCES. Nor does
        # the environmental read touch urgency — only `scene_hazards` does, as
        # before. Risks and access difficulty are scored once, by the Risk
        # stage; feeding them into urgency as well would count one description
        # twice and move a ranking Phase 9 tuned against the old signal.

    # -- urgency -----------------------------------------------------------

    def _urgency(self, target, environment, case, now):
        """How fast this target needs reaching, in 0..1.

        Two independent reasons to hurry: the weather is closing the survival
        window, or the scene shows something dangerous. They are combined with
        `max`, not a sum — both are coarse heuristics, and adding them together
        would compound two rough numbers into false precision.

        Returns the parts as well as the whole. The critic needs to ask what the
        ranking would have been without one agent, and re-deriving this maths
        outside fusion would let the two drift apart.
        """
        conditions = self._conditions_for(target, environment)
        weather = self._window_urgency(conditions, case, now,
                                       getattr(conditions, "survival_window_hours", None))
        baseline = self._window_urgency(
            conditions, case, now,
            getattr(conditions, "baseline_window_hours", None)
            or getattr(conditions, "survival_window_hours", None),
        )
        hazard = self._hazard_urgency(target)
        return max(weather, hazard), weather, hazard, max(baseline, hazard)

    def _window_urgency(self, conditions, case, now, window):
        if conditions is None or not conditions.hypothermia_risk or window is None:
            return 0.0
        remaining = float(window) - _hours_missing(case, now)
        if remaining <= 0.0:
            return 1.0  # the window has already closed
        return round(max(0.0, min(1.0, 1.0 - remaining / self.urgency_horizon_h)), 6)

    def _hazard_urgency(self, target):
        """What the scene description saw. Capped below 1.0: a hazard is a
        reason to hurry, never on its own a reason to drop everything."""
        if not target.scene_hazards:
            return 0.0
        step = self.params.hazard_urgency_step if self.params else HAZARD_URGENCY_STEP
        return round(min(HAZARD_URGENCY_CAP, step * len(target.scene_hazards)), 6)

    def _sector_prior(self, target, sectors, best_probability):
        """How well this target sits inside the Path model's prediction, 0..1.

        Normalised against the strongest sector, so the best sector scores 1.0
        and a target outside every sector scores 0. This is a *prior on where to
        look*, which is why it moves priority and never confidence: the
        simulation has not seen anything, it has only reasoned about terrain.
        """
        if not target.located or not sectors or best_probability <= 0.0:
            return 0.0, None
        for sector in sorted(sectors, key=lambda s: -s.probability):
            distance = ground_distance_m(target.position, (sector.latitude, sector.longitude))
            if distance <= sector.radius_m:
                return round(sector.probability / best_probability, 6), sector.rank
        return 0.0, None

    def _conditions_for(self, target, environment):
        """Conditions where this target is.

        An unlocated target still gets the case's conditions — being unplaceable
        is not a reason to treat someone as unhurried. It takes the most severe
        reading, since we cannot tell which sector they are in.
        """
        if not environment:
            return None
        if not target.located:
            return min(
                environment,
                key=lambda e: (not e.hypothermia_risk,
                               e.survival_window_hours if e.survival_window_hours is not None else 1e9),
            )

        nearest, best = None, self.weather_radius_m
        for entry in environment:
            distance = ground_distance_m(target.position, (entry.latitude, entry.longitude))
            if distance <= best:
                nearest, best = entry, distance
        return nearest

    def _combine(self, target):
        """Trust-weighted noisy-OR across distinct sources."""
        disbelief = 1.0
        for source, best in target.best_by_source.items():
            weight = self.trust.get(_source_enum(source), 1.0)
            disbelief *= 1.0 - min(1.0, max(0.0, best * weight))
        return round(1.0 - disbelief, 6)


def _hours_missing(case, now):
    """Hours since the subject was reported missing, 0 if the case never said."""
    missing_since = case.metadata.get("missing_since")
    if not isinstance(missing_since, datetime):
        return 0.0
    return max(0.0, (now - missing_since).total_seconds() / 3600.0)


def _track_id(clue):
    raw = clue.agent_metadata.get("track_id")
    return None if raw is None else str(raw)


def _position(clue):
    spatial = clue.spatial_context
    if spatial is None or spatial.latitude is None or spatial.longitude is None:
        return None
    return (spatial.latitude, spatial.longitude)


def _source_enum(value):
    try:
        return AgentSource(value)
    except ValueError:
        return value
