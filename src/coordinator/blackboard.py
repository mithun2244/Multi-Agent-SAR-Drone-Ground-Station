"""The case blackboard — the shared state store behind the command plane.

Phase 0 calls for "a thin case blackboard as the state store, and a case
lifecycle: open a case, mint the case_id, and carry it through everything
downstream". This is that store, and it is deliberately dumb: it holds cases,
accumulated targets, and each case's read position on the bus. All the fusion
logic lives in `coordinator/fusion.py`, so state and policy stay separable.

Holding the bus cursor here is what makes fusion restartable: a coordinator that
dies mid-search resumes from the last clue it actually absorbed rather than
re-reading the stream from the beginning or silently skipping it.
"""

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

# Redis Streams' "everything from the start" cursor.
STREAM_START = "0-0"


@dataclass
class Case:
    case_id: str
    opened_at: datetime
    status: str = "OPEN"
    metadata: dict = field(default_factory=dict)

    @property
    def is_open(self):
        return self.status == "OPEN"


@dataclass(frozen=True)
class Environment:
    """What the conditions are like at one point in the search area.

    Case-level context, not a target: weather is something the subject is *in*,
    not something the drone found.
    """

    latitude: float
    longitude: float
    hypothermia_risk: bool = False
    survival_window_hours: int | None = None
    apparent_c: float | None = None
    observed_at: datetime | None = None
    clue_id: str | None = None
    # Which agent set the window: the Weather agent's band table, or the Health
    # agent's subject-specific refinement of it.
    window_source: str | None = None
    # What the window was before Health refined it. Kept so the critic can ask
    # what the ranking would have been without the refinement, rather than
    # re-deriving fusion's urgency maths and letting the two drift apart.
    baseline_window_hours: int | None = None


@dataclass(frozen=True)
class SearchSector:
    """A patch of ground the Path model expects the subject to be in.

    A prediction, not an observation: it says where to look, never that anyone
    is there.
    """

    rank: int
    latitude: float
    longitude: float
    radius_m: float
    probability: float


@dataclass
class Target:
    """One physical thing the picture believes is out there.

    Several clues — several *tracks*, even — can collapse into one of these.
    `track_ids` and `clue_ids` keep the lineage so an operator can ask what the
    belief is actually built on.
    """

    target_id: str
    case_id: str
    class_label: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None
    bounding_box: tuple | None = None   # latest pixel extent, for detection IoU
    confidence: float = 0.0
    observations: int = 0
    best_by_source: dict = field(default_factory=dict)  # source -> best confidence seen
    track_ids: set = field(default_factory=set)
    clue_ids: list = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    range_source: str | None = None
    # Annotations from the reasoning plane. These describe a target, they are
    # not extra evidence that it is real — see coordinator/fusion.py.
    scene_description: str | None = None
    scene_hazards: list = field(default_factory=list)
    subject_state: str | None = None
    # The environment around this target, as the scene VLM read it. Hazards are
    # what is visible; `immediate_risks` is what they mean for the subject, and
    # that is what the Risk stage scores.
    scene_environment: str | None = None
    immediate_risks: list = field(default_factory=list)
    access_difficulty: str | None = None
    # Set when a picture is built, never stored on the live target: all three
    # depend on the clock and on context that changes independently of the
    # sighting itself.
    urgency: float = 0.0
    priority: float = 0.0
    sector_probability: float = 0.0
    sector_rank: int | None = None
    # Urgency broken out by where it came from, so the critic can ablate one
    # agent's contribution without guessing which signal it fed.
    weather_urgency: float = 0.0
    hazard_urgency: float = 0.0
    baseline_urgency: float = 0.0   # what weather alone would have said

    @property
    def located(self):
        return self.latitude is not None and self.longitude is not None

    @property
    def position(self):
        return (self.latitude, self.longitude) if self.located else None

    def snapshot(self):
        """A detached copy, containers included.

        Targets on the blackboard keep being updated as clues arrive. A picture
        handed to the operator must not change under them while they read it, so
        it is built from snapshots rather than live references.
        """
        return replace(
            self,
            best_by_source=dict(self.best_by_source),
            track_ids=set(self.track_ids),
            clue_ids=list(self.clue_ids),
            scene_hazards=list(self.scene_hazards),
            immediate_risks=list(self.immediate_risks),
        )


class Blackboard:
    """Per-case state. In-memory; a durable store swaps in behind this API."""

    def __init__(self):
        self._cases = {}
        self._targets = {}
        self._cursors = {}
        self._counters = {}
        self._environment = {}
        self._sectors = {}
        self._insights = {}
        self._profile = {}

    # -- case lifecycle ----------------------------------------------------

    def open_case(self, case_id=None, opened_at=None, **metadata):
        """Open a case, minting a case_id when one is not supplied."""
        case_id = case_id or f"case-{uuid.uuid4().hex[:8]}"
        if case_id in self._cases:
            raise ValueError(f"case {case_id} is already open")
        case = Case(
            case_id=case_id,
            opened_at=opened_at or datetime.now(timezone.utc),
            metadata=metadata,
        )
        self._cases[case_id] = case
        self._targets[case_id] = {}
        self._cursors[case_id] = STREAM_START
        self._counters[case_id] = 0
        self._environment[case_id] = {}
        self._sectors[case_id] = []
        self._insights[case_id] = []
        self._profile[case_id] = {}
        return case

    def case(self, case_id):
        try:
            return self._cases[case_id]
        except KeyError:
            raise KeyError(f"no such case: {case_id}") from None

    def close_case(self, case_id):
        self.case(case_id).status = "CLOSED"

    def cases(self):
        return list(self._cases.values())

    # -- bus cursor --------------------------------------------------------

    def cursor(self, case_id):
        self.case(case_id)
        return self._cursors[case_id]

    def set_cursor(self, case_id, entry_id):
        self.case(case_id)
        self._cursors[case_id] = entry_id

    # -- targets -----------------------------------------------------------

    def targets(self, case_id):
        self.case(case_id)
        return list(self._targets[case_id].values())

    def target(self, case_id, target_id):
        self.case(case_id)
        return self._targets[case_id].get(target_id)

    def add_target(self, target):
        self.case(target.case_id)
        self._targets[target.case_id][target.target_id] = target
        return target

    def drop_target(self, case_id, target_id):
        self.case(case_id)
        self._targets[case_id].pop(target_id, None)

    # -- environment -------------------------------------------------------

    def put_environment(self, case_id, environment):
        """Record conditions at a point. A later reading replaces an earlier one
        at the same place — the search wants current weather, not a history."""
        self.case(case_id)
        key = (round(environment.latitude, 3), round(environment.longitude, 3))
        existing = self._environment[case_id].get(key)
        stale = (
            existing is not None
            and existing.observed_at is not None
            and environment.observed_at is not None
            and environment.observed_at < existing.observed_at
        )
        if stale:
            return existing  # a late-arriving old forecast must not overwrite a newer one
        self._environment[case_id][key] = environment
        return environment

    def environment(self, case_id):
        self.case(case_id)
        return list(self._environment[case_id].values())

    # -- predicted sectors -------------------------------------------------

    def put_sectors(self, case_id, sectors):
        """Replace the search sectors. A newer projection supersedes the old one
        wholesale — sectors are one coherent distribution, not a pile."""
        self.case(case_id)
        self._sectors[case_id] = list(sectors)
        return self._sectors[case_id]

    def sectors(self, case_id):
        self.case(case_id)
        return list(self._sectors[case_id])

    # -- advisory notes ----------------------------------------------------

    def put_insight(self, case_id, insight):
        """Record a historical insight. Advisory context for the operator; it
        never asserts that anyone is anywhere."""
        self.case(case_id)
        self._insights[case_id].append(insight)
        return insight

    def insights(self, case_id):
        self.case(case_id)
        return list(self._insights[case_id])

    def update_profile(self, case_id, fields):
        """Merge facts about the subject. Only non-empty values overwrite, so a
        later statement that is silent on clothing does not erase what an
        earlier one said."""
        self.case(case_id)
        profile = self._profile[case_id]
        for key, value in fields.items():
            if value not in (None, "", [], ()):
                profile[key] = value
        return dict(profile)

    def profile(self, case_id):
        self.case(case_id)
        return dict(self._profile[case_id])

    def next_target_id(self, case_id):
        self.case(case_id)
        self._counters[case_id] += 1
        return f"T{self._counters[case_id]}"

    def __repr__(self):
        return (f"Blackboard({len(self._cases)} case(s), "
                f"{sum(len(t) for t in self._targets.values())} target(s))")
