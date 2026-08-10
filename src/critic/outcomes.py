"""Ground truth — where the subject actually was.

Architecture, Phase 8: "The critic compares each claim against what actually
turned out to be true and logs the gap as a labelled error."

An outcome is what a search learned after the fact: where each subject was
found, when, and in what state. It arrives from two places — a resolved live
case, or the Phase 1 evaluation dataset, where the truth is known by
construction.

The asymmetry the architecture warns about is built into the types here:

    "A person who is never found is a miss that nobody logs, so it can only
     learn from confirmed finds and from footage that gets reviewed afterwards."

`Subject.found` records whether the outcome is actually known. A case with
unfound subjects yields a report that says so rather than scoring recall
against a denominator nobody can vouch for — treating "we never found them" as
"they were not there" would train the critic to reward giving up.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Subject:
    """One person a search was looking for, and where they turned out to be."""

    subject_id: str
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = None
    found: bool = True
    found_at: datetime | None = None
    # How much this subject mattered: a graded relevance for ranking metrics.
    # A casualty needing evacuation outranks a walker who turned up at a pub.
    priority: float = 1.0
    bounding_box: tuple | None = None   # pixel extent, when frame truth exists
    frame_id: str | None = None
    label: str = "person"
    notes: str = ""

    @property
    def located(self):
        return self.found and self.latitude is not None and self.longitude is not None

    @property
    def position(self):
        return (self.latitude, self.longitude) if self.located else None


@dataclass(frozen=True)
class CaseOutcome:
    """What a case turned out to be. The critic's yardstick."""

    case_id: str
    subjects: tuple = ()
    resolved_at: datetime | None = None
    notes: str = ""

    @property
    def found_subjects(self):
        """Only subjects whose position is actually known can score anything."""
        return tuple(s for s in self.subjects if s.located)

    @property
    def unresolved_subjects(self):
        return tuple(s for s in self.subjects if not s.found)

    @property
    def is_scorable(self):
        """A case with nothing confirmed cannot tell the critic anything.

        Scoring against zero known subjects would make every phantom look like
        a correct rejection and reward a system that finds nobody.
        """
        return bool(self.found_subjects)

    @classmethod
    def from_split(cls, case_id, split, geo_only=True):
        """Build an outcome from a Phase 1 evaluation split.

        The harness's ground truth is exactly this: annotated positions that are
        true by construction. Reusing it means the critic can be exercised
        against a labelled dataset before a single real case has resolved.
        """
        subjects = []
        for index, truth in enumerate(split.ground_truth):
            if geo_only and truth.geo is None:
                continue
            subjects.append(Subject(
                subject_id=f"{truth.frame_id}#{index}",
                latitude=truth.geo[0] if truth.geo else None,
                longitude=truth.geo[1] if truth.geo else None,
                bounding_box=tuple(truth.box) if truth.box else None,
                frame_id=truth.frame_id,
                label=truth.label,
            ))
        return cls(case_id=case_id, subjects=tuple(subjects),
                   notes=f"from evaluation split {split.name!r}")


@dataclass
class OutcomeLog:
    """Resolved cases, keyed by case id. The critic reads from here."""

    outcomes: dict = field(default_factory=dict)

    def record(self, outcome):
        self.outcomes[outcome.case_id] = outcome
        return outcome

    def get(self, case_id):
        return self.outcomes.get(case_id)

    def scorable(self):
        return [o for o in self.outcomes.values() if o.is_scorable]

    def __len__(self):
        return len(self.outcomes)
