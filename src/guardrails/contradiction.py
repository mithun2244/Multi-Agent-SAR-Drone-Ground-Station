"""The contradiction guard — models do not get to deny what a sensor measured.

A language model summarising a search can write "no target was found" while the
perception plane holds two confirmed, geolocated tracks. That sentence in front
of an operator is worse than no summary at all: it is a fluent, confident denial
of the one thing the system actually knows.

So advisory prose is checked against perception facts before it is shown, and
the rules are ordered by how dangerous being wrong is:

  * **Denial** of a confirmed detection is overridden. The text is withheld and
    replaced with the facts; the original is kept for audit, never shown.
  * **Count** and **elevation** mismatches are flagged, not overridden. They are
    usually the model being loose rather than wrong, and suppressing a whole
    briefing over a rounded number would cost more than it saves.

The guard is deliberately narrow. It compares prose against numbers we hold; it
does not judge whether advice is good. A guard that tried to would be one more
unreliable opinion.
"""

import re
from dataclasses import dataclass, field

# Perception says someone is there; the prose says nobody is.
_DENIAL = re.compile(
    r"\b(?:"
    r"no\s+(?:target|person|persons|people|subject|subjects|casualt\w+|survivor\w*|sign|trace)s?\b"
    r"|nothing\s+(?:was\s+)?(?:found|detected|located|seen|spotted)"
    r"|(?:area|sector|search)\s+(?:is\s+|was\s+)?(?:clear|empty)"
    r"|no\s+one\s+(?:was\s+)?(?:found|detected|located|seen)"
    r"|(?:found|detected|located)\s+(?:no|none|nobody)\b"
    r"|unable\s+to\s+(?:find|locate|detect)\s+(?:any|anyone)"
    r")", re.I,
)

# "two targets", "3 people". Nouns kept narrow so "3 comparable cases" and
# "two hikers" from an archive summary do not read as claims about now.
_COUNT = re.compile(
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:confirmed\s+|located\s+)?(?:targets?|persons?|people|subjects?|casualt\w+|survivors?)\b",
    re.I,
)

# Only an explicit altitude claim, never a bare distance like "1400 m away".
_ELEVATION = re.compile(
    r"(?:(?:altitude|elevation)\s+(?:of\s+)?(\d{2,5})\s*(?:m|metres|meters)\b"
    r"|(\d{2,5})\s*(?:m|metres|meters)\s+(?:altitude|elevation|asl|above sea level))",
    re.I,
)

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

DENIAL = "denial_of_confirmed_detection"
COUNT_MISMATCH = "target_count_mismatch"
ELEVATION_MISMATCH = "elevation_mismatch"

ELEVATION_TOLERANCE_M = 150.0


@dataclass(frozen=True)
class PerceptionFacts:
    """What the sensors actually hold. The ground truth prose is checked against."""

    confirmed_targets: int = 0
    located_targets: int = 0
    elevations_m: tuple = ()
    labels: tuple = ()

    @classmethod
    def from_picture(cls, picture):
        targets = getattr(picture, "targets", ()) or ()
        return cls.from_targets(targets)

    @classmethod
    def from_targets(cls, targets):
        targets = list(targets)
        return cls(
            confirmed_targets=len(targets),
            located_targets=sum(1 for t in targets if getattr(t, "located", False)),
            elevations_m=tuple(t.elevation_m for t in targets
                               if getattr(t, "elevation_m", None) is not None),
            labels=tuple(sorted({t.class_label for t in targets
                                 if getattr(t, "class_label", None)})),
        )

    @property
    def has_targets(self):
        return self.confirmed_targets > 0

    def summarise(self):
        if not self.has_targets:
            return "Perception holds no confirmed targets."
        what = "/".join(self.labels) if self.labels else "target"
        where = f", {self.located_targets} geolocated" if self.located_targets else ""
        elevation = ""
        if self.elevations_m:
            elevation = f" at {min(self.elevations_m):.0f}-{max(self.elevations_m):.0f} m"
        return (f"Perception holds {self.confirmed_targets} confirmed {what} "
                f"track(s){where}{elevation}.")


@dataclass(frozen=True)
class Finding:
    rule: str
    detail: str
    severe: bool = False


@dataclass
class GuardResult:
    """The text as it may be shown, plus what was wrong with it."""

    text: str
    findings: list = field(default_factory=list)
    overridden: bool = False
    original: str | None = None

    @property
    def ok(self):
        return not self.findings

    @property
    def severe(self):
        return any(f.severe for f in self.findings)


def check(text, facts, override=True, elevation_tolerance_m=ELEVATION_TOLERANCE_M):
    """Check advisory prose against perception facts."""
    text = text or ""
    findings = []

    denial = _DENIAL.search(text)
    if denial and facts.has_targets:
        findings.append(Finding(
            DENIAL,
            f"summary says {denial.group(0).strip()!r} but {facts.summarise().lower()}",
            severe=True,
        ))

    for match in _COUNT.finditer(text):
        claimed = _to_int(match.group(1))
        if claimed is not None and claimed != facts.confirmed_targets:
            findings.append(Finding(
                COUNT_MISMATCH,
                f"summary claims {claimed} target(s), perception holds "
                f"{facts.confirmed_targets}",
            ))

    if facts.elevations_m:
        for match in _ELEVATION.finditer(text):
            claimed = float(match.group(1) or match.group(2))
            if all(abs(claimed - known) > elevation_tolerance_m for known in facts.elevations_m):
                findings.append(Finding(
                    ELEVATION_MISMATCH,
                    f"summary places a target at {claimed:.0f} m; perception has "
                    f"{', '.join(f'{e:.0f}' for e in facts.elevations_m)} m",
                ))

    severe = any(f.severe for f in findings)
    if severe and override:
        return GuardResult(
            text=(f"[withheld: contradicted confirmed perception] {facts.summarise()}"),
            findings=findings,
            overridden=True,
            original=text,
        )
    return GuardResult(text=text, findings=findings)


def _to_int(token):
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)
