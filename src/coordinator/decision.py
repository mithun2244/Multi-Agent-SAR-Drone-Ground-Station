"""The decision chain — Reason -> Risk -> Recommend -> Orchestrate -> Commander.

Phase 3 (docs/architecture.md). The data-gathering plane puts clues on the bus
and `coordinator/fusion.py` compiles them into one picture. This module is what
turns that picture into something a commander can act on, in four stages that
each do one job and can be tested on their own:

    reason(picture)                 -> Facts            what is objectively true
    risk(facts)                     -> Risk             how dangerous, 1-10
    recommend(facts, risk)          -> Recommendation   which protocol action
    orchestrate(facts, risk, reco)  -> CommanderBrief   consistent, or corrected

Why four stages and not one call
--------------------------------
A single "decide" function would fold three different kinds of judgement — what
is true, how bad it is, and what to do about it — into one number nobody can
argue with. Split, each stage is inspectable: an operator can disagree with the
risk score without disagreeing about the facts, and the critic can score the
stages separately.

Nothing here writes. Reason reads a picture snapshot, and a snapshot is detached
(`fusion.picture()`), so the chain cannot move the ranking it is reading.

Why Orchestrate re-derives instead of trusting
----------------------------------------------
`risk` and `recommend` are deterministic functions of the facts, so in the happy
path Orchestrate confirms and passes through. It earns its place when they are
*not* what the facts support: a tuned threshold, a hand-built brief, or a future
model-written recommendation. Orchestrate recomputes both from the facts and
corrects any drift, then re-checks its own correction — that is the feedback
loop. A chain that will not converge inside `MAX_PASSES` is not published as an
order; it goes to the commander marked for review, because a decision the system
cannot make consistent is exactly the one a human should see.
"""

from dataclasses import dataclass

from ..utils.ablation import DECISION, enabled

# Danger is scored on an explicit 1-10 scale. 1 is the floor, not zero: a case
# is only open because somebody is missing, and that is never no danger at all.
RISK_MIN, RISK_MAX = 1, 10
RISK_BASE = 1

# Survival-window bands, in hours.
WINDOW_CRITICAL_H = 6
WINDOW_TIGHT_H = 12

# Hazards are capped: several named hazards are a reason to hurry, never on
# their own the whole score.
HAZARD_CAP = 2

# What the environment around the subject costs, matched against the risks the
# scene VLM listed. Each category scores once however many phrases name it —
# "drowning risk" and "river below" are one river, not two.
#
# Matched against `immediate_risks` and nothing else. Scanning the environment
# prose for these words instead would score "no water nearby" as water; a false
# +2 on every description that mentions a river is worse than a missed one,
# because it is silent. The prompt asks for this list explicitly.
ENVIRONMENT_RISKS = (
    (("drown", "water", "river", "lake", "flood"), 2, "water at the scene"),
    (("fall", "cliff", "ledge", "slope", "scree"), 2, "fall exposure at the scene"),
)
OTHER_RISK_POINTS = 1        # entrapment, exposure, anything else it saw
DIFFICULT_ACCESS_POINTS = 1

# Urgency above this means the window is closing on the top target now.
URGENCY_PRESSING = 0.8

MAX_PASSES = 3

# Actions, from the field protocol.
IMMEDIATE_EXTRACTION = "IMMEDIATE_EXTRACTION"
DISPATCH_GROUND_TEAM = "DISPATCH_GROUND_TEAM"
MONITOR_AND_CONFIRM = "MONITOR_AND_CONFIRM"
RETASK_DRONE_FOR_FIX = "RETASK_DRONE_FOR_FIX"
EXPAND_SEARCH = "EXPAND_SEARCH"
CONTINUE_SEARCH = "CONTINUE_SEARCH"
HOLD_FOR_REVIEW = "HOLD_FOR_COMMANDER_REVIEW"


# -- 1. Reason -------------------------------------------------------------

@dataclass(frozen=True)
class Facts:
    """What is objectively true about the case right now.

    Compiled, not judged: every field here is either counted or copied from the
    clues fusion accumulated. Nothing in this dataclass says whether any of it
    is good or bad — that is the next stage's job.
    """

    case_id: str
    targets: int = 0
    located: int = 0
    unlocated: int = 0
    best_confidence: float = 0.0
    top_target_id: str | None = None
    top_position: tuple | None = None
    top_urgency: float = 0.0
    hazards: tuple = ()
    # What the scene VLM read off the ground around the targets: the risks those
    # hazards pose, and how hard the top target is to reach.
    immediate_risks: tuple = ()
    access_difficulty: str | None = None
    hypothermia_risk: bool = False
    survival_window_hours: int | None = None
    sectors: int = 0
    best_sector_probability: float = 0.0
    clues: int = 0
    guard_events: int = 0     # clues refused, commands flagged

    def summary(self):
        where = (f"{self.top_position[0]:.5f}, {self.top_position[1]:.5f}"
                 if self.top_position else "no fix")
        line = (f"{self.targets} target(s) from {self.clues} clue(s), "
                f"{self.located} located / {self.unlocated} not; "
                f"best confidence {self.best_confidence:.2f} at {where}")
        # What the ground looks like, when anything looked at it. Stated here,
        # scored in the next stage: the facts do not say whether a river is bad.
        ground = list(self.hazards) + [r for r in self.immediate_risks
                                       if r not in self.hazards]
        if ground:
            line += f"; scene: {', '.join(ground)}"
        if self.access_difficulty:
            line += f"; access {self.access_difficulty}"
        return line


def reason(picture):
    """Stage 1: compile the accumulated clues into objective facts."""
    targets = picture.targets
    top = targets[0] if targets else None
    hazards, risks = [], []
    for target in targets:
        for hazard in target.scene_hazards:
            if hazard not in hazards:
                hazards.append(hazard)
        for risk_seen in getattr(target, "immediate_risks", ()):
            if risk_seen not in risks:
                risks.append(risk_seen)

    return Facts(
        case_id=picture.case_id,
        targets=len(targets),
        located=len(picture.located),
        unlocated=len(picture.unlocated),
        best_confidence=max((t.confidence for t in targets), default=0.0),
        top_target_id=top.target_id if top else None,
        top_position=top.position if top else None,
        top_urgency=top.urgency if top else 0.0,
        hazards=tuple(hazards),
        immediate_risks=tuple(risks),
        # Hazards and risks are unioned across targets — any of them is a reason
        # to hurry. Access difficulty is not: it describes the walk in to one
        # place, and the place a team is being sent is the top target.
        access_difficulty=getattr(top, "access_difficulty", None) if top else None,
        hypothermia_risk=picture.hypothermia_risk,
        survival_window_hours=picture.survival_window_hours,
        sectors=len(picture.sectors),
        best_sector_probability=max((s.probability for s in picture.sectors), default=0.0),
        clues=picture.clues_ingested,
        guard_events=len(picture.security_events),
    )


# -- 2. Risk ---------------------------------------------------------------

@dataclass(frozen=True)
class Risk:
    """Situational danger, 1-10, with the points itemised.

    Itemised because a bare number is unarguable. An operator who can see that
    3 of the 8 came from "hypothermia risk" can tell the system it is wrong
    about the weather; one who sees only "8" cannot.
    """

    score: int
    drivers: tuple = ()

    @property
    def band(self):
        if self.score >= 9:
            return "CRITICAL"
        if self.score >= 7:
            return "HIGH"
        if self.score >= 4:
            return "ELEVATED"
        return "LOW"

    def explain(self):
        return "; ".join(f"{why} (+{points})" for why, points in self.drivers) or "baseline only"


def risk(facts):
    """Stage 2: score situational danger on the 1-10 scale."""
    drivers = []
    if facts.hypothermia_risk:
        drivers.append(("hypothermia risk in the search area", 3))

    window = facts.survival_window_hours
    if window is not None and window <= WINDOW_CRITICAL_H:
        drivers.append((f"survival window down to {window} h", 2))
    elif window is not None and window <= WINDOW_TIGHT_H:
        drivers.append((f"survival window {window} h", 1))

    if facts.hazards:
        drivers.append((f"hazards at the scene: {', '.join(facts.hazards)}",
                        min(HAZARD_CAP, len(facts.hazards))))

    # Each environmental risk is itemised with the phrases that raised it, so an
    # operator who thinks the model misread the ground can argue with that line
    # rather than with the total.
    named = set()
    for keywords, points, why in ENVIRONMENT_RISKS:
        hit = [r for r in facts.immediate_risks
               if any(word in r.lower() for word in keywords)]
        if hit:
            drivers.append((f"{why}: {', '.join(hit)}", points))
            named.update(hit)

    other = [r for r in facts.immediate_risks if r not in named]
    if other:
        drivers.append((f"other risks at the scene: {', '.join(other)}",
                        OTHER_RISK_POINTS))

    if facts.access_difficulty and "difficult" in facts.access_difficulty.lower():
        drivers.append((f"access to the top target is difficult "
                        f"({facts.access_difficulty})", DIFFICULT_ACCESS_POINTS))

    if facts.unlocated:
        # Seen and unreachable is worse than not yet seen: somebody is out there
        # and no team can be sent to them.
        drivers.append((f"{facts.unlocated} target(s) seen but not located", 1))

    if facts.top_urgency >= URGENCY_PRESSING:
        drivers.append(("the window is closing on the top target", 1))

    score = min(RISK_MAX, max(RISK_MIN, RISK_BASE + sum(points for _, points in drivers)))
    return Risk(score=score, drivers=tuple(drivers))


# -- 3. Recommend ----------------------------------------------------------

@dataclass(frozen=True)
class Recommendation:
    """One operational action, and the protocol rule that selected it."""

    action: str
    detail: str
    rule: str
    needs_fix: bool = False   # the action is meaningless without a position


@dataclass(frozen=True)
class ProtocolRule:
    rule: str
    action: str
    detail: str
    when: object              # (facts, risk) -> bool
    needs_fix: bool = False


# First match wins, so the order is the protocol. Anything needing a position is
# above anything that does not, and "found someone" outranks "still looking".
PROTOCOL = (
    ProtocolRule(
        "located-and-critical", IMMEDIATE_EXTRACTION,
        "Located subject in critical conditions — extract now, brief the team in the air.",
        lambda f, r: f.located and r.score >= 8, needs_fix=True,
    ),
    ProtocolRule(
        "located-and-pressing", DISPATCH_GROUND_TEAM,
        "Located subject with time pressure — send the nearest ground team to the fix.",
        lambda f, r: f.located and r.score >= 4, needs_fix=True,
    ),
    ProtocolRule(
        "located-and-quiet", MONITOR_AND_CONFIRM,
        "Located subject, conditions stable — confirm identity on the next pass before committing a team.",
        lambda f, r: f.located, needs_fix=True,
    ),
    ProtocolRule(
        "seen-but-no-fix", RETASK_DRONE_FOR_FIX,
        "Subject seen but not geolocated — retask the drone over the sighting for a fix.",
        lambda f, r: f.unlocated > 0,
    ),
    ProtocolRule(
        "nothing-found-and-dangerous", EXPAND_SEARCH,
        "Nothing found and conditions are closing in — widen to the next-ranked sectors.",
        lambda f, r: r.score >= 6,
    ),
    ProtocolRule(
        "nothing-found", CONTINUE_SEARCH,
        "Nothing found yet — hold the current search pattern.",
        lambda f, r: True,
    ),
)


def recommend(facts, assessment, protocol=PROTOCOL):
    """Stage 3: select the operational action the protocol calls for."""
    for rule in protocol:
        if rule.when(facts, assessment):
            return Recommendation(action=rule.action, detail=rule.detail,
                                  rule=rule.rule, needs_fix=rule.needs_fix)
    # Unreachable with PROTOCOL, whose last rule is unconditional. A custom
    # protocol that matches nothing must not silently produce "do nothing".
    return Recommendation(HOLD_FOR_REVIEW, "No protocol rule matched these facts.",
                          rule="no-rule-matched")


# -- 4. Orchestrate --------------------------------------------------------

@dataclass(frozen=True)
class CommanderBrief:
    """What goes to the commander: the facts, the score, the action, the audit."""

    case_id: str
    facts: Facts
    risk: Risk
    recommendation: Recommendation
    corrections: tuple = ()
    consistent: bool = True
    passes: int = 1

    def summary(self):
        """One line, for a caller that does not care which shape it got."""
        return (f"risk {self.risk.score}/10 {self.risk.band}, "
                f"{self.recommendation.action}")

    def render(self):
        lines = [
            "",
            f"  Commander brief — {self.case_id}",
            "  " + "-" * 84,
            f"  reason:     {self.facts.summary()}",
            f"  risk:       {self.risk.score}/10 {self.risk.band}  [{self.risk.explain()}]",
            f"  recommend:  {self.recommendation.action}  (protocol: {self.recommendation.rule})",
            f"              {self.recommendation.detail}",
        ]
        if self.facts.guard_events:
            lines.append(f"  security:   {self.facts.guard_events} event(s) at the guards — "
                         f"see the picture for what was refused")
        if self.corrections:
            lines.append(f"  orchestrate: {len(self.corrections)} correction(s) "
                         f"over {self.passes} pass(es)")
            for note in self.corrections:
                lines.append(f"      [corrected] {note}")
        else:
            lines.append("  orchestrate: consistent on the first pass")
        if not self.consistent:
            lines.append("  *** NOT PUBLISHED AS AN ORDER — the chain would not converge, "
                         "commander review required ***")
        lines.append("")
        return "\n".join(lines)


def _inconsistencies(facts, assessment, recommendation, protocol=PROTOCOL):
    """Check the three stages against each other. Returns corrections applied."""
    notes = []

    supported = risk(facts)
    if assessment.score != supported.score:
        notes.append(f"risk {assessment.score}/10 is not what the facts support "
                     f"({supported.score}/10): {supported.explain()}")
        assessment = supported

    expected = recommend(facts, assessment, protocol)
    if recommendation.action != expected.action:
        notes.append(f"{recommendation.action} does not follow from risk "
                     f"{assessment.score}/10: corrected to {expected.action}")
        recommendation = expected

    if recommendation.needs_fix and not facts.located:
        # Never send a team to a place nobody computed. This is the one check
        # that is not a re-derivation: it is the invariant from CLAUDE.md, that
        # no order may depend on a position that was never actually fixed.
        notes.append(f"{recommendation.action} needs a position and none was fixed: "
                     f"corrected to {RETASK_DRONE_FOR_FIX}")
        recommendation = Recommendation(
            RETASK_DRONE_FOR_FIX,
            "Subject seen but not geolocated — retask the drone over the sighting for a fix.",
            rule="orchestrate:no-fix",
        )

    return assessment, recommendation, notes


def orchestrate(facts, assessment, recommendation, max_passes=MAX_PASSES, protocol=PROTOCOL):
    """Stage 4: validate the three against each other, correcting until they
    agree, then hand the commander a brief."""
    corrections, passes = [], 0
    for passes in range(1, max_passes + 1):
        assessment, recommendation, notes = _inconsistencies(
            facts, assessment, recommendation, protocol)
        if not notes:
            return CommanderBrief(facts.case_id, facts, assessment, recommendation,
                                  tuple(corrections), consistent=True, passes=passes)
        corrections.extend(notes)

    # Still disagreeing after the loop. Publishing the last guess as an order
    # would be publishing something the system itself could not make consistent.
    return CommanderBrief(
        facts.case_id, facts, assessment,
        Recommendation(HOLD_FOR_REVIEW,
                       f"The decision chain did not converge in {max_passes} passes. "
                       f"Last action considered: {recommendation.action}.",
                       rule="orchestrate:no-convergence"),
        tuple(corrections), consistent=False, passes=passes,
    )


@dataclass(frozen=True)
class FlatReport:
    """What the commander gets with the chain switched off: the picture, as is.

    The ablation for the whole decision plane. No facts compiled, no risk
    scored, no action chosen, no consistency checked — fusion's ranked targets
    go straight through, which is what a system that stopped at Phase 2 would
    have handed an operator.

    It deliberately has no `risk` or `recommendation`: a caller reaching for
    them should fail loudly rather than read a zero and believe it. `summary()`
    is the one thing both shapes answer.
    """

    case_id: str
    targets: tuple = ()
    survival_window_hours: int | None = None
    hypothermia_risk: bool = False
    guard_events: int = 0

    def summary(self):
        return f"{len(self.targets)} target(s), no risk score (decision chain ablated)"

    def render(self):
        lines = [
            "",
            f"  Flat report — {self.case_id}   [ABLATED: no decision chain]",
            "  " + "-" * 84,
            f"  {len(self.targets)} target(s), best first, straight off fusion:",
        ]
        for rank, target in enumerate(self.targets, 1):
            where = (f"{target.latitude:.6f}, {target.longitude:.6f}"
                     if target.located else "NOT LOCATED")
            lines.append(f"    {rank:<4}{target.target_id:<6}conf {target.confidence:.3f}  "
                         f"prio {target.priority:.3f}  {where}")
        if self.survival_window_hours is not None:
            lines.append(f"  survival window {self.survival_window_hours} h"
                         + ("  HYPOTHERMIA RISK" if self.hypothermia_risk else ""))
        if self.guard_events:
            lines.append(f"  {self.guard_events} event(s) at the guards")
        lines.append("  no risk score, no recommended action — read the ranking and decide.")
        lines.append("")
        return "\n".join(lines)


def flatten(picture):
    """The picture as a commander-facing report, with nothing judged."""
    return FlatReport(
        case_id=picture.case_id,
        targets=tuple(picture.targets),
        survival_window_hours=picture.survival_window_hours,
        hypothermia_risk=picture.hypothermia_risk,
        guard_events=len(picture.security_events),
    )


def decide(picture, commander=None, protocol=PROTOCOL, chain=None):
    """The whole chain: picture in, commander brief out.

    `commander` is whoever receives the brief — the operator console, a log, a
    bus publisher. Left out, the brief is simply returned; the chain has no
    opinion about who reads it.

    `chain=False` (or `ABLATION_DECISION=off`) hands back a `FlatReport`
    instead: fusion's ranking with nothing reasoned over it.
    """
    if not enabled(DECISION, chain):
        report = flatten(picture)
        if commander is not None:
            commander(report)
        return report

    facts = reason(picture)
    assessment = risk(facts)
    brief = orchestrate(facts, assessment, recommend(facts, assessment, protocol),
                        protocol=protocol)
    if commander is not None:
        commander(brief)
    return brief
