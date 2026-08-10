"""The critic — score a finished picture against what actually happened.

Architecture, Phase 8: "The critic compares each claim against what actually
turned out to be true and logs the gap as a labelled error. It then retunes two
things: each agent's thresholds, and the source-trust table that fusion
consults."

This module does the scoring half. It produces the objective loss that Phase 9's
optimizer minimises, and per-agent scores saying which contributions earned
their place.

How an agent is scored
----------------------
By counterfactual, not by correlation. Each agent contributes a signal to the
ranking — urgency from weather and health, hazards from scene, a sector prior
from path — so the question "did this agent help?" has an exact answer: re-rank
the same targets with that signal removed and see whether the ranking got worse.

The signals are recomputed, never the picture. Fusion already worked out what
each target's urgency was and what it would have been on the unrefined window;
the critic re-weights those numbers rather than re-deriving them, so there is
one implementation of the urgency maths and it cannot drift.

Read-only, always
-----------------
Every input is a detached snapshot and nothing here writes back. An evaluation
pass over a live case must not perturb the search it is evaluating — the critic
runs while teams are on the hill, and a scoring run that nudged a ranking would
be changing the thing it claims to measure.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .metrics import (
    DEFAULT_MATCH_RADIUS_M,
    detection_iou,
    false_positive_penalty,
    geolocation_residuals,
    match_targets,
    ndcg,
    ranking_tau,
    relevance_vector,
)

# Which agents feed each ranking signal. Ablating a signal scores everything
# named against it; where two agents share one, they share the credit.
SIGNAL_AGENTS = {
    "urgency": ("weather", "health", "scene"),
    "weather_urgency": ("weather", "health"),
    "hazard_urgency": ("scene",),
    "health_refinement": ("health",),
    "sector_prior": ("path",),
}

# How the loss trades the four failures off. Missing somebody outweighs
# everything else; these are the knobs Phase 9 searches.
DEFAULT_LOSS_WEIGHTS = {
    "miss": 2.0,
    "false_positive": 1.0,
    "geolocation": 1.0,
    "ranking": 1.0,
}

# Residual at which the geolocation term is considered fully wrong.
GEO_SCALE_M = 100.0


@dataclass(frozen=True)
class AgentScore:
    """What removing one agent's signal would have done to the ranking."""

    signal: str
    agents: tuple
    ndcg_with: float | None
    ndcg_without: float | None
    delta: float | None
    verdict: str            # "helped" | "hurt" | "no effect" | "unscorable"
    active_targets: int = 0

    @property
    def helped(self):
        return self.verdict == "helped"


@dataclass
class CriticReport:
    """One evaluation pass. Everything Phase 9 needs to tune against."""

    case_id: str
    scorable: bool = True
    reason: str = ""
    generated_at: datetime | None = None

    subjects: int = 0
    targets: int = 0
    matched: int = 0
    missed: int = 0
    false_positives: int = 0
    unlocated_targets: int = 0

    recall: float | None = None
    detection_iou: float | None = None
    geolocation: dict = field(default_factory=dict)
    ndcg: float | None = None
    kendall_tau: float | None = None
    false_positive_penalty: float = 0.0

    loss: float | None = None
    loss_terms: dict = field(default_factory=dict)
    agent_scores: tuple = ()
    errors: tuple = ()          # labelled gaps, the architecture's "logged error"

    def render(self):
        lines = ["", f"  Critic report — {self.case_id}", "  " + "-" * 68]
        if not self.scorable:
            lines += [f"  not scorable: {self.reason}", ""]
            return "\n".join(lines)

        geo = self.geolocation or {}
        lines += [
            f"  subjects {self.subjects:<4} targets {self.targets:<4} "
            f"matched {self.matched:<4} missed {self.missed:<4} "
            f"false positives {self.false_positives}",
            "",
            f"  recall            {_fmt(self.recall)}",
            f"  detection IoU     {_fmt(self.detection_iou)}",
            f"  geo residual      median {_fmt(geo.get('median_m'), 'm')}  "
            f"p90 {_fmt(geo.get('p90_m'), 'm')}  max {_fmt(geo.get('max_m'), 'm')}",
            f"  NDCG              {_fmt(self.ndcg)}",
            f"  Kendall tau       {_fmt(self.kendall_tau)}",
            f"  FP penalty        {self.false_positive_penalty:.4f}",
            "",
            f"  LOSS              {_fmt(self.loss)}   "
            + "  ".join(f"{k} {v:.3f}" for k, v in sorted(self.loss_terms.items())),
        ]
        if self.agent_scores:
            lines += ["", "  agent contributions (NDCG with vs without)"]
            for score in self.agent_scores:
                lines.append(
                    f"    {score.signal:<20}{'/'.join(score.agents):<22}"
                    f"{_fmt(score.ndcg_with)} -> {_fmt(score.ndcg_without)}   "
                    f"{score.verdict}"
                )
        if self.errors:
            lines += ["", "  labelled errors"]
            for error in self.errors:
                lines.append(f"    {error}")
        lines.append("")
        return "\n".join(lines)


class Critic:
    """Scores pictures against outcomes. Holds no search state."""

    def __init__(self, match_radius_m=DEFAULT_MATCH_RADIUS_M, loss_weights=None,
                 geo_scale_m=GEO_SCALE_M, urgency_weight=1.0, sector_weight=0.5,
                 clock=None):
        self.match_radius_m = match_radius_m
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS if loss_weights is None else loss_weights)
        self.geo_scale_m = geo_scale_m
        # Must mirror the weights fusion ranked with, or the counterfactuals
        # would be answering a question about a different system.
        self.urgency_weight = urgency_weight
        self.sector_weight = sector_weight
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.history = []

    def evaluate(self, picture, outcome):
        """Score one picture. Returns a report; mutates nothing."""
        now = self.clock()
        if outcome is None or not outcome.is_scorable:
            report = CriticReport(
                case_id=getattr(picture, "case_id", "unknown"),
                scorable=False,
                generated_at=now,
                reason=("no outcome recorded" if outcome is None else
                        "no subject was confirmed found — a search that never located "
                        "anyone cannot say whether the picture was right"),
                targets=len(getattr(picture, "targets", ()) or ()),
            )
            self.history.append(report)
            return report

        targets = list(picture.targets)
        subjects = list(outcome.found_subjects)
        matches, unmatched_targets, missed = match_targets(
            targets, subjects, self.match_radius_m)

        relevances = relevance_vector(targets, matches)
        report = CriticReport(
            case_id=picture.case_id,
            generated_at=now,
            subjects=len(subjects),
            targets=len(targets),
            matched=len(matches),
            missed=len(missed),
            false_positives=sum(1 for _, t in unmatched_targets if getattr(t, "located", False)),
            unlocated_targets=sum(1 for _, t in unmatched_targets
                                  if not getattr(t, "located", False)),
            recall=round(len(matches) / len(subjects), 6) if subjects else None,
            detection_iou=detection_iou(matches),
            geolocation=geolocation_residuals(matches),
            ndcg=ndcg(relevances),
            kendall_tau=ranking_tau(relevances),
            false_positive_penalty=false_positive_penalty(targets, unmatched_targets),
        )
        report.loss, report.loss_terms = self._loss(report)
        report.agent_scores = self._score_agents(targets, subjects, report.ndcg)
        report.errors = self._label_errors(report, matches, missed, unmatched_targets)

        self.history.append(report)
        return report

    # -- loss --------------------------------------------------------------

    def _loss(self, report):
        """One scalar for Phase 9 to minimise. Lower is better."""
        geo = (report.geolocation or {}).get("median_m")
        terms = {
            "miss": 1.0 - (report.recall or 0.0),
            "false_positive": report.false_positive_penalty,
            "geolocation": 0.0 if geo is None else min(1.0, geo / self.geo_scale_m),
            "ranking": 0.0 if report.ndcg is None else 1.0 - report.ndcg,
        }
        weighted = {k: round(self.loss_weights.get(k, 1.0) * v, 6) for k, v in terms.items()}
        return round(sum(weighted.values()), 6), weighted

    # -- per-agent counterfactuals -----------------------------------------

    def _score_agents(self, targets, subjects, baseline_ndcg):
        scores = []
        for signal, agents in SIGNAL_AGENTS.items():
            scores.append(self._ablate(signal, agents, targets, subjects, baseline_ndcg))
        return tuple(scores)

    def _ablate(self, signal, agents, targets, subjects, baseline_ndcg):
        """Re-rank with one signal removed and compare."""
        active = sum(1 for t in targets if self._signal_value(t, signal) > 0.0)
        if baseline_ndcg is None or len(targets) < 2 or active == 0:
            return AgentScore(signal, agents, baseline_ndcg, None, None,
                              "unscorable", active)

        reordered = sorted(
            targets,
            key=lambda t: (-self._priority_without(t, signal), -t.confidence, t.target_id),
        )
        matches, _, _ = match_targets(reordered, subjects, self.match_radius_m)
        without = ndcg(relevance_vector(reordered, matches))
        if without is None:
            return AgentScore(signal, agents, baseline_ndcg, None, None, "unscorable", active)

        delta = round(baseline_ndcg - without, 6)
        verdict = "helped" if delta > 1e-9 else ("hurt" if delta < -1e-9 else "no effect")
        return AgentScore(signal, agents, baseline_ndcg, without, delta, verdict, active)

    @staticmethod
    def _signal_value(target, signal):
        if signal == "sector_prior":
            return getattr(target, "sector_probability", 0.0)
        if signal == "health_refinement":
            # Health only did something if the refined window moved urgency.
            return abs(getattr(target, "urgency", 0.0)
                       - getattr(target, "baseline_urgency", 0.0))
        return getattr(target, signal, 0.0)

    def _priority_without(self, target, signal):
        """The priority this target would have had without one signal.

        Uses the same expression fusion ranks with, so a counterfactual is
        comparable to the real thing rather than to a different formula.
        """
        urgency = target.urgency
        sector = target.sector_probability

        if signal == "urgency":
            urgency = 0.0
        elif signal == "weather_urgency":
            urgency = target.hazard_urgency
        elif signal == "hazard_urgency":
            urgency = target.weather_urgency
        elif signal == "health_refinement":
            urgency = target.baseline_urgency   # what weather alone would have said
        elif signal == "sector_prior":
            sector = 0.0

        return target.confidence * (1.0 + self.urgency_weight * urgency
                                    + self.sector_weight * sector)

    # -- labelled errors ---------------------------------------------------

    def _label_errors(self, report, matches, missed, unmatched_targets):
        """The architecture's "logs the gap as a labelled error"."""
        errors = []
        for subject in missed:
            errors.append(f"MISSED_SUBJECT {subject.subject_id}: never appeared in the picture")
        for rank, target in unmatched_targets:
            if getattr(target, "located", False):
                errors.append(f"FALSE_POSITIVE {target.target_id}: ranked {rank}, "
                              f"no subject within {self.match_radius_m:.0f} m")
        for rank, target, subject, distance in matches:
            if distance > self.match_radius_m / 2.0:
                errors.append(f"GEOLOCATION_DRIFT {target.target_id}: {distance:.0f} m from "
                              f"{subject.subject_id}")
        if report.ndcg is not None and report.ndcg < 0.8:
            # Say which kind of ranking failure it was. "Below a phantom" when
            # every target matched would send someone hunting a false positive
            # that does not exist.
            cause = ("a real subject was ranked below a phantom"
                     if report.false_positives
                     else "subjects were ranked out of priority order")
            errors.append(f"RANKING_ERROR: NDCG {report.ndcg:.3f}, {cause}")
        return tuple(errors)

    # -- aggregate ---------------------------------------------------------

    def summary(self):
        """Mean loss over every scorable pass, for tracking across a campaign."""
        scored = [r for r in self.history if r.loss is not None]
        if not scored:
            return {"passes": len(self.history), "scored": 0, "mean_loss": None}
        return {
            "passes": len(self.history),
            "scored": len(scored),
            "mean_loss": round(sum(r.loss for r in scored) / len(scored), 6),
            "mean_recall": round(sum(r.recall or 0.0 for r in scored) / len(scored), 6),
            "mean_ndcg": round(sum(r.ndcg or 0.0 for r in scored) / len(scored), 6),
        }

    def as_dict(self, report):
        """Report as plain data, for logging to a tuning run."""
        payload = asdict(report)
        payload["generated_at"] = (report.generated_at.isoformat()
                                   if report.generated_at else None)
        payload["agent_scores"] = [asdict(s) for s in report.agent_scores]
        payload["errors"] = list(report.errors)
        return payload


def _fmt(value, unit=""):
    return "n/a" if value is None else (f"{value:.3f}{unit}" if unit == "" else
                                        f"{value:.1f} {unit}")
