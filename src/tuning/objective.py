"""The Optuna study — search the configuration space against the critic's loss.

Architecture, Phase 9: "use Optuna, a Bayesian TPE search, to set the detection
parameters ... against a labelled validation set, with fitness defined as recall
minus lambda times the false-alarm rate."

The objective is `CriticReport.loss` — the same number Phase 8 produces, so the
thing being optimised is the thing the critic actually measures, not a proxy
invented for the optimiser.

Two things are layered on top of that loss, and both are constraints rather than
rewards:

  * **Folds.** Every configuration is scored on several seeds and averaged. A
    configuration that wins on one draw of the noise has not learned anything,
    and tuning to a single fold is how a search ends up worse in the field than
    the defaults it replaced.
  * **A false-alarm ceiling.** Loss alone will happily accept a flood of
    phantoms if it finds one more subject. The operating point exists to hold
    the false-alarm rate at what an operator can actually work, so exceeding
    `target_far` is penalised rather than traded away.
"""

import logging
from dataclasses import dataclass

import optuna

from ..critic.critic import Critic
from .params import TunedParams
from .scenario import run

# Optuna narrates every trial at INFO. A study of a hundred trials would bury
# whatever the caller is actually printing.
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)

DEFAULT_FOLDS = (0, 1, 2)
FAR_PENALTY = 4.0   # weight on exceeding the operator's false-alarm tolerance


@dataclass
class Fitness:
    """What one configuration scored, and why."""

    loss: float
    critic_loss: float
    far: float
    far_penalty: float
    recall: float
    ndcg: float
    targets: float

    def as_dict(self):
        return {k: round(v, 6) for k, v in self.__dict__.items()}


def suggest(trial, base=None):
    """The search space. Bounds are wide enough to be interesting and narrow
    enough to stay physically sensible."""
    base = base or TunedParams()
    floor = trial.suggest_float("health_multiplier_floor", 0.1, 0.6)
    return base.replace(
        # perception
        wbf_iou_threshold=trial.suggest_float("wbf_iou_threshold", 0.30, 0.80),
        wbf_score_threshold=trial.suggest_float("wbf_score_threshold", 0.0, 0.40),
        rgb_weight=trial.suggest_float("rgb_weight", 0.5, 2.0),
        lidar_weight=trial.suggest_float("lidar_weight", 0.5, 2.0),
        track_high_thresh=trial.suggest_float("track_high_thresh", 0.30, 0.70),
        track_new_thresh=trial.suggest_float("track_new_thresh", 0.40, 0.90),
        track_min_hits=trial.suggest_int("track_min_hits", 2, 5),
        # source trust
        trust_perception_fusion=trial.suggest_float("trust_perception_fusion", 0.50, 1.0),
        trust_drone_rgb=trial.suggest_float("trust_drone_rgb", 0.50, 1.0),
        trust_drone_lidar=trial.suggest_float("trust_drone_lidar", 0.50, 1.0),
        # ranking
        urgency_weight=trial.suggest_float("urgency_weight", 0.0, 2.0),
        sector_weight=trial.suggest_float("sector_weight", 0.0, 1.5),
        hazard_urgency_step=trial.suggest_float("hazard_urgency_step", 0.0, 0.6),
        # health guardrail — the ceiling is bounded above the floor, so the
        # clamp can never invert into a range that admits nothing.
        health_multiplier_floor=floor,
        health_multiplier_ceiling=trial.suggest_float(
            "health_multiplier_ceiling", max(1.05, floor + 0.5), 3.0),
        # operating point
        min_report_confidence=trial.suggest_float("min_report_confidence", 0.0, 0.80),
    )


def score(params, folds=DEFAULT_FOLDS, frames=6, critic=None):
    """Average fitness of one configuration over several folds."""
    critic = critic or Critic(urgency_weight=params.urgency_weight,
                              sector_weight=params.sector_weight)
    losses, fars, recalls, ndcgs, counts = [], [], [], [], []

    for seed in folds:
        result = run(params, seed=seed, frames=frames)
        report = critic.evaluate(result.picture, result.outcome)
        if not report.scorable:
            # Nothing confirmed means nothing to learn from; treat it as the
            # worst case rather than skipping it, or a configuration that finds
            # nobody would score as well as one that finds everyone.
            losses.append(4.0)
            fars.append(0.0)
            recalls.append(0.0)
            ndcgs.append(0.0)
            counts.append(0)
            continue

        losses.append(report.loss)
        fars.append(report.false_positives / max(1, result.frames))
        recalls.append(report.recall or 0.0)
        ndcgs.append(report.ndcg or 0.0)
        counts.append(report.targets)

    critic_loss = sum(losses) / len(losses)
    far = sum(fars) / len(fars)
    penalty = FAR_PENALTY * max(0.0, far - params.target_far)
    return Fitness(
        loss=round(critic_loss + penalty, 6),
        critic_loss=round(critic_loss, 6),
        far=round(far, 6),
        far_penalty=round(penalty, 6),
        recall=round(sum(recalls) / len(recalls), 6),
        ndcg=round(sum(ndcgs) / len(ndcgs), 6),
        targets=round(sum(counts) / len(counts), 3),
    )


def objective(trial, folds=DEFAULT_FOLDS, frames=6, base=None):
    """One Optuna trial: build a configuration, fly it, score it."""
    params = suggest(trial, base)
    fitness = score(params, folds=folds, frames=frames)
    for name, value in fitness.as_dict().items():
        trial.set_user_attr(name, value)
    return fitness.loss


def run_study(n_trials=80, folds=DEFAULT_FOLDS, frames=6, seed=0, base=None,
              show_progress=False):
    """Run the TPE search. Returns (study, best params, best fitness).

    Trial 0 is the shipped defaults, enqueued explicitly. Without it a study can
    report an improvement it never actually demonstrated against the baseline.
    """
    base = base or TunedParams()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="sar-operating-point",
    )
    study.enqueue_trial({
        "wbf_iou_threshold": base.wbf_iou_threshold,
        "wbf_score_threshold": base.wbf_score_threshold,
        "rgb_weight": base.rgb_weight,
        "lidar_weight": base.lidar_weight,
        "track_high_thresh": base.track_high_thresh,
        "track_new_thresh": base.track_new_thresh,
        "track_min_hits": base.track_min_hits,
        "trust_perception_fusion": base.trust_perception_fusion,
        "trust_drone_rgb": base.trust_drone_rgb,
        "trust_drone_lidar": base.trust_drone_lidar,
        "urgency_weight": base.urgency_weight,
        "sector_weight": base.sector_weight,
        "hazard_urgency_step": base.hazard_urgency_step,
        "health_multiplier_floor": base.health_multiplier_floor,
        "health_multiplier_ceiling": base.health_multiplier_ceiling,
        "min_report_confidence": base.min_report_confidence,
    })
    study.optimize(lambda trial: objective(trial, folds, frames, base),
                   n_trials=n_trials, show_progress_bar=show_progress)

    best = base.replace(**{k: v for k, v in study.best_params.items()})
    return study, best, score(best, folds=folds, frames=frames)
