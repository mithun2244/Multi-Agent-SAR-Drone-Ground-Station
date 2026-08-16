"""Phase 9: tune the system, then prove the tuning did something.

    python -m src.tuning.demo                 # run the study, save, compare
    python -m src.tuning.demo --trials 200    # search harder
    python -m src.tuning.demo --reuse         # score the saved config, no search
    python -m src.tuning.demo --seed 42       # pin the sampler's own randomness

Runs the shipped defaults and the tuned configuration over the same folds and
prints them side by side: mAP from the Phase 1 harness, and recall, NDCG,
geolocation residual, false-alarm rate and loss from the Phase 8 critic.

What this does and does not show
-------------------------------
The pipeline is real — detectors, weighted box fusion, BoT-SORT, geolocation,
the bus, coordinator fusion, the critic. The *imagery* is not: subjects are
projected from known world positions and detected by stubs with configured
recall. So an improvement here is an improvement in how the system is
configured, measured against a simulator. It is a starting point for a tuning
run against real recordings, not a substitute for one.
"""

import sys
import time

from ..critic.critic import Critic
from ..evaluation.metrics import evaluate as evaluate_detections
from ..utils.seed import DEFAULT_SEED, describe, set_global_seed
from .objective import DEFAULT_FOLDS, run_study, score
from .params import CONFIG_PATH, TunedParams
from .scenario import run


def measure(params, folds=DEFAULT_FOLDS, frames=6):
    """Score one configuration two ways: raw detection, and operational picture."""
    critic = Critic(urgency_weight=params.urgency_weight,
                    sector_weight=params.sector_weight)
    maps, ious = [], []
    for seed in folds:
        result = run(params, seed=seed, frames=frames)
        detections, ground_truth = result.evaluation_inputs()
        harness = evaluate_detections(detections, ground_truth, n_frames=result.frames,
                                      far_target=params.target_far)
        maps.append(harness["mAP"])
        ious.append(harness["recall_at_far"])

    fitness = score(params, folds=folds, frames=frames, critic=critic)
    residuals = [r.geolocation.get("median_m") for r in critic.history
                 if r.scorable and r.geolocation.get("median_m") is not None]
    return {
        "mAP": round(sum(maps) / len(maps), 4),
        "recall@FAR": round(sum(ious) / len(ious), 4),
        "recall": fitness.recall,
        "NDCG": fitness.ndcg,
        "geo_median_m": round(sum(residuals) / len(residuals), 2) if residuals else None,
        "FAR": fitness.far,
        "loss": fitness.loss,
    }


ROWS = (
    ("mAP (Phase 1 harness)", "mAP", "higher", 4),
    ("recall @ target FAR", "recall@FAR", "higher", 4),
    ("subject recall", "recall", "higher", 4),
    ("NDCG (ranking)", "NDCG", "higher", 4),
    ("geolocation median", "geo_median_m", "lower", 2),
    ("false alarms / frame", "FAR", "lower", 4),
    ("CRITIC LOSS", "loss", "lower", 4),
)


def _arrow(before, after, direction):
    if before is None or after is None or abs(after - before) < 1e-9:
        return "     ="
    better = (after > before) if direction == "higher" else (after < before)
    return "  better" if better else "   worse"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    reuse = "--reuse" in argv
    trials = 60
    if "--trials" in argv:
        trials = int(argv[argv.index("--trials") + 1])
    seed = DEFAULT_SEED
    if "--seed" in argv:
        seed = int(argv[argv.index("--seed") + 1])

    baseline = TunedParams()
    # Seeds the TPE sampler, so the same --seed searches the space in the same
    # order and lands on the same configuration. The *folds* are deliberately
    # not reseeded: they are the fixed validation set every configuration is
    # judged against, and moving them would make two studies incomparable —
    # which is the whole reason Phase 1 fixed the splits in the first place.
    print(f"\n  {describe(set_global_seed(seed), seed)}")
    print(f"  folds {list(DEFAULT_FOLDS)}   scenario: 3 subjects (1 faint casualty), "
          f"2 RGB-only decoys")

    if reuse:
        tuned = TunedParams.from_json()
        print(f"  reusing {CONFIG_PATH} ({'found' if CONFIG_PATH.exists() else 'MISSING'})")
    else:
        print(f"  running Optuna TPE study, {trials} trials …")
        started = time.time()
        study, tuned, fitness = run_study(n_trials=trials, seed=seed)
        print(f"  study done in {time.time() - started:.1f}s   "
              f"best trial {study.best_trial.number} of {len(study.trials)}   "
              f"loss {fitness.loss:.4f}")
        path = tuned.to_json()
        print(f"  saved {path}")

    print("\n  measuring baseline …")
    before = measure(baseline)
    print("  measuring tuned …")
    after = measure(tuned)

    print(f"\n  {'metric':<24}{'baseline':>12}{'tuned':>12}{'':>9}")
    print("  " + "-" * 57)
    for label, key, direction, places in ROWS:
        b, a = before[key], after[key]
        b_text = "n/a" if b is None else f"{b:.{places}f}"
        a_text = "n/a" if a is None else f"{a:.{places}f}"
        print(f"  {label:<24}{b_text:>12}{a_text:>12}{_arrow(b, a, direction):>9}")

    changes = baseline.differences(tuned)
    print(f"\n  {len(changes)} parameter(s) changed")
    for name, (was, now) in changes.items():
        was_text = f"{was:.3f}" if isinstance(was, float) else str(was)
        now_text = f"{now:.3f}" if isinstance(now, float) else str(now)
        print(f"    {name:<30}{was_text:>8} -> {now_text:>8}")

    delta = before["loss"] - after["loss"]
    verdict = ("improves on the defaults" if delta > 1e-6 else
               "does not beat the defaults — keep them")
    print(f"\n  loss {before['loss']:.4f} -> {after['loss']:.4f} "
          f"({delta:+.4f}); the tuned configuration {verdict}\n")
    return before, after


if __name__ == "__main__":
    main()
