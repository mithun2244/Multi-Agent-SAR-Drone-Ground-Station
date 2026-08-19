"""One experiment, several seeds, mean ± std.

    python experiments/run_multi_seed.py --experiment full_system
    python experiments/run_multi_seed.py --experiment no_wbf --seeds "42,123,456,789,101"
    python experiments/run_multi_seed.py --experiment no_weather --seeds 1,2,3 --frames 8
    python experiments/run_multi_seed.py --selfcheck

A single-seed ablation row is one draw of the detector noise. It can move for
reasons that have nothing to do with the component switched off, which is how a
component gets credited or blamed for a coin flip. Repeating the same
configuration across seeds and reporting the spread is what makes a difference
readable: if the standard deviation swamps the delta, the table was measuring
noise.

The seeds are named, not generated. A run reports the exact list it used and
writes it into the CSV, so "we averaged over five seeds" is a claim someone else
can reproduce rather than take on trust.

Results go to `experiments/results/multi_seed_{experiment}.csv`: one row per
seed, then a `mean` row and a `std` row, so the raw draws stay visible next to
the summary. An average that hides its inputs is how an outlier disappears.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._engines import METRICS                      # noqa: E402
from experiments.run_ablation import RESULTS, load_config     # noqa: E402

DEFAULT_SEEDS = (42, 123, 456, 789, 101)


def parse_seeds(text):
    seeds = [int(part) for part in str(text).replace(" ", "").split(",") if part]
    if not seeds:
        raise SystemExit("--seeds needs at least one integer")
    if len(set(seeds)) != len(seeds):
        # Averaging a repeated seed weights one draw twice and narrows the
        # spread for no reason — it would look more certain, not more measured.
        raise SystemExit(f"--seeds repeats a value: {sorted(seeds)}")
    return seeds


def find_experiment(name, config_path=None):
    _, runs = load_config(config_path) if config_path else load_config()
    for run in runs:
        if run["name"] == name:
            return run
    raise SystemExit(f"no experiment named {name!r}. Known: "
                     + ", ".join(r["name"] for r in runs))


def run_seed(config, seed, frames, timeout=900):
    """One seed of one configuration, in its own process."""
    command = [sys.executable, str(HERE / "run_ablation.py"), "--emit",
               "--engines", ",".join(config.get("measures", ["detection"])),
               "--seed", str(seed), "--frames", str(frames)]
    env = {**os.environ, **{k: str(v) for k, v in (config.get("env") or {}).items()}}
    env.update(REDIS_URL="", NVIDIA_API_KEY="")
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return {"error": tail[-1] if tail else "failed with no output"}
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"error": "no metrics on stdout"}


def summarise(per_seed):
    """mean and sample std per numeric metric. Returns (mean, std, n_used)."""
    mean, std, counted = {}, {}, {}
    for key, _ in METRICS:
        values = [row[key] for row in per_seed
                  if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        counted[key] = len(values)
        if not values:
            continue
        mean[key] = sum(values) / len(values)
        # Sample standard deviation: with one seed there is no spread to report,
        # and printing 0.0 would claim a certainty a single draw cannot support.
        std[key] = (math.sqrt(sum((v - mean[key]) ** 2 for v in values) / (len(values) - 1))
                    if len(values) > 1 else None)
    return mean, std, counted


def format_table(name, seeds, per_seed, mean, std):
    keys = [key for key, _ in METRICS if key in mean]
    width = max(12, max((len(k) for k in keys), default=12)) + 2
    lines = ["", f"  {name} over {len(seeds)} seed(s): {', '.join(map(str, seeds))}",
             "  " + "-" * (width + 34), f"  {'metric':<{width}}{'mean':>14}{'std':>12}{'n':>6}"]
    for key in keys:
        spread = "n/a" if std.get(key) is None else f"{std[key]:.4f}"
        lines.append(f"  {key:<{width}}{mean[key]:>14.4f}{spread:>12}"
                     f"{sum(1 for r in per_seed if isinstance(r.get(key), (int, float))):>6}")
    failures = [s for s, r in zip(seeds, per_seed) if r.get("error")]
    if failures:
        lines.append(f"  {len(failures)} seed(s) failed: {failures}")
    lines.append("")
    return "\n".join(lines)


def write_csv(path, name, seeds, per_seed, mean, std):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key, _ in METRICS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "seed"] + keys)
        for seed, row in zip(seeds, per_seed):
            writer.writerow([name, seed]
                            + [row.get(key, "") if row.get(key) is not None else ""
                               for key in keys])
        # The raw draws stay above these two rows on purpose: a mean that hides
        # its inputs is how an outlier disappears.
        writer.writerow([name, "mean"]
                        + [round(mean[key], 6) if key in mean else "" for key in keys])
        writer.writerow([name, "std"]
                        + [round(std[key], 6) if std.get(key) is not None else ""
                           for key in keys])
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="full_system")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    config = find_experiment(args.experiment, args.config)
    seeds = parse_seeds(args.seeds)
    frames = args.frames or 6

    switches = ", ".join(f"{k}={v}" for k, v in (config.get("env") or {}).items())
    print(f"\n  experiment {args.experiment}   {switches or 'no switches'}")
    print(f"  engines    {', '.join(config.get('measures', []))}")
    print(f"  seeds      {', '.join(map(str, seeds))}\n")

    per_seed = []
    for seed in seeds:
        metrics = run_seed(config, seed, frames)
        per_seed.append(metrics)
        note = metrics.get("error") or ", ".join(
            f"{k}={v:.4f}" for k, v in list(metrics.items())[:3]
            if isinstance(v, float))
        print(f"    seed {seed:<6} {note}")

    mean, std, _ = summarise(per_seed)
    if not mean:
        print("\n  every seed failed — nothing to average\n")
        return 1

    print(format_table(args.experiment, seeds, per_seed, mean, std))
    path = args.out or RESULTS / f"multi_seed_{args.experiment}.csv"
    print(f"  wrote {write_csv(path, args.experiment, seeds, per_seed, mean, std)}\n")
    return 1 if any(r.get("error") for r in per_seed) else 0


def selfcheck():
    """The statistics and the file, without running a single sortie."""
    assert parse_seeds("42,123, 456") == [42, 123, 456]
    assert parse_seeds(" 7 ") == [7]
    for bad in ("", " , "):
        try:
            parse_seeds(bad)
            raise AssertionError(f"{bad!r} must be refused")
        except SystemExit:
            pass
    try:
        parse_seeds("42,42,7")
        raise AssertionError("a repeated seed must be refused")
    except SystemExit as e:
        assert "repeats" in str(e)

    rows = [{"mAP": 0.80, "critic_loss": 0.10, "action": "IMMEDIATE_EXTRACTION"},
            {"mAP": 0.60, "critic_loss": 0.30, "action": "DISPATCH_GROUND_TEAM"},
            {"mAP": 0.70, "critic_loss": 0.20, "action": "IMMEDIATE_EXTRACTION"}]
    mean, std, counted = summarise(rows)
    assert abs(mean["mAP"] - 0.70) < 1e-9, mean
    assert abs(std["mAP"] - 0.1) < 1e-9, std          # sample std of .8/.7/.6
    assert "action" not in mean, "a string metric is not averaged"
    assert counted["mAP"] == 3

    # One seed has no spread, and must not claim zero.
    mean, std, _ = summarise([{"mAP": 0.5}])
    assert mean["mAP"] == 0.5 and std["mAP"] is None

    # A failed seed drops out of the average rather than counting as zero,
    # which would drag a mean down and look like a result.
    mean, std, counted = summarise([{"mAP": 0.8}, {"error": "boom"}, {"mAP": 0.6}])
    assert abs(mean["mAP"] - 0.7) < 1e-9 and counted["mAP"] == 2

    # Booleans are not numbers here: averaging chain_consistent would produce
    # "0.67 consistent", which means nothing.
    mean, _, _ = summarise([{"chain_consistent": True}, {"chain_consistent": False}])
    assert "chain_consistent" not in mean

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        mean, std, _ = summarise(rows)
        path = write_csv(Path(tmp) / "sub" / "multi_seed_x.csv", "x", [1, 2, 3],
                         rows, mean, std)
        body = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        assert [r["seed"] for r in body] == ["1", "2", "3", "mean", "std"], body
        assert body[0]["mAP"] == "0.8" and body[3]["mAP"] == "0.7"
        assert body[4]["mAP"] == "0.1", "the spread is written, not just the mean"

        rendered = format_table("x", [1, 2, 3], rows, mean, std)
        assert "over 3 seed(s)" in rendered and "42" not in rendered
        assert "mAP" in rendered and "0.7000" in rendered

    print("  ok  seeds parse, and a repeated one is refused")
    print("  ok  mean and sample std are right, strings and booleans left out")
    print("  ok  one seed reports no spread rather than claiming zero")
    print("  ok  a failed seed drops out instead of averaging as zero")
    print("  ok  the CSV keeps every draw above the mean and std rows")
    print("\n5 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
