"""Run every ablation in the config, and put the results side by side.

    python experiments/run_ablation.py
    python experiments/run_ablation.py --only full_system,no_wbf,no_weather
    python experiments/run_ablation.py --seed 42 --frames 8
    python experiments/run_ablation.py --selfcheck

Each configuration runs in its **own process**, with its switches in that
process's environment. That is not caution for its own sake: the switches are
read when a component is constructed, so setting them in-process would ablate
whatever had not been imported yet and silently leave the rest of the pipeline
whole. One process per row is the only way the row means what it says.

Results go to `experiments/results/ablation_table.csv` and to stdout as a table
with the delta against `full_system`, which is the row every other one is read
against.

Reading the table
-----------------
A cell is `n/a` when that engine did not run for that configuration — see
`measures` in the config. It is **not** the same as zero and it is not a number
carried over from the baseline: a switch the engine cannot see would otherwise
produce a row identical to `full_system`, which reads as "this component does
not matter" when it means "this experiment could not tell".
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._engines import METRICS, measure       # noqa: E402

CONFIG = HERE / "ablation_config.yaml"
RESULTS = HERE / "results"
BASELINE = "full_system"


def load_config(path=CONFIG):
    try:
        import yaml                                     # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "experiments need PyYAML for the config:  pip install pyyaml\n"
            "(it ships with ultralytics, so a training environment already has it)"
        ) from None
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = data.get("defaults") or {}
    runs = data.get("runs") or []
    if not runs:
        raise SystemExit(f"{path} defines no runs")
    names = [r["name"] for r in runs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise SystemExit(f"{path} repeats run name(s): {', '.join(sorted(duplicates))}")
    return defaults, runs


def run_one(config, seed, frames, timeout=900):
    """One configuration, in its own process. Returns the metrics dict."""
    command = [sys.executable, str(Path(__file__).resolve()), "--emit",
               "--engines", ",".join(config.get("measures", ["detection"])),
               "--seed", str(seed), "--frames", str(frames)]
    env = {**os.environ, **{k: str(v) for k, v in (config.get("env") or {}).items()}}
    # Offline and deterministic: an ablation table that depended on a live model
    # or a live Redis would be measuring the network as much as the system.
    env.update(REDIS_URL="", NVIDIA_API_KEY="")

    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        return {"error": (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
                or ["failed with no output"]}
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"error": ["no metrics on stdout"]}


def table(rows, metrics=METRICS):
    """The printed comparison, with deltas against the baseline row."""
    names = [r["name"] for r in rows]
    width = max(len(n) for n in names) + 2
    shown = [(key, direction) for key, direction in metrics
             if any(r["metrics"].get(key) is not None for r in rows)]

    baseline = next((r["metrics"] for r in rows if r["name"] == BASELINE), {})
    header = f"  {'run':<{width}}" + "".join(f"{key:>22}" for key, _ in shown)
    lines = ["", header, "  " + "-" * (width + 22 * len(shown))]

    for row in rows:
        cells = []
        for key, direction in shown:
            value = row["metrics"].get(key)
            if value is None:
                cells.append(f"{'n/a':>22}")
                continue
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            mark = ""
            base = baseline.get(key)
            if (direction and row["name"] != BASELINE and isinstance(value, (int, float))
                    and isinstance(base, (int, float)) and not isinstance(value, bool)):
                delta = value - base
                if abs(delta) > 1e-9:
                    better = (delta > 0) if direction == "higher" else (delta < 0)
                    mark = f" {'+' if delta > 0 else ''}{delta:.3f}{'^' if better else 'v'}"
            cells.append(f"{text + mark:>22}")
        lines.append(f"  {row['name']:<{width}}" + "".join(cells))
        if row["metrics"].get("error"):
            lines.append(f"  {'':<{width}}  FAILED: {row['metrics']['error'][0]}")

    lines += [
        "",
        "  ^ better than full_system, v worse. n/a means this engine did not run for",
        "  that configuration — see `measures` in the config. It is not a zero, and it",
        "  is not the baseline's number carried across.",
        "",
    ]
    return "\n".join(lines)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key, _ in METRICS] + ["agents_dispatched"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "description", "env"] + keys)
        for row in rows:
            writer.writerow(
                [row["name"], row.get("description", ""),
                 ";".join(f"{k}={v}" for k, v in (row.get("env") or {}).items())]
                + [row["metrics"].get(key, "") if row["metrics"].get(key) is not None else ""
                   for key in keys]
            )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--only", help="comma-separated run names")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--out", type=Path, default=RESULTS / "ablation_table.csv")
    parser.add_argument("--selfcheck", action="store_true")
    # The child half: run one configuration and print its metrics as JSON.
    parser.add_argument("--emit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engines", default="detection", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if args.emit:
        metrics = measure(args.engines.split(","), seed=args.seed or 0,
                          frames=args.frames or 6)
        print(json.dumps(metrics))
        return 0

    defaults, runs = load_config(args.config)
    seed = args.seed if args.seed is not None else defaults.get("seed", 0)
    frames = args.frames if args.frames is not None else defaults.get("frames", 6)
    wanted = {n.strip() for n in args.only.split(",")} if args.only else None

    print(f"\n  config  {args.config}")
    print(f"  seed {seed}, {frames} frame(s) per run, one process each")

    rows = []
    for config in runs:
        if wanted and config["name"] not in wanted:
            continue
        switches = config.get("env") or {}
        print(f"    {config['name']:<20} "
              + (", ".join(f"{k}={v}" for k, v in switches.items()) or "no switches"))
        rows.append({
            "name": config["name"],
            "description": config.get("description", ""),
            "env": switches,
            "metrics": run_one(config, seed, frames),
        })

    if not rows:
        raise SystemExit("nothing matched --only")

    print(table(rows))
    path = write_csv(rows, args.out)
    print(f"  wrote {path}\n")
    failed = [r["name"] for r in rows if r["metrics"].get("error")]
    if failed:
        print(f"  {len(failed)} run(s) failed: {', '.join(failed)}\n")
        return 1
    return 0


def selfcheck():
    """The runner's own logic, without running a single sortie."""
    defaults, runs = load_config()
    names = [r["name"] for r in runs]
    assert names[0] == BASELINE, "the baseline must be first so the table reads downward"
    for expected in ("rgb_only", "no_wbf", "no_cmc", "no_decision_chain", "no_weather",
                     "no_path", "no_scene", "no_health", "detection_only"):
        assert expected in names, expected

    # Every switch a run names must be one the code actually reads, or the row
    # is a no-op wearing a label.
    from src.utils.ablation import (
        CMC, DECISION, DISABLE_AGENTS, DISABLE_SENSORS, WBF,
    )
    from src.coordinator.router import ALL_AGENTS

    known = {WBF, CMC, DECISION, DISABLE_AGENTS, DISABLE_SENSORS}
    for config in runs:
        for key, value in (config.get("env") or {}).items():
            assert key in known, f"{config['name']} sets unknown switch {key}"
            if key == DISABLE_AGENTS:
                for agent in str(value).split(","):
                    assert agent.strip() in ALL_AGENTS, f"{config['name']}: no agent {agent!r}"
            if key == DISABLE_SENSORS:
                for sensor in str(value).split(","):
                    assert sensor.strip() in ("rgb", "lidar"), sensor
        assert config.get("measures"), f"{config['name']} says nothing about what can see it"
        for engine in config["measures"]:
            assert engine in ("detection", "flight", "command"), engine

    # The CMC switch is only meaningful where something moves the camera.
    cmc_runs = [c for c in runs if CMC in (c.get("env") or {})]
    assert cmc_runs, "no run ablates camera-motion compensation"
    for config in cmc_runs:
        assert "flight" in config["measures"], (
            f"{config['name']} ablates CMC but is not measured by a moving airframe, "
            f"so the row would show no change for the wrong reason"
        )
        assert "detection" not in config["measures"], (
            f"{config['name']}: the tuning scenario flies a static pose and cannot see CMC"
        )

    # A switch that only the command engine can see must not claim the detection
    # engine measures it — that is the mislabel this whole file warns about.
    for config in runs:
        switches = set((config.get("env") or {}))
        if switches and switches <= {DISABLE_AGENTS, DECISION}:
            assert "detection" not in config["measures"], (
                f"{config['name']} claims the detection engine sees an agent or "
                f"decision-chain switch, which it cannot"
            )

    # The table renders, marks direction, and never invents a missing cell.
    rows = [
        {"name": BASELINE, "metrics": {"mAP": 0.80, "critic_loss": 0.20, "targets": 2}},
        {"name": "no_wbf", "metrics": {"mAP": 0.60, "critic_loss": 0.35, "targets": 1}},
        {"name": "no_weather", "metrics": {"mAP": None, "critic_loss": 0.28, "targets": 2}},
    ]
    rendered = table(rows)
    assert "0.6000 -0.200v" in rendered, rendered
    assert "0.3500 +0.150v" in rendered, "a higher loss is worse"
    assert "n/a" in rendered and "not a zero" in rendered

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_csv([{**r, "description": "d", "env": {"A": "off"}} for r in rows],
                         Path(tmp) / "sub" / "table.csv")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("run,description,env,mAP"), text.splitlines()[0]
        assert "A=off" in text
        body = list(csv.DictReader(text.splitlines()))
        assert body[2]["mAP"] == "", "an unmeasured cell stays empty in the CSV too"
        assert body[1]["critic_loss"] == "0.35"

    print("  ok  the config defines every run, baseline first")
    print("  ok  every switch named is one the code reads, and every agent exists")
    print("  ok  no run claims an engine that cannot see its switch")
    print("  ok  the table marks better and worse against the baseline")
    print("  ok  an unmeasured cell reads n/a, never zero and never the baseline's number")
    print("  ok  the CSV round-trips, empty where unmeasured")
    print("\n6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
