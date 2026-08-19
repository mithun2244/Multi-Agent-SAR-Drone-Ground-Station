"""Pull every result in experiments/results/ into one table, and chart it.

    python experiments/baselines/compare.py
    python experiments/baselines/compare.py --metrics mAP,critic_loss,IDF1
    python experiments/baselines/compare.py --selfcheck

Reads whatever the runners have produced — the ablation table, both baselines,
any multi-seed sweeps — normalises them onto one set of columns, prints the
comparison, and writes `experiments/results/full_comparison.csv` plus
`comparison_chart.png`.

It reads results; it never produces them. Run the runners first:

    python experiments/run_ablation.py
    python experiments/baselines/plain_yolo.py
    python experiments/baselines/yolo_plus_tracker.py
    python experiments/run_multi_seed.py --experiment no_wbf --seeds 42,123,456

Comparing across sources honestly
---------------------------------
Every source measures the same scenario, but not at the same tap: `plain_yolo`
scores the detector's raw boxes, `yolo_plus_tracker` scores confirmed tracks,
and the ablation rows score the full pipeline. A tracker that drops
single-frame blips will show a *lower* mAP than the raw detector and be the
better system — so the table carries a `tap` column, and the chart is grouped by
it rather than pretending one ranking covers everything.

Missing cells stay empty. A source that never measured a metric is not a zero,
and averaging or plotting it as one is how a component gets blamed for a number
nobody took. Multi-seed files contribute their `mean` row and carry the spread
alongside, so a difference smaller than its own standard deviation is visible as
such.
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"

# Column order for the unified table. Only the ones some row actually has get
# printed, so the table narrows to what was measured rather than showing a wall
# of empties.
COLUMNS = ("mAP", "recall_at_far", "precision", "recall", "MOTA", "IDF1",
           "id_switches", "geolocation_error_m", "subject_recall", "ndcg",
           "critic_loss", "risk_score", "action")

# Which tap in the pipeline a source measures. See the module docstring on why
# this is a column and not a footnote.
TAPS = {
    "plain_yolo": "detector",
    "yolo_plus_tracker": "detector+tracker",
    "ablation": "full pipeline",
    "multi_seed": "full pipeline",
}


def _number(text):
    if text is None or text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect(results=RESULTS):
    """Every result file, normalised onto {source, run, tap, metrics, note}."""
    rows = []

    table = results / "ablation_table.csv"
    if table.is_file():
        for record in read_csv(table):
            rows.append({
                "source": "ablation",
                "run": record.get("run", ""),
                "tap": TAPS["ablation"],
                "note": record.get("env", ""),
                "metrics": {k: _number(record.get(k)) for k in COLUMNS},
            })

    for path in sorted(results.glob("baseline_*.csv")):
        records = read_csv(path)
        if not records:
            continue
        name = records[0].get("baseline") or path.stem.replace("baseline_", "")
        # Several seeds in one baseline file average into a single row, with the
        # count kept so nobody reads a three-seed mean as a single measurement.
        metrics = {}
        for key in COLUMNS:
            values = [_number(r.get(key)) for r in records]
            numeric = [v for v in values if isinstance(v, float)]
            metrics[key] = sum(numeric) / len(numeric) if numeric else None
        rows.append({
            "source": "baseline",
            "run": name,
            "tap": TAPS.get(name, "detector"),
            "note": f"mean of {len(records)} seed(s)" if len(records) > 1 else "1 seed",
            "metrics": metrics,
        })

    for path in sorted(results.glob("multi_seed_*.csv")):
        records = read_csv(path)
        mean = next((r for r in records if r.get("seed") == "mean"), None)
        std = next((r for r in records if r.get("seed") == "std"), None)
        if mean is None:
            continue
        name = mean.get("experiment") or path.stem.replace("multi_seed_", "")
        seeds = [r["seed"] for r in records if r.get("seed") not in ("mean", "std")]
        rows.append({
            "source": "multi_seed",
            "run": f"{name} (multi-seed)",
            "tap": TAPS["multi_seed"],
            "note": f"mean of {len(seeds)} seeds",
            "metrics": {k: _number(mean.get(k)) for k in COLUMNS},
            "spread": {k: _number((std or {}).get(k)) for k in COLUMNS},
        })

    return rows


def table(rows, metrics=None):
    keys = [k for k in (metrics or COLUMNS)
            if any(isinstance(r["metrics"].get(k), float) or r["metrics"].get(k)
                   for r in rows)]
    width = max((len(r["run"]) for r in rows), default=10) + 2
    tap_width = max((len(r["tap"]) for r in rows), default=8) + 2

    def cell(row, key):
        value = row["metrics"].get(key)
        if value is None or value == "":
            return "—"
        if isinstance(value, float):
            spread = (row.get("spread") or {}).get(key)
            return (f"{value:.3f}±{spread:.3f}"
                    if isinstance(spread, float) and spread > 0 else f"{value:.4f}")
        return str(value)

    # Column widths follow their contents. A fixed width silently truncated
    # `IMMEDIATE_EXTRACTION` to `IMMEDIATE_EXTRACTIO`, which is a different
    # action as far as anyone reading the table is concerned.
    widths = {k: max(len(k), max((len(cell(r, k)) for r in rows), default=0)) + 2
              for k in keys}

    lines = ["", f"  {'run':<{width}}{'tap':<{tap_width}}"
                 + "".join(f"{k:>{widths[k]}}" for k in keys),
             "  " + "-" * (width + tap_width + sum(widths.values()))]
    for row in rows:
        cells = "".join(f"{cell(row, k):>{widths[k]}}" for k in keys)
        lines.append(f"  {row['run']:<{width}}{row['tap']:<{tap_width}}" + cells)

    lines += [
        "",
        "  A dash is a metric that source never measured — not a zero. Taps differ:",
        "  a tracker that drops single-frame blips scores a lower mAP than the raw",
        "  detector and is the better system, so compare within a tap first.",
        "",
    ]
    return "\n".join(lines)


def write_csv(rows, path, metrics=None):
    keys = list(metrics or COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "run", "tap", "note"] + keys)
        for row in rows:
            writer.writerow([row["source"], row["run"], row["tap"], row.get("note", "")]
                            + [row["metrics"].get(k) if row["metrics"].get(k) is not None
                               else "" for k in keys])
    return path


def chart(rows, path, metrics=("mAP", "critic_loss")):
    """One panel per metric, bars grouped by tap. Skips what was not measured."""
    try:
        import matplotlib                                        # noqa: PLC0415
        matplotlib.use("Agg")                                    # no display needed
        import matplotlib.pyplot as plt                          # noqa: PLC0415
    except ImportError:
        print("  no chart: matplotlib is not installed (pip install matplotlib)")
        return None

    panels = [m for m in metrics
              if any(isinstance(r["metrics"].get(m), float) for r in rows)]
    if not panels:
        print("  no chart: none of the requested metrics were measured anywhere")
        return None

    figure, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)),
                                squeeze=False)
    for axis, metric in zip((a[0] for a in axes), panels):
        present = [r for r in rows if isinstance(r["metrics"].get(metric), float)]
        labels = [r["run"] for r in present]
        values = [r["metrics"][metric] for r in present]
        # Error bars only where a spread was actually measured; a bar with no
        # whisker is one draw, and should not look like a converged number.
        errors = [(r.get("spread") or {}).get(metric) or 0.0 for r in present]
        colours = ["#c44e52" if r["source"] == "baseline" else "#4c72b0"
                   for r in present]

        axis.bar(range(len(values)), values, yerr=errors, capsize=3, color=colours)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} — baselines in red, system runs in blue", fontsize=9)
        axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--metrics", default=None,
                        help="comma-separated columns; default is everything measured")
    parser.add_argument("--chart-metrics", default="mAP,critic_loss")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--chart", type=Path, default=None)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    rows = collect(args.results)
    if not rows:
        raise SystemExit(
            f"nothing in {args.results}. Produce some results first:\n"
            "  python experiments/run_ablation.py\n"
            "  python experiments/baselines/plain_yolo.py\n"
            "  python experiments/baselines/yolo_plus_tracker.py"
        )

    metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None
    print(f"\n  {len(rows)} row(s) from {args.results}")
    print(table(rows, metrics))

    out = args.out or args.results / "full_comparison.csv"
    print(f"  wrote {write_csv(rows, out, metrics)}")
    drawn = chart(rows, args.chart or args.results / "comparison_chart.png",
                  tuple(m.strip() for m in args.chart_metrics.split(",")))
    if drawn:
        print(f"  wrote {drawn}")
    print()
    return 0


def selfcheck():
    """Normalising, rendering and writing — on files this makes itself."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp)
        (results / "ablation_table.csv").write_text(
            "run,description,env,mAP,recall_at_far,critic_loss,risk_score,action\n"
            "full_system,everything,,0.7778,0.7778,0.0030,9,IMMEDIATE_EXTRACTION\n"
            "no_weather,no conditions,ABLATION_DISABLE_AGENTS=weather,,,0.0030,3,MONITOR_AND_CONFIRM\n",
            encoding="utf-8")
        (results / "baseline_plain_yolo.csv").write_text(
            "baseline,detector,seed,mAP,precision,recall\n"
            "plain_yolo,stub,0,0.60,0.53,0.89\n"
            "plain_yolo,stub,42,0.70,0.55,0.89\n",
            encoding="utf-8")
        (results / "multi_seed_no_wbf.csv").write_text(
            "experiment,seed,mAP,recall_at_far\n"
            "no_wbf,42,0.59,0.39\nno_wbf,123,0.60,0.00\n"
            "no_wbf,mean,0.595,0.195\nno_wbf,std,0.007,0.276\n",
            encoding="utf-8")

        rows = collect(results)
        by_run = {r["run"]: r for r in rows}
        assert set(by_run) == {"full_system", "no_weather", "plain_yolo",
                               "no_wbf (multi-seed)"}, sorted(by_run)

        # An ablation row that never measured mAP stays empty, not zero.
        assert by_run["no_weather"]["metrics"]["mAP"] is None
        assert by_run["full_system"]["metrics"]["mAP"] == 0.7778

        # Several seeds in one baseline file average, and say how many.
        assert abs(by_run["plain_yolo"]["metrics"]["mAP"] - 0.65) < 1e-9
        assert "2 seed(s)" in by_run["plain_yolo"]["note"]
        assert by_run["plain_yolo"]["tap"] == "detector"

        # A multi-seed file contributes its mean, and keeps the spread.
        multi = by_run["no_wbf (multi-seed)"]
        assert multi["metrics"]["mAP"] == 0.595
        assert multi["spread"]["recall_at_far"] == 0.276
        assert "2 seeds" in multi["note"]

        rendered = table(rows)
        assert "0.595±0.007" in rendered, "a measured spread is shown with the mean"
        assert "—" in rendered and "not a zero" in rendered
        assert "IMMEDIATE_EXTRACTION" in rendered

        out = write_csv(rows, results / "full_comparison.csv")
        body = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert body[1]["mAP"] == "", "an unmeasured cell is empty in the CSV too"
        assert body[0]["tap"] == "full pipeline"

        drawn = chart(rows, results / "chart.png")
        if drawn is not None:
            assert drawn.is_file() and drawn.stat().st_size > 1000, "a real png"

        # An empty results directory is a clear message, not a traceback.
        try:
            main(["--results", str(results / "empty")])
            raise AssertionError("an empty results directory must be refused")
        except SystemExit as e:
            assert "run_ablation.py" in str(e)

    print("  ok  ablation, baseline and multi-seed files all normalise onto one shape")
    print("  ok  an unmeasured metric stays empty in the table and the CSV")
    print("  ok  several seeds in one baseline file average, and say how many")
    print("  ok  a multi-seed mean carries its spread into the table")
    print("  ok  the chart writes a real png, or says why it did not")
    print("  ok  an empty results directory tells you which runner to run")
    print("\n6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
