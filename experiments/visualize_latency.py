"""The latency budget as a picture.

    python experiments/visualize_latency.py
    python experiments/visualize_latency.py --selfcheck

Reads `experiments/results/latency_budget.csv` and writes
`experiments/results/latency_breakdown.png`. It reads results; it never produces
them — run the profiler first:

    python experiments/profile_latency.py

Two panels, because one cannot show both things
-----------------------------------------------
The stack on the left is the frame budget, one bar per airframe, each segment a
component priced at what it costs *per frame* (a per-track stage counted once
per track). That answers "where does the time go", and the answer is usually one
enormous segment.

Which is exactly why the right panel exists. The components span four orders of
magnitude, so on the stack everything except the detector is a hairline. The
right panel is per-call cost on a log axis, where a stage that takes 90 µs is
still legible — and legible is what stops someone optimising the wrong one.

The device the numbers came off is in the subtitle, from the CSV's own column. A
chart of milliseconds without the hardware behind them is a chart of nothing.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

# One colour per component, fixed, so a component keeps its colour between the
# two panels and between runs. Same palette as `baselines/compare.py`.
COLOURS = {
    "yolo11m_rgb": "#4c72b0",
    "wbf": "#dd8452",
    "botsort_cmc": "#55a868",
    "lidar_range_projection": "#c44e52",
    "dem_ray_march": "#8172b3",
    "decision_chain": "#937860",
}
FALLBACK = "#8c8c8c"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    """A blank cell is not a zero — it is a measurement nobody took."""
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def split(rows):
    """(components, pipelines) — pipelines keep only those with a total."""
    components = {r["component"]: r for r in rows
                  if not r["component"].startswith("pipeline_")}
    pipelines = [r for r in rows
                 if r["component"].startswith("pipeline_")
                 and number(r["mean_ms"]) is not None]
    return components, pipelines


def stacks(components, pipelines):
    """Per pipeline: (label, [(component, ms this frame)], total, fps)."""
    built = []
    for row in pipelines:
        parts = [p for p in row["parts"].split("+") if p]
        segments = []
        for part in parts:
            source = components.get(part)
            per_call = number((source or {}).get("mean_ms"))
            if per_call is None:
                continue
            calls = number(source.get("calls_per_frame")) or 1.0
            segments.append((part, per_call * calls))
        built.append((row["component"].replace("pipeline_", ""), segments,
                      number(row["mean_ms"]), number(row["fps_equivalent"])))
    return built


def chart(rows, path):
    try:
        import matplotlib                                        # noqa: PLC0415
        matplotlib.use("Agg")                                    # no display needed
        import matplotlib.pyplot as plt                          # noqa: PLC0415
    except ImportError:
        print("  no chart: matplotlib is not installed (pip install matplotlib)")
        return None

    components, pipelines = split(rows)
    if not pipelines:
        raise SystemExit(
            "no pipeline totals in the CSV — every budget was n/a.\n"
            "  That means a component did not run, most likely the detector.\n"
            "  Run `python experiments/profile_latency.py` and read its header."
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5),
                                         gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: the frame budget, stacked.
    built = stacks(components, pipelines)
    labelled = set()          # one legend entry each, across every bar: the
    for x, (label, segments, total, fps) in enumerate(built):   # airframes differ
        bottom = 0.0
        for name, milliseconds in segments:
            left.bar(x, milliseconds, bottom=bottom, width=0.55,
                     color=COLOURS.get(name, FALLBACK),
                     label=None if name in labelled else name)
            labelled.add(name)
            bottom += milliseconds
        left.text(x, bottom * 1.02, f"{total:.1f} ms\n{fps:.1f} FPS",
                  ha="center", va="bottom", fontsize=9)

    left.set_xticks(range(len(built)))
    left.set_xticklabels([label for label, _, _, _ in built], fontsize=9)
    left.set_ylabel("milliseconds per frame")
    left.set_title("frame budget by component — airframe on the x axis", fontsize=9)
    left.set_ylim(0, max(total for _, _, total, _ in built) * 1.25)
    left.grid(axis="y", alpha=0.3)
    # Below the axes: inside, it sits on top of the tallest bar and hides the
    # one number on this panel anyone came for.
    left.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2,
                frameon=False)

    # Right: per-call cost, log axis, so the sub-millisecond stages are visible.
    measured = [r for r in components.values() if number(r["mean_ms"]) is not None]
    measured.sort(key=lambda r: number(r["mean_ms"]))
    labels = [f"{r['component']}  (per {r['per']})" for r in measured]
    means = [number(r["mean_ms"]) for r in measured]
    p95s = [number(r["p95_ms"]) for r in measured]

    right.barh(range(len(means)), means,
               color=[COLOURS.get(r["component"], FALLBACK) for r in measured])
    # p95 as a tick on the bar, not a second bar: it is the same measurement's
    # bad frame, not a different quantity.
    right.scatter(p95s, range(len(p95s)), marker="|", s=140, color="#222222",
                  zorder=3, label="p95")
    for i, value in enumerate(means):
        right.text(value * 1.15, i, f"{value:.3f} ms", va="center", fontsize=8)

    right.set_yticks(range(len(labels)))
    right.set_yticklabels(labels, fontsize=8)
    right.set_xscale("log")
    right.set_xlim(min(means) / 3, max(p95s) * 3)
    right.set_xlabel("milliseconds per call (log scale)")
    right.set_title("cost of one call — log axis, or the small stages vanish",
                    fontsize=9)
    right.grid(axis="x", alpha=0.3)
    right.legend(fontsize=7, loc="lower right")

    device = next((r.get("device") for r in rows if r.get("device")), "device not recorded")
    figure.suptitle(f"SAR perception latency — {device}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=RESULTS / "latency_budget.csv")
    parser.add_argument("--out", type=Path, default=RESULTS / "latency_breakdown.png")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    if not args.csv.is_file():
        raise SystemExit(f"no {args.csv} — run `python experiments/profile_latency.py` first")

    rows = read_rows(args.csv)
    drawn = chart(rows, args.out)
    if drawn:
        print(f"\n  {len(rows)} row(s) from {args.csv}")
        print(f"  wrote {drawn}\n")
    return 0


def selfcheck():
    """The budget arithmetic the chart draws, without drawing it."""
    rows = [
        {"component": "yolo11m_rgb", "per": "frame", "calls_per_frame": "1",
         "mean_ms": "200.0", "p95_ms": "240.0", "parts": "", "device": "cpu — test"},
        {"component": "wbf", "per": "frame", "calls_per_frame": "1",
         "mean_ms": "0.5", "p95_ms": "0.8", "parts": "", "device": "cpu — test"},
        {"component": "dem_ray_march", "per": "track", "calls_per_frame": "4",
         "mean_ms": "3.0", "p95_ms": "3.6", "parts": "", "device": "cpu — test"},
        {"component": "decision_chain", "per": "query", "calls_per_frame": "0",
         "mean_ms": "0.04", "p95_ms": "0.06", "parts": "", "device": "cpu — test"},
        {"component": "pipeline_rgb_only", "per": "frame", "calls_per_frame": "",
         "mean_ms": "212.0", "fps_equivalent": "4.72", "p95_ms": "254.4",
         "parts": "yolo11m_rgb+dem_ray_march", "device": "cpu — test"},
        {"component": "pipeline_rgb_lidar", "per": "frame", "calls_per_frame": "",
         "mean_ms": "", "fps_equivalent": "", "p95_ms": "",
         "parts": "yolo11m_rgb+wbf+lidar_range_projection", "device": "cpu — test"},
    ]
    components, pipelines = split(rows)
    assert set(components) == {"yolo11m_rgb", "wbf", "dem_ray_march", "decision_chain"}
    assert [p["component"] for p in pipelines] == ["pipeline_rgb_only"], \
        "a pipeline with no total is not drawn as a zero-height bar"

    built = stacks(components, pipelines)
    assert len(built) == 1
    label, segments, total, fps = built[0]
    assert label == "rgb_only"
    assert dict(segments) == {"yolo11m_rgb": 200.0, "dem_ray_march": 12.0}, segments
    assert round(sum(v for _, v in segments), 3) == 212.0 == total, "the stack is the total"
    assert fps == 4.72

    # A blank cell is not a zero.
    assert number("") is None and number(None) is None and number("0") == 0.0

    # A component the profiler could not measure is left out of the stack rather
    # than charged as free.
    components["dem_ray_march"]["mean_ms"] = ""
    _, thin, thin_total, _ = stacks(components, pipelines)[0]
    assert [n for n, _ in thin] == ["yolo11m_rgb"], thin
    assert sum(v for _, v in thin) != thin_total, "and the gap stays visible"

    # Every component the profiler emits has a colour, so nothing changes hue
    # between panels or between runs.
    from profile_latency import PIPELINES                       # noqa: PLC0415

    for _name, parts, _note in PIPELINES:
        for part in parts:
            assert part in COLOURS, part

    print("  ok  a pipeline with no total is not drawn")
    print("  ok  the stacked segments add up to the pipeline total")
    print("  ok  a per-track stage is charged once per track in the stack")
    print("  ok  a blank cell is not a zero, in the stack or out of it")
    print("  ok  every component the profiler emits has a fixed colour")
    print("\n5 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
