"""What one frame costs, component by component, and what that leaves for FPS.

    python experiments/profile_latency.py
    python experiments/profile_latency.py --warmup 10 --iterations 50
    python experiments/profile_latency.py --selfcheck

Five components, each called on its own with realistic inputs: the RGB detector,
weighted box fusion, BoT-SORT with camera-motion compensation, the geolocation
step in both of its forms, and the coordinator's decision chain. 50 warm-up
calls, then 200 measured ones, `time.perf_counter` around each.

**Real weights, real frame.** The detector row is `PERCEPTION_MODE=real` —
`config/weights/yolo11m_visdrone.pt` over a VisDrone test image. A stub answers
in microseconds and says nothing whatever about whether an airframe can fly this
pipeline, so a missing checkpoint produces `n/a` rows and a printed reason,
never a stub's number wearing YOLO11m's name. Content matters as well as
resolution: NMS costs what the frame's candidate boxes cost, which is why this
profiles a real image rather than noise.

Two pipelines, because there are two airframes
----------------------------------------------
With two sensors fitted, WBF runs and a measured LiDAR range makes the fix. With
one, there is nothing to fuse and the fix comes off the DEM ray march — a very
different bill. Reporting a single FPS would be quoting whichever airframe
flattered the number.

The decision chain is **not in either budget**. It runs per operator query on a
picture, not per frame, and adding it to a frame budget would understate FPS.

What the numbers are not
------------------------
  * A pipeline row's `p95` is the *sum* of its components' p95s — an upper bound
    on the frame's p95, not the frame's p95. Components do not all have their
    bad frame at the same time.
  * p99 over 200 samples is the second-slowest sample. A hint about the tail,
    not an estimate of it.
  * The device is whatever ran it, printed in the header. CPU numbers here are
    not Jetson numbers; the shape of the budget transfers, the absolute times do
    not.

ponytail: single-frame, single-threaded, batch of one, no TensorRT or half
precision. That is the airframe's worst case and the useful one to plan against;
if a real drone build wants the optimised figure, export the checkpoint and
re-run this against it — nothing here reads the model except through
`build_detectors`.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordinator.decision import decide                            # noqa: E402
from src.perception.detectors import lidar_stub, yolo11m_stub          # noqa: E402
from src.perception.fusion import weighted_box_fusion                  # noqa: E402
from src.perception.geolocation import (                               # noqa: E402
    RangeEstimate,
    RangeSource,
    geolocate,
)
from src.perception.models import REAL, RGB_WEIGHTS, build_detectors   # noqa: E402
from src.perception.tracking import Affine, BoTSORT                    # noqa: E402
from src.tuning.params import load_params                              # noqa: E402
from src.tuning.scenario import (                                      # noqa: E402
    CAMERA,
    TELEMETRY,
    _stub_targets,
    search_area_dem,
)
from src.tuning.scenario import run as run_scenario                    # noqa: E402

RESULTS = ROOT / "experiments" / "results"
FRAMES_DIR = ROOT / "datasets" / "VisDrone" / "images" / "test"
CASE = "case-profile"
WARMUP, ITERATIONS = 50, 200

# A frame of drone drift, for the compensation to have something to compensate.
# Sub-pixel on purpose: the fixture holds the *detections* still while the affine
# pushes the tracks, which is backwards from a sortie, where both move together.
# At 290 m AGL on a 2400 px lens a person is about 5 px wide, so a realistic 6 px
# pan against static detections tears every track down and this would profile
# respawning rather than tracking. The compensation itself costs the same per
# track whatever the magnitude — it is the association behind it that needs the
# geometry to stay sane.
CAMERA_MOTION = Affine.from_camera_delta(dx_px=0.4, dy_px=-0.15,
                                         rotation_rad=0.0002, scale=1.00005)

COLUMNS = ("component", "per", "calls_per_frame", "iterations",
           "mean_ms", "p95_ms", "p99_ms", "fps_equivalent", "parts", "device", "note")

PIPELINES = (
    ("pipeline_rgb_lidar",
     ("yolo11m_rgb", "wbf", "botsort_cmc", "lidar_range_projection"),
     "two sensors fitted: WBF runs and a measured range beats the terrain march"),
    ("pipeline_rgb_only",
     ("yolo11m_rgb", "botsort_cmc", "dem_ray_march"),
     "one feed: nothing to fuse, and the fix comes off the DEM"),
)


def percentile(samples, q):
    """Nearest-rank percentile: an actual observed sample, never interpolated."""
    ordered = sorted(samples)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def time_it(call, warmup=WARMUP, iterations=ITERATIONS):
    """Warm up, then time each call. Returns milliseconds, in the order run.

    `call` takes the iteration number, so a component that must not be handed
    the same frame id twice — the tracker — still gets a fresh one every time.
    """
    for i in range(warmup):
        call(i)
    samples = []
    for i in range(iterations):
        start = time.perf_counter()
        call(warmup + i)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def detector_and_frame(frames_dir=FRAMES_DIR):
    """The real detector and something real to point it at, or why not.

    Returns (detector, frame, note, reason). `detector` is None when real
    weights cannot be had, and `reason` says what was missing.
    """
    if not RGB_WEIGHTS.is_file():
        return None, None, "", f"no checkpoint at {RGB_WEIGHTS}"
    try:
        detector = build_detectors(REAL, sensors=("rgb",))["rgb"]
        detector.model  # load now: the first frame must not pay for the weights
    except Exception as error:                       # ultralytics, torch, a bad file
        return None, None, "", f"{type(error).__name__}: {error}"

    images = sorted(frames_dir.glob("*.jpg")) if frames_dir.is_dir() else []
    if images:
        return detector, str(images[0]), f"{images[0].name}, conf {detector.conf}", None

    # No dataset checked out. A grey frame still measures the model, and NMS has
    # nothing to chew on, so the number is a floor — said so in the note.
    import numpy as np

    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    return detector, frame, "synthetic 1280x720 grey frame — a floor, NMS sees nothing", None


def build_components(frames_dir=FRAMES_DIR):
    """Every component with realistic inputs, ready to be called in a loop.

    Returns (components, context). A component is
    `(name, per, calls_per_frame, call, note)`, and `call` is None for one that
    cannot run here.
    """
    dem = search_area_dem()
    clear, faint, decoys, _truth = _stub_targets(dem)

    # Boxes for the stages that only take boxes. Where they came from does not
    # matter to WBF or the tracker — count and geometry do, and these are the
    # scenario's own five contacts, three of them people.
    rgb_clues = yolo11m_stub(seed=0, recall=1.0, fp_per_frame=0.0).detect(
        "frame_0000", list(clear) + list(faint) + list(decoys), CASE)
    lidar_clues = lidar_stub(seed=0, recall=1.0, fp_per_frame=0.0).detect(
        "frame_0000", list(clear) + list(faint), CASE)
    fused = weighted_box_fusion([rgb_clues, lidar_clues])

    # A tracker already at steady state, which is what a sortie spends its time
    # in: tracks confirmed, association doing real work every frame.
    tracker = BoTSORT()
    for i in range(tracker.min_hits + 1):
        confirmed = tracker.update(fused, frame_id=f"warm_{i:04d}",
                                   camera_motion=CAMERA_MOTION)
    tracks_per_frame = max(1, len(confirmed))

    # One track's box, and the range a LiDAR measured for it.
    x1, _y1, x2, y2 = confirmed[0].box
    pixel = ((x1 + x2) / 2.0, y2)
    ranged = (RangeEstimate(confirmed[0].range_m or 400.0,
                            RangeSource.MEASURED_LIDAR, 0.5),)

    picture = run_scenario(load_params(), seed=0, frames=6).picture
    detector, frame, frame_note, reason = detector_and_frame(frames_dir)

    components = [
        ("yolo11m_rgb", "frame", 1,
         None if detector is None else
         (lambda i: detector.detect(f"frame_{i:05d}", frame, CASE)),
         frame_note if detector is not None else f"n/a — {reason}"),
        ("wbf", "frame", 1,
         lambda i: weighted_box_fusion([rgb_clues, lidar_clues]),
         f"{len(rgb_clues)} RGB + {len(lidar_clues)} LiDAR boxes"),
        ("botsort_cmc", "frame", 1,
         lambda i: tracker.update(fused, frame_id=f"frame_{i:05d}",
                                  camera_motion=CAMERA_MOTION),
         f"{len(fused)} detections, {tracks_per_frame} confirmed track(s), CMC on"),
        ("lidar_range_projection", "track", tracks_per_frame,
         lambda i: geolocate(pixel, CAMERA, TELEMETRY, measured_ranges=ranged, dem=dem),
         "measured range: no DEM march attempted"),
        ("dem_ray_march", "track", tracks_per_frame,
         lambda i: geolocate(pixel, CAMERA, TELEMETRY, dem=dem),
         "RGB-only fix: ray marched against the DEM"),
        ("decision_chain", "query", 0,
         lambda i: decide(picture),
         f"Reason-Risk-Recommend-Orchestrate over {len(picture.targets)} target(s)"),
    ]
    return components, {"detector": detector, "tracks_per_frame": tracks_per_frame,
                        "reason": reason}


def pipeline_rows(measured):
    """Add up the frame budget for each airframe. `measured` is {name: row}."""
    rows = []
    for name, parts, note in PIPELINES:
        available = [measured[p] for p in parts if measured.get(p, {}).get("mean_ms") != ""]
        if len(available) != len(parts):
            rows.append({"component": name, "per": "frame", "calls_per_frame": "",
                         "iterations": "", "mean_ms": "", "p95_ms": "", "p99_ms": "",
                         "fps_equivalent": "", "parts": "+".join(parts),
                         "note": "n/a — a component in this pipeline did not run"})
            continue

        def budget(key):
            return sum(row[key] * row["calls_per_frame"] for row in available)

        mean = budget("mean_ms")
        rows.append({
            "component": name,
            "per": "frame",
            "calls_per_frame": "",
            "iterations": "",
            "mean_ms": round(mean, 3),
            "p95_ms": round(budget("p95_ms"), 3),
            "p99_ms": round(budget("p99_ms"), 3),
            "fps_equivalent": round(1000.0 / mean, 2) if mean else "",
            "parts": "+".join(parts),
            "note": note + "; p95/p99 are summed per component, an upper bound",
        })
    return rows


def profile(warmup=WARMUP, iterations=ITERATIONS, frames_dir=FRAMES_DIR):
    """Time every component, then total the two frame budgets."""
    components, context = build_components(frames_dir)

    rows = []
    for name, per, calls, call, note in components:
        if call is None:
            rows.append({"component": name, "per": per, "calls_per_frame": calls,
                         "iterations": 0, "mean_ms": "", "p95_ms": "", "p99_ms": "",
                         "fps_equivalent": "", "parts": "", "note": note})
            continue
        samples = time_it(call, warmup, iterations)
        mean = sum(samples) / len(samples)
        rows.append({
            "component": name,
            "per": per,
            "calls_per_frame": calls,
            "iterations": len(samples),
            "mean_ms": round(mean, 4),
            "p95_ms": round(percentile(samples, 95), 4),
            "p99_ms": round(percentile(samples, 99), 4),
            # Blank on purpose. A frame rate is a property of the whole
            # pipeline, and "BoT-SORT: 11,854 FPS" is a true sentence that will
            # be quoted as though the drone could fly at it.
            "fps_equivalent": "",
            "parts": "",
            "note": note,
        })

    rows.extend(pipeline_rows({row["component"]: row for row in rows}))

    # On every row, not in a header: a millisecond is meaningless without the
    # hardware that produced it, and a chart or a quoted figure gets separated
    # from this file the first time anyone copies a number out of it.
    hardware = device_line()
    for row in rows:
        row["device"] = hardware
    return rows, context


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def device_line():
    """What hardware produced these numbers — the first thing to read them by."""
    try:
        import torch
    except ImportError:
        return "unknown (torch not importable)"
    if torch.cuda.is_available():
        return f"cuda — {torch.cuda.get_device_name(0)}, torch {torch.__version__}"
    return f"cpu — {torch.get_num_threads()} thread(s), torch {torch.__version__}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS / "latency_budget.csv")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()

    print(f"\n  latency profile — {args.warmup} warm-up, {args.iterations} measured, "
          f"batch of one")
    rows, context = profile(args.warmup, args.iterations, args.frames_dir)
    # Read off the rows, not taken before the run: ultralytics caps the torch
    # thread count when it loads, so asking first reports threads that never
    # touched an inference.
    print(f"  device: {rows[0]['device']}")
    if context["reason"]:
        print(f"  detector: n/a — {context['reason']}\n"
              f"            a stub's latency would say nothing about an airframe, "
              f"so it is not substituted")
    else:
        print(f"  detector: real YOLO11m — {RGB_WEIGHTS.name}")
    print()

    fmt = lambda v: "n/a" if v == "" else f"{v:10.3f}"          # noqa: E731
    print(f"  {'component':<24}{'per':<8}{'x/frame':>9}{'mean ms':>11}"
          f"{'p95 ms':>11}{'p99 ms':>11}")
    for row in rows:
        if row["component"].startswith("pipeline_"):
            continue
        calls = row["calls_per_frame"] or "-"
        print(f"  {row['component']:<24}{row['per']:<8}{calls:>9}"
              f"{fmt(row['mean_ms'])}{fmt(row['p95_ms'])}{fmt(row['p99_ms'])}")

    print(f"\n  {'frame budget':<24}{'':<8}{'':>9}{'mean ms':>11}{'p95 ms':>11}{'FPS':>11}")
    for row in rows:
        if not row["component"].startswith("pipeline_"):
            continue
        fps = "n/a" if row["fps_equivalent"] == "" else f"{row['fps_equivalent']:10.2f}"
        print(f"  {row['component']:<24}{'':<8}{'':>9}{fmt(row['mean_ms'])}"
              f"{fmt(row['p95_ms'])}{fps:>11}")
    print("  the decision chain is per operator query, not per frame — not in either budget")

    print(f"\n  wrote {write_csv(rows, args.out)}\n")
    return 0


def selfcheck():
    """The arithmetic, with no sortie and no model: percentiles and budgets."""
    # Nearest-rank percentiles are observed samples, never interpolated.
    ten = [float(v) for v in range(1, 11)]           # 1..10
    assert percentile(ten, 95) == 10.0, "ceil(0.95*10) = 10th of 10"
    assert percentile(ten, 50) == 5.0
    assert percentile([4.0], 99) == 4.0, "one sample is its own p99"
    assert percentile(list(reversed(ten)), 95) == 10.0, "order in must not matter"
    hundred = [float(v) for v in range(1, 101)]
    assert percentile(hundred, 99) == 99.0 and percentile(hundred, 95) == 95.0

    # time_it runs the warm-up outside the measurement and measures the rest.
    seen = []
    samples = time_it(seen.append, warmup=3, iterations=5)
    assert len(samples) == 5 and len(seen) == 8, "warm-up calls are not measured"
    assert seen == list(range(8)), "the iteration number keeps counting through"
    assert all(s >= 0 for s in samples)

    # A budget is per-frame calls x per-call cost, so a per-track component
    # priced once per track counts once per track.
    measured = {
        "yolo11m_rgb": {"mean_ms": 200.0, "p95_ms": 240.0, "p99_ms": 260.0,
                        "calls_per_frame": 1},
        "wbf": {"mean_ms": 0.5, "p95_ms": 0.8, "p99_ms": 1.0, "calls_per_frame": 1},
        "botsort_cmc": {"mean_ms": 1.0, "p95_ms": 1.4, "p99_ms": 2.0,
                        "calls_per_frame": 1},
        "lidar_range_projection": {"mean_ms": 0.25, "p95_ms": 0.3, "p99_ms": 0.4,
                                   "calls_per_frame": 4},
        "dem_ray_march": {"mean_ms": 3.0, "p95_ms": 3.6, "p99_ms": 4.0,
                          "calls_per_frame": 4},
    }
    rows = {row["component"]: row for row in pipeline_rows(measured)}
    both = rows["pipeline_rgb_lidar"]
    assert both["mean_ms"] == 202.5, both          # 200 + 0.5 + 1 + 4x0.25
    assert both["fps_equivalent"] == round(1000 / 202.5, 2)
    assert both["p95_ms"] == 240.0 + 0.8 + 1.4 + 4 * 0.3

    # The single-sensor airframe pays no WBF and marches the DEM per track.
    single = rows["pipeline_rgb_only"]
    assert single["mean_ms"] == 213.0, single      # 200 + 1 + 4x3
    assert single["mean_ms"] > both["mean_ms"], "the DEM march is the expensive fix"
    assert "wbf" not in single["parts"], "one feed has nothing to fuse"

    # A component that could not run makes the pipeline n/a — never a total
    # that quietly left something out.
    missing = dict(measured)
    missing["yolo11m_rgb"] = {"mean_ms": "", "p95_ms": "", "p99_ms": "",
                              "calls_per_frame": 1}
    degraded = {row["component"]: row for row in pipeline_rows(missing)}
    assert degraded["pipeline_rgb_lidar"]["mean_ms"] == "", degraded
    assert degraded["pipeline_rgb_lidar"]["fps_equivalent"] == ""
    assert "n/a" in degraded["pipeline_rgb_lidar"]["note"]

    # Every pipeline names components this file actually profiles.
    profiled = {"yolo11m_rgb", "wbf", "botsort_cmc", "lidar_range_projection",
                "dem_ray_march", "decision_chain"}
    for _name, parts, _note in PIPELINES:
        assert set(parts) <= profiled, parts
        assert "decision_chain" not in parts, "per query, not per frame"

    print("  ok  percentiles are observed samples, and independent of input order")
    print("  ok  warm-up calls run but are not measured")
    print("  ok  a frame budget charges a per-track component once per track")
    print("  ok  the single-sensor airframe skips WBF and pays for the DEM march")
    print("  ok  a component that could not run makes the whole budget n/a")
    print("  ok  no pipeline budget includes the per-query decision chain")
    print("\n6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
