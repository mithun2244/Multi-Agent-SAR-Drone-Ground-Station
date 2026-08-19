"""The two measurement engines the experiment runners share.

**detection** — one repeatable sortie through the perception pipeline
(`tuning/scenario.py`): stub detectors, weighted box fusion, BoT-SORT,
geolocation, coordinator fusion. Scored by the Phase 1 harness for mAP and
recall at a fixed false-alarm rate, and by the Phase 8 critic for geolocation
residual against where the subjects actually were. This is the engine that sees
WBF, CMC and a missing sensor.

**command** — the full command plane (`coordinator/demo.py`): the same
perception chain, then the router, the four data-gathering agents, fusion, and
the decision chain. Scored by the critic against known subject positions, plus
what the chain itself produced. This is the engine that sees a disabled agent or
an ablated decision chain.

Neither engine sees everything, and pretending otherwise is the trap. A switch
the engine never reaches produces a row identical to the baseline, which reads
as "this component does not matter" when it means "this experiment could not
tell". `measures` in the config says which is which, and the runner marks any
cell it did not actually measure as `n/a` rather than copying a number across.

Nothing here reads the environment: switches are picked up by the components
themselves as they are constructed, which is why the runner spawns one process
per configuration.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.critic.critic import Critic                                  # noqa: E402
from src.critic.outcomes import CaseOutcome, Subject                  # noqa: E402
from src.evaluation.metrics import evaluate as evaluate_detections    # noqa: E402
from src.geometry import ground_distance_m                            # noqa: E402
from src.tuning.params import load_params                             # noqa: E402
from src.tuning.scenario import run as run_scenario                   # noqa: E402

# Every metric the runners know how to report, and which way is better. The CSV
# columns and the printed table are both built from this, so a metric added here
# appears in both or neither.
METRICS = (
    ("mAP", "higher"),
    ("recall_at_far", "higher"),
    ("geolocation_error_m", "lower"),
    ("subject_recall", "higher"),
    ("ndcg", "higher"),
    ("critic_loss", "lower"),
    ("risk_score", None),
    ("targets", None),
    ("chain_consistent", None),
    ("action", None),
    ("clues_published", "higher"),
    ("tracked_targets", "higher"),
    ("flight_fix_error_m", "lower"),
)


def detection_metrics(seed=0, frames=6):
    """One sortie through the perception pipeline, scored by the Phase 1 harness."""
    params = load_params()
    result = run_scenario(params, seed=seed, frames=frames)
    detections, ground_truth = result.evaluation_inputs()
    harness = evaluate_detections(detections, ground_truth, n_frames=result.frames,
                                  far_target=params.target_far)

    critic = Critic(urgency_weight=params.urgency_weight,
                    sector_weight=params.sector_weight)
    report = critic.evaluate(result.picture, result.outcome)

    return {
        "mAP": round(harness["mAP"], 4),
        "recall_at_far": round(harness["recall_at_far"], 4),
        "geolocation_error_m": _round(report.geolocation.get("median_m"), 2),
        "targets": len(result.picture.targets),
    }


def command_metrics(seed=None, frames=6):
    """The full command plane: agents, fusion, and the decision chain.

    Scored against where the demo's subjects actually are — the same resolution
    `critic/demo.py` uses, so an ablation here is comparable with the number
    that demo prints.
    """
    from src.coordinator.demo import SUBJECTS, build_case

    wiring = build_case(seed=seed)
    dispatch = wiring.orchestrator.handle("full briefing", wiring.case.case_id, frames=frames)
    picture = wiring.fusion.picture(wiring.case.case_id)

    now = datetime.now(timezone.utc)
    outcome = CaseOutcome(
        case_id=wiring.case.case_id,
        resolved_at=now,
        subjects=(
            Subject("hiker-1", SUBJECTS[0][0], SUBJECTS[0][1], priority=1.0, found_at=now),
            Subject("hiker-2", SUBJECTS[1][0], SUBJECTS[1][1], priority=3.0,
                    notes="hypothermic, needed evacuation"),
        ),
    )
    report = Critic().evaluate(picture, outcome)

    # The decision chain's own output. A flat report has no risk and no action —
    # reported as None rather than 0, which would read as "no danger found".
    brief = dispatch.brief
    risk = getattr(brief, "risk", None)
    recommendation = getattr(brief, "recommendation", None)

    located = [t for t in picture.targets if t.located]
    residuals = [min(ground_distance_m(t.position, s.position) for s in outcome.subjects)
                 for t in located]

    return {
        "subject_recall": _round(report.recall, 4),
        "ndcg": _round(report.ndcg, 4),
        "critic_loss": _round(report.loss, 4),
        "geolocation_error_m": _round(_median(residuals), 2),
        "targets": len(picture.targets),
        "risk_score": None if risk is None else risk.score,
        "action": "n/a (chain ablated)" if recommendation is None else recommendation.action,
        "chain_consistent": None if not hasattr(brief, "consistent") else brief.consistent,
        "agents_dispatched": ",".join(dispatch.agents),
    }


def flight_metrics(seed=None, frames=8):
    """A moving airframe: the only engine that supplies camera motion.

    `tuning/scenario.py` flies a *static* pose, so nothing there ever calls the
    compensation and ablating it changes nothing — a row that would read as
    "camera-motion compensation does not matter" when the experiment simply
    could not see it. The mock drone orbits, which is what CMC exists for.
    """
    from src.coordinator.mock_drone_publisher import SUBJECT, _fly

    agent, fusion, case_id = _fly(ticks=frames, use_gmc=True, quiet=True,
                                  **({} if seed is None else {"seed": seed}))
    # refresh, not picture: the flight published to the bus and nothing has
    # absorbed it yet, so asking for the picture first would report zero targets
    # for every configuration and call it a result.
    picture = fusion.refresh(case_id)
    located = [t for t in picture.targets if t.located]
    residuals = [ground_distance_m(t.position, SUBJECT) for t in located]

    return {
        "clues_published": agent.published,
        "tracked_targets": len(picture.targets),
        "flight_fix_error_m": _round(_median(residuals), 2),
    }


def measure(engines, seed=0, frames=6):
    """Run whichever engines this configuration can be seen by."""
    out = {}
    if "detection" in engines:
        out.update(detection_metrics(seed=seed, frames=frames))
    if "flight" in engines:
        out.update(flight_metrics(seed=None, frames=8))
    if "command" in engines:
        # The command engine's own detection metrics are the better ones when
        # both ran: same pipeline, but scored on the picture an operator sees.
        out.update(command_metrics(seed=seed, frames=frames))
    return out


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _round(value, places):
    return None if value is None else round(value, places)
