"""Phase 8 end to end: score a real picture against a resolved outcome.

    python -m src.critic.demo

Runs the full pipeline for a case, resolves it against where the subjects
actually were, and prints what the critic makes of the result — including which
agent's contribution earned its place in the ranking.
"""

from datetime import datetime, timedelta, timezone

from ..coordinator.demo import SUBJECTS
from ..coordinator.demo import build_case as build_wiring
from ..geometry import ground_distance_m
from .critic import Critic
from .outcomes import CaseOutcome, OutcomeLog, Subject


def main():
    wiring = build_wiring()
    case = wiring.case

    # Fly a sortie and take the picture the operator would be working from.
    wiring.orchestrator.handle("full briefing", case.case_id, frames=6)
    picture = wiring.fusion.picture(case.case_id)

    # The search resolves: both subjects are found, one of them a casualty.
    outcome = CaseOutcome(
        case_id=case.case_id,
        resolved_at=datetime.now(timezone.utc),
        subjects=(
            Subject("hiker-1", SUBJECTS[0][0], SUBJECTS[0][1], priority=1.0,
                    found_at=datetime.now(timezone.utc)),
            Subject("hiker-2", SUBJECTS[1][0], SUBJECTS[1][1], priority=3.0,
                    notes="hypothermic, needed evacuation"),
        ),
    )
    log = OutcomeLog()
    log.record(outcome)

    print(picture.render())
    print("  resolved outcome")
    for subject in outcome.subjects:
        print(f"    {subject.subject_id}: {subject.latitude:.6f}, {subject.longitude:.6f}"
              f"  priority {subject.priority}"
              + (f"  ({subject.notes})" if subject.notes else ""))

    critic = Critic()
    report = critic.evaluate(picture, log.get(case.case_id))
    print(report.render())

    for rank, target in enumerate(picture.targets, 1):
        nearest = min(outcome.found_subjects,
                      key=lambda s: ground_distance_m(target.position, s.position))
        distance = ground_distance_m(target.position, nearest.position)
        print(f"    rank {rank}: {target.target_id} is {distance:5.1f} m from "
              f"{nearest.subject_id}")

    print(f"\n  campaign so far: {critic.summary()}\n")


if __name__ == "__main__":
    main()
