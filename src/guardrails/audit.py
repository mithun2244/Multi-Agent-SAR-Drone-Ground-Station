"""The security audit log — rejected clues are evidence, not litter.

A clue turned away at the provenance guard is the most interesting thing that
happens all sortie. Dropping it silently would mean an operator never learns
that something tried to write to their blackboard, and a post-incident review
would have nothing to look at.

So every rejection is recorded with what was claimed and why it was refused.
The detail buffer is bounded — a flood must not exhaust memory on a ground
station — but the counters are not, so the *number* of attempts survives even
when the oldest details are dropped. Losing evidence quietly is the failure this
module exists to avoid, so `dropped` says how much detail was lost.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

PROVENANCE_REJECTED = "provenance_rejected"
IMAGE_REJECTED = "image_rejected"
COMMAND_FLAGGED = "command_flagged"


@dataclass(frozen=True)
class SecurityEvent:
    at: datetime
    kind: str
    reason: str
    detail: str = ""
    case_id: str | None = None
    clue_id: str | None = None
    source_agent: str | None = None
    provenance_tag: str | None = None

    def describe(self):
        if self.source_agent is None and self.provenance_tag is None:
            # Not a clue — an operator command, say. Naming an "unknown agent
            # claiming no tag" would invent an attacker that was never there.
            return f"{self.reason}: {self.detail}"
        who = self.source_agent or "unknown agent"
        tag = self.provenance_tag or "no tag"
        return f"{self.reason}: {who} claiming {tag!r} — {self.detail}"


class AuditLog:
    """Bounded record of security events, with unbounded counters."""

    def __init__(self, maxlen=500, clock=None):
        self._events = deque(maxlen=maxlen)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.counts = {}
        self.total = 0
        self.dropped = 0

    def record(self, kind, reason, detail="", case_id=None, clue_id=None,
               source_agent=None, provenance_tag=None):
        event = SecurityEvent(
            at=self.clock(), kind=kind, reason=reason, detail=detail,
            case_id=case_id, clue_id=clue_id, source_agent=source_agent,
            provenance_tag=provenance_tag,
        )
        if len(self._events) == self._events.maxlen:
            self.dropped += 1
        self._events.append(event)
        self.counts[reason] = self.counts.get(reason, 0) + 1
        self.total += 1
        return event

    def reject_clue(self, clue, verdict, case_id=None):
        """Record a clue refused by the provenance guard."""
        return self.record(
            PROVENANCE_REJECTED,
            verdict.reason,
            verdict.detail,
            case_id=case_id or getattr(clue, "case_id", None),
            clue_id=getattr(clue, "clue_id", None),
            source_agent=getattr(getattr(clue, "source_agent", None), "value", None),
            provenance_tag=getattr(clue, "provenance_tag", None),
        )

    @property
    def events(self):
        return list(self._events)

    def for_case(self, case_id):
        return [e for e in self._events if e.case_id == case_id]

    def clear(self):
        self._events.clear()
        self.counts = {}
        self.total = 0
        self.dropped = 0

    def __len__(self):
        return self.total

    def __repr__(self):
        return (f"AuditLog({self.total} event(s), {len(self._events)} retained, "
                f"{self.dropped} detail(s) dropped)")
