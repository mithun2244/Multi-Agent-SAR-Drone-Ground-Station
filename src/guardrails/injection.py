"""Injection tells, and the guard on operator commands.

Two untrusted texts reach this system: a witness statement (quoted into a model
prompt) and an operator command (which steers dispatch). They trip on the same
phrases, so the pattern list lives here and both callers import it rather than
keeping a copy each.

The command guard is deliberately lightweight. An operator command cannot write
to the blackboard — it only chooses which agents run — so the damage an injected
one can do is to *narrow* the search: "ignore the ridge, stand down". The guard
therefore never drops the command and never obeys it either. It flags it, the
caller widens to the full agent set, and the attempt is logged. Refusing outright
would be worse: a real operator typing "stand down the north sector" would get
silence in the middle of a search.

ponytail: keyword patterns, same ceiling as before — a paraphrase gets through.
The upgrade is a classifier, and the failure direction is safe either way,
because a missed flag still routes the query normally and a false flag only
costs the extra agents of a full briefing.
"""

import re
from dataclasses import dataclass

from .audit import COMMAND_FLAGGED

_INJECTION_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bignore (all |any )?(previous|prior|above)\b",
    r"\bdisregard (the |all )?(previous|prior|above|instructions)\b",
    r"\byou are now\b",
    r"\bsystem\s*:",
    r"\bnew instructions?\b",
    r"\boverride\b.*\b(instruction|rule|safety)\b",
    r"\bstand down\b",
    r"\bcall off the search\b",
))


def looks_like_injection(text):
    """Which injection tells the text trips, if any."""
    return [p.pattern for p in _INJECTION_PATTERNS if p.search(text or "")]


@dataclass(frozen=True)
class CommandCheck:
    text: str
    flags: tuple = ()

    @property
    def safe(self):
        return not self.flags


def guard_command(text, audit=None, case_id=None):
    """Check an operator command before it is allowed to steer a dispatch."""
    flags = tuple(looks_like_injection(text))
    if flags and audit is not None:
        audit.record(
            COMMAND_FLAGGED,
            "operator command carries injection tells",
            f"{len(flags)} tell(s) in {(text or '')[:120]!r}; widening instead of obeying",
            case_id=case_id,
        )
    return CommandCheck(text=text or "", flags=flags)
