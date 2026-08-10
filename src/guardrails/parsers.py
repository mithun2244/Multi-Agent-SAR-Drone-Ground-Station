"""Strict parsing of model replies, with one repair attempt and then a stop.

The policy is deliberately blunt: parse it, and if that fails, repair the
*syntax* once and parse again. Two failures and the reply is discarded — the
caller gets its safe default and a reason, never a half-populated object.

Repair only ever fixes how a reply is written, never what it says. Fences,
trailing commas, Python literals and prose wrapped around the JSON are all
formatting the model got wrong. Nothing here supplies a missing field: a scene
description invented to satisfy a schema would read exactly like one the model
actually wrote.
"""

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

OK = "ok"
REPAIRED = "repaired"
DISCARDED = "discarded"

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_PY_LITERAL = re.compile(r"(?<![\"\w])(True|False|None)(?![\"\w])")

_LITERALS = {"True": "true", "False": "false", "None": "null"}


@dataclass
class ParseResult:
    """What came back, and how much work it took."""

    value: object = None
    outcome: str = DISCARDED
    error: str | None = None
    attempts: int = 0
    raw: str = ""

    @property
    def ok(self):
        return self.outcome in (OK, REPAIRED)

    @property
    def discarded(self):
        return self.outcome == DISCARDED


def extract_json(text):
    """The outermost JSON object in a reply, fences and prose stripped."""
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return cleaned
    return cleaned[start:end + 1]


def repair_json(text):
    """Fix how a reply is written. Never what it says."""
    repaired = extract_json(text)
    repaired = _TRAILING_COMMA.sub(r"\1", repaired)
    repaired = _PY_LITERAL.sub(lambda m: _LITERALS[m.group(1)], repaired)
    # Single-quoted JSON, but only when there is no double quote to confuse it.
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def parse_reply(text, model, default=None):
    """Validate a model reply against `model`. Two attempts, then give up."""
    attempts = 0
    last_error = None

    for candidate in (extract_json(text), repair_json(text)):
        attempts += 1
        try:
            value = model.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
        return ParseResult(
            value=value,
            outcome=OK if attempts == 1 else REPAIRED,
            attempts=attempts,
            raw=text or "",
        )

    return ParseResult(
        value=default,
        outcome=DISCARDED,
        error=last_error,
        attempts=attempts,
        raw=text or "",
    )


def parse_text(text, model, default=None):
    """Validate a prose reply — no JSON involved, same two-strikes policy.

    The first attempt is the tidied text: fences and a chat preamble are
    formatting nobody wants in an operator's briefing. The second falls back to
    the raw reply, in case tidying took too much.
    """
    attempts = 0
    last_error = None

    for candidate in (_strip_preamble(text), (text or "").strip()):
        attempts += 1
        try:
            value = model.model_validate({"text": candidate})
        except (ValidationError, TypeError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
        return ParseResult(
            value=value,
            outcome=OK if attempts == 1 else REPAIRED,
            attempts=attempts,
            raw=text or "",
        )

    return ParseResult(value=default, outcome=DISCARDED, error=last_error,
                       attempts=attempts, raw=text or "")


# Chat lead-ins only. Deliberately narrow: a general "anything before a colon"
# rule would eat the first line of "Sector 1: work this first", which is content.
_PREAMBLE = re.compile(
    r"^(?:sure|certainly|of course|okay|ok|here(?:'s| is)|below is|this is)\b[^\n]{0,80}?:\s*\n+",
    re.I,
)


def _strip_preamble(text):
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    return _PREAMBLE.sub("", cleaned).strip()
