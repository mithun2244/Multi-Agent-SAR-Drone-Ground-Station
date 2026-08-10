"""Pydantic schemas for what each model is allowed to say back.

Every LLM and VLM reply crosses one of these before it can become a clue. A
model that returns the wrong shape gets rejected here rather than downstream,
where a missing field would surface as `None` in an operator's briefing.

Validators do the small coercions models actually make — a lone string where a
list belongs, a number as text — and nothing more. Repairing *shape* is fair;
inventing *content* is not, so nothing here supplies a default for a field the
model was asked to fill.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


def _as_list(value):
    """A model asked for a list often returns one item, or a comma string."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _blank_to_none(value):
    """Models write "null", "N/A" and "" for "I do not know"."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "null", "none", "n/a", "unknown", "not stated") else text


class SceneReply(BaseModel):
    """What the VLM saw around a confirmed detection."""

    description: str = Field(..., min_length=1, max_length=2000)
    terrain: Optional[str] = None
    visibility: Optional[str] = None
    hazards: List[str] = Field(default_factory=list)
    subject_state: Optional[str] = None

    @field_validator("hazards", mode="before")
    @classmethod
    def _hazards_as_list(cls, value):
        return _as_list(value)

    @field_validator("terrain", "visibility", "subject_state", mode="before")
    @classmethod
    def _optional_text(cls, value):
        return _blank_to_none(value)


class HealthReply(BaseModel):
    """A scaling factor on an already-computed survival window.

    Deliberately unbounded here: the Health agent clamps to its own guardrail
    range and records that it did. Rejecting an out-of-range multiplier at the
    schema would throw away the rationale along with it and lose the audit
    trail of what the model actually said.
    """

    multiplier: float
    rationale: str = ""

    @field_validator("rationale", mode="before")
    @classmethod
    def _rationale_text(cls, value):
        return "" if value is None else str(value).strip()[:400]


class InterviewReply(BaseModel):
    """Named entities pulled from a witness statement."""

    time_last_seen: Optional[str] = None
    clothing: Optional[str] = None
    direction_of_travel: Optional[str] = None
    confidence: float = 0.5

    @field_validator("time_last_seen", "clothing", "direction_of_travel", mode="before")
    @classmethod
    def _optional_text(cls, value):
        return _blank_to_none(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_in_range(cls, value):
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.5


class TextReply(BaseModel):
    """A prose reply — a field briefing or a historical insight.

    Bounded on both ends: an empty completion is a failure dressed as an answer,
    and a runaway one is a model that has stopped following the prompt.
    """

    text: str = Field(..., min_length=10, max_length=4000)

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, value):
        return "" if value is None else str(value).strip()
