# Phase 0 — Clue Contract

The typed interface every agent emits against. Fixing this first is what makes the
vertical slices work: if it is wrong, each agent ends up with its own output shape
and fusion becomes impossible (see [architecture.md](architecture.md), Phase 0).

Producers publish `ClueContract` instances to the Redis Streams bus; the case
blackboard stores them and Coordinator fusion consumes them.

## Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class AgentSource(str, Enum):
    DRONE_RGB = "DRONE_RGB"
    DRONE_LIDAR = "DRONE_LIDAR"
    PERCEPTION_FUSION = "PERCEPTION_FUSION"
    WEATHER_API = "WEATHER_API"
    PATH_MODEL = "PATH_MODEL"
    SCENE_VLM = "SCENE_VLM"
    HEALTH_LLM = "HEALTH_LLM"
    HISTORY_RAG = "HISTORY_RAG"
    INTERVIEW_LLM = "INTERVIEW_LLM"


class SpatialContext(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    altitude_m: Optional[float] = Field(None, ge=-1000.0, le=20000.0)
    bounding_box: Optional[List[float]] = Field(
        None, min_length=4, max_length=4, description="[x_min, y_min, x_max, y_max]"
    )


class ClueContract(BaseModel):
    clue_id: str = Field(..., description="Unique UUID for this observation")
    case_id: str = Field(..., description="Case this clue belongs to, minted at case open")
    parent_clue_ids: Optional[List[str]] = Field(None, description="clue_ids this clue was merged from, e.g. the RGB and LiDAR detections combined by Weighted Box Fusion")
    timestamp: datetime = Field(..., description="Observation timestamp ISO 8601")
    source_agent: AgentSource = Field(..., description="Agent emitting this clue")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Certainty score 0.0-1.0")
    finding_summary: str = Field(..., description="Human readable summary")
    spatial_context: Optional[SpatialContext] = Field(None)
    frame_id: Optional[str] = Field(None, description="Source frame; set by detection producers, None for agents that do not observe frames")
    class_label: Optional[str] = Field(None, description="Detected class, e.g. 'person'; None for non-detection agents")
    provenance_tag: str = Field(..., description="Origin tag, checked against the provenance allow-list")
    agent_metadata: Dict[str, Any] = Field(default_factory=dict)
```

`AgentSource` covers the perception producers and every ground-side agent. Two of
them, `HISTORY_RAG` and `INTERVIEW_LLM`, are out of the active pipeline
(docs/architecture.md, Phase 5) and stay in the enum: a source that has ever
published is part of the schema's history, and removing it would make an archived
clue unreadable.
`agent_metadata` is the per-agent escape hatch — anything not common to all
producers (track IDs, LiDAR range, model version, retrieval hits) goes there
rather than widening the shared contract.

The schema lives in [`src/contracts/clue.py`](../src/contracts/clue.py); the block
above mirrors it. Change the module, not this document.

## Detection fields

`frame_id` and `class_label` are first-class but **optional**, and the asymmetry is
deliberate. Phase 2 has two detection producers — `DRONE_RGB` and `DRONE_LIDAR` —
whose clues get merged by Weighted Box Fusion, and fusion can only pair detections
that agree on the frame they came from. A typed field gives both detectors one
spelling of that key instead of two conventions that drift apart. Meanwhile a
weather forecast observes no frame and detects no class, so `WEATHER_API` (Phase 4)
and the other data-gathering agents simply leave both `None` and validate cleanly.

| Field | Set by | Meaning |
|---|---|---|
| `frame_id` | detection producers | Frame the observation came from. Fusion pairs on it; the evaluator matches per frame. |
| `class_label` | detection producers | Detected class (`"person"`). Consumers fall back to `source_agent` when absent. |

Optional on the bus does not mean optional everywhere. A consumer that genuinely
needs one enforces it at its own boundary — see below.

`agent_metadata` remains the escape hatch for anything not common to producers:
track IDs, LiDAR range, model version, retrieval hits.

## Lineage

`parent_clue_ids` is `None` for a raw observation and holds the merged clues'
ids for a derived one. Weighted Box Fusion (Phase 2) is the first producer of
derived clues: an RGB detection and a LiDAR detection of the same target become
one fused clue naming both parents, so "what did the drone actually see" is
still answerable after fusion has collapsed them.

The critic (Phase 8) needs this to attribute a wrong claim back to the sensor
that made it, and the operator needs it to ask why a target is on the map.
Fused clue ids are derived deterministically from the sorted parent ids, so
re-fusing the same detections cannot create duplicates on the blackboard.

## Relationship to the evaluation harness

`src/evaluation/` consumes clues, not a parallel schema. `Detection.from_clue()`
projects a `ClueContract` — or the JSON form that arrives off the bus — onto the
five fields scoring needs:

| Detection | Contract |
|---|---|
| `frame_id` | `frame_id` |
| `label` | `class_label`, falling back to `source_agent` |
| `box` | `spatial_context.bounding_box` |
| `confidence` | `confidence_score` |
| `geo` | `spatial_context.latitude` / `longitude` |

The projection is a trust boundary. A clue with no bounding box, a malformed one,
or no `frame_id` is rejected loudly rather than silently dropped, since a dropped
clue would deflate recall and hide a broken producer. This is where "optional on
the bus, required here" is enforced: the contract lets Weather omit `frame_id`,
and the evaluator refuses to score a detection that lacks it.

`case_id` scopes a run to one search — `--case-id` filters out clues belonging to
other cases.

## `case_id` and `provenance_tag`

Both are required, not optional. That is the point of each:

- **`case_id`** is minted by the case lifecycle at case open and carried through
  everything downstream. Fusion scopes the current picture by it, so a clue that
  cannot name its case cannot be fused. Required means a producer physically
  cannot emit an unattributable clue.
- **`provenance_tag`** is what the provenance allow-list guard checks (Phase 5,
  hardened in Phase 7). A security field that defaults to absent gives the guard
  nothing to check, so it carries no default. The schema guarantees the tag is
  *present*; it does not decide whether the tag is *trusted* — the allow-list is
  runtime data owned by the guard, which is why this stays a plain `str` rather
  than a second enum that would need a schema change per new source.
