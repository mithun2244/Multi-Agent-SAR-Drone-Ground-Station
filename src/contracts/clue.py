"""The Phase 0 clue contract — the typed interface every agent emits against.

Single source of truth for the schema documented in docs/phase0_contract.md.
Producers publish these to the bus; the blackboard stores them; Coordinator
fusion and the evaluation harness consume them.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    # Bounded at the contract, not just where they are used: a clue claiming
    # latitude 999 is not a bad fix, it is a forged payload, and it should be
    # impossible to construct rather than caught by whoever reads it first.
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

    def detection_box(self):
        """This clue's bounding box as an (x_min, y_min, x_max, y_max) tuple.

        Raises ValueError if the clue is not a locatable detection: no box, a
        malformed one, or no frame to place it in. `frame_id` and
        `bounding_box` are optional on the contract so non-detection agents can
        omit them, so every consumer that must locate a clue — fusion,
        evaluation — calls this rather than reaching into spatial_context and
        inventing its own rules.
        """
        box = self.spatial_context.bounding_box if self.spatial_context else None
        if box is None:
            raise ValueError(f"clue {self.clue_id}: no bounding_box to locate")
        if len(box) != 4:
            raise ValueError(
                f"clue {self.clue_id}: bounding_box must be "
                f"[x_min, y_min, x_max, y_max], got {box!r}"
            )
        if self.frame_id is None:
            raise ValueError(f"clue {self.clue_id}: frame_id is required to locate a detection")
        return tuple(box)
