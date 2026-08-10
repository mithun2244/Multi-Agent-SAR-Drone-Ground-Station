"""The Scene Description agent — a VLM, but only where it earns its cost.

Architecture, Phase 5: "Scene Description — VLM, triggered only on frames where
detection already confirmed something."

The gate is the design, not an optimisation. A drone flies thousands of frames
of empty hillside per sortie; describing them would cost a fortune and tell
nobody anything. This agent looks only at frames where the perception plane
already published a *confirmed* track, and looks at each such frame once.

What it is not
--------------
Scene descriptions are **not** independent corroboration. This agent only runs
because detection already fired, so its agreement is guaranteed and worth
nothing as evidence. It is an annotation source: it enriches a target's profile
and can flag hazards that change how fast a team must get there. Coordinator
fusion treats it that way deliberately — see `CONTEXT`/`ANNOTATION` sources
there.

It also never describes a frame it could not actually see. No image, no clue.
"""

import uuid
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.parsers import parse_reply
from ..guardrails.provenance import TAG_SCENE
from ..guardrails.schemas import SceneReply
from .llm import DEFAULT_VLM_MODEL, LLMUnavailable

# Names the component, not the model behind it: swapping models must not
# invalidate every clue this agent has ever published.
SCENE_PROVENANCE = TAG_SCENE
CONFIRMED = "CONFIRMED"

PROMPT = """You are looking at one frame from a search-and-rescue drone.
An automatic detector has already confirmed a {label} in this frame, inside the
box {box} (pixel coordinates x1, y1, x2, y2).

Describe what is around that detection so a rescue team knows what they are
walking into. Reply with JSON only, no prose outside it:

{{"description": "one or two sentences",
  "terrain": "short phrase",
  "visibility": "short phrase",
  "hazards": ["short phrase", "..."],
  "subject_state": "short phrase or null"}}

Only list hazards you can actually see. An empty list is a good answer."""


class SceneAgent:
    """Describes the surroundings of confirmed detections, once per frame."""

    def __init__(self, bus, case_id, describe=None, image_loader=None, stream=None):
        self.bus = bus
        self.case_id = case_id
        self.describe = describe
        self.image_loader = image_loader
        self.stream = stream

        self.described = set()      # frames already looked at
        self.api_calls = 0
        self.skipped_unconfirmed = 0
        self.skipped_already_seen = 0
        self.skipped_no_image = 0
        self.failures = 0

    def process(self, clues):
        """Look at the frames worth looking at. Returns the clues published."""
        published = []
        for frame_id, trigger in self._frames_to_describe(clues):
            image = self.image_loader(frame_id) if self.image_loader else None
            if image is None:
                # Never describe a frame we could not see. A VLM asked about an
                # image it was not given will happily write something anyway.
                self.skipped_no_image += 1
                continue

            self.described.add(frame_id)
            try:
                reply = self.describe(_prompt_for(trigger), image=image)
                self.api_calls += 1
            except LLMUnavailable:
                self.failures += 1
                continue

            clue = self.build_clue(frame_id, trigger, reply)
            self.bus.publish(clue, stream=self.stream)
            published.append(clue)
        return published

    def _frames_to_describe(self, clues):
        """One trigger per frame: the highest-confidence confirmed detection.

        Everything else is counted and dropped, so the cost of a sortie is one
        call per frame that actually contained somebody.
        """
        best = {}
        for clue in clues:
            if not _is_confirmed_detection(clue):
                self.skipped_unconfirmed += 1
                continue
            if clue.frame_id in self.described:
                self.skipped_already_seen += 1
                continue
            current = best.get(clue.frame_id)
            if current is None or clue.confidence_score > current.confidence_score:
                best[clue.frame_id] = clue
        return sorted(best.items())

    def build_clue(self, frame_id, trigger, reply):
        result = parse_reply(reply, SceneReply)
        parsed = result.value
        # A reply that failed the schema twice is kept as prose rather than
        # discarded: the words are still the model's account of the frame, they
        # just carry no structure worth trusting.
        description = (parsed.description if parsed else str(reply).strip())
        hazards = list(parsed.hazards) if parsed else []

        spatial = trigger.spatial_context
        return ClueContract(
            clue_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"sar:scene:{self.case_id}:{frame_id}"
            )),
            case_id=self.case_id,
            # The frame's own time, not now: this describes that moment.
            timestamp=trigger.timestamp or datetime.now(timezone.utc),
            source_agent=AgentSource.SCENE_VLM,
            # Deliberately inherits the detection's confidence rather than
            # asserting its own. The VLM is describing something already
            # believed; it is not a second opinion on whether it is there.
            confidence_score=trigger.confidence_score,
            finding_summary=description[:400] or "Scene described, no detail returned",
            spatial_context=SpatialContext(
                latitude=spatial.latitude if spatial else None,
                longitude=spatial.longitude if spatial else None,
                bounding_box=list(spatial.bounding_box) if spatial and spatial.bounding_box else None,
            ),
            frame_id=frame_id,
            class_label=trigger.class_label,
            provenance_tag=SCENE_PROVENANCE,
            agent_metadata={
                "description": description,
                "terrain": parsed.terrain if parsed else None,
                "visibility": parsed.visibility if parsed else None,
                "hazards": hazards,
                "subject_state": parsed.subject_state if parsed else None,
                "track_id": trigger.agent_metadata.get("track_id"),
                "triggered_by": trigger.clue_id,
                "structured": parsed is not None,
                "parse_outcome": result.outcome,
                "vlm_model": DEFAULT_VLM_MODEL,
            },
        )


def _is_confirmed_detection(clue):
    return (
        clue.source_agent is AgentSource.PERCEPTION_FUSION
        and clue.frame_id is not None
        and clue.agent_metadata.get("track_state") == CONFIRMED
    )


def _prompt_for(trigger):
    box = trigger.spatial_context.bounding_box if trigger.spatial_context else None
    return PROMPT.format(
        label=trigger.class_label or "subject",
        box=[round(v) for v in box] if box else "unknown",
    )
