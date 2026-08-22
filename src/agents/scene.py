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

Level 2: person *and* environment
---------------------------------
The model is asked about the subject's state and about the ground around them —
terrain, hazards, access difficulty — because a team walking in needs to know
what they are walking into. It gets two images for that: a crop of the detection
box, where a subject who is forty pixels tall in the frame fills the picture,
and the whole frame behind it for context. The prompt names the ~3x region
around the box as well, so the model knows which part of the wide image is the
immediate surroundings.

The crop is cut in-process from the frame the ledger already verified, so it
needs no capture record of its own — it is the same pixels, and nothing between
the check and the crop can substitute them. Two things it degrades to rather
than failing: no decoder installed, or an endpoint that will not take a second
inline image, both of which send the frame alone.
"""

import io
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

# How far around the detection counts as "the immediate surroundings", as a
# multiple of the box. 3x is roughly the 50 m the prompt asks about at the demo
# camera's altitude and focal length.
CONTEXT_FACTOR = 3.0

# Under this many pixels a side there is nothing in the crop to look at, and the
# second image is payload for no information.
MIN_CROP_PX = 8

TWO_IMAGES = ("You are given two images: first a crop of that box on its own, "
              "then the whole frame it was cut from.")
ONE_IMAGE = "You are given the whole frame."

PROMPT = """You are analysing aerial imagery from a search-and-rescue drone.
An automatic detector has already confirmed a {label} in this frame, inside the
box {box} (pixel coordinates x1, y1, x2, y2). {images}
Within the frame, that box is the subject, the region {region} around it is
their immediate surroundings, and the rest is the wider context. Read all three.

PERSON STATE
- Posture: standing, sitting, lying, crouching, or unclear
- Movement: moving, stationary, or unclear
- Visible condition: any sign of injury, distress or incapacitation
- Clothing or gear visible

ENVIRONMENT (within roughly 50 m of the person)
- Terrain type: open field, forest, rocky slope, water edge, road, building
- Ground conditions: dry, wet, muddy, snow-covered, flooded
- Nearby hazards: water, cliff edges, dense vegetation, structures, power lines, vehicles
- Access difficulty: easy (open ground), moderate (vegetation or slope), difficult (cliff, water, dense forest)
- Visibility: clear, hazy, low light, shadows

IMMEDIATE RISKS — list only those actually present: drowning risk (near water),
fall risk (near a cliff or steep slope), exposure risk (no shelter visible),
entrapment risk (near a collapsed structure), and any other hazard you can see.

Be factual and specific. Do not speculate beyond what is visible. Where
something is unclear, say "unclear" rather than guessing.

Reply with JSON only, no prose outside it:

{{"description": "one or two sentences",
  "person_state": "posture, movement, visible condition, clothing",
  "terrain": "short phrase",
  "environment": "terrain, ground conditions and access, one or two sentences",
  "visibility": "short phrase",
  "hazards": ["short phrase", "..."],
  "immediate_risks": ["drowning risk", "..."],
  "access_difficulty": "easy | moderate | difficult",
  "subject_state": "short phrase or null"}}

Only list hazards and risks you can actually see. Empty lists are good answers."""


class SceneAgent:
    """Describes the surroundings of confirmed detections, once per frame."""

    def __init__(self, bus, case_id, describe=None, image_loader=None, stream=None,
                 frame_size=None, send_crop=True):
        self.bus = bus
        self.case_id = case_id
        self.describe = describe
        self.image_loader = image_loader
        self.stream = stream
        # (width, height) of the frames this sortie produces, when known. Only
        # used to keep the context region inside the image; unset, it is still
        # clamped at zero and may run past the far edge.
        self.frame_size = frame_size
        # Cleared for the rest of the sortie the first time the endpoint refuses
        # a second image, so one unsupported model costs one retry, not one per
        # frame. Pass False to never crop.
        self.send_crop = send_crop

        self.described = set()      # frames already looked at
        self.api_calls = 0
        self.skipped_unconfirmed = 0
        self.skipped_already_seen = 0
        self.skipped_no_image = 0
        self.failures = 0
        self.crops_sent = 0
        self.crops_dropped = 0      # cut, then refused by the endpoint
        self.crops_unavailable = 0  # no decoder, or nothing worth cropping

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
                reply, images = self._ask(trigger, image)
                self.api_calls += 1
            except LLMUnavailable:
                self.failures += 1
                continue

            clue = self.build_clue(frame_id, trigger, reply, images=images)
            self.bus.publish(clue, stream=self.stream)
            published.append(clue)
        return published

    def _ask(self, trigger, image):
        """Ask the VLM about the crop and the frame. Returns (reply, images).

        Falls back to the frame alone when the endpoint will not take two inline
        images — NIM's vision models differ on this, and it is not something the
        suite can settle offline. Losing the crop costs detail; treating it as a
        failure would cost the description, which is the thing worth having.
        """
        crop = crop_region(image, _box_of(trigger)) if self.send_crop else None
        if crop is None:
            if self.send_crop:
                self.crops_unavailable += 1
            return self.describe(_prompt_for(trigger, self.frame_size, cropped=False),
                                 image=image), 1

        try:
            reply = self.describe(_prompt_for(trigger, self.frame_size, cropped=True),
                                  image=[crop, image])
            self.crops_sent += 1
            return reply, 2
        except LLMUnavailable:
            # Raises again if the model is simply unreachable, which is not a
            # verdict on the second image and must not switch cropping off.
            reply = self.describe(_prompt_for(trigger, self.frame_size, cropped=False),
                                  image=image)
            self.crops_dropped += 1
            self.send_crop = False
            return reply, 1

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

    def build_clue(self, frame_id, trigger, reply, images=1):
        result = parse_reply(reply, SceneReply)
        parsed = result.value
        # A reply that failed the schema twice is kept as prose rather than
        # discarded: the words are still the model's account of the frame, they
        # just carry no structure worth trusting.
        description = (parsed.description if parsed else str(reply).strip())
        hazards = list(parsed.hazards) if parsed else []
        # Two names for the same observation: the Level 2 prompt asks for
        # `person_state`, the older one asked for `subject_state`, and fusion
        # reads the latter. Whichever the model filled is the one it saw.
        subject_state = (parsed.subject_state or parsed.person_state) if parsed else None

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
                "subject_state": subject_state,
                "person_state": parsed.person_state if parsed else None,
                "environment": parsed.environment if parsed else None,
                "immediate_risks": list(parsed.immediate_risks) if parsed else [],
                "access_difficulty": parsed.access_difficulty if parsed else None,
                "context_region": context_region(_box_of(trigger), self.frame_size),
                # What the model was actually shown, so a thin description can
                # be read against whether it got the crop.
                "images_sent": images,
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


def context_region(box, frame_size=None, factor=CONTEXT_FACTOR):
    """The detection box grown `factor`x about its centre, clipped to the frame.

    The environmental half of the prompt is about what surrounds the subject,
    not the subject, so the model is told which part of the frame that is. This
    one stays coordinates rather than pixels: the subject gets the crop, and a
    third image of the middle distance would be payload for what the frame the
    model already has can answer.
    """
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    grow_x = (x2 - x1) * (factor - 1.0) / 2.0
    grow_y = (y2 - y1) * (factor - 1.0) / 2.0
    width, height = frame_size if frame_size else (None, None)
    return [
        round(max(0.0, x1 - grow_x)),
        round(max(0.0, y1 - grow_y)),
        round(x2 + grow_x if width is None else min(float(width), x2 + grow_x)),
        round(y2 + grow_y if height is None else min(float(height), y2 + grow_y)),
    ]


def crop_region(image, box, min_px=MIN_CROP_PX):
    """The detection box cut out of the frame, encoded as the frame was.

    A subject forty pixels tall in a 1280-wide frame survives a vision model's
    downscale badly. The crop is the same pixels at the same scale, but they are
    the whole picture, which is the point.

    Returns None — never raises, and never a partial image — when there is no
    decoder installed, no usable box, or nothing big enough to look at. Every
    caller treats that as "send the frame alone", so a missing Pillow degrades
    the description rather than the sortie.
    """
    if not image or not box or len(box) != 4:
        return None
    try:
        from PIL import Image      # noqa: PLC0415 — optional, see the module docstring
    except ImportError:
        return None

    buffer = io.BytesIO()
    try:
        with Image.open(io.BytesIO(image)) as frame:
            x1, y1, x2, y2 = (round(float(v)) for v in box)
            # Clip to the frame: a detector box can overhang the edge, and PIL
            # would pad the overhang with black rather than refuse.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.width, x2), min(frame.height, y2)
            if x2 - x1 < min_px or y2 - y1 < min_px:
                return None
            # Re-encoded as it arrived, so both images in the request are the
            # one mime type the caller declared.
            frame.crop((x1, y1, x2, y2)).convert("RGB").save(
                buffer, format=frame.format or "JPEG")
    except (OSError, ValueError, KeyError):
        # Not an image, truncated, or an encoding PIL will not read or write.
        # The frame is still whatever the ledger verified; only the crop is lost.
        return None
    return buffer.getvalue()


def _box_of(trigger):
    return trigger.spatial_context.bounding_box if trigger.spatial_context else None


def _prompt_for(trigger, frame_size=None, cropped=False):
    box = _box_of(trigger)
    region = context_region(box, frame_size)
    return PROMPT.format(
        label=trigger.class_label or "subject",
        box=[round(v) for v in box] if box else "unknown",
        region=region if region else "the whole frame",
        images=TWO_IMAGES if cropped else ONE_IMAGE,
    )
