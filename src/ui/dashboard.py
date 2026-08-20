"""Ground station — the whole pipeline behind one button.

    streamlit run src/ui/dashboard.py
    python -m src.ui.dashboard --selfcheck   # the wiring, without a browser

Nothing here is a second implementation of the system. The sidebar sets the same
`ABLATION_*` variables the command line sets, `build_case()` from the coordinator
demo stands up the same bus, guards, agents and orchestrator, and the three
columns render what came back. The only new component is the detection handler
for an *uploaded* frame: the demo's handler feeds stubs a set of projected truth
targets, and a photograph does not come with any.

Two honest limits, both stated in the UI rather than papered over:

  * one frame is not a sortie. The uploaded path tracks with `min_hits=1`,
    because a track cannot be confirmed over three frames when there is one.
  * a photograph carries no telemetry. Altitude and pitch are operator inputs
    and the intrinsics are derived from the image size at the demo camera's
    field of view — geolocation is only ever as good as those numbers.

ponytail: `sys.path` shim. `streamlit run` executes this as a script, so the
relative imports the rest of `src/` uses do not resolve. Drop it if the app is
ever launched through `python -m streamlit run` from an installed package.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from PIL import Image, ImageDraw
from redis.exceptions import RedisError

from src.contracts.clue import AgentSource
from src.coordinator.demo import DRONE, DRONE_ID, _search_area_dem, build_case
from src.coordinator.router import ALL_AGENTS
from src.perception.agent import DetectionAgent
from src.perception.geolocation import Camera, Telemetry
from src.guardrails.provenance import TAG_RGB
from src.perception.models import DEFAULT_CONF, RealDetector, build_detectors
from src.perception.tracking import BoTSORT
from src.utils.ablation import CMC, DECISION, DISABLE_AGENTS, OFF, ON, WBF, describe

# The demo camera is 1000 px focal length over a 1280 px frame. Keeping that
# ratio means an uploaded image is treated as the same lens at a different
# resolution, rather than silently as a 1280x720 frame it is not.
FOV_FX_RATIO = 1000.0 / 1280.0
SYNTHETIC_FRAME = (1280, 720)
UPLOAD_FRAME_ID = "upload_0000"


def camera_for(width, height):
    """Intrinsics for a frame of this size, at the demo camera's field of view."""
    focal = width * FOV_FX_RATIO
    return Camera(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0)


@st.cache_resource
def rgb_detector():
    """Real YOLO11m weights, loaded once per session rather than per press."""
    return build_detectors(mode="real", sensors=("rgb",), device_id=DRONE_ID)["rgb"]


def single_frame_tracker(conf):
    """BoT-SORT with every confirmation threshold collapsed onto one number.

    The shipped tracker earns a track over frames: a detection has to clear
    `new_track_thresh` (0.6) to start one and `min_hits` (3) frames to confirm
    it. Both exist because a sortie has later frames to settle the question
    with, and neither can do its job on a single photograph — on a real VisDrone
    frame the detector reported five subjects at 0.25-0.5 and the tracker
    correctly published none of them.

    So on this path the operator's confidence floor is the whole decision: what
    the detector reports at or above it becomes a track, because nothing else is
    coming. The airframe keeps the shipped values (`perception/agent.py`); this
    is a single-frame concession, and the UI says so.
    """
    return BoTSORT(min_hits=1, high_thresh=conf, new_track_thresh=conf)


def uploaded_detection_agent(bus, case_id, image, altitude_m, pitch_deg, conf=DEFAULT_CONF,
                             detector=None):
    """Detection over one uploaded frame: YOLO11m -> BoT-SORT -> geolocation -> bus.

    Replaces the demo's stub handler, which is handed the truth targets that were
    projected into a synthetic frame. A real detector gets the frame and nothing
    else, and `DetectionAgent` does the geolocation exactly as it does on a
    sortie — the only thing that changed is where the boxes came from.
    """
    detector = detector or rgb_detector()
    # ponytail: the cached detector is shared, so two browser sessions moving the
    # slider at once share a threshold. Key the cache on `conf` if this is ever
    # more than one operator at a desk — it costs a model load per value.
    detector.conf = conf
    camera = camera_for(*image.size)
    telemetry = Telemetry(DRONE[0], DRONE[1], altitude_m=altitude_m,
                          yaw_deg=15.0, pitch_deg=pitch_deg)
    agent = DetectionAgent(bus, case_id, camera, dem=_search_area_dem(),
                           tracker=single_frame_tracker(conf), device_id=DRONE_ID)

    def handler(case_id, context):
        found = detector.detect(UPLOAD_FRAME_ID, image, case_id)
        # One feed: no second sensor to reconcile, so WBF does not run here at
        # all — which is what a camera-only airframe does.
        published = agent.process_frame(UPLOAD_FRAME_ID, [found], telemetry)
        return {
            "detections": len(found),
            "clues_published": len(published),
            "without_fix": agent.without_fix,
            "detector": f"YOLO11m {detector.weights.name}",
        }

    return handler


def track_boxes(bus, case_id):
    """The pixel boxes the pipeline actually published, newest state per track."""
    boxes = {}
    for _, clue in bus.read(f"clues:{case_id}"):
        if clue.source_agent is not AgentSource.PERCEPTION_FUSION:
            continue
        box = clue.spatial_context.bounding_box if clue.spatial_context else None
        if box:
            boxes[clue.agent_metadata.get("track_id", "?")] = (
                box, clue.confidence_score, clue.agent_metadata.get("geolocation"))
    return boxes


def draw_boxes(image, boxes):
    """Boxes over the frame. Green located, amber seen but not placed."""
    canvas = image.copy().convert("RGB")
    pen = ImageDraw.Draw(canvas)
    for track_id, (box, confidence, fix) in boxes.items():
        colour = "#22c55e" if fix == "located" else "#f59e0b"
        pen.rectangle([box[0], box[1], box[2], box[3]], outline=colour, width=3)
        pen.text((box[0] + 4, max(0.0, box[1] - 12)),
                 f"track {track_id}  {confidence:.2f}", fill=colour)
    return canvas


def run_pipeline(query, image=None, frames=4, altitude_m=1570.0, pitch_deg=55.0,
                 conf=DEFAULT_CONF):
    """Stand up a case under the current ablation environment and dispatch once.

    The ablation variables are read when the router and the trackers are built,
    so the case is rebuilt per run — a toggle that only took effect on the next
    press would be worse than no toggle.
    """
    wiring = build_case()
    if image is not None:
        wiring.orchestrator.register("detection", uploaded_detection_agent(
            wiring.bus, wiring.case.case_id, image, altitude_m, pitch_deg, conf))
    return wiring, wiring.orchestrator.handle(query, wiring.case.case_id, frames=frames)


# -- the app ---------------------------------------------------------------

def sidebar():
    """Ablation switches, written to the environment the pipeline already reads."""
    st.sidebar.title("Ablations")
    st.sidebar.caption("Off by default is the shipped system. Anything switched "
                       "off is stated in the run header.")
    for label, name in (("Weighted box fusion (WBF)", WBF),
                        ("Camera motion compensation (CMC)", CMC),
                        ("Decision chain", DECISION)):
        os.environ[name] = ON if st.sidebar.toggle(label, value=True, key=name) else OFF

    st.sidebar.subheader("Agents")
    off = [name for name in ALL_AGENTS
           if not st.sidebar.toggle(name, value=True, key=f"agent_{name}")]
    os.environ[DISABLE_AGENTS] = ",".join(off)

    st.sidebar.subheader("Airframe")
    # A photograph carries no telemetry. These are the calibration knobs the fix
    # actually depends on; everything else is the demo camera.
    altitude = st.sidebar.number_input("Altitude (m AMSL)", 1000.0, 4000.0, 1570.0, 10.0)
    pitch = st.sidebar.number_input("Camera pitch (deg below horizon)", 5.0, 89.0, 55.0, 1.0)
    conf = st.sidebar.slider("Detection confidence", 0.05, 0.95, DEFAULT_CONF, 0.05,
                             help="Uploaded frames only. One frame has no later "
                                  "frames to confirm a track over, so this floor "
                                  "is the whole decision — see single_frame_tracker.")
    return altitude, pitch, conf


def vision_column(image, boxes):
    st.subheader("Vision")
    if image is None:
        image = Image.new("RGB", SYNTHETIC_FRAME, "#0b1220")
        st.caption("Synthetic sortie — stub detectors over projected targets, "
                   "no frame to show. Boxes are where they landed in the frame.")
    st.image(draw_boxes(image, boxes), width="stretch")
    st.caption(f"{len(boxes)} confirmed track(s). "
               "Green located, amber seen but not placed.")


def telemetry_column(picture):
    st.subheader("Telemetry")
    if not picture.targets:
        st.info("No targets in the picture.")
        return
    st.dataframe([{
        "target": t.target_id[:8],
        "tracks": ", ".join(sorted(t.track_ids)) or "—",
        "lat": None if t.latitude is None else round(t.latitude, 6),
        "lon": None if t.longitude is None else round(t.longitude, 6),
        "elev m": None if t.elevation_m is None else round(t.elevation_m, 1),
        "range via": t.range_source or "no fix",
        "conf": round(t.confidence, 3),
        "urgency": round(t.urgency, 3),
        "priority": round(t.priority, 3),
    } for t in picture.targets], width="stretch", hide_index=True)

    located = picture.located
    if located:
        st.map([{"lat": t.latitude, "lon": t.longitude} for t in located], size=8)
    if picture.unlocated:
        st.warning(f"{len(picture.unlocated)} target(s) seen but not located — "
                   "no coordinates are invented for them.")
    window = picture.survival_window_hours
    if window is not None:
        st.metric("Survival window", f"{window} h",
                  "hypothermia risk" if picture.hypothermia_risk else "no cold risk",
                  delta_color="inverse" if picture.hypothermia_risk else "normal")


def command_column(brief):
    st.subheader("Command")
    assessment = getattr(brief, "risk", None)
    if assessment is None:
        # ABLATION_DECISION=off: a FlatReport, with no risk score to reach for.
        st.warning("Decision chain ablated — fusion's ranking, nothing reasoned over it.")
        st.code(brief.render())
        return

    st.metric("Risk", f"{assessment.score}/10", assessment.band,
              delta_color="inverse")
    st.markdown(f"**Recommended action**\n\n`{brief.recommendation.action}`")
    st.caption(f"{brief.recommendation.detail}  (protocol: {brief.recommendation.rule})")

    st.markdown("**Reasoning**")
    st.write(brief.facts.summary())
    for why, points in assessment.drivers:
        st.write(f"- {why}  `+{points}`")
    if not assessment.drivers:
        st.write("- baseline only")

    if brief.corrections:
        st.markdown(f"**Orchestrate** — {len(brief.corrections)} correction(s) "
                    f"over {brief.passes} pass(es)")
        for note in brief.corrections:
            st.write(f"- {note}")
    if not brief.consistent:
        st.error("NOT PUBLISHED AS AN ORDER — the chain would not converge, "
                 "commander review required.")


def main():
    st.set_page_config(page_title="SAR Ground Station", layout="wide")
    st.title("SAR Drone Ground Station")
    altitude, pitch, conf = sidebar()
    st.caption(describe())

    upload = st.file_uploader("Test frame", type=["jpg", "jpeg", "png"])
    query = st.text_input("Operator query", "Give me a full briefing")
    left, right = st.columns([1, 3])
    synthetic = left.button("Synthetic stub sortie", width="stretch")
    run = right.button("Run System", type="primary", width="stretch")

    if run or synthetic:
        image = None
        if upload is not None and not synthetic:
            image = Image.open(upload).convert("RGB")
        with st.spinner("Perception -> Router -> Agents -> Decision..."):
            try:
                wiring, dispatch = run_pipeline(
                    query, image=image, frames=0 if image is not None else 4,
                    altitude_m=altitude, pitch_deg=pitch, conf=conf)
            except FileNotFoundError as missing:      # weights, named with a way out
                st.error(str(missing))
                return
            except RedisError as unreachable:
                st.error(f"REDIS_URL={os.environ.get('REDIS_URL')} is not answering "
                         f"({unreachable}). Start the server, or unset REDIS_URL to "
                         f"run against the mocked connection.")
                return
        st.session_state["run"] = (wiring, dispatch, image)

    if "run" not in st.session_state:
        st.info("Upload a frame and press **Run System**, or fly the synthetic "
                "sortie the demo ships with.")
        return

    wiring, dispatch, image = st.session_state["run"]
    st.success(f"routed **{dispatch.scenario.value}** to {list(dispatch.agents)} "
               f"— {dispatch.route.reason}")
    if dispatch.command_flags:
        st.error(f"COMMAND GUARD: {len(dispatch.command_flags)} injection tell(s) — "
                 "widened rather than obeyed.")

    vision, telemetry, command = st.columns(3)
    with vision:
        vision_column(image, track_boxes(wiring.bus, dispatch.case_id))
    with telemetry:
        telemetry_column(dispatch.picture)
    with command:
        command_column(dispatch.brief)

    with st.expander("Agent results, guards and the raw picture"):
        st.json(dispatch.results)
        st.write(f"{len(dispatch.picture.security_events)} security event(s), "
                 f"{len(dispatch.picture.guard_findings)} contradiction(s) with "
                 f"confirmed perception")
        st.code(dispatch.picture.render())


def selfcheck():
    """The wiring, headless: toggles reach the pipeline and the columns get data."""
    offline = ("REDIS_URL", "NVIDIA_API_KEY")
    env = {k: os.environ.get(k)
           for k in (WBF, CMC, DECISION, DISABLE_AGENTS) + offline}
    try:
        # Offline, like every other check in this repository: the mocked
        # connection inside a real RedisBus, and stub completers rather than a
        # third party's uptime, key or quota. Emptied rather than popped —
        # `.env` sets both, and `load_dotenv` fills a key back in on the next
        # import but leaves one that is already present alone.
        for name in offline:
            os.environ[name] = ""
        os.environ[DISABLE_AGENTS] = "path"
        wiring, dispatch = run_pipeline("Give me a full briefing")
        assert "path" not in dispatch.agents, dispatch.agents
        assert "detection" in dispatch.agents
        boxes = track_boxes(wiring.bus, dispatch.case_id)
        assert boxes, "the synthetic sortie must publish confirmed tracks"
        box, confidence, fix = next(iter(boxes.values()))
        assert len(box) == 4 and 0.0 <= confidence <= 1.0 and fix in ("located", "no_fix")
        assert draw_boxes(Image.new("RGB", SYNTHETIC_FRAME), boxes).size == SYNTHETIC_FRAME
        assert dispatch.picture.targets, "telemetry column would be empty"
        assert 1 <= dispatch.brief.risk.score <= 10, "command column needs a score"
        assert dispatch.brief.recommendation.action

        os.environ[DISABLE_AGENTS] = ""
        os.environ[DECISION] = OFF
        _, flat = run_pipeline("Give me a full briefing")
        assert getattr(flat.brief, "risk", None) is None, "ablated chain has no score"
        assert flat.brief.render(), "the flat report still renders"

        camera = camera_for(1920, 1080)
        assert (camera.cx, camera.cy) == (960.0, 540.0)
        assert camera.fx == camera.fy == 1500.0, "same field of view, more pixels"

        # One frame, a subject the detector is only 0.30 sure of. The shipped
        # tracker publishes nothing — it wants 0.6 to spawn and three frames to
        # confirm, and on a photograph neither is coming. This is the real
        # VisDrone case: five subjects found, none published.
        clue = RealDetector("unused.pt", AgentSource.DRONE_RGB, TAG_RGB).to_clues(
            UPLOAD_FRAME_ID, [([10.0, 20.0, 30.0, 60.0], 0.30, "person")], "case-0000")
        assert not BoTSORT().update(clue, frame_id=UPLOAD_FRAME_ID), \
            "the shipped tracker cannot confirm anything on one frame"
        confirmed = single_frame_tracker(0.25).update(clue, frame_id=UPLOAD_FRAME_ID)
        assert len(confirmed) == 1 and confirmed[0].confidence == 0.30
        assert not single_frame_tracker(0.5).update(clue, frame_id=UPLOAD_FRAME_ID), \
            "the confidence floor still has to mean something"
    finally:
        for name, value in env.items():
            os.environ.pop(name, None) if value is None else os.environ.update({name: value})

    print("  ok  a sidebar toggle keeps an agent out of the dispatch")
    print("  ok  the vision column draws the boxes the pipeline published")
    print("  ok  the telemetry column has targets, the command column a score")
    print("  ok  ABLATION_DECISION=off leaves no risk score to reach for")
    print("  ok  intrinsics scale with the frame at a fixed field of view")
    print("  ok  one frame confirms at the operator's floor, and only at it")
    print("\n6 checks passed")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv[1:]:
        sys.exit(selfcheck())
    main()
