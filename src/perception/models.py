"""Which detectors the pipeline runs: the stubs, or real YOLO11m weights.

    PERCEPTION_MODE=stub  python -m src.perception.agent     # the default
    PERCEPTION_MODE=real  python -m src.perception.agent
    python -m src.perception.agent --real
    python -m src.perception.models --selfcheck

**Stub is the default, deliberately.** Every test, demo and tuning fold in this
repository runs on projected stubs and must keep doing so: they need ground
truth to score against, and there is no dataset of real sorties here to replace
it with. `PERCEPTION_MODE=real` is opt-in, and an unset variable never loads
torch, never reads a checkpoint, and never changes a number.

The seam already existed
------------------------
CLAUDE.md: "Detectors are injected as callables, never imported by name into the
harness, so a real model replaces a stub without touching scoring code." That is
what this module cashes in — `build_detectors()` hands back the same pair of
objects either way, and `DetectionAgent.process_frame` cannot tell which it got.

What differs is what each is looking at. A stub is handed the *targets* that were
projected into the frame; a real model is handed the *frame*. Both answer the
same question — what is in front of this sensor right now — so both expose
`detect(frame_id, source, case_id)` and return `ClueContract`s. The second
argument is the only thing that changes, and it changes because a real detector
cannot be told where the answer is.

Positions are not the detector's business
-----------------------------------------
A real detector emits a box, a class and a confidence, and **no coordinates**:
`DetectionAgent` geolocates from the box, the telemetry and the DEM. That is the
CLAUDE.md rule — never emit a position that was not computed — and it is also
why swapping the model changes nothing downstream. The LiDAR model likewise only
reports a range when something actually measured one; without a range image to
sample, the clue carries no range and geolocation falls back to the terrain
intersection, which is the honest outcome rather than a plausible number.

ponytail: no batching, no half precision, no TensorRT export, no warm-up pass.
This is the correctness seam, not the airframe's inference loop — the drone will
want all four, and none of them belong in a factory that a test imports.
"""

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.provenance import TAG_LIDAR, TAG_RGB
from .detectors import lidar_stub, yolo11m_stub

STUB, REAL = "stub", "real"
MODES = (STUB, REAL)
MODE_ENV = "PERCEPTION_MODE"

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "config" / "weights"
RGB_WEIGHTS = WEIGHTS_DIR / "yolo11m_visdrone.pt"
LIDAR_WEIGHTS = WEIGHTS_DIR / "yolo11m_lidar.pt"

# Below this the box is not worth tracking. Deliberately low: BoT-SORT's own
# thresholds decide what becomes a track, and this only keeps the obvious noise
# out of the association step. The Phase 9 study tunes the ones that matter.
DEFAULT_CONF = 0.25


class PerceptionModeError(ValueError):
    """An unrecognised PERCEPTION_MODE."""


class MissingWeights(FileNotFoundError):
    """Real mode was asked for and the checkpoint is not there."""


def perception_mode(override=None, env=None):
    """Resolve the mode: explicit argument, then the environment, then stub."""
    raw = override if override is not None else (env or os.environ).get(MODE_ENV, STUB)
    mode = str(raw).strip().lower()
    if mode not in MODES:
        raise PerceptionModeError(
            f"{MODE_ENV}={raw!r} is not a mode; expected one of {', '.join(MODES)}"
        )
    return mode


def require_weights(path, sensor):
    """Fail with the path and how to produce it, before anything heavy loads."""
    path = Path(path)
    if path.is_file():
        return path
    raise MissingWeights(
        f"{sensor} weights not found at {path}\n"
        f"  {MODE_ENV}=real needs a trained checkpoint at exactly that path.\n"
        f"  Train one with `python train_perception.py`, then copy the resulting\n"
        f"  runs/detect/*/weights/best.pt to {path}.\n"
        f"  Or leave {MODE_ENV} unset to run on the stubs."
    )


def build_detectors(mode=None, seed=0, device_id=None, sensors=("rgb", "lidar"),
                    rgb_weights=RGB_WEIGHTS, lidar_weights=LIDAR_WEIGHTS, **kwargs):
    """The detector pair for this run. Returns {sensor: detector}.

    Only the sensors named are built, so a single-sensor airframe never loads a
    model it has no feed for — the cost that actually matters on a drone.
    """
    mode = perception_mode(mode)
    builders = {
        STUB: {
            "rgb": lambda: yolo11m_stub(seed=seed, device_id=device_id, **kwargs),
            "lidar": lambda: lidar_stub(seed=seed, device_id=device_id, **kwargs),
        },
        REAL: {
            "rgb": lambda: RealDetector(
                require_weights(rgb_weights, "RGB"), AgentSource.DRONE_RGB, TAG_RGB,
                device_id=device_id, **kwargs),
            "lidar": lambda: RealDetector(
                require_weights(lidar_weights, "LiDAR"), AgentSource.DRONE_LIDAR, TAG_LIDAR,
                device_id=device_id, emits_range=True, **kwargs),
        },
    }[mode]

    unknown = set(sensors) - set(builders)
    if unknown:
        raise ValueError(f"no detector for sensor(s): {', '.join(sorted(unknown))}")
    return {sensor: builders[sensor]() for sensor in sensors}


class RealDetector:
    """A trained YOLO11m checkpoint, wrapped to emit `ClueContract`s.

    Weights are checked at construction and loaded lazily on the first frame, so
    building one is cheap and a missing checkpoint fails at wiring time rather
    than mid-sortie.
    """

    def __init__(self, weights, source_agent, provenance_tag, conf=DEFAULT_CONF,
                 device_id=None, label=None, emits_range=False, range_lookup=None,
                 clock=None):
        self.weights = Path(weights)
        self.source_agent = source_agent
        self.provenance_tag = provenance_tag
        self.conf = conf
        self.device_id = device_id
        self.label = label
        self.emits_range = emits_range
        # Given a frame and a box, returns the measured range in metres, or None.
        # Absent, the clue carries no range and the DEM ray march does the work.
        self.range_lookup = range_lookup
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._model = None
        self.frames = 0

    @property
    def model(self):
        if self._model is None:
            # Imported here, not at module scope: ultralytics pulls torch,
            # torchvision and opencv — several gigabytes that the ground station
            # never touches and that every test would otherwise pay for.
            from ultralytics import YOLO      # noqa: PLC0415  (training-only dep)

            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, frame_id, frame, case_id="case-0000"):
        """Detect over one frame. `frame` is anything ultralytics accepts."""
        self.frames += 1
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        names = getattr(result, "names", {}) or {}
        rows = []
        for box in result.boxes:
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            rows.append((xyxy, float(box.conf[0]),
                         self.label or names.get(int(box.cls[0]), "contact")))
        return self.to_clues(frame_id, rows, case_id, frame=frame)

    def to_clues(self, frame_id, rows, case_id, frame=None):
        """(box, confidence, label) rows -> clues. The whole conversion, in one
        place a test can reach without a checkpoint."""
        clues = []
        for box, confidence, label in rows:
            metadata = {"device_id": self.device_id} if self.device_id else {}
            if self.emits_range and self.range_lookup is not None:
                measured = self.range_lookup(frame, box)
                if measured is not None:
                    metadata["range_m"] = round(float(measured), 2)

            clues.append(ClueContract(
                clue_id=str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"sar:{self.source_agent.value}:{case_id}:{frame_id}:{box}")),
                case_id=case_id,
                timestamp=self.clock(),
                source_agent=self.source_agent,
                confidence_score=round(min(1.0, max(0.0, confidence)), 4),
                finding_summary=f"Possible {label} on {self.source_agent.value}",
                # No coordinates. The agent geolocates from this box, the
                # telemetry and the DEM; a detector that guessed a position
                # would be inventing one.
                spatial_context=SpatialContext(latitude=None, longitude=None,
                                               bounding_box=box),
                frame_id=frame_id,
                class_label=label,
                provenance_tag=self.provenance_tag,
                agent_metadata=metadata,
            ))
        return clues

    def __repr__(self):
        return (f"RealDetector({self.source_agent.value}, {self.weights.name}, "
                f"{'loaded' if self._model else 'not loaded'})")


def describe(mode, detectors):
    """One line for a script header."""
    if mode == STUB:
        return f"detectors: stub ({', '.join(sorted(detectors))}) — {MODE_ENV}=real for weights"
    return ("detectors: real — "
            + ", ".join(f"{name} {d.weights.name}" for name, d in sorted(detectors.items())))


def selfcheck():
    """No checkpoint, no ultralytics, no network."""
    assert perception_mode() == STUB, "an unset environment must mean stubs"
    assert perception_mode(env={}) == STUB
    assert perception_mode(env={MODE_ENV: "real"}) == REAL
    assert perception_mode(env={MODE_ENV: " REAL "}) == REAL, "tolerant of case and space"
    assert perception_mode("real", env={MODE_ENV: "stub"}) == REAL, "argument beats env"

    for bad in ("REALISTIC", "yes", "1", ""):
        try:
            perception_mode(env={MODE_ENV: bad})
            raise AssertionError(f"{bad!r} must not be accepted as a mode")
        except PerceptionModeError as e:
            assert MODE_ENV in str(e) and "stub, real" in str(e)

    # Stubs, which is what every test and demo gets.
    stubs = build_detectors()
    assert set(stubs) == {"rgb", "lidar"}
    assert stubs["rgb"].source_agent is AgentSource.DRONE_RGB
    assert stubs["lidar"].source_agent is AgentSource.DRONE_LIDAR
    assert stubs["lidar"].emits_range and not stubs["rgb"].emits_range
    assert build_detectors(seed=3)["rgb"].detect("f", [], "c") == []

    # One sensor asks for one model: nothing is loaded for a feed we do not have.
    assert set(build_detectors(sensors=("rgb",))) == {"rgb"}
    try:
        build_detectors(sensors=("thermal",))
        raise AssertionError("an unknown sensor must be refused")
    except ValueError as e:
        assert "thermal" in str(e)

    # Real mode with no checkpoint: the error names the path and the way out.
    try:
        build_detectors(mode=REAL)
        raise AssertionError("missing weights must be refused")
    except MissingWeights as e:
        assert str(RGB_WEIGHTS) in str(e), e
        assert "train_perception.py" in str(e) and MODE_ENV in str(e)
    assert issubclass(MissingWeights, FileNotFoundError), "catchable as a missing file"

    # The LiDAR checkpoint is named separately, and its absence says so.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rgb_only = Path(tmp) / "rgb.pt"
        rgb_only.write_bytes(b"not a real checkpoint, never loaded here")
        try:
            build_detectors(mode=REAL, rgb_weights=rgb_only)
            raise AssertionError("the LiDAR checkpoint must be checked too")
        except MissingWeights as e:
            assert "LiDAR" in str(e) and str(LIDAR_WEIGHTS) in str(e)

        # A real detector builds without touching ultralytics, and converts
        # boxes to clues without a model.
        detector = build_detectors(mode=REAL, sensors=("rgb",), rgb_weights=rgb_only,
                                   device_id="AA:BB:CC:DD:EE:01")["rgb"]
        assert detector._model is None, "the checkpoint is loaded lazily"
        clues = detector.to_clues("frame_0001", [([10.0, 20.0, 30.0, 60.0], 0.87, "person")],
                                  "case-0000")
        assert len(clues) == 1
        clue = clues[0]
        assert clue.spatial_context.bounding_box == [10.0, 20.0, 30.0, 60.0]
        assert clue.spatial_context.latitude is None, "a detector never invents a position"
        assert clue.confidence_score == 0.87 and clue.class_label == "person"
        assert clue.provenance_tag == TAG_RGB
        assert clue.agent_metadata == {"device_id": "AA:BB:CC:DD:EE:01"}
        assert "range_m" not in clue.agent_metadata

        # Replaying a frame mints the same clue id rather than a duplicate.
        again = detector.to_clues("frame_0001", [([10.0, 20.0, 30.0, 60.0], 0.87, "person")],
                                  "case-0000")
        assert again[0].clue_id == clue.clue_id

        # LiDAR only reports a range when something actually measured one.
        lidar = RealDetector(rgb_only, AgentSource.DRONE_LIDAR, TAG_LIDAR, emits_range=True)
        assert "range_m" not in lidar.to_clues(
            "f", [([0.0, 0.0, 4.0, 8.0], 0.5, "person")], "c")[0].agent_metadata
        lidar.range_lookup = lambda frame, box: 41.276
        assert lidar.to_clues("f", [([0.0, 0.0, 4.0, 8.0], 0.5, "person")],
                              "c")[0].agent_metadata["range_m"] == 41.28

    print("  ok  an unset PERCEPTION_MODE means stubs, and an unknown value is refused")
    print("  ok  build_detectors returns the same shape in both modes")
    print("  ok  only the sensors asked for are built")
    print("  ok  missing weights name the path, the sensor and the way out")
    print("  ok  a real detector builds without importing ultralytics")
    print("  ok  a detector emits a box and never a position")
    print("  ok  a range is reported only when something measured one")
    print("\n7 checks passed")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv[1:]:
        sys.exit(selfcheck())
    mode = perception_mode("real" if "--real" in sys.argv else
                           "stub" if "--stub" in sys.argv else None)
    print(f"  {describe(mode, build_detectors(mode))}")
