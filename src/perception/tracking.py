"""BoT-SORT tracking — link fused detections across sequential frames.

Phase 2 (docs/architecture.md): "BoT-SORT links detections across frames, and
its camera-motion compensation helps because the drone camera is always moving.
Tracks are confirmed on hit-count and confidence."

Implements the three things BoT-SORT adds over plain SORT:

  1. a constant-velocity Kalman filter on (cx, cy, w, h) rather than SORT's
     (x, y, aspect, height),
  2. ByteTrack's two-stage association, so a target that dims for a few frames
     is recovered from low-confidence detections instead of being dropped,
  3. camera motion compensation: track predictions are warped into the current
     frame before association, so drone movement is not mistaken for target
     movement.

Not implemented: BoT-SORT's ReID appearance embedding, which needs image crops
and a re-identification network. IoU association only. That costs identity
switches when two targets cross, which in a search context is far less
damaging than a missed target.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from ..geometry import iou

# ByteTrack/BoT-SORT noise scaling. Position noise grows with target size: a
# large nearby target moves more pixels per frame than a distant one.
_STD_POSITION = 1.0 / 20.0
_STD_VELOCITY = 1.0 / 160.0

# ponytail: lineage capped so a long track does not grow without bound.
_MAX_LINEAGE = 32


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"  # seen, not yet trusted
    CONFIRMED = "CONFIRMED"  # survived min_hits, publishable
    LOST = "LOST"            # missed recently, still predicted
    REMOVED = "REMOVED"


@dataclass(frozen=True)
class Affine:
    """2-D affine camera motion: [[a, b, tx], [c, d, ty]].

    Identity means a static camera. A real Phase 2 estimator produces this by
    registering consecutive frames (ORB features + RANSAC, or ECC); until frames
    exist, `from_camera_delta` builds one from telemetry.
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    @classmethod
    def from_camera_delta(cls, dx_px=0.0, dy_px=0.0, rotation_rad=0.0, scale=1.0):
        """Affine for a camera that panned, rolled, or zoomed between frames.

        Sign convention: `dx_px`/`dy_px` are how far the *scene* shifts in the
        image. A drone flying forward makes the scene flow backward.
        """
        cos_r, sin_r = math.cos(rotation_rad) * scale, math.sin(rotation_rad) * scale
        return cls(a=cos_r, b=-sin_r, c=sin_r, d=cos_r, tx=dx_px, ty=dy_px)

    @property
    def is_identity(self):
        return (self.a, self.b, self.c, self.d, self.tx, self.ty) == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @property
    def scale(self):
        """Uniform scale factor, as the square root of the linear part's area."""
        return math.sqrt(abs(self.a * self.d - self.b * self.c)) or 1.0

    def apply_point(self, x, y):
        return (self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty)

    def apply_box(self, box):
        """Warp a box and re-fit an axis-aligned box around the result."""
        x1, y1, x2, y2 = box
        corners = [self.apply_point(x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        return (min(xs), min(ys), max(xs), max(ys))


class _Dim:
    """One dimension of a constant-velocity Kalman filter: position + velocity.

    The 8-D BoT-SORT filter decomposes *exactly* into four of these. Its
    transition matrix is block-diagonal per dimension and its process,
    measurement, and initial covariances are all diagonal, so no off-diagonal
    covariance is ever created between dimensions. Four 2x2 filters, not an
    approximation — and it removes the need for any matrix inversion.
    """

    __slots__ = ("x", "v", "p11", "p12", "p22")

    def __init__(self, x, var_pos, var_vel):
        self.x, self.v = float(x), 0.0
        self.p11, self.p12, self.p22 = var_pos, 0.0, var_vel

    def predict(self, q_pos, q_vel):
        self.x += self.v
        self.p11 = self.p11 + 2.0 * self.p12 + self.p22 + q_pos
        self.p12 = self.p12 + self.p22
        self.p22 = self.p22 + q_vel

    def update(self, z, r):
        s = self.p11 + r
        if s <= 0.0:  # degenerate; trust the measurement outright
            self.x, self.v = float(z), 0.0
            return
        k1, k2 = self.p11 / s, self.p12 / s
        residual = z - self.x
        self.x += k1 * residual
        self.v += k2 * residual
        p11, p12 = self.p11, self.p12
        self.p11 = p11 - k1 * p11
        self.p12 = p12 - k1 * p12
        self.p22 = self.p22 - k2 * p12


class _KalmanBox:
    """Constant-velocity filter over (cx, cy, w, h)."""

    def __init__(self, box):
        cx, cy, w, h = _to_cxcywh(box)
        self.dims = [
            _Dim(cx, (2 * _STD_POSITION * w) ** 2, (10 * _STD_VELOCITY * w) ** 2),
            _Dim(cy, (2 * _STD_POSITION * h) ** 2, (10 * _STD_VELOCITY * h) ** 2),
            _Dim(w, (2 * _STD_POSITION * w) ** 2, (10 * _STD_VELOCITY * w) ** 2),
            _Dim(h, (2 * _STD_POSITION * h) ** 2, (10 * _STD_VELOCITY * h) ** 2),
        ]

    @property
    def box(self):
        cx, cy, w, h = (d.x for d in self.dims)
        return _to_xyxy(cx, cy, max(w, 1e-6), max(h, 1e-6))

    def _scales(self):
        w, h = max(self.dims[2].x, 1e-6), max(self.dims[3].x, 1e-6)
        return (w, h, w, h)

    def predict(self):
        for dim, scale in zip(self.dims, self._scales()):
            dim.predict((_STD_POSITION * scale) ** 2, (_STD_VELOCITY * scale) ** 2)

    def update(self, box):
        for dim, z, scale in zip(self.dims, _to_cxcywh(box), self._scales()):
            dim.update(z, (_STD_POSITION * scale) ** 2)

    def apply_affine(self, transform):
        """Warp the filter into the current frame's coordinates."""
        cx, cy, w, h = _to_cxcywh(transform.apply_box(self.box))
        for dim, value in zip(self.dims, (cx, cy, w, h)):
            dim.x = value
        # Velocity is a displacement, so only the linear part applies.
        vx, vy = self.dims[0].v, self.dims[1].v
        self.dims[0].v = transform.a * vx + transform.b * vy
        self.dims[1].v = transform.c * vx + transform.d * vy
        # ponytail: variances rescaled but rotation's x/y covariance coupling is
        # ignored — it is second-order for the small inter-frame rotations GMC
        # sees. Move to a full 8x8 P = M P Mᵀ if the gimbal ever rolls hard.
        var_scale = transform.scale ** 2
        for dim in self.dims:
            dim.p11 *= var_scale
            dim.p12 *= var_scale
            dim.p22 *= var_scale


@dataclass
class Track:
    track_id: int
    box: tuple
    confidence: float
    class_label: str | None
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 0                 # frames since the last successful update
    frames_seen: int = 1
    first_frame_id: str | None = None
    last_frame_id: str | None = None
    range_m: float | None = None  # latest measured range, if any parent had one
    clue_ids: list = field(default_factory=list)
    kalman: _KalmanBox = None

    @property
    def is_active(self):
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)


class BoTSORT:
    """Multi-object tracker over fused detection clues.

    Feed it one frame at a time, in order. Every knob is a tuning decision the
    Phase 9 optimizer will search, not a constant.
    """

    def __init__(
        self,
        high_thresh=0.5,
        new_track_thresh=0.6,
        match_thresh=0.8,
        second_match_thresh=0.5,
        min_hits=3,
        max_age=30,
    ):
        self.high_thresh = high_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh  # max IoU *cost* (1 - IoU) for stage one
        self.second_match_thresh = second_match_thresh
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks = []
        self._next_id = 1

    def update(self, clues, frame_id=None, camera_motion=None):
        """Advance one frame. Returns the confirmed tracks after this frame."""
        detections = [_Detection.from_clue(c) for c in clues]

        for track in self.tracks:
            track.kalman.predict()
            track.box = track.kalman.box
            if camera_motion is not None and not camera_motion.is_identity:
                track.kalman.apply_affine(camera_motion)
                track.box = track.kalman.box

        high = [d for d in detections if d.confidence >= self.high_thresh]
        low = [d for d in detections if d.confidence < self.high_thresh]

        # Lost tracks stay in the pool: they are still predicted, still
        # matchable, and still ageing toward removal. Dropping them from
        # association is how a recoverable target becomes a duplicate id.
        candidates = list(self.tracks)
        matches, unmatched_tracks, unmatched_high = _associate(candidates, high, self.match_thresh)
        for track, det in matches:
            self._apply(track, det, frame_id)

        # ByteTrack stage two: try the leftovers against low-confidence
        # detections. A target half-hidden in canopy still deserves its track.
        retry = [candidates[i] for i in unmatched_tracks]
        second, still_unmatched, _ = _associate(retry, low, self.second_match_thresh)
        for track, det in second:
            self._apply(track, det, frame_id)

        for i in still_unmatched:
            self._miss(retry[i])

        for i in unmatched_high:
            det = high[i]
            if det.confidence >= self.new_track_thresh:
                self.tracks.append(self._spawn(det, frame_id))

        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]
        return self.confirmed_tracks()

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.state is TrackState.CONFIRMED and t.age == 0]

    def _spawn(self, det, frame_id):
        track = Track(
            track_id=self._next_id,
            box=det.box,
            confidence=det.confidence,
            class_label=det.class_label,
            first_frame_id=frame_id,
            last_frame_id=frame_id,
            range_m=det.range_m,
            clue_ids=[det.clue_id],
            kalman=_KalmanBox(det.box),
        )
        self._next_id += 1
        if self.min_hits <= 1:
            track.state = TrackState.CONFIRMED
        return track

    def _apply(self, track, det, frame_id):
        track.kalman.update(det.box)
        track.box = track.kalman.box
        track.confidence = det.confidence
        track.class_label = det.class_label or track.class_label
        track.hits += 1
        track.frames_seen += 1
        track.age = 0
        track.last_frame_id = frame_id
        if det.range_m is not None:
            track.range_m = det.range_m
        track.clue_ids = (track.clue_ids + [det.clue_id])[-_MAX_LINEAGE:]
        if track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED
        elif track.state is TrackState.LOST:
            track.state = TrackState.TENTATIVE

    def _miss(self, track):
        track.age += 1
        if track.state is TrackState.TENTATIVE:
            # Never confirmed, missed once: it was probably noise.
            track.state = TrackState.REMOVED
        elif track.age > self.max_age:
            track.state = TrackState.REMOVED
        else:
            track.state = TrackState.LOST


@dataclass
class _Detection:
    box: tuple
    confidence: float
    class_label: str | None
    clue_id: str
    range_m: float | None

    @classmethod
    def from_clue(cls, clue):
        return cls(
            box=clue.detection_box(),  # trust boundary
            confidence=clue.confidence_score,
            class_label=clue.class_label,
            clue_id=clue.clue_id,
            range_m=clue.agent_metadata.get("range_m"),
        )


def _associate(tracks, detections, max_cost):
    """Match tracks to detections on IoU cost, best pair first.

    ponytail: greedy rather than the Hungarian assignment BoT-SORT specifies.
    Identical for the sparse, well-separated targets a search produces, and it
    can pick a globally sub-optimal set when targets overlap heavily. Swap in
    scipy.optimize.linear_sum_assignment if crowding ever shows up in the
    identity-switch count.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    candidates = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            if track.class_label is not None and det.class_label is not None:
                if track.class_label != det.class_label:
                    continue
            cost = 1.0 - iou(track.box, det.box)
            if cost <= max_cost:
                candidates.append((cost, ti, di))

    candidates.sort()
    matches, used_tracks, used_dets = [], set(), set()
    for cost, ti, di in candidates:
        if ti in used_tracks or di in used_dets:
            continue
        used_tracks.add(ti)
        used_dets.add(di)
        matches.append((tracks[ti], detections[di]))

    return (
        matches,
        [i for i in range(len(tracks)) if i not in used_tracks],
        [i for i in range(len(detections)) if i not in used_dets],
    )


def _to_cxcywh(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def _to_xyxy(cx, cy, w, h):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
