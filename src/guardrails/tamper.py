"""The imagery guard — is this frame the one the sensor actually captured?

Architecture, Phase 7: "The imagery guard runs a behavioural check: it compares
detections on the raw frame against a transformed copy, and a large divergence
flags tampering."

Two layers, cheapest first.

**Integrity.** The drone records a SHA-256 of every frame as it captures it. A
frame arriving at a consumer must hash to the digest the sensor recorded, or it
is not the frame the sensor took. This is the check that runs on every frame:
one hash, no model, no inference. It catches a substituted or edited image
outright, and it catches a payload that is not an image at all before that
payload reaches a VLM.

**Behaviour.** Integrity cannot help when the tampering happened *before*
capture — a projected pattern, a printed target on the ground. That is what the
divergence check is for: run detection on the raw frame and on a transformed
copy, and a genuine object survives the transform while an artefact tuned to the
detector usually does not. It needs real inference, so it is the expensive layer
and runs on demand.

ponytail: the header check reads magic bytes only, so it identifies a format
rather than validating a whole file. It exists to stop a text payload or an
executable reaching a model, not to be an image parser.
"""

import hashlib
from dataclasses import dataclass, field

from .audit import IMAGE_REJECTED

# Rejection reasons.
UNKNOWN_FRAME = "UNKNOWN_FRAME"
DIGEST_MISMATCH = "DIGEST_MISMATCH"
MALFORMED_IMAGE = "MALFORMED_IMAGE"
EMPTY_IMAGE = "EMPTY_IMAGE"
SIZE_MISMATCH = "SIZE_MISMATCH"

# Magic bytes for the formats a drone camera actually produces.
_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


@dataclass(frozen=True)
class FrameRecord:
    """What the sensor says it captured."""

    frame_id: str
    digest: str
    size_bytes: int
    device_id: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class FrameVerdict:
    allowed: bool
    reason: str | None = None
    detail: str = ""
    record: FrameRecord | None = None

    def __bool__(self):
        return self.allowed


def digest_of(image):
    """SHA-256 of a frame, as the sensor records it."""
    return hashlib.sha256(image).hexdigest()


def sniff_mime(image):
    """The format a payload claims to be by its magic bytes, or None."""
    for prefix, mime in _MAGIC:
        if image.startswith(prefix):
            return mime
    return None


@dataclass
class FrameLedger:
    """Digests of the frames the sensor captured, checked before a frame is used.

    `strict_header` refuses anything that does not look like an image. Leave it
    off only where the frames are synthetic — a stub sortie has no real JPEGs,
    and a check that must be bypassed in testing is a check nobody trusts.
    """

    strict_header: bool = True
    records: dict = field(default_factory=dict)
    verified: int = 0
    rejected: int = 0

    def record(self, frame_id, image, device_id=None):
        """Register a frame at capture time. Returns the record."""
        entry = FrameRecord(
            frame_id=frame_id,
            digest=digest_of(image),
            size_bytes=len(image),
            device_id=device_id,
            mime_type=sniff_mime(image),
        )
        self.records[frame_id] = entry
        return entry

    def verify(self, frame_id, image):
        """Check a frame about to be used against what the sensor captured."""
        if not image:
            self.rejected += 1
            return FrameVerdict(False, EMPTY_IMAGE, f"frame {frame_id!r} carries no bytes")

        known = self.records.get(frame_id)
        if known is None:
            # No sensor ever reported capturing this. Describing it would mean
            # a VLM narrating an image that entered from somewhere unknown.
            self.rejected += 1
            return FrameVerdict(False, UNKNOWN_FRAME,
                                f"no capture record for frame {frame_id!r}")

        if self.strict_header and sniff_mime(image) is None:
            self.rejected += 1
            return FrameVerdict(False, MALFORMED_IMAGE,
                                f"frame {frame_id!r} is not a recognised image format",
                                known)

        if len(image) != known.size_bytes:
            self.rejected += 1
            return FrameVerdict(
                False, SIZE_MISMATCH,
                f"frame {frame_id!r} is {len(image)} bytes, sensor recorded "
                f"{known.size_bytes}", known,
            )

        actual = digest_of(image)
        if actual != known.digest:
            self.rejected += 1
            return FrameVerdict(
                False, DIGEST_MISMATCH,
                f"frame {frame_id!r} hashes to {actual[:12]}…, sensor recorded "
                f"{known.digest[:12]}…", known,
            )

        self.verified += 1
        return FrameVerdict(True, record=known)

    def loader(self, source, audit=None, case_id=None):
        """Wrap an image loader so only verified frames ever reach a consumer.

        Returns None for anything that fails, which is what the Scene agent
        already treats as "no image, do not describe" — so a tampered frame
        costs nothing and produces no clue.
        """
        def load(frame_id):
            image = source(frame_id)
            if image is None:
                return None
            verdict = self.verify(frame_id, image)
            if verdict:
                return image
            if audit is not None:
                audit.record(
                    IMAGE_REJECTED, verdict.reason, verdict.detail,
                    case_id=case_id, clue_id=frame_id,
                    provenance_tag=getattr(verdict.record, "device_id", None),
                )
            return None

        return load


def detection_divergence(raw, transformed, tolerance_px=8.0):
    """How much detection disagrees between a frame and a transformed copy.

    Returns 0.0 when every box in one has a partner in the other within
    `tolerance_px`, rising toward 1.0 as they disagree. A real object survives a
    mild transform; an artefact crafted against the detector often does not.

    Boxes are (x1, y1, x2, y2) in the *same* coordinate frame — the caller
    un-transforms the second set before comparing, since the point is whether
    the same things were seen, not whether they moved.
    """
    raw, transformed = list(raw), list(transformed)
    if not raw and not transformed:
        return 0.0
    if not raw or not transformed:
        return 1.0

    matched = 0
    remaining = list(transformed)
    for box in raw:
        best, best_gap = None, tolerance_px
        for candidate in remaining:
            gap = max(abs(a - b) for a, b in zip(box, candidate))
            if gap <= best_gap:
                best, best_gap = candidate, gap
        if best is not None:
            remaining.remove(best)
            matched += 1

    # Symmetric difference over union: one of two detections vanishing is a
    # divergence of 0.5, not 0.33. Counting both lists in the denominator would
    # halve every score and let real losses sit under the threshold.
    unmatched = (len(raw) - matched) + len(remaining)
    return round(unmatched / (matched + unmatched), 6)


def behavioural_check(raw, transformed, threshold=0.35, tolerance_px=8.0):
    """Flag tampering when detections do not survive a transform."""
    divergence = detection_divergence(raw, transformed, tolerance_px)
    if divergence > threshold:
        return FrameVerdict(
            False, DIGEST_MISMATCH,
            f"detections diverge by {divergence:.2f} across the transform "
            f"(threshold {threshold})",
        )
    return FrameVerdict(True)
