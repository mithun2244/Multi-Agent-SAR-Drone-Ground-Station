"""The tunable parameters, and where they live on disk.

Architecture, Phase 9: "use Optuna, a Bayesian TPE search, to set the detection
parameters ... against a labelled validation set, with fitness defined as recall
minus lambda times the false-alarm rate."

Every number here was a hand-picked default somewhere in the system. Collecting
them in one dataclass does three things: the search space is obvious, a tuned
run is one JSON file, and nothing has to be edited in source to deploy a new
setting.

Defaults are exactly the values the code shipped with, so `TunedParams()` is the
baseline. A tuning run that cannot beat it should change nothing.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_PATH = Path("config/tuned_params.json")


@dataclass(frozen=True)
class TunedParams:
    """One complete operating configuration."""

    # -- perception: what counts as a detection --------------------------
    wbf_iou_threshold: float = 0.55        # boxes this close are one target
    wbf_score_threshold: float = 0.0       # drop detections weaker than this
    rgb_weight: float = 1.0                # WBF trust in the RGB detector
    lidar_weight: float = 1.0              # WBF trust in the LiDAR detector
    track_high_thresh: float = 0.5         # ByteTrack's high/low split
    track_new_thresh: float = 0.6          # confidence needed to start a track
    track_min_hits: int = 3                # frames before a track is confirmed

    # -- fusion: how much each source's word is worth --------------------
    trust_perception_fusion: float = 1.0
    trust_drone_rgb: float = 0.9
    trust_drone_lidar: float = 0.9
    # How close two tracks must be before they are treated as one target. Too
    # wide and two people standing together become one; too tight and a
    # re-acquired track becomes a second person.
    merge_distance_m: float = 25.0

    # -- ranking ----------------------------------------------------------
    urgency_weight: float = 1.0            # how far urgency may move priority
    sector_weight: float = 0.5             # how far the path prior may move it
    hazard_urgency_step: float = 0.3       # urgency added per visible hazard

    # -- health: bounds on the survival-window multiplier -----------------
    health_multiplier_floor: float = 0.25
    health_multiplier_ceiling: float = 2.0

    # -- reporting --------------------------------------------------------
    # The operating point. Targets below this never reach the operator, which is
    # how the false-alarm rate is actually held down.
    min_report_confidence: float = 0.0
    target_far: float = 0.10               # tolerated false alarms per frame

    def to_json(self, path=CONFIG_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path=CONFIG_PATH):
        """Load a tuned configuration, or the shipped defaults if none exists.

        Missing file means untuned, not broken: a fresh checkout must fly on the
        defaults rather than refuse to start.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, values):
        """Build from a dict, ignoring keys this version does not know.

        A config written by a newer build must not stop an older one starting —
        unknown settings are dropped, known ones applied.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    @property
    def trust_table(self):
        """The source-trust table fusion consults, in its own terms."""
        from ..contracts.clue import AgentSource
        return {
            AgentSource.PERCEPTION_FUSION: self.trust_perception_fusion,
            AgentSource.DRONE_RGB: self.trust_drone_rgb,
            AgentSource.DRONE_LIDAR: self.trust_drone_lidar,
        }

    @property
    def wbf_weights(self):
        return [self.rgb_weight, self.lidar_weight]

    def replace(self, **changes):
        from dataclasses import replace as _replace
        return _replace(self, **changes)

    def differences(self, other):
        """Fields where two configurations disagree, for a before/after report."""
        mine, theirs = asdict(self), asdict(other)
        return {
            name: (mine[name], theirs[name])
            for name in sorted(mine)
            if not _close(mine[name], theirs[name])
        }


@dataclass
class ParamStore:
    """Loads once at startup and hands the same configuration to every module."""

    path: Path = CONFIG_PATH
    params: TunedParams = field(default=None)

    def __post_init__(self):
        if self.params is None:
            self.params = TunedParams.from_json(self.path)

    @property
    def is_tuned(self):
        return Path(self.path).exists()

    def reload(self):
        self.params = TunedParams.from_json(self.path)
        return self.params


def _close(a, b, tolerance=1e-9):
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tolerance
    return a == b


def load_params(path=CONFIG_PATH):
    """Convenience for startup paths."""
    return TunedParams.from_json(path)
