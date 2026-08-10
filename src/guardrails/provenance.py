"""The provenance allow-list — who is allowed to put a clue on the blackboard.

Architecture, Phase 7: "the provenance guard checks retrieved content against an
allow-list."

Every clue names its origin in `provenance_tag`. This registry is the list of
origins the system will accept, and what each one is allowed to claim. A clue
whose tag is unknown, or whose tag does not match the agent that sent it, never
reaches fusion.

Two layers, because they defend different things
------------------------------------------------
**Tag and agent binding** needs no configuration and is always on. It stops the
obvious forgery: a clue arriving as `PERCEPTION_FUSION` while carrying the
Weather agent's tag, or any tag nobody has ever registered. Reusing a legitimate
tag on the wrong agent is exactly what an injected clue looks like.

**Identity credentials** — drone device ids, API endpoints, operator ids — are
checked where data crosses a trust boundary from outside the system: onboard
sensors, third-party APIs, and statements taken by a human. Internal
computations (fusion, the Monte-Carlo model, the LLM agents) have no external
identity to check, so they are allow-listed by tag alone.

An unconfigured credential set means that check is skipped, and `unconfigured`
says which — a deployment can assert it is empty before flying. A configured set
is fail-closed: present, and a member, or the clue is rejected. An empty
configured set therefore trusts nothing, which is a legitimate way to ground a
fleet.
"""

from dataclasses import dataclass, field

from ..contracts.clue import AgentSource
from ..geometry import ground_distance_m

# The registered origins. Tags name a *component*, not a model version, so
# swapping the model behind an agent does not silently invalidate its clues.
TAG_RGB = "onboard:rgb-camera"
TAG_LIDAR = "onboard:lidar"
TAG_WBF = "fusion:wbf"
TAG_TRACK = "perception:track"
TAG_WEATHER = "api:open-meteo"
TAG_PATH = "model:monte-carlo-v1"
TAG_SCENE = "vlm:scene-description"
TAG_HEALTH = "llm:health-assessment"
TAG_HISTORY = "rag:case-archive"
TAG_INTERVIEW = "llm:witness-interview"

# Rejection reasons, in the order they are checked.
MISSING_TAG = "MISSING_TAG"
UNKNOWN_TAG = "UNKNOWN_TAG"
AGENT_MISMATCH = "AGENT_MISMATCH"
MISSING_IDENTITY = "MISSING_IDENTITY"
UNTRUSTED_IDENTITY = "UNTRUSTED_IDENTITY"
OUT_OF_BOUNDS = "OUT_OF_BOUNDS"

DEVICE = "device"
ENDPOINT = "endpoint"
OPERATOR = "operator"

# The one third-party endpoint the system is allowed to take data from.
OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class SourceRule:
    """One registered origin and what it may claim."""

    tag: str
    agents: frozenset
    kind: str = "internal"
    identity_key: str | None = None   # metadata field carrying the credential
    credential: str | None = None     # which configured set it is checked against

    def allows(self, source_agent):
        return source_agent in self.agents


DEFAULT_RULES = (
    # Onboard sensors — data from hardware we own, over a radio link.
    SourceRule(TAG_RGB, frozenset({AgentSource.DRONE_RGB}), "onboard", "device_id", DEVICE),
    SourceRule(TAG_LIDAR, frozenset({AgentSource.DRONE_LIDAR}), "onboard", "device_id", DEVICE),
    SourceRule(TAG_TRACK, frozenset({AgentSource.PERCEPTION_FUSION}), "onboard",
               "device_id", DEVICE),
    # Third-party APIs.
    SourceRule(TAG_WEATHER, frozenset({AgentSource.WEATHER_API}), "api", "endpoint", ENDPOINT),
    # Human-entered.
    SourceRule(TAG_INTERVIEW, frozenset({AgentSource.INTERVIEW_LLM}), "operator",
               "operator_id", OPERATOR),
    # Internal computation — nothing external to authenticate.
    SourceRule(TAG_WBF, frozenset({AgentSource.PERCEPTION_FUSION}), "internal"),
    SourceRule(TAG_PATH, frozenset({AgentSource.PATH_MODEL}), "internal"),
    SourceRule(TAG_SCENE, frozenset({AgentSource.SCENE_VLM}), "internal"),
    SourceRule(TAG_HEALTH, frozenset({AgentSource.HEALTH_LLM}), "internal"),
    SourceRule(TAG_HISTORY, frozenset({AgentSource.HISTORY_RAG}), "internal"),
)


@dataclass(frozen=True)
class Geofence:
    """The ground this search covers.

    The contract already refuses an impossible latitude. This refuses a
    *possible* one that is nowhere near the search: a clue placing a subject in
    the next valley is either a badly broken producer or a deliberate attempt to
    pull teams off the ground they should be on. Either way it does not belong
    on the blackboard.
    """

    latitude: float
    longitude: float
    radius_m: float

    def contains(self, position):
        return ground_distance_m((self.latitude, self.longitude), position) <= self.radius_m

    def distance_m(self, position):
        return ground_distance_m((self.latitude, self.longitude), position)


@dataclass(frozen=True)
class Verdict:
    """Whether a clue may pass, and why not if it may not."""

    allowed: bool
    reason: str | None = None
    detail: str = ""
    rule: SourceRule | None = None

    def __bool__(self):
        return self.allowed


@dataclass
class ProvenanceRegistry:
    """The allow-list. Construct with the fleet, endpoints and operators in use."""

    rules: tuple = DEFAULT_RULES
    devices: frozenset | None = None
    endpoints: frozenset | None = None
    operators: frozenset | None = None
    geofence: Geofence | None = None
    _by_tag: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        self.devices = _maybe_set(self.devices)
        self.endpoints = _maybe_set(self.endpoints)
        self.operators = _maybe_set(self.operators)
        self._by_tag = {rule.tag: rule for rule in self.rules}

    @property
    def tags(self):
        return tuple(sorted(self._by_tag))

    @property
    def unconfigured(self):
        """Credential kinds with no allow-list, whose checks are therefore off.

        A deployment should assert this is empty before it flies.
        """
        missing = []
        for name, values in ((DEVICE, self.devices), (ENDPOINT, self.endpoints),
                             (OPERATOR, self.operators)):
            if values is None and any(r.credential == name for r in self.rules):
                missing.append(name)
        return tuple(missing)

    def credentials(self, kind):
        return {DEVICE: self.devices, ENDPOINT: self.endpoints, OPERATOR: self.operators}[kind]

    def verify(self, clue):
        """Check one clue against the allow-list."""
        tag = (clue.provenance_tag or "").strip()
        if not tag:
            return Verdict(False, MISSING_TAG, "clue carries no provenance_tag")

        rule = self._by_tag.get(tag)
        if rule is None:
            return Verdict(False, UNKNOWN_TAG, f"{tag!r} is not a registered origin")

        if not rule.allows(clue.source_agent):
            return Verdict(
                False, AGENT_MISMATCH,
                f"{clue.source_agent.value} may not use {tag!r} "
                f"(registered to {', '.join(sorted(a.value for a in rule.agents))})",
                rule,
            )

        if rule.credential is not None:
            allowed = self.credentials(rule.credential)
            if allowed is not None:
                claimed = clue.agent_metadata.get(rule.identity_key)
                if claimed in (None, ""):
                    return Verdict(
                        False, MISSING_IDENTITY,
                        f"{tag!r} requires agent_metadata[{rule.identity_key!r}]", rule,
                    )
                if str(claimed) not in allowed:
                    return Verdict(
                        False, UNTRUSTED_IDENTITY,
                        f"{rule.credential} {str(claimed)!r} is not on the allow-list", rule,
                    )

        if self.geofence is not None:
            position = _position_of(clue)
            if position is not None and not self.geofence.contains(position):
                return Verdict(
                    False, OUT_OF_BOUNDS,
                    f"claims {position[0]:.5f}, {position[1]:.5f} — "
                    f"{self.geofence.distance_m(position) / 1000.0:.1f} km outside the "
                    f"search area", rule,
                )

        return Verdict(True, rule=rule)


def _maybe_set(values):
    return None if values is None else frozenset(str(v) for v in values)


def _position_of(clue):
    spatial = clue.spatial_context
    if spatial is None or spatial.latitude is None or spatial.longitude is None:
        return None
    return (spatial.latitude, spatial.longitude)
