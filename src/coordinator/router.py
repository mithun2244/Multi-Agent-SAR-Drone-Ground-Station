"""Scenario routing — dispatch only the agents that can answer the question.

Architecture, Phase 6: "call only the smallest set of agents that can answer the
current question. In practice an agent runs only if its answer could change the
next action."

Seven agents on every operator query is seven times the API cost and seven times
the exposure to a rate limit, to answer "what is the weather doing?".

The bias is deliberately asymmetric
-----------------------------------
Routing too narrowly loses an agent's contribution to a live search. Routing too
widely costs an API call. Those are not comparable, so ambiguity always widens:

  * nothing recognised  -> every agent runs;
  * several things recognised -> the union of what matched, never a pick;
  * one thing recognised, clearly -> just that.

Classification is keyword matching, not a model. A router that called an LLM to
decide what to dispatch would put an unreliable component in front of every
dispatch, cost a call to save calls, and fail exactly when the API is rate
limited — which is the situation it exists to handle.

ponytail: keyword sets. They will miss a paraphrase ("is it going to turn cold"
has no weather keyword), and when they do the query falls through to the full
route, which is the safe direction. Swap in a classifier when the miss rate is
measured, not guessed.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

# Canonical dispatch order. Routing changes *which* agents run, never the order
# they run in: scene needs detection to have published, health needs weather.
ALL_AGENTS = ("interview", "detection", "weather", "health", "history", "path", "scene")


class RouteScenario(str, Enum):
    PERCEPTION_EVENT = "PERCEPTION_EVENT"            # drone airborne, a sighting
    WEATHER_QUERY = "WEATHER_QUERY"                  # "what is the outlook?"
    WITNESS_INPUT = "WITNESS_INPUT"                  # a transcript arrived
    HISTORICAL_QUERY = "HISTORICAL_QUERY"            # "anything like this before?"
    FULL_SEARCH_BRIEFING = "FULL_SEARCH_BRIEFING"    # everything, or unclear


AGENT_SETS = {
    # Weather is in here because Health refines the window Weather computes.
    # Without it a perception event on a fresh case gives Health nothing to work
    # from, and the survival window stays at the coarse band table.
    RouteScenario.PERCEPTION_EVENT: ("detection", "weather", "scene", "health", "path"),
    RouteScenario.WEATHER_QUERY: ("weather",),
    RouteScenario.WITNESS_INPUT: ("interview",),
    RouteScenario.HISTORICAL_QUERY: ("history",),
    RouteScenario.FULL_SEARCH_BRIEFING: ALL_AGENTS,
}

# Word-boundary matched, so "windows" does not read as "wind".
_SIGNALS = {
    RouteScenario.WEATHER_QUERY: (
        r"weather", r"forecast", r"temperature", r"wind", r"rain", r"snow",
        r"outlook", r"hypothermia", r"how cold", r"conditions",
    ),
    RouteScenario.WITNESS_INPUT: (
        r"witness", r"statement", r"transcript", r"interview", r"caller",
        r"someone saw", r"reported seeing", r"last seen by",
    ),
    RouteScenario.HISTORICAL_QUERY: (
        r"past cases?", r"similar cases?", r"previous searches?", r"historical",
        r"history", r"archive", r"precedent", r"happened before", r"comparable",
    ),
    RouteScenario.PERCEPTION_EVENT: (
        r"drone", r"airborne", r"sortie", r"detection", r"detected", r"sighting",
        r"spotted", r"camera", r"lidar", r"thermal", r"frame", r"flight",
    ),
}

_FULL_SIGNALS = (
    r"full brief\w*", r"everything", r"sitrep", r"situation report", r"brief me",
    r"overall", r"complete picture", r"all agents", r"status report",
)

_PATTERNS = {
    scenario: tuple(re.compile(rf"\b{p}\b", re.I) for p in patterns)
    for scenario, patterns in _SIGNALS.items()
}
_FULL_PATTERNS = tuple(re.compile(rf"\b{p}\b", re.I) for p in _FULL_SIGNALS)


@dataclass(frozen=True)
class Route:
    """Which agents to run, and why."""

    scenario: RouteScenario
    agents: tuple
    reason: str
    matched: tuple = field(default_factory=tuple)

    @property
    def is_narrow(self):
        return len(self.agents) < len(ALL_AGENTS)

    @property
    def agents_skipped(self):
        return tuple(a for a in ALL_AGENTS if a not in self.agents)


class ScenarioRouter:
    """Picks the smallest agent set that can answer the current question."""

    def __init__(self, agent_sets=None, all_agents=ALL_AGENTS):
        self.agent_sets = dict(AGENT_SETS if agent_sets is None else agent_sets)
        self.all_agents = tuple(all_agents)
        self.routed = {}

    def route(self, trigger=None, context=None):
        """Route a trigger: a RouteScenario, a free-text query, or nothing.

        `context` is the dispatch context. A transcript in it is treated as a
        witness input on its own — a statement arriving *is* the trigger,
        whatever words came with it.
        """
        context = context or {}
        route = self._classify(trigger, context)
        self.routed[route.scenario.value] = self.routed.get(route.scenario.value, 0) + 1
        return route

    def _classify(self, trigger, context):
        if isinstance(trigger, RouteScenario):
            return self._build({trigger}, f"explicit scenario {trigger.value}", ())

        text = trigger if isinstance(trigger, str) else ""
        text = " ".join(filter(None, [text, str(context.get("query") or "")]))

        signals, matched = set(), []
        if context.get("statement") or context.get("transcript"):
            signals.add(RouteScenario.WITNESS_INPUT)
            matched.append("transcript in context")

        for pattern in _FULL_PATTERNS:
            hit = pattern.search(text)
            if hit:
                return self._build({RouteScenario.FULL_SEARCH_BRIEFING},
                                   f"asked for a full briefing ({hit.group(0)!r})",
                                   (hit.group(0),))

        for scenario, patterns in _PATTERNS.items():
            for pattern in patterns:
                hit = pattern.search(text)
                if hit:
                    signals.add(scenario)
                    matched.append(hit.group(0))
                    break

        if not signals:
            # Nothing recognised. Missing an agent in a live search costs more
            # than an API call, so an unrecognised query runs everything.
            return self._build({RouteScenario.FULL_SEARCH_BRIEFING},
                               "nothing recognised, widening to every agent", ())

        if len(signals) > 1:
            return self._build(
                signals,
                "several topics recognised, running the union rather than choosing",
                tuple(matched),
            )

        only = next(iter(signals))
        return self._build({only}, f"matched {matched[0]!r}", tuple(matched))

    def _build(self, scenarios, reason, matched):
        wanted = set()
        for scenario in scenarios:
            wanted.update(self.agent_sets.get(scenario, self.all_agents))
        # Canonical order always, so dependencies hold whatever matched.
        agents = tuple(a for a in self.all_agents if a in wanted)
        scenario = (next(iter(scenarios)) if len(scenarios) == 1
                    else RouteScenario.FULL_SEARCH_BRIEFING)
        return Route(scenario=scenario, agents=agents, reason=reason, matched=matched)
