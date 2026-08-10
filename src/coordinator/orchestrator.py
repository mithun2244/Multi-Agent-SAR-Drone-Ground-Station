"""The orchestrator — routes a scenario to the agents that can answer it.

Phase 3 (docs/architecture.md): "At this point the orchestrator is still a
simple router: the 'drone airborne' scenario dispatches detection."

So this is a routing table and a dispatch loop, and nothing more. It is
deliberately not the Phase 6 version, which decides "call only the smallest set
of agents that can answer the current question" across four scenarios and seven
agents. Only detection exists, so every route here names only detection.

The shape is already the right one for that though: routes are data, agents
register by name, and adding an agent in Phase 4 is a line in `ROUTES` plus a
`register` call, not a change to the dispatch logic.
"""

from dataclasses import dataclass, field
from enum import Enum

from .router import ALL_AGENTS, RouteScenario, ScenarioRouter


class Scenario(str, Enum):
    OPERATOR_QUERY = "OPERATOR_QUERY"    # "what have you got?"
    DRONE_AIRBORNE = "DRONE_AIRBORNE"    # a sortie is flying, run detection


# A sortie launching is a perception event whatever words came with it. A bare
# operator query is whatever its words turn out to mean, so it goes to the
# router to classify rather than being pinned here.
LEGACY_SCENARIOS = {
    Scenario.DRONE_AIRBORNE: RouteScenario.PERCEPTION_EVENT,
    Scenario.OPERATOR_QUERY: None,
}


@dataclass
class Dispatch:
    """What one scenario run actually did."""

    scenario: Scenario
    case_id: str
    agents: tuple = ()
    results: dict = field(default_factory=dict)
    picture: object = None
    route: object = None

    @property
    def target_count(self):
        return 0 if self.picture is None else len(self.picture.targets)

    @property
    def agents_skipped(self):
        return () if self.route is None else self.route.agents_skipped


class Orchestrator:
    """Routes scenarios to agents, then asks fusion for the current picture."""

    def __init__(self, fusion, blackboard, router=None, params=None):
        self.fusion = fusion
        self.blackboard = blackboard
        self.router = router if router is not None else ScenarioRouter()
        # The tuned configuration, handed to every agent through the dispatch
        # context so a handler reads settings from one place rather than each
        # agent loading its own copy off disk.
        self.params = params
        self._agents = {}

    def register(self, name, handler):
        """Register an agent. `handler(case_id, context) -> anything`."""
        if not callable(handler):
            raise TypeError(f"agent {name!r} handler must be callable")
        self._agents[name] = handler
        return self

    @property
    def agents(self):
        return sorted(self._agents)

    def handle(self, trigger, case_id, **context):
        """Dispatch a trigger and return the resulting picture.

        `trigger` may be a legacy `Scenario`, a `RouteScenario`, or the
        operator's own words — the router works out which agents can answer it.

        Order matters: agents run first and publish to the bus, then fusion
        reads what they published. Asking for the picture first would answer
        with the state from before the query.
        """
        case = self.blackboard.case(case_id)
        if not case.is_open:
            raise ValueError(f"case {case_id} is {case.status}, refusing to dispatch")

        if self.params is not None:
            context.setdefault("params", self.params)
        route = self.router.route(self._as_trigger(trigger, context), context)

        results = {}
        for name in route.agents:
            handler = self._agents.get(name)
            if handler is None:
                # A route naming an agent nobody registered means the picture
                # would silently be built on missing evidence.
                raise LookupError(
                    f"scenario {route.scenario.value} routes to {name!r}, which is not "
                    f"registered (have: {self.agents or 'none'})"
                )
            results[name] = handler(case_id, context)

        return Dispatch(
            scenario=route.scenario,
            case_id=case_id,
            agents=tuple(route.agents),
            results=results,
            picture=self.fusion.refresh(case_id),
            route=route,
        )

    @staticmethod
    def _as_trigger(trigger, context):
        """Legacy `Scenario` values map onto route scenarios; anything else goes
        to the router as-is."""
        if isinstance(trigger, Scenario):
            mapped = LEGACY_SCENARIOS.get(trigger)
            return mapped if mapped is not None else context.get("query")
        return trigger
