"""The dispatcher — operator trigger in, agents run, decision chain out.

Two jobs, in order:

  1. dispatch: ask the router which data-gathering agents can answer this
     trigger, run them, and let fusion absorb what they published;
  2. decide: run the picture through Reason -> Risk -> Recommend -> Orchestrate
     (`decision.py`) so the dispatch returns a commander brief and not just a
     table of targets.

Not to be confused with the *Orchestrate stage* inside the decision chain: this
class orchestrates agents, that stage orchestrates the three judgements about
what they found. This one is named for the command plane it fronts.

The operator's own words are the one untrusted input left in the build, so they
cross a guard before they are allowed to steer anything (`guardrails/injection`).
A flagged command is never obeyed and never dropped: it widens to the full agent
set and is logged, because narrowing a live search is the damage an injected
command could actually do.
"""

from dataclasses import dataclass, field
from enum import Enum

from ..guardrails.injection import guard_command
from .decision import decide
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
    brief: object = None            # what the decision chain sends the commander
    command_flags: tuple = ()       # injection tells in the operator's own words

    @property
    def target_count(self):
        return 0 if self.picture is None else len(self.picture.targets)

    @property
    def agents_skipped(self):
        return () if self.route is None else self.route.agents_skipped


class Orchestrator:
    """Routes a trigger to agents, then runs the picture through the decision chain."""

    def __init__(self, fusion, blackboard, router=None, params=None, commander=None,
                 audit=None):
        self.fusion = fusion
        self.blackboard = blackboard
        self.router = router if router is not None else ScenarioRouter()
        # The tuned configuration, handed to every agent through the dispatch
        # context so a handler reads settings from one place rather than each
        # agent loading its own copy off disk.
        self.params = params
        # Whoever receives the brief. Nothing is required — the brief comes back
        # on the dispatch either way.
        self.commander = commander
        # Same log fusion writes rejected clues to, so an operator sees one
        # security picture rather than two.
        self.audit = audit if audit is not None else getattr(fusion, "audit", None)
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
        """Dispatch a trigger and return the picture and the commander brief.

        `trigger` may be a legacy `Scenario`, a `RouteScenario`, or the
        operator's own words — the router works out which agents can answer it.

        Order matters: agents run first and publish to the bus, then fusion
        reads what they published, then the decision chain reads the picture.
        Asking for the picture first would answer with the state from before
        the query, and deciding on it would decide on stale facts.
        """
        case = self.blackboard.case(case_id)
        if not case.is_open:
            raise ValueError(f"case {case_id} is {case.status}, refusing to dispatch")

        if self.params is not None:
            context.setdefault("params", self.params)

        # The one untrusted input left: what the operator typed. Flagged text
        # still gets a dispatch — it just does not get to choose a narrow one.
        wanted = self._as_trigger(trigger, context)
        check = guard_command(wanted if isinstance(wanted, str) else "",
                              audit=self.audit, case_id=case_id)
        route = self.router.route(
            RouteScenario.FULL_SEARCH_BRIEFING if check.flags else wanted, context)

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

        picture = self.fusion.refresh(case_id)
        return Dispatch(
            scenario=route.scenario,
            case_id=case_id,
            agents=tuple(route.agents),
            results=results,
            picture=picture,
            route=route,
            brief=decide(picture, commander=self.commander),
            command_flags=check.flags,
        )

    @staticmethod
    def _as_trigger(trigger, context):
        """Legacy `Scenario` values map onto route scenarios; anything else goes
        to the router as-is."""
        if isinstance(trigger, Scenario):
            mapped = LEGACY_SCENARIOS.get(trigger)
            return mapped if mapped is not None else context.get("query")
        return trigger
