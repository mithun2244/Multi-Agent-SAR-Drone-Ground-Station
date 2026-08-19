"""Ablation switches — turn one component off and measure what it was worth.

    ABLATION_WBF=off              python -m src.perception.agent
    ABLATION_CMC=off              python -m src.coordinator.mock_drone_publisher --check
    ABLATION_DISABLE_AGENTS=weather,path  python -m src.coordinator.demo
    ABLATION_DECISION=off         python -m src.coordinator.demo
    python -m src.utils.test_ablation

Every switch defaults to **on**, meaning the system built in Phases 0-9. An
unset environment is the shipped behaviour, which is why no existing test had to
change: a flag nobody sets does nothing.

Why this is not the critic's ablation
-------------------------------------
`critic/critic.py` already ablates *signals*: it re-ranks the same targets with
one agent's contribution removed and asks whether the ranking got worse. That is
cheap, exact, and runs on a picture that has already been produced.

These switches remove a *component* — the merge step, the motion model, an
agent's dispatch, the whole decision chain — so the pipeline actually runs
without it and the difference shows up in the metrics rather than in a
re-weighting. The critic answers "was this signal worth including?"; this
answers "was this stage worth building?", which is the question a write-up needs
and the one the Phase 9 study cannot ask.

Reading them here rather than at each site
------------------------------------------
One module means one place to look up what can be switched off, one spelling of
on/off, and one function a test can drive without setting process-wide
environment variables. Each site takes an explicit argument that defaults to
`None`, and `None` means "ask the environment" — the same shape as
`perception/models.perception_mode`.

Anything switched off is **stated in the run header**, never silent. An ablated
run that looks like a normal one is how a wrong number ends up in a report.
"""

import os
import sys

ON, OFF = "on", "off"

WBF = "ABLATION_WBF"
CMC = "ABLATION_CMC"
DECISION = "ABLATION_DECISION"
DISABLE_AGENTS = "ABLATION_DISABLE_AGENTS"

SWITCHES = (WBF, CMC, DECISION)

_TRUE = {"on", "1", "true", "yes", "enabled"}
_FALSE = {"off", "0", "false", "no", "disabled"}


class AblationError(ValueError):
    """An ablation variable that is not on or off."""


def enabled(name, override=None, env=None):
    """Is this component switched on? Explicit argument, then env, then on."""
    if override is not None:
        return bool(override)
    raw = (env if env is not None else os.environ).get(name)
    if raw is None or not str(raw).strip():
        return True
    value = str(raw).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise AblationError(
        f"{name}={raw!r} is not a switch; expected one of "
        f"{', '.join(sorted(_TRUE | _FALSE))}"
    )


def disabled_agents(override=None, env=None):
    """Agent names to keep out of every dispatch. Empty by default."""
    if override is not None:
        raw = override if isinstance(override, str) else ",".join(override)
    else:
        raw = (env if env is not None else os.environ).get(DISABLE_AGENTS, "")
    return frozenset(name.strip().lower() for name in str(raw).split(",") if name.strip())


def active(env=None):
    """Everything currently switched off, for a run header. Empty means shipped."""
    off = [name for name in SWITCHES if not enabled(name, env=env)]
    agents = disabled_agents(env=env)
    if agents:
        off.append(f"{DISABLE_AGENTS}={','.join(sorted(agents))}")
    return tuple(off)


def describe(env=None):
    """One line for a script header. Silence is not an option here."""
    off = active(env)
    return "ablations: none — full system" if not off else f"ABLATED: {', '.join(off)}"


if __name__ == "__main__":
    print(f"  {describe()}")
    sys.exit(0)
