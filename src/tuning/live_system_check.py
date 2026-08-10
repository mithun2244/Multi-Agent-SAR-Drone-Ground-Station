"""Pre-flight check against the live services, before a demo or a sortie.

    python -m src.tuning.live_system_check

Everything else in this repository runs offline against stubs on purpose. This
is the one place that deliberately touches the real world, so it exists to
answer the question the test suite structurally cannot: *is the key valid, is
the endpoint reachable, and does a round trip actually work right now?*

Each check does a real round trip and verifies the value that came back:

  * **NVIDIA NIM** — a minimal chat completion on each model in use, asserting
    non-empty text. A 200 with an empty body is a failure, not a pass.
  * **Redis** — publish a probe clue to a throwaway stream, read it back, and
    compare the parsed contract to the one sent. A ping only proves the socket
    opened; this proves the contract survives the wire.

Nothing is skipped silently. An unset variable is reported as SKIPPED with the
name of the variable, because "no output" and "no key" must not look alike.

Cleans up after itself: the probe stream is deleted, pass or fail. It never
touches a `clues:*` stream, so it cannot disturb a live search.
"""

import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..agents.llm import DEFAULT_LLM_MODEL, DEFAULT_VLM_MODEL, FAST_LLM_MODEL, NimClient
from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.provenance import TAG_PATH

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Deliberately not "clues:*": a probe must never land on a stream a coordinator
# is reading, even by accident.
PROBE_STREAM = "healthcheck:live-system-check"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == PASS

    def render(self):
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}[self.status]
        timing = f"{self.seconds:6.2f}s" if self.seconds else "       "
        return f"  [{mark}] {self.name:<34}{timing}  {self.detail}"


def check_nim(models=None, timeout=30.0):
    """One minimal completion per model, verifying real text comes back."""
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return [Check("NVIDIA NIM", SKIP, "NVIDIA_API_KEY is not set")]

    models = models or (DEFAULT_LLM_MODEL, FAST_LLM_MODEL)
    client = NimClient(key, timeout=timeout)
    checks = []

    for model in models:
        started = time.monotonic()
        try:
            reply = client.using(model=model).complete(
                "Reply with the single word: READY")
        except Exception as e:                        # noqa: BLE001 - report, never raise
            checks.append(Check(f"NIM {model}", FAIL, _reason(e),
                                time.monotonic() - started))
            continue

        elapsed = time.monotonic() - started
        text = (reply or "").strip()
        if not text:
            checks.append(Check(f"NIM {model}", FAIL, "empty completion", elapsed))
        else:
            checks.append(Check(f"NIM {model}", PASS,
                                f"{len(text)} chars: {text[:40]!r}", elapsed,
                                {"model": model}))
    return checks


def check_nim_vision(timeout=30.0):
    """The vision path separately: it uses a different model and payload shape.

    A 1x1 PNG is enough to prove the inline-image encoding is accepted — the
    failure this catches is a rejected request, not a poor description.
    """
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return Check("NIM vision", SKIP, "NVIDIA_API_KEY is not set")

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
        "00000049454e44ae426082"
    )
    started = time.monotonic()
    try:
        reply = NimClient(key, timeout=timeout).complete(
            "Reply with the single word: SEEN", image=png, mime_type="image/png")
    except Exception as e:                            # noqa: BLE001
        return Check("NIM vision", FAIL, _reason(e), time.monotonic() - started)

    elapsed = time.monotonic() - started
    text = (reply or "").strip()
    if not text:
        return Check("NIM vision", FAIL, "empty completion", elapsed)
    return Check(f"NIM {DEFAULT_VLM_MODEL}", PASS, f"{text[:40]!r}", elapsed)


def check_redis(timeout=5.0):
    """Publish a real clue, read it back, and compare it to what was sent."""
    url = os.environ.get("REDIS_URL")
    if not url:
        return [Check("Redis Streams", SKIP, "REDIS_URL is not set")]

    try:
        import redis
    except ImportError:
        return [Check("Redis Streams", FAIL, "pip install redis")]

    from ..bus import RedisBus

    checks = []
    started = time.monotonic()
    try:
        client = redis.Redis.from_url(url, decode_responses=True,
                                      socket_connect_timeout=timeout,
                                      socket_timeout=timeout)
        client.ping()
    except Exception as e:                            # noqa: BLE001
        return [Check("Redis connect", FAIL, _reason(e), time.monotonic() - started)]

    checks.append(Check("Redis connect", PASS, _server_line(client),
                        time.monotonic() - started))

    stream = f"{PROBE_STREAM}:{uuid.uuid4().hex[:8]}"
    bus = RedisBus(client)
    probe = _probe_clue()
    started = time.monotonic()
    try:
        entry_id = bus.publish(probe, stream=stream)
        delivered = bus.read(stream)
        elapsed = time.monotonic() - started

        if len(delivered) != 1:
            checks.append(Check("Redis round trip", FAIL,
                                f"published 1, read back {len(delivered)}", elapsed))
        elif delivered[0][1].model_dump() != probe.model_dump():
            checks.append(Check("Redis round trip", FAIL,
                                "clue did not survive the wire unchanged", elapsed))
        else:
            checks.append(Check("Redis round trip", PASS,
                                f"XADD/XREAD ok, entry {entry_id}", elapsed))
    except Exception as e:                            # noqa: BLE001
        checks.append(Check("Redis round trip", FAIL, _reason(e),
                            time.monotonic() - started))
    finally:
        # Always, so a failed probe does not leave rubbish on someone's server.
        try:
            client.delete(stream)
        except Exception:                             # noqa: BLE001
            checks.append(Check("Redis cleanup", FAIL, f"could not delete {stream}"))
    return checks


def _probe_clue():
    """A real ClueContract, so the check exercises the actual serialisation."""
    return ClueContract(
        clue_id=f"healthcheck-{uuid.uuid4()}",
        case_id="healthcheck",
        timestamp=datetime.now(timezone.utc),
        source_agent=AgentSource.PATH_MODEL,
        confidence_score=0.5,
        finding_summary="live system check probe",
        spatial_context=SpatialContext(latitude=46.8182, longitude=8.2275),
        provenance_tag=TAG_PATH,
        agent_metadata={"probe": True},
    )


def _server_line(client):
    try:
        info = client.info("server")
        return f"redis {info.get('redis_version', '?')}"
    except Exception:                                 # noqa: BLE001
        return "connected"


def _reason(error):
    """A short reason that cannot leak the API key."""
    text = f"{type(error).__name__}: {error}"
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        text = text.replace(key, "***")
    return text[:160]


def run_all():
    checks = list(check_nim())
    checks.append(check_nim_vision())
    checks.extend(check_redis())
    return checks


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    checks = run_all()

    print("\n  Live system check — the only part of this repo that touches the network\n")
    for check in checks:
        print(check.render())

    failed = [c for c in checks if c.status == FAIL]
    skipped = [c for c in checks if c.status == SKIP]
    passed = [c for c in checks if c.ok]

    print(f"\n  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print("  set NVIDIA_API_KEY and REDIS_URL to check the services that were skipped")
    if failed:
        print("  the system will still run offline on stubs; live agents will degrade\n")
        return 1
    if not passed:
        print("  nothing was checked\n")
        return 2
    print("  live endpoints are good\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
