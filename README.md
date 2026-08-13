# Autonomous Multi-Agent SAR Drone Ground Station

An event-driven ground station for mountain search-and-rescue. A drone's RGB and
LiDAR sensors feed a perception pipeline; a coordinator fuses what it finds with
weather, terrain simulation, historical cases and witness statements; and an
operator gets one ranked picture of who is out there, where, and who to reach
first.

Built in nine phases, each with its own exit criteria and test suite.
**393 automated checks** across eight suites, no network required.

```
sensors ─▶ WBF ─▶ BoT-SORT ─▶ geolocation ─▶ Redis Streams ─▶ fusion ─▶ picture
                                                  ▲                        │
   weather · path · scene · health · history ─────┘                        ▼
                                                                    critic ─▶ Optuna
```

---

## System overview

### Perception plane — on the drone, at frame rate

| Stage | What it does |
|---|---|
| **YOLO11m + LiDAR** | RGB classifies, LiDAR ranges. Two modalities, so a target visible to one and not the other is suspect. |
| **Weighted Box Fusion** | Merges both sensors' boxes into one confirmed target carrying an RGB class *and* a measured LiDAR range. Support is counted per **distinct detector**, not per box — two RGB boxes are one opinion. |
| **BoT-SORT tracking** | Constant-velocity Kalman filter on `(cx, cy, w, h)`, ByteTrack two-stage association, and camera-motion compensation so drone movement is not mistaken for target movement. |
| **Geodesic geolocation** | WGS84 geodesics via `pyproj`, and **DEM ray-marching** to find where the line of sight actually strikes the ground. A *measured* LiDAR range always overrides a terrain-inferred one. |

**Terrain** comes from real elevation tiles. `SrtmHgtDEM` reads NASA `.hgt`
files with **no dependency at all** — the format is a raw big-endian int16
square with its corner in the filename, so there is nothing to guess.
`GeoTiffDEM` handles GeoTIFF/Copernicus tiles through `rasterio`, which stays
optional because GeoTIFF is a container with dozens of legal encodings and a
half-correct parser would produce a silently wrong altitude rather than a crash.
Both sit behind the same three-member interface as `ConstantDEM` and `GridDEM`,
so ray-marching swaps between them without a line changing.

SRTM voids are not elevations: a hole is excluded from interpolation, and a ray
that marches into one produces **no fix** rather than an assumed one.

The Kalman filter is four independent 2×2 filters, not an approximation:
BoT-SORT's 8-D filter decomposes exactly, since its transition and covariances
are diagonal per dimension — which also removes any need for matrix inversion.

`world → pixel → geolocate → world` round-trips to **under 1 cm**, and the
RGB-only path recovers a point standing on terrain from its pixel using the DEM
alone to under 5 cm.

### Spine — Redis Streams

One stream per case (`clues:<case_id>`). Every producer emits the same Phase 0
`ClueContract`; consumers resume from the last entry id they saw. Clues are
serialised on publish, so anything unserialisable fails in a test rather than in
flight.

### Reasoning plane — NVIDIA NIM

Six agents, all reached through an injected completer so the suite never touches
the network.

| Agent | Model | Role |
|---|---|---|
| **Weather** | Open-Meteo (no LLM) | Wind chill, hypothermia risk, survival window |
| **Path** | `meta/llama-3.1-70b-instruct` | **Monte-Carlo** sector simulation; the LLM only writes the briefing |
| **Scene** | `meta/llama-3.2-90b-vision-instruct` | VLM, **only** on frames with a confirmed detection |
| **Health** | `meta/llama-3.1-70b-instruct` | Subject-specific survival window |
| **History** | `meta/llama-3.1-70b-instruct` | TF-IDF RAG over a case archive |
| **Interview** | `meta/llama-3.1-8b-instruct` | NER over untrusted witness transcripts |

NIM is reached with stdlib `urllib` against its OpenAI-compatible
`chat/completions` endpoint — **no SDK, no `openai` package**.

### Command plane — the coordinator

A scenario router dispatches only the agents that can answer the question, then
fusion accumulates clues into a ranked picture.

```
priority = confidence × (1 + urgency_weight·urgency + sector_weight·sector_prior)
```

Three separate numbers, deliberately:

- **Confidence** is belief. Only *detection* sources may move it.
- **Urgency** is how fast to get there — weather, a closing survival window, visible hazards.
- **Sector prior** is where the Monte-Carlo model expects the subject to be.

A cold night does not make a detection more likely to be *real*; it changes
which believed target you reach first. Folding them into one score would corrupt
the number the critic learns against.

---

## Design decisions worth knowing

These shaped the build more than any library choice.

- **Repeated sightings never inflate confidence.** Ten frames of one camera is
  one opinion observed ten times. Confidence is the best observation *per
  source*, combined across *distinct* sources with a trust-weighted noisy-OR.
- **An agent that only runs because another fired is not corroboration.** Scene
  is triggered by detection, so its agreement is guaranteed and worth nothing as
  evidence. It annotates and can raise urgency; it never touches confidence.
- **LLMs write prose over computed facts, never the facts.** Path fixes its
  sectors before the model is called and the prompt says so. Where a model must
  touch a number, it returns a **bounded adjustment** — Health returns a
  multiplier clamped to `[0.25, 2.0]`, never a survival window of its own.
- **Never emit a position that was not computed.** No placeholder coordinates.
  A sighting that cannot be geolocated is published with `latitude`/`longitude`
  as `None` and `geolocation: "no_fix"` — not dropped, because "seen but not
  located" is real information in a search.
- **A case where nobody was found is not scorable.** Treating "never found" as
  "was not there" would train the critic to reward giving up.

---

## Security and guardrails

| Layer | What it stops |
|---|---|
| **Pydantic coordinate bounds** | `latitude ∈ [-90, 90]`, `longitude ∈ [-180, 180]`, bounded altitude, 4-element boxes. A forged payload is impossible to *construct*. |
| **Geofence** | *Plausible* coordinates nowhere near the search — the classic way to pull teams off the right ground. |
| **Provenance allow-list** | Registered origins, each bound to the agent permitted to use it, plus device / endpoint / operator identity where data crosses in from outside. |
| **Frame integrity** | SHA-256 recorded at capture; a frame that does not hash to it is not the frame the sensor took. Header sniffing stops a non-image payload reaching a VLM. |
| **Behavioural divergence** | Detections that do not survive a transform — for tampering that happened *before* capture. |
| **Contradiction guard** | A model summary denying a confirmed detection is **withheld and replaced with the facts**. |
| **Prompt-injection isolation** | Interview clues carry **no `spatial_context`**, so fusion structurally cannot turn a witness statement into a sighting. |
| **Output validation** | Every model reply crosses a Pydantic schema. One repair pass fixes *syntax* only; nothing invents a missing field. |
| **Audit log** | Every rejection recorded. The detail buffer is bounded; the counters are not. |

**27/27 adversarial vectors blocked and logged** — forged tags, borrowed tags,
rogue airframes, unauthorised endpoints, poisoned archive records, four
stand-down injections, tampered and substituted frames, shell-script payloads,
impossible coordinates, and wrong-valley positions. Every test asserts three
things: the attack does not land, **legitimate work continues**, and the attempt
is in the audit log.

> Every bus consumer filters by provenance, not just the blackboard. An
> unverified clue reaching any other consumer still spends an API call and can
> put a forged frame in front of a model.

---

## Optimization results

Optuna TPE search wired directly to `CriticReport.loss` — the objective is the
critic's own loss, not a proxy invented for the optimiser.

```
metric                      baseline       tuned
---------------------------------------------------------
mAP (Phase 1 harness)         0.4259      0.7778   better
recall @ target FAR           0.4259      0.7778   better
subject recall                0.7778      1.0000   better
NDCG (ranking)                1.0000      1.0000        =
geolocation median              0.11 m      0.09 m better
false alarms / frame          0.0000      0.0000        =
CRITIC LOSS                   0.4455      0.0009   better
```

- **Loss 0.4455 → 0.0009**, recall 0.778 → 1.000, over a 60-trial study (~6 s).
- **Sub-metre geolocation**: median residual 0.09 m in the tuning scenario, 0.6 m
  end-to-end in the critic demo.
- **Response cache: 9 of 17 model calls avoided (53%)** on a two-dispatch run,
  with TTL expiry and per-image keys. Failures are never cached — an outage is a
  fact about one moment, not an answer to remember.
- Scenario routing cuts a targeted question from **7 agents to 1**.

Tuned parameters persist to `config/tuned_params.json` and load at startup.
Sixteen parameters changed; the search independently discovered the
**modality cross-check**, leaning trust toward LiDAR (0.963 vs RGB 0.741)
because the scenario's decoys are visible to RGB alone.

> **Honest caveat.** The pipeline under test is real end to end, but the
> **imagery is stubbed** — subjects are projected from known world positions and
> detected by stubs with configured recall. These results measure how the system
> is *configured*, against a simulator. They are a starting point for a tuning
> run on real recordings, not a substitute for one.

---

## Tech stack

| | |
|---|---|
| **Language** | Python 3.12, standard library first |
| **Contracts** | `pydantic` v2 |
| **Geodesy** | `pyproj` (WGS84) |
| **Terrain** | stdlib `.hgt` reader; `rasterio` *optional*, for GeoTIFF |
| **Bus** | `redis` (Redis Streams) |
| **Optimisation** | `optuna` (TPE) |
| **Inference** | NVIDIA NIM via stdlib `urllib` — no SDK |
| **Weather** | Open-Meteo via stdlib `urllib` — no key |
| **Tests** | `assert`-based, no framework, no network |

Four third-party packages total. Metrics, geometry, fusion, tracking, terrain,
guardrails and the critic are **pure standard library**.

---

## Directory map

```
src/
├── geometry.py              IoU and geodesic distance, shared by every plane
├── bus.py                   RedisBus over Redis Streams (+ FakeRedisStreams)
├── contracts/clue.py        ClueContract — one schema for every agent
├── evaluation/              Phase 1 — mAP, recall@FAR, geolocation error
├── perception/              Phase 2 — WBF, BoT-SORT, DEM geolocation, agent
├── coordinator/             Phase 3 — blackboard, fusion, orchestrator, router
├── agents/                  Phases 4-5 — weather, path, scene, health, history, interview
├── guardrails/              Phases 6-7 — schemas, cache, contradiction, provenance, tamper
├── critic/                  Phase 8 — outcomes, metrics, loss, agent counterfactuals
└── tuning/                  Phase 9 — params, scenario, Optuna objective
                             + live_system_check.py — the one online script
config/tuned_params.json     the tuned operating point
docs/architecture.md         the design this was built from
```

### Phase map

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Clue contract + Redis Streams spine | ✅ |
| 1 | Evaluation harness (mAP, recall@FAR, geo error) | ✅ |
| 2 | Detection end-to-end: WBF, BoT-SORT, geolocation | ✅ |
| 3 | Minimal command plane: blackboard, fusion, orchestrator | ✅ |
| 4 | Weather agent and real corroboration | ✅ |
| 5 | Five reasoning agents (Path, Scene, Health, History, Interview) | ✅ |
| 6 | Output parsers, response cache, contradiction guard, scenario routing | ✅ |
| 7 | Provenance, geofence, tamper checks, adversarial suite | ✅ |
| 8 | Critic: metrics, per-agent counterfactuals, objective loss | ✅ |
| 9 | Optuna study, tuned config persistence, before/after report | ✅ |

---

## Quickstart

```bash
pip install pydantic pyproj redis optuna
```

### Run the demos

```bash
python -m src.evaluation.harness       # Phase 1 baseline metrics
python -m src.perception.fusion        # a worked Weighted Box Fusion merge
python -m src.perception.agent         # sensors → geolocation → bus
python -m src.coordinator.demo         # operator query → routing → ranked picture
python -m src.critic.demo              # fly a case, resolve it, score it
python -m src.tuning.demo              # Optuna study, baseline vs tuned
```

### Run the tests

```bash
python -m src.evaluation.test_evaluation     #  23 checks
python -m src.perception.test_perception     #  75 checks
python -m src.coordinator.test_coordinator   #  95 checks
python -m src.agents.test_agents             #  63 checks
python -m src.guardrails.test_guardrails     #  51 checks
python -m src.guardrails.test_adversarial    #  27 crafted attacks
python -m src.critic.test_critic             #  34 checks
python -m src.tuning.test_tuning             #  25 checks
```

### Go live

Everything external is an injected callable, so each of these is a substitution
rather than a rewrite:

```bash
export NVIDIA_API_KEY=...                       # live NIM instead of stubs
export REDIS_URL=redis://localhost:6379/0       # real Redis instead of the fake

python -m src.tuning.live_system_check          # verify both, before you fly
python -m src.coordinator.demo --live-weather   # live Open-Meteo
```

`live_system_check` is the only thing in this repository that deliberately
touches the network. It answers what the offline suite structurally cannot: is
the key valid, is the endpoint reachable, does a round trip work *right now*. It
sends a minimal completion to each NIM model in use (text and vision), publishes
a probe `ClueContract` to a throwaway Redis stream and reads it back, compares
the parsed contract to what was sent, and deletes the stream afterwards — pass
or fail. An unset variable is reported as `SKIP` with the variable's name, never
as silence, and the API key is redacted from every error.

```
  [PASS] NIM meta/llama-3.1-70b-instruct     0.44s  'READY'
  [PASS] Redis round trip                    0.01s  XADD/XREAD ok, entry 1786…-0
```

Exit code is `0` when everything checked passed, `1` if anything failed, `2` if
nothing was checked at all. Without the variables the system still runs fully
offline on fixed replies — deterministic, and never dependent on a third party's
uptime or quota.

---

## What is not done

Stated plainly, because the code cannot supply it for itself:

- **Trained detector weights.** `yolo11m_stub` / `lidar_stub` model the *shape*
  of RGB and LiDAR behaviour — recall, confidence spread, the LiDAR range
  advantage — not a real network's output.
- **Real recorded RGB and LiDAR footage.** The Scene agent refuses to describe a
  frame it was not handed, and the frame ledger hashes captures at source, so
  both need real recordings.
- **A tuning run on real data.** The Optuna study optimises against a simulator;
  the numbers above are a starting point, not a field result.

Everything external is an injected callable, so each of these is a substitution
rather than a rewrite.

---

## Licence

[MIT](LICENSE).
