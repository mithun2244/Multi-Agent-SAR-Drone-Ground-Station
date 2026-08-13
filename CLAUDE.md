# Project: SAR Drone Ground Station

## Rules
- Read `docs/architecture.md` before starting work on any phase.
- All agent outputs must strictly conform to the schema in `docs/phase0_contract.md`.
- Keep code clean, modular, and minimal using standard Python libraries.
- Do not move to the next phase until the exit criteria of the current phase are satisfied.

## Phase status

- **Phase 0 — Contracts & spine: schema done.** `ClueContract` lives in
  `src/contracts/clue.py`; `docs/phase0_contract.md` mirrors it. The Redis Streams
  bus and case blackboard are **not** built yet.
- **Phase 1 — Data & evaluation harness: complete.** Exit criteria met: the harness
  reports mAP, recall at a fixed false-alarm rate, and geolocation error for a
  baseline RGB-only run.
- **Phase 2 — Detection end-to-end: complete.** A sensor-agnostic pipeline
  (only fitted sensors run a model; WBF only with two feeds), detector stubs,
  BoT-SORT tracking, WGS84 geodesic geolocation with DEM ray intersection, and
  clue emission to Redis Streams. Deferred to later phases: real GMC from
  frame registration, a real raster DEM source, and real detector models.
- **Phase 3 — Minimal command plane: complete.** Orchestrator (routing table),
  case blackboard, and coordinator fusion. Exit criteria met: an operator query
  routes through the orchestrator, fusion reads the resulting clues off the bus,
  and a ranked picture comes back.
- **Phase 4 — Weather agent & real fusion: complete.** Weather agent (Open-Meteo
  over stdlib urllib), weather absorbed as case context, urgency-weighted
  ranking. Exit criteria met: two sources fuse into one picture and the survival
  window measurably reorders it.
- **Phase 5 — Reasoning agents: complete.** All five built (Path, Scene, Health,
  History, Interview), all emitting `ClueContract` to the bus and consumed by
  fusion. Exit criteria met per agent: each one's output measurably changes the
  fused picture. Deferred to Phase 7: the real prompt-injection classifier and
  the hardened provenance guard (minimal versions are in place).
- **Phase 6 — Guardrails, rate protection and scenario routing: complete.**
  Strict Pydantic parsing with one repair pass, a TTL response cache, the
  contradiction guard, and the scenario router. A targeted question now
  dispatches one agent instead of seven.
- **Phase 7 — Security, provenance & trust boundaries: complete.** Provenance
  allow-list, blackboard guard, security audit log, geofence, contract-level
  coordinate bounds, the imagery tamper check (SHA-256 integrity + header sniff
  + behavioural divergence), and an adversarial suite covering every vector.
- **Phase 8 — Critic / evaluator loop: complete.** Outcomes, metrics, per-agent
  counterfactuals and the objective loss. The retuning it feeds is Phase 9.
- **Phase 9 — Hyperparameter optimization & retuning: complete.** Optuna TPE
  study against `CriticReport.loss`, tuned config persisted to
  `config/tuned_params.json` and loaded at startup, with a before/after report.

**All nine phases are complete.** What remains is real-world work the code
cannot do for itself: real detector weights, real recorded footage, a raster
DEM, live NIM and Redis endpoints, and a tuning run against real data rather
than the simulator.

## Directory map

```
src/
├── geometry.py              iou(), shared by perception and evaluation
├── bus.py                   RedisBus over Redis Streams (+ FakeRedisStreams)
├── contracts/
│   └── clue.py              ClueContract — single source of truth for the schema
├── evaluation/              Phase 1 harness
│   ├── metrics.py           mAP, recall@FAR, geolocation error
│   ├── dataset.py           loaders, fixed splits, mock data, Detection.from_clue
│   ├── harness.py           CLI + report
│   └── test_evaluation.py   23 checks
├── perception/              Phase 2
│   ├── fusion.py            Weighted Box Fusion (pixel-level, one frame)
│   ├── detectors.py         YOLO11m / LiDAR stubs
│   ├── tracking.py          BoT-SORT: Kalman, two-stage association, CMC
│   ├── terrain.py           DEM sources (constant, lat/lon grid, JSON loader)
│   ├── geolocation.py       WGS84 geodesy, DEM ray march, range selection
│   ├── agent.py             the vertical slice: sensors → … → bus
│   └── test_perception.py   66 checks
├── coordinator/             Phase 3 — the command plane
│   ├── blackboard.py        cases, targets, environment, bus cursor (dumb store)
│   ├── fusion.py            clue accumulation, dedupe, urgency, ranked picture
│   ├── orchestrator.py      dispatch loop; asks the router which agents to run
│   ├── router.py            trigger/query → smallest agent set that answers it
│   ├── demo.py              operator query → dispatch → picture
│   ├── mock_drone_publisher.py  mock airframe → real pipeline → bus (dev feed)
│   └── test_coordinator.py  85 checks
└── agents/                  Phase 4-5 — the reasoning plane
    ├── llm.py               NVIDIA NIM over stdlib urllib (text + vision)
    ├── weather.py           Open-Meteo, wind chill, hypothermia risk + window
    ├── path.py              Monte-Carlo sectors; LLM writes the briefing only
    ├── scene.py             VLM, gated on confirmed detections
    ├── health.py            subject-specific survival window (clamped multiplier)
    ├── history.py           TF-IDF RAG over a case archive + provenance allow-list
    ├── interview.py         NER over untrusted witness text (fast model)
    └── test_agents.py       63 checks
└── guardrails/              Phase 6 — what stands between a model and the bus
    ├── schemas.py           Pydantic reply models (Scene/Health/Interview/Text)
    ├── parsers.py           strict parse, one repair pass, then discard
    ├── cache.py             LRU + TTL response cache for NIM rate limits
    ├── contradiction.py     model prose checked against perception facts
    ├── provenance.py        the allow-list: origins, identities, geofence
    ├── tamper.py            frame integrity (SHA-256, header) + divergence
    ├── audit.py             security events for anything refused
    ├── test_guardrails.py   51 checks
    └── test_adversarial.py  27 crafted attacks, each blocked and logged
├── critic/                  Phase 8 — scoring the picture against reality
│   ├── outcomes.py          ground truth: Subject, CaseOutcome, OutcomeLog
│   ├── metrics.py           matching, residuals, NDCG, Kendall tau, FP penalty
│   ├── critic.py            the report, the loss, per-agent counterfactuals
│   ├── demo.py              fly a case, resolve it, score it
│   └── test_critic.py       34 checks
└── tuning/                  Phase 9 — search the configuration space
    ├── params.py            every tunable, and config/tuned_params.json
    ├── scenario.py          one repeatable search, run end to end
    ├── objective.py         Optuna TPE study against CriticReport.loss
    ├── demo.py              baseline vs tuned, side by side
    └── test_tuning.py       25 checks
```

How fusion treats each source (`coordinator/fusion.py`). Every `AgentSource` has
exactly one role, and a test fails if one is ever left unrouted:

| Kind | Sources | Effect |
|---|---|---|
| detection | `PERCEPTION_FUSION`, `DRONE_*` | creates/updates targets, sets confidence |
| context | `WEATHER_API`, `HEALTH_LLM` | conditions + survival window → urgency |
| prediction | `PATH_MODEL` | search sectors → priority prior |
| annotation | `SCENE_VLM` | target profile + hazards → urgency |
| advisory | `HISTORY_RAG`, `INTERVIEW_LLM` | operator notes; change nothing else |

Only **detection** sources touch confidence.

Full flow: `fitted detectors → [WBF] → tracking → geolocation → bus → coordinator
fusion → picture`. Each stage takes and returns plain values or `ClueContract`,
so stages can be tested and replaced one at a time. WBF is in brackets because
it is conditional: the pipeline is sensor-agnostic, runs only the detectors for
sensors actually fitted and delivering a frame, and fuses only when there is
more than one feed to reconcile. Everything after the detectors takes boxes and
does not care where they came from.

Two different things are called fusion. `perception/fusion.py` merges *boxes*
from two sensors on one frame; `coordinator/fusion.py` merges *clues* from any
producer across the whole case.

## Commands

```
python -m src.evaluation.harness            # baseline metrics report
python -m src.evaluation.test_evaluation    # Phase 1 checks
python -m src.perception.test_perception    # Phase 2 checks
python -m src.perception.fusion             # worked WBF example
python -m src.perception.agent              # full pipeline, clues landing on the bus
REDIS_URL=redis://localhost:6379/0 python -m src.perception.agent   # against a real server
python -m src.coordinator.test_coordinator  # Phase 3 checks
python -m src.agents.test_agents            # Phase 4-5 checks
python -m src.guardrails.test_guardrails    # Phase 6-7 checks
python -m src.guardrails.test_adversarial   # Phase 7 crafted attacks
python -m src.critic.test_critic            # Phase 8 checks
python -m src.coordinator.demo              # operator query → dispatch → ranked picture
python -m src.critic.demo                   # fly a case, resolve it, score it
python -m src.tuning.test_tuning            # Phase 9 checks
python -m src.tuning.demo                   # run the study, baseline vs tuned
python -m src.tuning.demo --reuse           # score the saved config without searching
python -m src.coordinator.demo --live-weather   # hits Open-Meteo instead of fixed conditions
python -m src.coordinator.mock_drone_publisher          # mock airframe onto a live Redis
python -m src.coordinator.mock_drone_publisher --check  # offline: the feed reaches a picture
CASE_ID=case-mock-drone python -m src.coordinator.demo  # a coordinator joins that case
```

## Conventions

- Run modules with `python -m src.…` from the project root; the packages use
  relative imports and rely on namespace packages (no `__init__.py`).
- Tests are `assert`-based, dependency-free, and run as `python -m`. No pytest.
- Third-party dependencies: `pydantic` (contracts), `pyproj` (geodesy in
  `perception/geolocation.py`), `redis` (the bus). Metrics, geometry, fusion,
  tracking and terrain are pure stdlib — keep it that way.
- Mock the Redis *connection*, never the bus. Tests inject `FakeRedisStreams`
  into a real `RedisBus`, so entry-id parsing, field encoding, XREAD resume
  semantics and bytes/str decoding are all still under test. A test double one
  layer up would only test itself.
- Never emit a position that was not actually computed. No placeholder, assumed,
  or flat-earth coordinates: a wrong position sends a ground team to the wrong
  valley. When a fix cannot be had, publish the sighting with `latitude`/
  `longitude` as `None` and `agent_metadata["geolocation"] = "no_fix"` — do not
  drop the detection either, since "seen but not located" is real information.
- Detectors are injected as callables, never imported by name into the harness,
  so a real model replaces a stub without touching scoring code.
- Optional on the contract, required at the consumer: `frame_id` and
  `bounding_box` are optional so non-detection agents can omit them, and
  `ClueContract.detection_box()` enforces them where a clue must be located.
- Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling and the
  upgrade path.
- The RGB detector is **YOLO11m** (Medium), everywhere and without exception —
  code, docs and the Phase 1 baseline all name the same model. A smaller variant
  was trialled for demo latency and dropped: what it gives up is recall on small,
  distant and partly occluded subjects, which is the case a search exists for. If
  latency on the airframe forces the question again, settle it with the harness
  (mAP and recall@FAR, both models) rather than by swapping the name.
- Repeated observations from one source never raise confidence. Correlated
  evidence combined as if independent manufactures false certainty. Confidence is
  the best observation *per source*, combined across *distinct* sources with a
  trust-weighted noisy-OR in `coordinator/fusion.py`.
- Confidence, urgency and the sector prior are separate numbers. Ranking uses
  `priority = confidence x (1 + urgency_weight x urgency + sector_weight x
  sector_prior)`. Nothing but a detection may move confidence — a cold night or
  a likely sector does not make a sighting more real, it changes which believed
  target you reach first.
- An agent that only runs *because* another agent fired is not corroboration.
  Scene is triggered by detection, so its agreement is guaranteed and worth
  nothing as evidence; it annotates and can raise urgency, never confidence.
- LLMs write prose over computed facts, never the facts. The Path model fixes
  its sectors before the model is called, and the prompt says so. If the model
  is unreachable, the sectors still ship with a clearly labelled
  `computed-fallback` summary.
- Where a model must touch a number, it returns a **bounded adjustment**, never
  the number. Health returns a multiplier on the computed survival window,
  clamped to [0.25, 2.0], and anything unparseable falls back to 1.0. The worst
  an unreliable model can do is scale a defensible number; it can never invent
  one.
- Witness text is the only untrusted input in the system. Interview clues carry
  **no `spatial_context`**, so fusion structurally cannot turn a statement into a
  sighting — only a sensor puts someone on the map. Suspected injections are
  flagged and discounted, not dropped.
- Retrieval is filtered by a provenance allow-list *before* anything reaches a
  model, and blocked records do not even influence IDF scoring.
- Every model reply crosses a Pydantic schema in `guardrails/`. One repair pass
  fixes *syntax* only — fences, trailing commas, Python literals. Nothing ever
  invents a missing field, and two failures discard the reply for the caller's
  safe default.
- A model may not deny what a sensor measured. `guardrails/contradiction.py`
  checks advisory prose against perception facts: denial of a confirmed
  detection is **overridden**, count and elevation mismatches are **flagged**.
  The guard runs at picture time on snapshots, so the blackboard keeps the raw
  model output for audit while the operator sees the guarded text.
- Wrap completers in a shared `ResponseCache` when running live. Failures are
  never cached, images are part of the key, and entries carry a TTL so a
  weather-derived briefing cannot go stale.
- Routing is biased to widen, never narrow. An unrecognised query runs every
  agent and a multi-topic query runs the union — losing an agent's contribution
  to a live search is not comparable to spending an API call. Classification is
  keyword matching in `router.py`, deliberately not a model call.
- `ALL_AGENTS` in `router.py` is the canonical dispatch order and encodes
  dependencies: detection before scene, weather before health. Every route is
  built by filtering that tuple, so routing can never reorder agents.
- `provenance_tag` values live in `guardrails/provenance.py` and agents import
  them. A tag names a **component**, never a model version, so changing the
  model behind an agent does not invalidate every clue it has ever published.
- **Every bus consumer filters by provenance, not just fusion.** Fusion guards
  the blackboard, but an unverified clue reaching another consumer still spends
  an API call and can put a forged frame in front of a VLM. See `_trusted` in
  `coordinator/demo.py`.
- A rejected clue is evidence: log it as a security event with what was claimed
  and why it was refused. `AuditLog` bounds the detail buffer but never the
  counters, so a flood cannot hide how often it happened.
- Physically impossible values are refused by the **contract**, not by whoever
  reads them first — `SpatialContext` bounds latitude, longitude, altitude and
  box arity. A `Geofence` then refuses *possible* coordinates that are nowhere
  near the search.
- Never hand a model a frame you have not verified. `FrameLedger.loader()` wraps
  an image source so a tampered or unknown frame returns `None`, which the Scene
  agent already treats as "do not describe" — no API call, no clue.
- Every new guard needs a crafted attack in `test_adversarial.py`, asserting all
  three of: the attack does not land, legitimate work continues, and the attempt
  is in the audit log.
- The critic is strictly read-only. It scores detached picture snapshots and
  never writes back — it runs while teams are on the hill, and a scoring pass
  that nudged a ranking would be changing what it claims to measure.
- Agents are scored by **counterfactual, not correlation**: re-rank the same
  targets with one signal removed and see if the ranking got worse. Fusion
  computes the signal components (`weather_urgency`, `hazard_urgency`,
  `baseline_urgency`, `sector_probability`) so the critic re-weights them rather
  than re-deriving the urgency maths and letting the two drift.
- A case where nobody was found is **not scorable**. Treating "never found" as
  "was not there" would train the critic to reward giving up.
- Every tunable lives in `tuning/params.py`, and its default is the value the
  code shipped with — so `TunedParams()` is the baseline and a study that cannot
  beat it changes nothing. A missing `config/tuned_params.json` means untuned,
  not broken.
- Tune against the critic's own loss, never a proxy invented for the optimiser,
  and always over several folds. A configuration that wins on one draw of the
  noise has learned nothing.
- Tuning results come from a **simulator**: real pipeline, stubbed imagery. They
  are a starting point for a run against real recordings, not a substitute.
- External APIs are called through an injected fetcher/completer (see
  `agents/weather.py`, `agents/llm.py`). Tests and demos run offline against
  fixed data; nothing in the suite depends on a third party's uptime, a key, or
  a quota. `NVIDIA_API_KEY` opts into live NIM calls.
- LLM and VLM inference is **NVIDIA NIM** at `https://integrate.api.nvidia.com/v1`
  (`meta/llama-3.1-70b-instruct` for text, `meta/llama-3.2-90b-vision-instruct`
  for vision). It is OpenAI chat-completions shaped and reached with stdlib
  `urllib` — no `openai` package. Images inline as an `<img src="data:...">` tag
  in the message content, which is NIM's documented form, and one over the inline
  size limit is refused rather than truncated.
- The Scene VLM runs once per frame that already holds a *confirmed* track, and
  never on a frame whose image it was not handed. Empty hillside costs nothing.
- A picture is a snapshot, not a live view. `fusion.picture()` returns detached
  copies so state cannot change under whoever is reading it.
- A measured LiDAR range always overrides an RGB range inferred from terrain.
  This is enforced in one place, `geolocation.select_range`, and it is never a
  confidence comparison — a confident inference must not outrank a measurement.