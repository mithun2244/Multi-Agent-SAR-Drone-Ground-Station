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
- **Phase 3 — Decision chain: complete.** Case blackboard, coordinator fusion,
  the dispatcher (routing table), and the four-stage chain in
  `coordinator/decision.py`: Reason → Risk → Recommend → Orchestrate →
  Commander. Exit criteria met: an operator query routes through the
  dispatcher, fusion reads the resulting clues off the bus, and a clue travels
  the whole chain to a validated commander brief.
- **Phase 4 — Weather agent: complete.** Weather agent (Open-Meteo over stdlib
  urllib), weather absorbed as case context, urgency-weighted ranking. Exit
  criteria met: two sources fuse into one picture and the survival window
  measurably reorders it.
- **Phase 5 — Data-gathering agents: complete.** Path, Scene and Health built,
  all emitting `ClueContract` to the bus and consumed by fusion. Exit criteria
  met per agent: each one's output measurably changes the fused picture. With
  Weather that is the whole data-gathering plane — four agents, plus Detection
  on the drone. History and Interview were built and are **out of the active
  pipeline**: nothing routes to them, the demo does not register them, and their
  guards were kept (provenance on every bus consumer, the injection heuristic on
  operator commands).
- **Phase 6 — Guardrails, rate protection and scenario routing: complete.**
  Strict Pydantic parsing with one repair pass, a TTL response cache, the
  contradiction guard, and the scenario router. A targeted question now
  dispatches one agent instead of seven.
- **Phase 7 — Security, provenance & trust boundaries: complete.** Provenance
  allow-list, blackboard guard, security audit log, geofence, contract-level
  coordinate bounds, the on-drone imagery tamper check (SHA-256 integrity +
  header sniff + behavioural divergence), the input guard on operator commands,
  and an adversarial suite covering every vector.
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
├── utils/
│   ├── seed.py              set_global_seed() for torch/numpy under ultralytics
│   ├── ablation.py          ABLATION_* switches: turn one component off
│   └── test_ablation.py     25 checks, one per switch, on and off
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
│   ├── models.py            stub-or-real factory, PERCEPTION_MODE=stub|real
│   ├── tracking.py          BoT-SORT: Kalman, two-stage association, CMC
│   ├── terrain.py           DEM sources (constant, lat/lon grid, JSON loader)
│   ├── geolocation.py       WGS84 geodesy, DEM ray march, range selection
│   ├── agent.py             the vertical slice: sensors → … → bus
│   └── test_perception.py   66 checks
├── coordinator/             Phase 3 — the decision plane
│   ├── blackboard.py        cases, targets, environment, bus cursor (dumb store)
│   ├── fusion.py            clue accumulation, dedupe, urgency, ranked picture
│   ├── decision.py          Reason → Risk → Recommend → Orchestrate → Commander
│   ├── orchestrator.py      dispatch loop + command guard; then the chain
│   ├── router.py            trigger/query → smallest agent set that answers it
│   ├── demo.py              operator query → dispatch → picture → brief
│   ├── mock_drone_publisher.py  mock airframe → real pipeline → bus (dev feed)
│   └── test_coordinator.py  105 checks
└── agents/                  Phase 4-5 — the data-gathering plane
    ├── llm.py               NVIDIA NIM over stdlib urllib (text + vision)
    ├── weather.py           Open-Meteo, wind chill, hypothermia risk + window
    ├── path.py              Monte-Carlo sectors; LLM writes the briefing only
    ├── scene.py             VLM, gated on confirmed detections
    ├── health.py            subject-specific survival window (clamped multiplier)
    ├── history.py           not in the pipeline — TF-IDF RAG over a case archive
    ├── interview.py         not in the pipeline — NER over untrusted witness text
    └── test_agents.py       63 checks
└── guardrails/              Phase 6 — what stands between a model and the bus
    ├── schemas.py           Pydantic reply models (Scene/Health/Interview/Text)
    ├── parsers.py           strict parse, one repair pass, then discard
    ├── cache.py             LRU + TTL response cache for NIM rate limits
    ├── contradiction.py     model prose checked against perception facts
    ├── provenance.py        the allow-list: origins, identities, geofence
    ├── injection.py         injection tells + the operator-command guard
    ├── tamper.py            frame integrity (SHA-256, header) + divergence
    ├── audit.py             security events for anything refused or flagged
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
└── ui/                      the ground station, over the same pipeline
    └── dashboard.py         Streamlit: ablation toggles, a frame in, 3 columns out
```

How fusion treats each source (`coordinator/fusion.py`). Every `AgentSource` has
exactly one role, and a test fails if one is ever left unrouted:

| Kind | Sources | Effect |
|---|---|---|
| detection | `PERCEPTION_FUSION`, `DRONE_*` | creates/updates targets, sets confidence |
| context | `WEATHER_API`, `HEALTH_LLM` | conditions + survival window → urgency |
| prediction | `PATH_MODEL` | search sectors → priority prior |
| annotation | `SCENE_VLM` | target profile + hazards → urgency |
| advisory | `HISTORY_RAG`, `INTERVIEW_LLM` | operator notes; change nothing else (agents retired, handling kept) |

Only **detection** sources touch confidence.

Full flow: `fitted detectors → [WBF] → tracking → geolocation → bus → coordinator
fusion → picture → Reason → Risk → Recommend → Orchestrate → commander brief`. Each stage takes and returns plain values or `ClueContract`,
so stages can be tested and replaced one at a time. WBF is in brackets because
it is conditional: the pipeline is sensor-agnostic, runs only the detectors for
sensors actually fitted and delivering a frame, and fuses only when there is
more than one feed to reconcile. Everything after the detectors takes boxes and
does not care where they came from.

Two different things are called fusion. `perception/fusion.py` merges *boxes*
from two sensors on one frame; `coordinator/fusion.py` merges *clues* from any
producer across the whole case.

Two different things are called orchestration, too. `Orchestrator` in
`orchestrator.py` dispatches *agents*; the Orchestrate stage in `decision.py`
validates the three *judgements* about what those agents found. The class fronts
the command plane; the stage is the last gate before the commander.

## Commands

```
python -m src.evaluation.harness            # baseline metrics report
python -m src.evaluation.test_evaluation    # Phase 1 checks
python -m src.perception.test_perception    # Phase 2 checks
python -m src.perception.fusion             # worked WBF example
python -m src.perception.agent              # full pipeline, clues landing on the bus
python -m src.perception.agent --real       # real weights instead of stubs (needs checkpoints)
PERCEPTION_MODE=real python -m src.perception.agent
python -m src.perception.models --selfcheck # the stub/real switch, without a checkpoint
REDIS_URL=redis://localhost:6379/0 python -m src.perception.agent   # against a real server
python -m src.coordinator.test_coordinator  # Phase 3 checks
python -m src.agents.test_agents            # Phase 4-5 checks
python -m src.guardrails.test_guardrails    # Phase 6-7 checks
python -m src.guardrails.test_adversarial   # Phase 7 crafted attacks
python -m src.critic.test_critic            # Phase 8 checks
python -m src.coordinator.demo              # operator query → dispatch → picture → commander brief
python -m src.critic.demo                   # fly a case, resolve it, score it
python -m src.tuning.test_tuning            # Phase 9 checks
python -m src.tuning.demo                   # run the study, baseline vs tuned
python -m src.tuning.demo --reuse           # score the saved config without searching
python -m src.tuning.demo --seed 42         # pin the TPE sampler; folds stay fixed
python -m src.tuning.scenario --seed 42     # one repeatable sortie
python -m src.agents.path --seed 42         # Monte-Carlo sectors, reproducible
python -m src.coordinator.demo --seed 42    # one draw of the stub noise, reproducible
python -m src.coordinator.mock_drone_publisher --seed 42
python -m src.utils.seed --selfcheck        # what a global seed does and does not reach
python -m src.utils.test_ablation           # ablation switches, on and off
ABLATION_WBF=off python -m src.perception.agent          # no box fusion
ABLATION_CMC=off python -m src.coordinator.mock_drone_publisher --check
ABLATION_DISABLE_AGENTS=weather,path python -m src.coordinator.demo
ABLATION_DECISION=off python -m src.coordinator.demo     # flat report, no chain
python train_perception.py --mode rgb --epochs 50 --seed 42   # fine-tune, then score it
python train_perception.py --mode lidar     # skips gracefully until there is LiDAR data
python train_perception.py --selfcheck      # the wiring, without a dataset or a GPU
python -m src.coordinator.demo --live-weather   # hits Open-Meteo instead of fixed conditions
python -m src.coordinator.mock_drone_publisher          # mock airframe onto a live Redis
python -m src.coordinator.mock_drone_publisher --check  # offline: the feed reaches a picture
CASE_ID=case-mock-drone python -m src.coordinator.demo  # a coordinator joins that case
streamlit run src/ui/dashboard.py           # the ground station UI
python -m src.ui.dashboard --selfcheck      # its wiring, without a browser
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
- **Stub is the default detector, always.** `PERCEPTION_MODE=real` is opt-in and
  `perception/models.py` is the only place that decides; unset, nothing imports
  ultralytics, reads a checkpoint or changes a number. Tests and tuning folds
  need projected ground truth to score against, so they stay on stubs whatever
  the environment says. A missing checkpoint fails at wiring time with the path
  it wanted, never mid-sortie.
- A real detector emits a box, a class and a confidence — and **no coordinates**.
  `DetectionAgent` geolocates from the box, the telemetry and the DEM, and the
  LiDAR model reports a range only when something actually measured one.
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
- Operator commands are the untrusted input in this build. They cross
  `guardrails/injection.guard_command` before they steer a dispatch, and a
  flagged command is **widened, never obeyed and never dropped** — the damage an
  injected command can do is to narrow a live search, and a real operator typing
  "stand down the north sector" still deserves an answer. Same rule as the
  retired witness path: flagged and discounted, not dropped. Interview clues
  carry **no `spatial_context`**, so fusion structurally cannot turn a statement
  into a sighting — only a sensor puts someone on the map.
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
- `ALL_AGENTS` in `router.py` is the canonical dispatch order, the roster of the
  active pipeline, and the encoding of its dependencies: detection before scene,
  weather before health. Every route is built by filtering that tuple, so routing
  can never reorder agents, and a route naming an unregistered agent fails
  loudly rather than building a picture on missing evidence.
- The decision chain is four stages, not one call: Reason states facts, Risk
  scores 1-10 and itemises every point, Recommend picks one action from the
  protocol table, Orchestrate validates the three against each other. Keep them
  separable — an operator has to be able to disagree with the score without
  disagreeing about the facts.
- Orchestrate **re-derives rather than trusts**: it recomputes risk and the
  recommendation from the facts, corrects the drift, and re-checks its own
  correction. Two things it will not do — order a team to a position nobody
  actually fixed, and publish an order the chain could not make consistent
  (that one goes to the commander marked for review).
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
- Ablations are **off by default and always announced**. `utils/ablation.py`
  resolves every `ABLATION_*` switch; unset means the shipped system, so no test
  had to change. Each site takes an explicit argument that defaults to `None`,
  and `None` asks the environment. A run with something switched off says so in
  its header — an ablated run that looks normal is how a wrong number reaches a
  report. This is not the critic's ablation: that re-weights a *signal* on a
  finished picture, this removes a *component* so the pipeline runs without it.
- Randomness is **injected, never global**. Every RNG in `src/` is an isolated
  `random.Random(seed)`, so a fixed split means the same thing whatever else the
  process did first. `utils/seed.set_global_seed` exists for the libraries we do
  not own — torch and numpy under ultralytics — and a `--seed` flag must do both:
  call it *and* thread the number into the explicit seed the code already takes.
  A flag that only set the global generators would be decorative everywhere
  except training.
- A `--seed` never moves the **folds**. They are the fixed validation set every
  configuration is judged against; reseeding them per run would make two studies
  incomparable, which is the thing Phase 1 fixed the splits to prevent.
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
  copies so state cannot change under whoever is reading it — and both readers,
  the critic and the decision chain, are read-only for the same reason.
- A measured LiDAR range always overrides an RGB range inferred from terrain.
  This is enforced in one place, `geolocation.select_range`, and it is never a
  confidence comparison — a confident inference must not outrank a measurement.