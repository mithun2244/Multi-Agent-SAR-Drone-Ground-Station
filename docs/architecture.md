# Search & Rescue Multi-Agent System

**Revised Architecture & Step-by-Step Build Process**

*RGB + LiDAR detection core · coordinator (orchestrate + fuse) · seven specialist agents*

## 1. Target Architecture

The design has three planes, separated by where each part runs and how fast it has to react. The perception plane runs on the drone at frame rate. The reasoning plane holds the specialist agents on the ground. The command plane reads the shared state and drives the human decision loop. Between the producers and the command plane sits a shared spine: the event bus and a versioned case blackboard. Security guards sit at each untrusted edge, and the critic and provenance services run across all three planes.

![Figure 1. The three-plane architecture](architecture-figure1.png)

*Figure 1. The three-plane architecture. Arrows show the primary data path. The orchestrator's dispatch back to the producer planes and the critic feedback are described in Section 2.*

## 2. Step-by-Step Build Process

The build order follows one rule: put the spine and the data contract in place first, then add producers one vertical slice at a time. The cross-cutting parts, the guards and the critic, come last, once the pieces they defend and score exist. Every phase ends with a concrete exit test, and the next phase does not start until the current one passes it. The phases fall into two groups. Phases 0 to 5 are the assessed core and should be built and measured first; Phases 6 to 10 are stretch goals, worth doing if time allows but not needed for the core system to work. The aim through the early phases is a thin path that runs end to end, from detection through the bus to a result, before any single agent is deepened.

### Phase 0 — Contracts & spine (nothing works without this)

Define the clue / evidence schema that every agent emits against: case_id, source, timestamp, confidence, geo, payload, provenance_tag. This typed interface is what makes the vertical slices work. If it is wrong, each agent ends up with its own output shape and fusion becomes impossible. With the schema fixed, stand up Redis Streams as the bus and a thin case blackboard as the state store, and add a case lifecycle: open a case, mint the case_id, and carry it through everything downstream.

**Exit criteria** You can publish a hand-written clue to a stream and read the current picture back off the blackboard.

### Phase 1 — Data & evaluation harness

Before any detector is built, settle the data and the way results will be measured. RGB and LiDAR search-and-rescue data is scarce, and the two datasets here are independent, so a small set of paired frames, the same scene captured by both sensors, is needed to check whether the Weighted Box Fusion merge actually beats RGB alone. Alongside the data, stand up the evaluation harness with fixed train and validation splits and the metrics that every later exit test refers back to: mean average precision, recall at a fixed false-alarm rate, and geolocation error in metres. Set the RGB-only detector as the baseline, so the value of adding LiDAR can be shown as a number rather than asserted.

**Exit criteria** The harness reports mAP, recall at a fixed false-alarm rate, and geolocation error for a baseline RGB-only run.

### Phase 2 — Detection end-to-end, on recorded footage

This is the core of the system and the riskiest part, so it is built first and on its own. YOLO11n runs on the RGB frame to classify targets, a second detector runs on the LiDAR range image for depth and range, and their detections are combined with Weighted Box Fusion so each confirmed target carries both an RGB class and a measured LiDAR range. A measured range beats one inferred from the terrain, which improves geolocation and keeps detection working in low light, haze, and thin canopy where RGB alone struggles. BoT-SORT links detections across frames, and its camera-motion compensation helps because the drone camera is always moving. Tracks are confirmed on hit-count and confidence, geolocation combines the LiDAR range with telemetry, and the agent emits one clue per confirmed track to the bus. Work from recorded clips that carry RGB, LiDAR, and telemetry rather than a live drone, so this stage exercises the perception logic and not the radio link. Geolocation on a moving drone, which depends on camera calibration, the sensor extrinsics, and terrain height, is the hard and error-prone part of this stage and the main risk to watch.

**Exit criteria** Video in → correctly geolocated clues land on the blackboard.

### Phase 3 — Minimal command plane against detection only

Build the two Coordinator modules, but connect them to detection only. At this point the orchestrator is still a simple router: the "drone airborne" scenario dispatches detection. Fusion reads clues off the blackboard, removes duplicate tracks, and keeps a current picture. There is nothing to corroborate yet with a single source. The point of the phase is to get the accumulation and confidence-weighting logic in place so it is ready once the second source arrives.

**Exit criteria** Operator query → orchestrator dispatches → fusion emits a ranked picture.

### Phase 4 — Second agent (Weather) & real fusion

Weather comes second, and the choice is deliberate. It is API-only, so it brings no LLM, no untrusted input, and needs no guard. What it does bring is a second source, which is the point: corroboration and confidence-weighting now do real work instead of sitting idle. The hypothermia-risk indicator and the survival window are derived at this stage.

**Exit criteria** Two sources fused into one picture; the search window is reflected in the ranking.

### Phase 5 — Remaining reasoning agents, one slice each, guard co-located

Add the reasoning agents one at a time, in dependency order, rather than several half-built at once:

- Path — Monte-Carlo core; the LLM only writes the field briefing over computed sectors.
- Scene Description — VLM, triggered only on frames where detection already confirmed something.
- Health — LLM; needs weather and subject profile already flowing.
- History — Qdrant RAG with BGE-M3 embeddings, a reranker, and hybrid dense plus keyword search; bring the provenance allow-list guard online with it.
- Interview — LLM; bring the prompt-injection guard online with it.

The LLM- and VLM-backed agents (Scene, Health, the Path briefing, and Interview) run on NVIDIA NIM: `meta/llama-3.1-70b-instruct` for text and `meta/llama-3.2-90b-vision-instruct` for vision. The loop is the same for each agent: build it, emit against the Phase-0 contract, let fusion consume the output, check it, and move on.

**Exit criteria** Per agent — its output changes the fused picture in a way you can point at.

**Stretch goals —** the phases below extend the system but are not part of the assessed core. Take them on only once Phases 0 to 5 are working and measured.

### Phase 6 — Scenario routing

With the agents in place, add the routing rule: call only the smallest set of agents that can answer the current question. In practice an agent runs only if its answer could change the next action. Wire up the four scenarios (fresh call, drone airborne, possible sighting, and subject located).

*Built as five routing scenarios rather than the four above, driven by what triggers a dispatch rather than by what stage the search is at:* `PERCEPTION_EVENT` (detection, scene, health, path), `WEATHER_QUERY` (weather), `WITNESS_INPUT` (interview), `HISTORICAL_QUERY` (history), and `FULL_SEARCH_BRIEFING` (all seven, and the fallback for anything unrecognised). "Drone airborne" maps onto `PERCEPTION_EVENT`; the other three original scenarios are stages of a search rather than distinct agent sets, and splitting the routes by trigger avoids inventing three near-identical ones. Routing changes which agents run, never the order — the dispatch order is fixed by dependency (scene needs detections published, health needs a weather window to refine).

**Exit criteria** Each scenario dispatches only its listed agents, verifiably.

### Phase 7 — Security hardening pass

The guards were added next to their agents in Phase 5. This phase is the dedicated adversarial pass. The imagery guard runs a behavioural check: it compares detections on the raw frame against a transformed copy, and a large divergence flags tampering. Two modalities give a second check for free, because RGB and LiDAR should agree. A target that shows up in one but not the other is suspect, and spoofing both consistently is much harder. The injection guard pairs a classifier with a firm rule that witness text can never trigger a state change, and the provenance guard checks retrieved content against an allow-list. Test each one with a crafted attack: a stand-down injection, a perturbed frame, and a poisoned record.

**Exit criteria** Each attack is caught at its own boundary.

### Phase 8 — Critic & learning loop

This is the last of the runtime work, since it needs logged outcomes to learn from. The critic compares each claim against what actually turned out to be true and logs the gap as a labelled error. It then retunes two things: each agent's thresholds, and the source-trust table that fusion consults. Weather is the place to start, because the real conditions arrive on schedule and are easy to score against. Detection is harder. A person who is never found is a miss that nobody logs, so it can only learn from confirmed finds and from footage that gets reviewed afterwards.

**Exit criteria** A logged weather error measurably shifts the next survival-window estimate.

### Phase 9 — Optimization & resilience

This phase covers two separate pieces of hardening. The first is tuning: use Optuna, a Bayesian TPE search, to set the detection parameters (confidence threshold, hit-count to confirm, NMS IoU, track age and gap, and the Weighted Box Fusion weight between the RGB and LiDAR detectors) against a labelled validation set, with fitness defined as recall minus λ times the false-alarm rate. The second is resilience. Health data falls back through three tiers, from the online source to a ground-station cache to an encrypted subset held on the drone, and it is fetched when a case opens. If the link drops, the perception plane buffers its clues locally and resyncs the blackboard once the connection returns.

**Exit criteria** Detector tuned from data, not by hand; the system degrades gracefully when the link drops.

### Phase 10 — Live drone & hybrid deployment

Only at this point does the system leave recorded footage. YOLO11n runs on the drone for real-time confirmation, the ground station handles heavier re-scoring and the command plane, and the cloud does post-mission fusion. Because everything before this was tested on recordings, a failure that appears now points to the deployment rather than the logic, which is where you want the problem to sit.

**Exit criteria** The full pipeline runs on a live flight over a real sector.

## 3. Core, stretch, and the five sprints

The eleven phases divide into an assessed core and a set of stretch goals, and the core maps onto the five sprints in the deck. Phases 6 to 10 (routing, security, the critic loop, optimization, and live deployment) are the parts the current plan does not budget for, so it is better to flag them now than to run into them in the final sprint.

- Sprint 1 → Phases 0–1 (contract, spine, data and the evaluation harness)
- Sprint 2 → Phase 2 (detection end to end on recorded footage, including geolocation)
- Sprint 3 → Phase 3 (minimal command plane) and Phase 4 (Weather and real fusion)
- Sprint 4 → Phase 5 (the remaining reasoning agents) and the health-data fallback from Phase 9
- Sprint 5 → Phases 7–8 (security layer and critic loop)
- Stretch / overflow → Phase 6 (routing), Phase 9 (tuning and resilience), Phase 10 (live deployment)

**Future work.** YOLO26, which is NMS-free and faster on edge hardware, is a candidate to replace YOLO11n once the pipeline is stable. YOLO11n stays the baseline for now because it is well documented and easy to cite.

**Build status.** All eleven phases are implemented and tested: the Phase 0 contract and the Redis Streams spine, the Phase 1 evaluation harness, the Phase 2 detection chain, the Phase 3 command plane, Weather in Phase 4, the five reasoning agents in Phase 5, scenario routing with output guardrails and rate protection in Phase 6, the security and provenance work in Phase 7, the critic in Phase 8, and the Optuna study in Phase 9. What is not done is the part code cannot supply for itself: trained detector weights, real recorded RGB and LiDAR footage, a raster DEM over the search area, live NVIDIA NIM and Redis endpoints, and a tuning run against real data rather than the simulator. Every model, sensor and external service is reached through an injected callable, so each of those is a substitution rather than a rewrite.

**Tuning as built.** The Optuna objective is the critic's own loss from Phase 8, not a fitness function invented for the optimiser — the thing being minimised is the thing the critic measures. Every configuration is scored over several folds, because one that wins on a single draw of the noise has learned nothing, and the shipped defaults are evaluated as trial zero so an improvement is always measured against them. The false-alarm target enters as a constraint rather than a reward: loss alone will trade a flood of phantoms for one more subject, and the operating-point threshold exists to stop it. Results carry one honest caveat — the pipeline under test is real end to end, but the imagery is stubbed, so a tuned configuration is a starting point for a run against real recordings rather than a finished answer.

**Trust boundaries as built.** The imagery guard is two layers, not one. Integrity comes first and runs on every frame: the sensor records a SHA-256 at capture, and a frame that does not hash to it is not the frame the sensor took. That also stops a payload which is not an image at all from reaching a VLM, which costs nothing and closes the cheapest attack. The behavioural divergence check described above is the second layer, for tampering that happened before capture — a projected pattern or a printed target, where the bytes are genuine and the scene is not. Provenance is checked by an allow-list of registered origins, each bound to the agent permitted to use it, with device, endpoint and operator identities verified where data crosses in from outside. Every bus consumer applies it, not only the one writing to the blackboard: an unverified clue reaching any consumer still spends an API call and can put a forged frame in front of a model. Coordinate payloads are bounded twice — the contract refuses impossible values outright, and a geofence refuses possible ones that are nowhere near the search. Everything refused is recorded as a security event; a guard that blocks silently teaches an operator nothing about being probed.

**Routing signals.** The router classifies a trigger by keyword, not by asking a model. A model in front of every dispatch would add an unreliable component to the one path that must always work, cost an API call to save API calls, and fail precisely when the provider is rate limited — the situation routing exists to handle. Classification is deliberately biased: an unrecognised query runs every agent, and a query matching several topics runs the union of them rather than picking one. Routing too narrowly loses an agent's contribution to a live search; routing too widely costs a call, and those are not comparable.

**Inference provider.** The reasoning agents were originally specified on Gemini 3.6 Flash and were moved to NVIDIA NIM, whose free tier is what the live demonstration runs on. NIM exposes an OpenAI-compatible chat-completions endpoint at `https://integrate.api.nvidia.com/v1`, so the integration is a single HTTPS POST and adds no dependency. The models are larger but hosted, and the practical differences to watch are rate limits on the free tier and weaker instruction-following on structured output: the Scene agent asks for JSON and falls back to treating the reply as prose when it does not get it, rather than inventing structure.

**Model size.** The detector was originally specified as YOLO11m and was changed to YOLO11n (Nano) for lower latency during live demonstrations. Nano is the smallest of the YOLO11 family, so it trades detection accuracy for speed: expect lower recall on small, distant, or partly occluded subjects, which is exactly the case that matters in search. The Phase 1 harness measures this directly — re-run the baseline and compare mAP and recall at a fixed false-alarm rate against the YOLO11m numbers before treating Nano as the operational model rather than the demo one.

**The part to get right first:** if the clue contract from Phase 0 is not stable before the build phases begin, every later phase pays for it. That is where the early effort should go.
