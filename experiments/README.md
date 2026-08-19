# experiments/

Ablation and multi-seed runners. **The config and the scripts are committed;
`experiments/results/` is not** — a CSV is a render of whatever the code did
that afternoon, and committing one invites quoting a number nobody can
reproduce. Regenerate instead.

```bash
python experiments/run_ablation.py                       # every run in the config
python experiments/run_ablation.py --only full_system,no_wbf
python experiments/run_multi_seed.py --experiment no_wbf --seeds "42,123,456"
python experiments/run_ablation.py --selfcheck           # the runner, no sorties
python experiments/run_multi_seed.py --selfcheck
```

Needs PyYAML for the config (`pip install pyyaml`; it ships with ultralytics).

## One process per row

Ablation switches are read when a component is *constructed*, so setting them
in-process would ablate whatever had not been imported yet and leave the rest of
the pipeline whole. Each row is a subprocess with its own environment. That is
the only way a row means what its label says.

## Three engines, none of which sees everything

| Engine | What runs | Sees |
|---|---|---|
| `detection` | `tuning/scenario.py` — stub detectors, WBF, BoT-SORT, geolocation, fusion | WBF, a missing sensor |
| `flight` | `coordinator/mock_drone_publisher.py` — an orbiting airframe | camera-motion compensation |
| `command` | `coordinator/demo.py` — router, agents, fusion, decision chain | disabled agents, the decision chain |

`measures:` in the config names which engines can see a run, and the runner
prints `n/a` for the rest. **`n/a` is not zero and it is not the baseline's
number carried across.** A switch an engine cannot see produces a row identical
to `full_system`, which reads as "this component does not matter" when it means
"this experiment could not tell" — the selfcheck refuses a config that claims an
engine it cannot be seen by.

The `flight` engine exists because of exactly that trap: `tuning/scenario.py`
flies a *static* pose and never supplies camera motion, so `ABLATION_CMC=off`
measured there changes nothing at all.

## Why multi-seed

A single ablation row is one draw of the detector noise, and it moves for
reasons that have nothing to do with the component switched off. Measured over
three seeds, `no_wbf` gives `recall_at_far` **0.278 ± 0.242** — the spread is
comparable to the effect, so a one-seed row claiming a precise drop would be
reporting a coin flip. The per-seed rows stay above the mean and std in the CSV,
because an average that hides its inputs is how an outlier disappears.

## Baselines

What the full system has to justify itself against, and the comparison over all
of it.

```bash
python experiments/baselines/plain_yolo.py --seeds 0,42,123        # detector alone
python experiments/baselines/yolo_plus_tracker.py --seeds 0,42,123 # + BoT-SORT
python experiments/baselines/compare.py                            # everything, one table
```

| Baseline | What runs | Scored on |
|---|---|---|
| `plain_yolo` | the RGB detector's raw per-frame boxes | mAP, recall@FAR, precision, recall |
| `yolo_plus_tracker` | one sensor (so no fusion) + BoT-SORT | the above plus MOTA, IDF1, ID switches |

Both **refuse to run under `PERCEPTION_MODE=real`**. The scenario projects
subjects into a frame that has no pixels, so a real checkpoint could not be
handed anything to look at, and a CSV row labelled `real` that came from a stub
is the one output worth preventing. Real weights are measured by
`train_perception.py`, which runs them over real validation images.

MOTA and IDF1 are computed in `yolo_plus_tracker.py` rather than pulled from a
package — the whole scoring path here is stdlib, and a tracking-metrics
dependency would exist only for one table. MOTA is allowed to go negative;
clamping it would hide the case worth seeing.

### Comparing across taps

`compare.py` carries a `tap` column because the sources do not measure the same
stage. Measured over three seeds:

| run | tap | mAP | MOTA | IDF1 |
|---|---|---|---|---|
| `plain_yolo` | detector | 0.680 | — | — |
| `yolo_plus_tracker` | detector + tracker | 0.568 | 0.222 | 0.650 |
| `full_system` | full pipeline | 0.778 | — | — |

The tracker's mAP is *lower* than the raw detector's and it is the better
system: BoT-SORT drops single-frame blips that the detector happily reports, so
it trades a little mAP for far fewer phantoms on an operator's map. Compare
within a tap first; a dash is a metric that source never measured, not a zero.
