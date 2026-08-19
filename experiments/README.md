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
