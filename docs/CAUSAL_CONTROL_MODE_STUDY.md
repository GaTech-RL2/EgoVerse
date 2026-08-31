# Causal Action Models across PushShapes Control Modes — What Was Built

Implements `docs/CAUSAL_MODEL_HANDOVER.md`. This file records what actually
shipped, where it differs from the handover and why, and how to read the
results. Branch: `algo/causal-action-models`.

---

## 1. The question

One embodiment (`gripper`), six controller modes. Train on four, hold out two.
Success rate under a controller the model never trained on is the dependent
variable.

Modes ordered by sensing-noise floor, which is the axis the held-out question
is asked along:

| mode    | noise_std | other gap terms                      | role |
|---------|-----------|--------------------------------------|------|
| ideal   | 0.0       | none                                  | **UNSEEN** — below the training range |
| tight   | 0.3       | lag 0.25                              | seen |
| laggy   | 0.4       | latency 6, lag 0.35                   | seen |
| loose   | 0.8       | latency 2, lag 0.55, deadband 1.5, gain 0.95 | seen |
| sticky  | 0.0       | deadband 4.0, gain 0.88, lag 0.15     | seen |
| jittery | 2.5       | none                                  | **UNSEEN** — 3x above the range; the headline test |

Training spans 0.3–0.8. Both holdouts sit outside it, on opposite sides, so
generalization is bracketed rather than probed in one direction.

`jittery` is the honest extrapolation test because it is pure zero-mean noise
with no bias to learn — "can it cope with an irreducible floor it has never
seen". The seen modes all have *structured* biases a model can plausibly infer
and compensate.

---

## 2. Four arms

| # | arm | attention | generation | purpose |
|---|-----|-----------|------------|---------|
| 1 | `arm1_dp_flow` | bidirectional | flow matching | established baseline |
| 2 | `arm2_causal_bidir` | **bidirectional** | regression + MSE | **the control** |
| 3 | `arm3_state_action_ar` | causal | regression + MSE, row fed back | the causal model |
| 4 | `arm4_state_idm` | causal | regression + MSE over the pose path + IDM | causal path, action via inverse dynamics |

### Read arm 2 first

The handover specified flow matching for all four arms. Per the user's call,
arms 2–4 use regression + MSE instead. That makes **arm 2 load-bearing**: arm 1
now differs from arms 3/4 in BOTH objective and attention, so arm 1 vs arm 3
cannot separate "causal generation helps" from "the MSE head helps".

Arm 2 shares the backbone, the head, the loss, the optimizer and the parameter
count with arms 3/4 and differs ONLY in the attention mask and in consuming
learned queries instead of shifted rows. **Arm 2 vs arm 3 is the experiment.**
Arm 1 is an external reference point for absolute SR, not the control.

### What "causal" means here

One training sample is a single observation plus one planar arc token of shape
`(17, 5)`: 16 waypoints plus a trailing velocity row (`append` layout), with
channels `[x, y, cos, sin, grip]`. The sequence the causal arms are
autoregressive OVER is the **rows of that token**, not time. Row *m* is a point
further along the same arc, so feeding row *m−1* back is meaningful: it is the
model's own committed path so far.

`state_idm` feeds back the POSE only. Its action channels are a readout of the
`p_m → p_{m+1}` transition, so at row *m* they are still provisional; feeding
them back made the fed-back sequence differ from the final output — training
would have optimized a function the rollout never runs. Caught by a
train/rollout consistency test, not by inspection.

---

## 3. Capacity — measured, not derived

`12 * n_layers * d_model^2` misses the heads, the per-domain projections and
`ff_mult`. Totals below include the shared 11.20M observation encoder.

| capacity | arms 2–4 | arm 1 | spread |
|----------|----------|-------|--------|
| large | `d_model=1024, n_layers=24, n_heads=16` → **313.63M** | `CrossTransformer nblocks=31, hidden=384` → **309.12M** | 1.44% |
| small | `d_model=512, n_layers=12, n_heads=8` → **49.09M** | `CrossTransformer nblocks=4, hidden=384` → **50.01M** | 1.88% |

This **corrects handover §4**, which assumed arm 1 was ~20M and needed scaling
up. CrossTransformer at the shipped `nblocks=16/hidden=384` is already 153.6M,
so arms 2–4 come DOWN to meet it rather than the reverse. §4's "~30M control"
is really ~49M once the encoder is counted.

Arm 1 keeps `hidden_dim=384` in both rows deliberately: `hidden=256/nblocks=71`
matched tighter (0.31%) but changes width *and* depth, making arm 1 a third
architecture rather than the established baseline made deeper.

**Run both capacities and report both.** With ~2,188 training episodes the
large arm sits at ~143k params/episode. If large wins on seen modes and loses
on the holdouts, that is overfitting, not causality — and without the small
arm you cannot tell those apart.

---

## 4. Data

`s3://rldb/processed_v3/pushshapes_sim/control_gap_dedup_gripper_simv2_20260830`

**Dedupe-only**: real source demos with near-duplicates removed, capped at 547
per mode, and **no MimicGen-generated fill**. See `MANIFEST.json` under that
prefix for per-mode counts.

`ds_gen` is ~60% real / ~40% generated, and the generated episodes are kept
only if they re-reach the goal under their own cell's control gap. That is a
success filter sitting directly on top of the quantity this study measures —
"generalizes to a held-out controller" would partly become "generalizes from a
distribution pre-selected for robustness". It would not fake a difference
*between* arms (same data for all four), but it would change what the absolute
numbers mean. Hence dedupe-only.

547 is `min(from_source)` across the modes (`loose`), so every mode supplies
that many real episodes and the 19% between-mode imbalance costs nothing to
remove.

**Caveat, do not overstate:** dedupe removes duplicates, it does NOT raise
intrinsic dimensionality — PC95 is 4–8 before and after. Read it as better
sample efficiency per epoch, not broader behaviour coverage.

### Why `ideal` is a holdout

Not by design. 714/1000 of its generated episodes are unreadable: zarr 3.1.0
silently wrote corrupt `actions` and `observations.state` while reporting
success (the JPEG/reward/goal arrays take a different codec path and stay
readable, so a partial check passes). Training on its 286 survivors would have
put a 3.5x per-mode imbalance under the comparison.

Holding it out turned out to be the stronger design, but note the honest
framing: dropping `ideal` does **not** move the training set toward `jittery`.
It removes the lower anchor and narrows the axis being extrapolated from.
`jittery` is 3x outside either way.

When a clean `ideal` cell lands, rerun with it in training and report both —
the 4-mode vs 5-mode difference measures what controller diversity buys.

---

## 5. Reading the results

```bash
python scripts/control_modes/collect_results.py --step-matched
```

Per run, the evaluator reports SR for all six modes under these wandb keys:

```
Valid/seen_tight_sim_success_rate     Valid/unseen_ideal_sim_success_rate
Valid/seen_laggy_sim_success_rate     Valid/unseen_jittery_sim_success_rate
Valid/seen_loose_sim_success_rate
Valid/seen_sticky_sim_success_rate
```

(plus `_sim_coverage` for the mean IoU behind each.) The `seen_`/`unseen_`
prefix is a `metric_tag`, not decoration: metrics default to keying by
embodiment, and all six evaluators are the SAME embodiment, so without distinct
tags the six collapse to whichever ran last — one plausible number instead of
six, with nothing raised.

- **seen**: `tight`, `loose`, `laggy`, `sticky` — in-distribution BC
- **unseen**: `ideal`, `jittery` — controller generalization

Compare at **matched steps**, not wall-clock: runs start at different times, so
the last logged value of a run that started earlier is not comparable to one
that started later. `--step-matched` does this.

Questions, in order of what the study was built to answer:

1. **arm 3 vs arm 2 on `jittery`** — does causal generation help under an
   unseen noise floor, holding backbone/head/loss/capacity fixed? This is the
   result.
2. **arm 4 vs arm 3** — is predicting the path and recovering the action by
   inverse dynamics better than predicting the action directly?
3. **large vs small, same arm** — if large wins seen and loses unseen, it is
   overfitting.
4. **arm 1 vs the rest** — absolute reference. A gap here mixes objective and
   attention; do not read it as an attention result.

---

## 6. Gates (all passing)

| gate | what it protects |
|------|------------------|
| `tests/test_ar_action_decoder.py` (19) | causal mask leaks nothing; feedback is not severed; bidirectional never reads the target; teacher forcing on the model's own rollout reproduces the rollout |
| `tests/test_eval_sim_control_gap.py` (6) | the named gap reaches the agent and survives `reset()` |
| `tests/test_control_mode_configs.py` (38) | every config instantiates and runs a real forward; arms parameter-matched; adapter layout matches the data config |
| `tests/test_model_config_preflight.py` | repo-wide `act_seq == action_horizon`, cross-stage horizon/latent agreement, instantiate-and-forward |
| `egomimic/rldb/zarr/planar_arc_sr_gate_d10m16.py` | tokenize at every timestep, execute one action, SR must equal the untokenized baseline. **D10 M16 r0 append = 93%, concat = 93%, baseline = 93%** on gripper |

The SR gate matching the baseline *including where the baseline itself fails*
is the property that matters, not a high number.

---

## 7. Bugs found and fixed along the way

1. **`act_seq` vs `action_horizon`.** CrossTransformer adds a
   `(1, act_seq, D)` positional table to the token sequence, so nine `append`
   layout arc configs died on their first forward pass — after the R2 pull and
   the norm-stats phase, i.e. ~2.5h in. Fixed on `sim/arc-sweep-cotrain`
   (`0796b40c`); those fixed copies are taken here.
2. **`init_mode: seed`.** `eval_sim_pushshapes.yaml` uses `seed`;
   `SimRolloutEval` validates against `{replay, random, seeds}` and rejects it,
   so that evaluator cannot instantiate. The new evaluator uses `seeds`.
3. **`control_gap` was not plumbed** (handover TODO-7, correctly flagged as an
   assumption). It is a class attribute on `PushShapesEnv`, so `env_kwargs`
   cannot carry it — passing it there raises "unexpected PushShapesEnv
   option(s)". `SimRolloutEval` now takes it and applies it to the agent,
   validated at construction so a typo fails in seconds rather than hours in.
4. **`planar_arc_sr_gate.py` cannot run from a clean checkout.** It imports
   `apply_source_control_gap` from `Tsimulation.sim_v2.generate.mimicgen`,
   which exists only in the untracked `sim_run/runtime` tree. The variant here
   derives the gap from the episode's own `control_gap_mode` and has no
   untracked dependency.
5. **zarr 3.1.0 corruption** (found, owned by the peer session). Discriminator
   is mtime: cells written 08-29 are suspect, 08-30 are clean. A written count
   is not evidence of a good write.

Handover **TODO-3 is obsolete**: the decoders are a pipeline `Stage` occupying
the sampler's slot, so they inherit `PipelineAlgo.inference_step`, the rollout
adapters and norm-stat handling rather than needing an `Algo` wrapper.

---

## 8. Running it

```bash
# 8 runs: 4 arms x 2 capacities. small first (see submit.sh).
scripts/control_modes/submit.sh --dry-run     # inspect
scripts/control_modes/submit.sh small         # or: small large

# Re-upload / extend the dataset (e.g. once a clean ideal cell lands)
python scripts/control_modes/upload_dedup_gripper.py [modes...]

# Regenerate every config from one place
python egomimic/hydra_configs/data/pusht/_gen_control_modes.py
```

Checkpoints sync to
`s3://rldb/staged/pushshapes_control_modes/<job_name>/checkpoints/` every 7
minutes — the container is ephemeral and declares no outputs, and two completed
20h runs were lost that way previously.
