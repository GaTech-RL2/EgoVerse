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

**Credentials:** the runs log to `rl2-group`, using the node's wandb key from
Secrets Manager. A personal key with a different default entity cannot read
them — and wandb reports that as a missing PROJECT rather than a permissions
error, which is misleading. The script now says so explicitly and prints
`https://wandb.ai/rl2-group/zarr_test` as the fallback.

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

## 5b. One thing to check in the logs before trusting the numbers

Grep the run logs for `[sim] WATCHDOG:`.

Each rollout has a 120s SIGALRM watchdog; a rollout that trips it is recorded
as **0 coverage**. The causal arms are structurally slower per action than the
bidirectional one — free-running decode does 17 sequential backbone passes per
replan, where `causal_bidir` does one — so if the watchdog ever fires, it fires
on arms 3/4 first and hands arm 2 an advantage that has nothing to do with
attention.

Estimated margin is comfortable (~10-15s per episode against a 120s budget,
even with 8 ranks contending for CPU on renders and physics), which is why the
timeout was left at its default rather than raised. But it is an assumption,
and it is cheap to falsify: **any** WATCHDOG line means the seen/unseen SR for
that arm is biased low and the arm comparison is contaminated. If they appear,
raise `rollout_timeout_s` and rerun rather than reasoning about how many.

## 5c. Known waste, deliberately not fixed in this batch

Every rollout renders a frame per step and the frames are encoded to mp4 — and
**the videos are never synced anywhere**. The checkpoint loop pushes only
`checkpoints/` and `norm_stats.json`, so the videos are written to node-local
ephemeral storage, raced across all 8 ranks writing identical filenames, and
then discarded with the container.

Cost: 60 episodes/rank x up to 600 renders per validation, plus h264 encoding
of the result. Plausibly a large fraction of validation wall-clock, which
directly reduces how many SR points a night produces.

The fix is one line — `env_kwargs: {render_mode: null}` — which is permitted
(`env.py:164` allows None, and `render()` then returns None so no frames are
collected and no video is written). It is NOT applied here because this batch
was already relaunched five times for correctness, and this is a throughput
issue, not a correctness one. Apply it next launch.

(Video writing itself is safe: `av` and `torchvision` are both in uv.lock, so
`tvio.write_video` will not raise, and the frame buffer flushes at 1000 frames
so it cannot grow unbounded.)

## 5d. FIRST RESULT — and it is not yet interpretable

`arm2_causal_bidir` small, 100 epochs / 250k steps, evaluated offline:
**0% success on all six control modes**, coverage ~0.0000 (not merely below the
0.95 bar — the object barely moves).

Before reading anything into that, four things were checked and all pass:

| check | result |
|---|---|
| checkpoint weights loaded | 222/222 tensors match name AND shape |
| optimizer worked | train loss 0.139 → 0.00417, monotone |
| all four modes trained on | launcher guard: `all 4 seen modes staged at 547 episodes each` |
| observation path | same 6 channels, same order, same units at train and rollout |

And the model itself is **good**, open-loop, on the row that is actually
executed:

```
ROW 0, raw units      x        y        cos      sin      grip
model FVE           +0.997   +0.991   +0.998   +0.998   +0.977
persistence FVE     +0.917   +0.936   +0.937   +0.917    n/a
median |error|       2.5 px   2.3 px   0.010    0.012    0.002
```

It beats a persistence baseline on every channel and predicts the commanded
action to ~2.5 px in a 512-unit world.

### ROOT CAUSE — the rollout executed the whole token (fix pending validation)

`inference_step` defaults `replan_every=None`, which consumes the ENTIRE
decoded chunk before querying a new observation. Measured:

```
expert motion per env step : 3.537 px
spacing between waypoints  : 0.656 px    (= D/M = 10/16, as designed)
ratio                      : 0.19x
```

Waypoints are spaced by ARC LENGTH; env steps happen at the control rate. So
executing all 16 consecutively drives the pusher at about a FIFTH of expert
speed — ~375 px over a 600-step episode where the expert covers ~2100. The
pusher crawls, the object barely moves, every rollout scores ~0 coverage, and
nothing raises.

`PlanarArcRolloutAdapter`'s own docstring states the contract: *"the deployed
controller predicts, executes waypoint 0, and re-predicts"* — waypoint 0 IS the
action at `t` by construction, and that is exactly the loop
`planar_arc_sr_gate` replays to reproduce the untokenized baseline at 93% SR.

**The gate validated a loop the deployment did not use.** `replan_every` was
reachable only through a CLI flag in `ckpt_loading.py` that neither the
in-training evaluator nor the offline eval path touches. It is now a
`PipelineAlgo` constructor argument, set to 1 in all eight configs and asserted
by a test.

Ruled out first, each with evidence: weights load (222/222 by name and shape);
the model generalizes rather than memorizes (open-loop FVE +0.998 at ~2.5 px on
trained modes, +0.72..+0.90 at ~6 px on HELD-OUT modes); the rollout adapter
agrees with the gate's decoder to 1.2e-5; observation normalization is
consistent between training and rollout; and `norm_mode` is quantile on both
sides (the `zscore` constructor default never fires because trainHydra passes
it explicitly and `set_norm_stats_from` propagates it).

**Status: THIS FIX DID NOT RESOLVE THE ZEROS.** Re-running the identical
eval on the archived checkpoint with `replan_every: 1` present in the composed
config (verified in the job log; `PipelineAlgo.__init__` takes no `**kwargs`, so
a stale clone would have raised rather than swallowed it) still returns coverage
0.0000 on 60/60 rollouts across all six control modes. The 0.19x speed defect is
real and the fix is correct on the merits, but it is NOT the cause of the zeros.
The root cause is still open.

### What has since been RULED OUT, each with evidence

| candidate | evidence it is not the fault |
|---|---|
| harness / env / physics / coverage metric | replaying RECORDED actions from RECORDED `episode_init` reaches final coverage 0.9529 / 0.9295 / 0.9538 / 0.9520 on 4 of 5 episodes — the harness reproduces its own data |
| action semantics | gripper `action_spec=('x','y','angle','grip')`, 4-wide, replayed verbatim above |
| state vector | `pushshapes_sim_gripper` -> `_env_to_zarr_pushshapes_oriented`, 6-wide `[ax,ay,a_th,ox,oy,o_th]`, matches the dataset's `observations.state` (N,6) channel-for-channel; object pose identical |
| observation images | rollout frames vs dataset frames agree to 0.03/255 mean; `_skip_obs_render` is never set in the eval path and `render_mode` governs only `env.render()` |
| eval init distribution | recorded `|obj-goal|` 121.8..414.5 (mean 212.3) vs `env.reset(seed)` 122.3..400.4 (mean 233.8); position and angle marginals overlap |
| weights / open-loop | 222/222 load by name+shape; FVE +0.998 on trained modes, +0.72..+0.90 on HELD-OUT modes |
| `replan_every` | applied and measured; no change (this section) |

### Two traps that produce a FALSE harness failure

Both cost a wrong zero during this investigation:

1. `scripted_collect.scripted_action` returns `(2,)`, but the gripper action
   space is `(4,)`. Padding `angle`/`grip` by hand yields ~0 coverage that is an
   artifact of the padding. Recorded `grip` spans 0.14..1.00, `angle` 6.46..8.14.
   Use `Tsimulation/sim_v2/generate/planner.py::plan_for(agent)` — that is the
   planner that generated the 4-wide data.
2. `episode_init` in the zarr attrs is a JSON **string**, not a dict.
   `env.set_state(agent_pos=init["agent_pos"], ...)` raises
   `string indices must be integers`; swallow that and you replay correct
   actions from the WRONG initial state, which looks exactly like a broken
   harness. `json.loads` it.

### Open landmine (not the current fault, but silent)

`goal_pose` is absent from the pushshapes keymap, so it never reaches the val
batch. `SimRolloutEval.batch_to_env_init` then passes `goal_pose=None` and
`env.set_state` leaves the goal where the seeded reset put it. Under
`init_mode: replay` that gives agent+object from the dataset and the GOAL FROM A
RANDOM SEED — a different task than the data represents, scored silently. This
study uses `init_mode: seeds`, where agent/object/goal all derive from one seed
and stay self-consistent, so it is not affected.

### Still open

Training inits are SUCCESS-FILTERED (`scripted_collect` discards episodes below
threshold as `low_coverage`), while the evaluator scores UNFILTERED
`env.reset(seed=0..9)`. Measuring the expert's ceiling on those exact seeds
decides whether any policy could score above zero through this evaluator. Also
note the policy's state carries no goal — the goal is only in the 96x96 image,
and coverage 0.95 requires position AND angle. Exactly-0.0000 is zero IoU
overlap: the object is not merely misaligned, it is nowhere near the goal.

**Original status line (superseded):** — the same archived
checkpoint re-evaluated with `replan_every=1` against a prior run of the
identical eval that scored 0.0000 on 60/60 rollouts.

### Methodological warning, paid for four times

Every intermediate verdict in this diagnosis was WRONG, and always the same
way: **an aggregate error computed across mismatched units.**

- normalized predictions vs RAW targets → "training did not learn" (false)
- mean/std-normalized targets vs a QUANTILE-normalized model → "orientation not
  learned, sin FVE +0.07" (false; it is +0.998)
- `MultiDataset.infer_norm_from_dataset` silently no-ops on a hand-built dataset
  (`No proprio/action keys for embodiment=25`), so a "normalized" rerun produced
  byte-identical numbers to the unnormalized one

`norm_mode` here is **quantile**: `2*(x-q1)/(q99-q1) - 1`, not mean/std. Check
`norm_mode` before comparing anything to a model's output.

What caught each error was never the aggregate — it was **per-channel
disaggregation**. Two channels in the tens of thousands beside three near 1.0 is
a units bug; no summary statistic shows you that.

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
6. **A comma in a hydra override value is a SWEEP.** `RUN_DESC` contained
   `train tight,loose,laggy,sticky`, so config composition aborted with
   `Ambiguous value for argument` — in phase 1, ~40 minutes in, after the image
   pull, uv sync, R2 pull and staging had all succeeded. Bash quoting does not
   help; the value reaches hydra intact and hydra objects.

   Worth separating from the others: this is a CLI-composition error, not a
   config error, so the config preflight structurally cannot catch it — it
   passed on the node minutes before this failed. `tests/test_submit_overrides.py`
   asks hydra's own parser instead. Note the predicate: `parse_overrides`
   accepts the broken string happily and returns a *sweep*; it is
   `validate_sweep_overrides_legal` that refuses it. A "does it parse?" test
   passes the broken string, so the gate checks `is_sweep_override()`.
   Verified safe in a value: `|`, `+`, `-`, spaces. Only commas break.

7. **The simulator's deps are not in EgoVerse's environment.** `pymunk`,
   `gymnasium`, `pygame` and `shapely` are in neither `pyproject.toml` nor
   `uv.lock`, yet `Tsimulation/sim_v2/pushshapes/env.py` imports all four at
   module scope. `uv sync --frozen` therefore yields an env in which
   `SimRolloutEval` cannot construct an env at all — ANY sim rollout eval dies
   at its first rollout, in phase 2, after training has begun. The launcher now
   installs them (pymunk pinned to 7.3.0 per §9) and then constructs a real env
   to verify.

   Nearly missed: the on-node preflight reported "42 passed, **2 skipped**",
   and those 2 skips were exactly the `pytest.importorskip("pymunk")` tests. On
   a machine whose whole job is rollouts, an `importorskip` is a failure
   wearing a pass's clothes.
8. **A `python -c "` block broke the launcher YAML.** Unindented continuation
   lines escaped the literal block and the spec stopped parsing. Embedded
   interpreters belong in an indented heredoc. Now gated: the launcher must
   parse as YAML and every embedded script must pass `bash -n`.

9. **The evaluator was never reached at all.** A run trained all 100 epochs,
   exited `COMPLETED`, wrote a valid 589MB checkpoint — and logged ZERO success
   rates. Three shipped defaults, each independently fatal, none of which
   raises:

   | setting | ships as | effect |
   |---|---|---|
   | `check_val_every_n_epoch` | 200 (`trainer/ddp.yaml`) | > `max_epochs=100`, so the first validation is scheduled for an epoch that never arrives; `num_sanity_val_steps: 0` means nothing runs at step 0 either |
   | `limit_val_batches` | 80 (`trainer/default.yaml`) | `on_validation_step` runs the WHOLE evaluator per batch → 80 × 6 × 10 = 4,800 rollouts for one validation |
   | `model_checkpoint.every_n_epochs` | 100 | writes only at the very end, so an in-progress run has nothing recoverable |

   **This is the sharpest failure of the set, because the run SUCCEEDS.** Exit
   0, empty error log, checkpoint present, five GPU-hours spent. Every signal
   that normally means "healthy" was present and the one thing the run existed
   to produce was absent. Status could not have caught it. The check that did:

   ```bash
   grep -c '\[sim\]' <run log>    # 0 ⇒ no evaluation happened, whatever the status says
   ```

   Overridden in phase 2 and asserted in `tests/test_hydra_composition.py`.

These sit in THREE distinct layers, and each is invisible to the layer below's
gate:

  1. **Config** (1-5, 7's registry cousin) — caught by constructing every
     object the run will construct, before the run.
  2. **Invocation** (6, and the config-group prefix) — the overrides are
     upstream of every object, so no amount of instantiating configs sees them.
     Caught by composing the real config with the real override list.
  3. **Environment** (7, 8) — upstream of the interpreter itself. No config
     test can see `uv.lock`. Caught by installing and then USING the dependency
     in the launcher.
  4. **Reachability** (9) — the code is correct, the config is correct, and the
     schedule never calls it. Nothing upstream can see this: the only evidence
     is the ABSENCE of an expected output. Caught by asserting the metric
     appears, not that the job succeeded.

All of them share a failure shape: validation happens on construction, and the
expensive objects are constructed late, so a mistake survives the pull and the
staging and presents as a training failure hours after the fact. The defence is
to move construction earlier — but "earlier" has to mean all three layers, not
just the configs.

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


---

## RESULT PATH: closed-loop coverage is dead; score the DELTA

Closed-loop coverage returns exactly 0.0000 for every arm, every control gap,
every seed — 180 rollouts confirmed. It cannot separate the arms and never will,
for a reason that is now understood rather than suspected.

### Why coverage is saturated

`_grasp_planner` commands `tx = px + clip(gx-ox, -3, 3)`: the action is the
pusher's CURRENT POSITION plus a <=3px delta, and the pusher's xy is channels
0-1 of the policy's own state. A PERSISTENCE baseline — "command where I already
am" — therefore scores:

| mode | FVE_persist | median error |
|---|---|---|
| tight | 0.8539 | 0.58 px |
| loose | 0.9544 | 2.62 px |
| laggy | 0.9047 | 4.70 px |
| sticky | 0.9027 | 2.65 px |
| ideal | 0.8500 | **0.00 px** |
| jittery | 0.9111 | 3.41 px |

So MSE on the ABSOLUTE action spends nearly all its gradient on a quantity the
policy reads off its own input. All four arms consequently over-command by
3-12x (median 6-12px against the expert's 0.6-5.3px), leave the data manifold
within ~50 steps, and land in an ABSORBING FIXED POINT where the commanded
target equals the current position. Action is a desired xy that the pusher walks
toward, so commanding your own position is zero velocity: the pusher freezes,
the object is never engaged, and coverage stays at EXACTLY its initial 0.0000.

Exact zeros mean "nothing happened", not "the metric is broken". Replaying
recorded actions from recorded inits reaches 0.93-0.95.

### The metric that does discriminate

    delta = action - pusher_xy

Score **row 0 only**. Row 0 is the first autoregressive output, produced with no
action context, so causal arms get no teacher-forcing advantage over the
bidirectional control even though the batch carries `actions`.

### Matched-epoch comparison (all four small arms @ epoch 11 of 100)

FVE_delta:

| arm | tight | loose | laggy | sticky | ideal | jittery | seen | held |
|---|---|---|---|---|---|---|---|---|
| arm1 dp_flow | 0.732 | 0.221 | 0.721 | 0.724 | -0.008 | -0.519 | 0.599 | -0.264 |
| arm2 causal_bidir (CONTROL) | 0.914 | 0.580 | 0.650 | 0.764 | -0.118 | -0.387 | **0.727** | -0.252 |
| arm3 state_action_ar (CAUSAL) | 0.792 | 0.217 | 0.735 | 0.773 | -0.161 | -0.254 | 0.629 | **-0.208** |
| arm4 state_idm | 0.602 | 0.096 | 0.550 | 0.425 | -0.350 | -0.230 | 0.418 | -0.290 |

**arm3 - arm2 = -0.098 on SEEN, +0.044 on HELD-OUT.** Causal generation fits the
training gaps worse and generalizes slightly better. arm3 also over-commands
least (7.4px vs arm2's 10.0px on tight, expert 0.6px), consistent with more
conservative out-of-distribution behaviour.

### What this does NOT yet support

- Epoch 11 of 100. arm2's held-out FVE_delta moved from -0.252 (ep11) to -0.188
  (ep15), ~0.064 per 4 epochs, so these numbers are still moving fast.
- Every held-out value is NEGATIVE — worse than predicting zero delta. The
  arm3-vs-arm2 comparison is currently between two failures.
- One seed per arm, n=96 samples per mode.
- Small capacity only. arm1-large is at 46 min/epoch (77h to 100) and will not
  produce a matched large-capacity row on any useful timescale.

### Recommended fix to the action representation

MSE on absolute targets is the wrong objective for this data. Two options:

1. **Residual head (cheap):** predict `action - pusher_xy` and add the current
   position back. The loss then lands entirely on the delta. No retokenization —
   a change at the head and target only. Validate on one arm before the grid.
2. **Retokenize in delta space (clean, expensive):** regenerate tokens and
   retrain all eight runs.


---

## THREE-WAY SPLIT: the control gap is the entire failure

The two-way seen/held-out split conflated two different things. "Seen" sampled
the whole 1000-episode ds_gen folder while training used n_per_mode=547, so it
mixed memorised episodes with new episodes from trained gaps. Decomposing:

| group | data | measures |
|---|---|---|
| A | episodes 0-546 of tight/loose/laggy/sticky | train fit |
| B | episodes 547-999 of the same gaps | NEW episodes, SAME control gap |
| C | ideal / jittery | NEW control gap |

FVE_delta, all four small arms, matched at epoch 11, n=96 per folder:

| arm | A trained eps | B new eps same gap | C new gap | A-B | B-C |
|---|---|---|---|---|---|
| arm1 dp_flow | 0.764 | 0.733 | -0.113 | 0.031 | 0.846 |
| arm2 causal_bidir | 0.705 | 0.671 | -0.175 | 0.034 | 0.845 |
| arm3 state_action_ar | 0.635 | 0.633 | -0.182 | 0.002 | 0.816 |
| arm4 state_idm | 0.421 | 0.350 | -0.328 | 0.071 | 0.679 |

**A-B is ~0 (0.002-0.071): there is essentially no memorisation.** Every arm
transfers cleanly to unseen episodes from trained gaps.

**B-C is 0.68-0.85: the entire collapse is the control-gap change.** Against a
noise band of roughly +/-0.07 this is a large, robust effect, and it isolates
the study's question — the failure is specifically about control-gap transfer,
not about overfitting to episodes.

### Retracted

An earlier revision reported `arm3 - arm2 = +0.044` on held-out and read it as
"causal generation generalises slightly better". Under the cleaner group-C
sampling it is **-0.007**, i.e. arm2 marginally ahead. The sign flips between
two reasonable sampling choices, so **arm3 and arm2 are indistinguishable on
held-out at epoch 11** and the directional claim was over-read from noise.

What survives is stronger and less convenient: **no arm generalises across
control gaps at all.** Every C value sits at or below predicting zero delta.

### Unexpected

**arm1 (dp_flow) leads every column**, including the best held-out (-0.113) —
the flow-matching baseline the study moved away from. One seed at epoch 11 is
not grounds to act, but it inverts the assumed arm ordering and should be
checked before more capacity goes into the causal arms.

### Does training longer help? No.

arm2 small, FVE_delta by epoch:

| epoch | A/B-style seen | held-out |
|---|---|---|
| 3 | 0.343 | -0.115 |
| 7 | 0.603 | -0.114 |
| 11 | 0.727 | -0.252 |
| 15 | 0.730 | -0.188 |

Seen fit more than doubles then plateaus by epoch 11. Held-out is flat within
noise across the whole range and never goes positive. Epochs 15-100 buy
in-distribution fit the study does not need and no transfer that it does.
