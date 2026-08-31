# Causal Action Models on PushShapes Control Modes — Implementation Handover

You are implementing and launching a controlled comparison between a diffusion/
flow baseline and several **causal (autoregressive) action models**, co-trained
across simulator *controller* modes with one mode held out.

Everything below is verified against the repo unless marked ASSUMPTION or TODO.
Where this document says a thing does not exist, it has been checked.

---

## 1. The experiment

One embodiment (`gripper`), six controller modes. Train on five, hold out one.
The held-out mode measures **controller generalization**; the five trained modes
measure in-distribution BC.

### Held-out mode: `jittery`

`jittery` is `ControlGap(noise_std=2.5)` — pure zero-mean sensing noise with no
bias to learn. The other gaps have *structured* biases a model can plausibly
infer and compensate:

| mode    | ControlGap                                                   |
|---------|--------------------------------------------------------------|
| ideal   | (all zero)                                                    |
| tight   | lag 0.25, noise_std 0.3                                       |
| loose   | latency 2, lag 0.55, deadband 1.5, gain 0.95, noise_std 0.8   |
| laggy   | latency 6, lag 0.35, noise_std 0.4                            |
| sticky  | deadband 4.0, gain 0.88, lag 0.15                             |
| jittery | noise_std 2.5                                                 |

Holding out `jittery` therefore asks "can it cope with an irreducible noise floor
it has never seen", which is the honest extrapolation test. Holding out `sticky`
or `laggy` instead would ask a *learnable*-bias question — a legitimate but
different experiment. If you change the held-out mode, say so in the run name.

### Four arms (identical encoder, optimizer, data, parameter budget)

| # | arm | attention | generation | purpose |
|---|-----|-----------|------------|---------|
| 1 | `dp_baseline`   | bidirectional | flow matching | the established baseline |
| 2 | `causal_bidir`  | **bidirectional** | flow matching | **architecture ablation** |
| 3 | `state_action_ar` | causal | flow matching, action fed back | the causal model |
| 4 | `state_idm`     | causal | flow matching over states + IDM | causal states, actions via inverse dynamics |

**Arm 2 is the important control and must not be dropped.** It is the same
backbone as arms 3/4 but bidirectional and non-autoregressive. Without it, any
win by arm 3 confounds "causal generation helps" with "this backbone helps", and
arm 1 alone cannot separate those.

---

## 2. Data — ALREADY GENERATED, do not regenerate

Local, on this machine:

```
~/Desktop/GEAR/sim_run/ds_gen/{ideal,tight,loose,laggy,sticky,jittery}/gripper/T/
```

1000 episodes per mode, 6000 total. Distinct source demos per mode (dedupe at
radius 40 in a concat object+pusher path signature):

```
ideal 625 | tight 602 | loose 554 | laggy 632 | sticky 661 | jittery 644
```

`gripper` was chosen because it holds 554–661 distinct in *every* mode. Do not
substitute `stick`: it has 11–22 distinct sources per cell and is degenerate.

Each episode zarr has: `observations.images.front_img_1` (JPEG bytes, 96x96x3),
`observations.state` (T,6) = `[px, py, pangle, ox, oy, oth]`, `actions` (T,4)
for gripper, `reward`, `goal_pose`, and an `episode_init` attr carrying
`control_gap`, `control_gap_mode`, `reset_seed`. `total_frames` is authoritative
— arrays are chunk-padded beyond it, so ALWAYS slice `[:total_frames]`.

To stage on a training node, mirror the pattern in
`osmo/pushshapes_cotrain11_l40s.yaml`: pull from R2 (see §7) then trim.

---

## 3. Action representation and loss

### 3.1 The baseline is flow matching, not regression

`bf_cotrain11_dense_h16.yaml` composes:

```
egomimic.pipeline.stages_sampler.GaussianLatentNoise
egomimic.pipeline.stages_sampler.MultiJActionSampler   (num_inference_steps: 16)
egomimic.pipeline.stages_sampler.NativeActionMSELoss
```

i.e. noise -> iterative sampler -> MSE on the denoising target. The causal arms
**must use the same generative head**, or the comparison measures flow-vs-
regression instead of causal-vs-bidirectional.

### 3.2 The AR variants currently do NOT do this

`egomimic/models/ar/variants.py` (branch `algo/ar-state-action`) has:

```python
self.action_head = nn.Linear(cfg.d_model, cfg.action_dim)   # direct regression
state_loss_weight / action_loss_weight                       # weighted MSE
```

There is **no** discrete tokenization, no codebook, no cross-entropy — and no
flow matching. This is the single biggest piece of work: replace the direct
regression head with the flow-matching head so arms 2–4 match arm 1.

**TODO-1**: give `_ARBase` a flow-matching action head. Reuse
`GaussianLatentNoise` + `MultiJActionSampler` + `NativeActionMSELoss` rather than
writing a new sampler, so the noise schedule and inference steps are identical
to the baseline by construction.

### 3.3 Target representation: arc tokens

Targets are **planar arc tokens**, not raw action chunks. Use
`egomimic.rldb.zarr.arc_length_tokenizer.TokenizePlanarArcLength`
(branch `sim/planar-arc-tokenizer-v2`, merged into `sim/arc-sweep-cotrain`).

Config knobs, all reachable through
`egomimic.rldb.embodiment.pushshapes.get_planar_arc_length_transform_list`:

```
min_distance_unit      D, arc length per token
resampled_vector_length M waypoints
rotation_radius        0 = translation-only arc, theta rides along;
                       >0 adds lambda_for_radius(r)*mu(dtheta), lambda=2*sqrt(2)*r
hybrid_rotation_unit   rotation gets its own budget; token spans
                       min(D/v_trans, D_rot/v_rot)
velocity_mode          mean_scalar | per_step_scalar
velocity_layout        append -> (M+1, 5) | concat -> (M, 5+V)
```

**Start with `D=10, M=16, rotation_radius=0, velocity_layout=append`.** M=16
matches the dense baseline's 16-step width, so the arc and dense arms have the
same scalar output budget — an arc token and a time chunk covering different
amounts of motion are not comparable (this exact mistake produced an implausible
"130x better" result in a previous study; see §9 of the arc-tokenization note).

Critical property to preserve: a token is produced at EVERY timestep, anchored
at `t`, and waypoint 0 is taken explicitly from index `t` rather than by arc
lookup. See §6 for why.

---

## 4. Capacity

Current `ARVariantConfig`: `d_model=256, n_layers=4, n_heads=8` — about **3M
params**. Target is ~300M.

Params ≈ `12 * n_layers * d_model^2`:

| config     | d_model | layers | heads | params |
|------------|---------|--------|-------|--------|
| current    | 256     | 4      | 8     | ~3M    |
| **target** | **1024**| **24** | **16**| ~302M  |
| alt (wide) | 1280    | 16     | 16    | ~315M  |

Use **1024 / 24 / 16, ff_mult=4**. Depth suits causal sequence modeling better
than width.

**Run a ~30M control (d_model=512, n_layers=12) alongside.** 300M on 6000
episodes is ~50k params per episode. It will fit the training modes easily; the
risk is that held-out-controller SR gets *worse*, and without the smaller model
you cannot distinguish "causal structure helps" from "we overfit". Report both.

**TODO-2**: match parameter count across all four arms to within ~5%. Arm 1's
capacity lives in `bf_pipeline_sampler_*` (CrossTransformer, `n_heads: 8`); you
will need to scale it up to sit next to a 300M causal model. A comparison where
the baseline is 20M and the causal model is 300M measures capacity.

---

## 5. Integration — the actual blocker

`egomimic/models/ar/variants.py` and `sequence.py` have 18 passing unit tests
(`tests/models/test_ar_variants.py`) but were **never wired to the `Algo`
interface or to real data**. They are standalone `nn.Module`s.

**TODO-3**: wrap them so hydra can instantiate them the way
`egomimic.pipeline.algo.PipelineAlgo` is instantiated. Look at how
`bf_cotrain11_dense_h16.yaml` declares `robomimic_model` with `domains`,
`ac_keys`, `stages`, `action_dims`, `rollout_adapters` and mirror that surface.

Required by the sim evaluator (`egomimic/eval/core/eval_sim.py`):
`inference_step(obs_zarr, t, emb_id) -> np.ndarray`, with `t=0` resetting any
recurrent/AR state. This is the single entry point the rollout loop uses.

**TODO-4**: rollout adapter. Training widens every effector to a shared
`[x, y, cos, sin, grip]`; the simulator wants NATIVE width (gripper = 4). Use
`egomimic.pipeline.pushshapes.PlanarArcRolloutAdapter(embodiment=..., velocity_layout=...)`.
`velocity_layout` MUST match the data config: `append` carries a trailing
velocity ROW that is not a waypoint and must be dropped; `concat` carries
velocity as extra CHANNELS. Getting this wrong is silent — a mis-shaped command
is not rejected, just misinterpreted.

---

## 6. Correctness gate — run this before any training

`egomimic/rldb/zarr/planar_arc_sr_gate.py`.

Tokenize at every timestep, execute exactly ONE action, and the success rate
must equal the untokenized baseline per embodiment. It currently does (100% vs
100% on circle/u_socket/umi; gripper 80% vs 80%, reproducing the baseline's own
failures rather than hiding them).

This gate caught a bug that is easy to reintroduce: grip was interpolated along
the ARC parameter, and in a stationary cluster many timesteps share one
cumulative arc value, so the lookup returned the first one and dropped the 0->1
grip transition. **Exactly one row per episode — and it was the grasp trigger.**
That single row took stride-1 replay SR from 100% to 0%. The fix is anchoring
waypoint 0 at the true index (`start_idx` in the gr00t reference).

If you touch the tokenizer, re-run this gate. A tokenizer that fails it will
train fine and produce meaningless SR.

---

## 7. Launch

Launcher: `osmo/pushshapes_cotrain11_l40s.yaml`. It clones `$BRANCH` from
GitHub, `uv sync`, `setup_secret.sh`, wandb preflight, norm-stats on 1 GPU, then
DDP training.

Data pull inside the launcher — **`s3://rldb` is Cloudflare R2, not AWS S3**:

```bash
--endpoint-url "$R2_ENDPOINT_URL"     # from ~/.egoverse_env
unset AWS_SESSION_TOKEN                # stale token -> InvalidArgument
```

Pointing valid AWS credentials at an AWS endpoint returns `AccessDenied` and
looks exactly like a permissions problem. This cost hours in a previous session.

### Pools (measured)

| pool | access | note |
|------|--------|------|
| `groot-l40s-03` | yes | quota 1098/1100 — effectively full despite ~425 idle |
| `groot-l40s-01` | yes | **4 GPU/node** — `num_gpu=8` is rejected outright |
| `isaac-dex-l40s-*` | **403 denied** | not in that DL |

So: `--pool groot-l40s-01 --priority HIGH --set num_gpu=2 batch_size=16`.
`--priority LOW` bypasses quota onto idle capacity but is preemptible.
Submissions intermittently return 503; just retry.

**TODO-5**: before submitting, run
`egomimic/hydra_configs/data/pusht/_preflight_configs.py`. It instantiates every
config's transform exactly as hydra will. A config key with no matching factory
parameter raises `TypeError` only once the job is on the node, AFTER the data
pull — that killed 8 of 9 runs in a previous batch on an unexpected
`rotation_radius`.

---

## 8. Evaluation

`egomimic/hydra_configs/evaluator/eval_sim_pushshapes.yaml` composes N
`SimRolloutEval` instances with `EvalVideoList` (there is **no**
`MultiEvaluatorWrapper` — `trainHydra` instantiates a single evaluator object).

**TODO-6**: write the control-mode analogue. One `SimRolloutEval` per mode, all
with `pusher_shape: gripper`, differing in the control gap. Name them
`seen_<mode>` for the five trained and `unseen_jittery` for the held-out one.

Two things to get right:
- `coverage_threshold: 0.95` — matches the simulator's own `SUCCESS_THRESHOLD`.
  A lower bar reports successes the environment does not count.
- **The evaluator must actually be enabled.** The launcher passes `~evaluator`
  in the norm-stats phase (correct — 1 batch, no rollout) and must pass
  `evaluator=<your config>` in the training phase. Passing `~evaluator` in both
  produces runs that report loss only, finish cleanly, and cannot answer the
  question they were launched for.

**TODO-7**: `SimRolloutEval` needs the control gap applied to its env. Check
whether `env_kwargs` reaches `PushShapesEnv` in a way that sets the gap; if not,
apply it the way `mimicgen.apply_source_control_gap` does. ASSUMPTION: this is
not currently plumbed and will need a small change.

---

## 9. Pitfalls already paid for

1. **Embodiment registration.** Every `pushshapes_sim_*` must exist in the
   `EMBODIMENT` enum (`egomimic/rldb/embodiment/embodiment.py`) or you get
   `KeyError: 'PUSHSHAPES_SIM_X'`. IDs <= 20 are pinned in existing datasets and
   checkpoints — never reuse; new IDs start at 21. Also add to `_ENV_TO_ZARR`
   and the oriented set in `pushshapes_sim.py` (all PushShapes cells store
   6-dim state).
2. **Version pins.** `sim_run/requirements.txt`: `pymunk==7.3.0` (the sim needs
   7.x — `Space.on_collision` is 7.x API; 6.9.0 fails `reset`), `zarr==3.1.3`
   (3.3.0 raises `Chunk size must be positive, got 0` on the empty
   `annotations` array; 3.0.8 lacks `zarr.core.dtype`; 3.1.0 fails writing the
   longest-episode cells).

   **zarr 3.1.0 also corrupts SILENTLY.** It visibly failed on 4 long-episode
   cells with checksum errors, but cells that *appeared* to succeed under it
   were written corrupt too — 11 of 13 `ideal` cells, 52–88% of episodes
   unreadable, while reporting `WRITTEN=1000`. The damage is per-array: the
   zstd-compressed numeric arrays (`actions`, `observations.state`) raise
   `Zstd decompression error` or a checksum mismatch, while the JPEG
   `front_img_1`, `reward` and `goal_pose` in the SAME episode read fine — so a
   spot check that only opens images will not see it. Anything under
   `ds_gen/` dated 2026-08-29 is suspect; 08-30 onward is zarr 3.1.3 and clean.
   Verify a regenerated cell by actually reading `actions` and
   `observations.state`, not by trusting the written count.

3. **Deleting only the result JSON re-runs a cell but does NOT replace it.**
   `run_cell.py` skips a cell whose `results/<gap>__<emb>__T.json` exists, so
   removing the JSON forces a re-run — but the writer then APPENDS to the
   existing output directory. That produced cells with 1801 and 1834 episodes
   against a target of 1000, containing two passes mixed together. Always
   `rm -rf ds_gen/<gap>/<emb>` as well.
4. **Do not put long-running output in `/tmp`** — macOS wipes it on reboot. A
   15-hour dataset build and a day of uncommitted work were lost that way.
   Commit before running anything long.
5. **`nohup` alone did not survive harness teardown**; `setsid` does not exist
   on macOS. Use `nohup bash -c "exec ..." &` + `disown`.
6. Verify a `_target_` exists before writing a config against it. Two configs in
   this session referenced classes that were never in the tree.

---

## 10. Definition of done

- [ ] Flow-matching action head on `_ARBase`, reusing the baseline's sampler
- [ ] All four arms instantiate through hydra, parameter-matched within ~5%
- [ ] 300M and ~30M variants of arms 2–4
- [ ] `planar_arc_sr_gate.py` passes
- [ ] `_preflight_configs.py` passes for every new config
- [ ] Control-mode evaluator reporting `seen_*` and `unseen_jittery` SR
- [ ] Runs launched on `groot-l40s-01`, `num_gpu=2`, with the evaluator ENABLED
- [ ] Reported: SR on 5 seen modes and on held-out `jittery`, for all 4 arms at
      both capacities, with parameter counts stated
