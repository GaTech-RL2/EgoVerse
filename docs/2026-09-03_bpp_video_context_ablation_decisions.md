# Decisions made while setting up arms A and C

Companion to `docs/2026-09-03_bpp_video_context_ablation.md`. Every item below is a
choice the plan did not spell out, or a place where the plan was internally
inconsistent and one reading had to be picked. Each names the file it
affects so it can be reversed. Smoke-run numbers are in the last section.

## Data config (`data/bpp_folding_clothes.yaml`)

1. **Standalone file, not a child of `bpp_cup_on_saucer.yaml`.** The plan
   says "built from `bpp_cup_on_saucer.yaml`". The cup config's valid block
   is `prompt: ${data.train_datasets...prompt}`, and hydra cannot partially
   override an interpolated node (a child `prompt: {heldout_groups: [...]}`
   would replace the whole block, dropping `chunk_n_actions`). So the file
   copies the resolver and writes the train and valid `prompt` blocks out
   in full. The two blocks differ only in `heldout_groups`.
2. **Batch size 32.** Section 4's yaml snippet says 16, sections 3, 5 and 7
   say 32 is the target that the smoke run measures. Went with 32; 16 with
   `trainer.accumulate_grad_batches=2` is the documented fallback.
3. **`shuffle: true` on the valid loader.** Section 5 assumes it ("with
   `shuffle` on the valid loader every eval sees a different draw") but the
   section 4 snippet does not list it, and the cup config has it off.
   Without it, `limit_val_batches: 80` at batch 32 (2560 samples) would only
   ever cover the first one or two of the twelve val episodes in hash order,
   and the unseen metrics would come from a single operator or be missing.
   Side effect: the validation video shows different frames at every eval.
4. **`valid_ratio: 0.0` written explicitly** on both datasets. In `total`
   mode the value is unused; it is set to 0 so nobody reads the cup
   config's 0.2 and expects a split.
5. **`prompt.seed: 0` written explicitly.** The plan says "seeded by
   `prompt.seed` (default 0)", but the dataset's actual default is
   `SEED = 42` from `zarr_dataset_multi.py`. The config pins 0 so the
   prompt-draw RNG matches what the plan states.
6. **`balance_by: group` is also set on the valid dataset** for symmetry
   with the plan's snippet, but it is inert there: only the train loader
   builds a `WeightedRandomSampler`. Validation is a uniform shuffle.
7. **`min_episodes_per_group: 2`, `cache_bytes: 2e9`, `image_size: [224,
   224]`, `num_workers: 8`** carried over from the cup config unchanged. At
   45 chunks a cached prompt is about 7 MB (uint8), so 36 training
   episodes are 245 MB per worker, well inside the cap.

## Code

8. **`prompt.ignore` data knob implemented** (`prompt_dataset.py`). The plan
   marks it optional. It costs three lines and saves arm A the prompt IO
   and the 32 x 45 x 3 x 224 x 224 float collate per batch (about 870 MB
   per batch moving between workers and the main process). Arm A is
   launched with
   `data.train_datasets.human_bimanual.prompt.ignore=true data.valid_datasets.human_bimanual.prompt.ignore=true`.
   Tags and episode lists are unchanged (`test_ignore_knob_skips_prompt_but_keeps_tags`).
   Forgetting the overrides only makes arm A slower, not wrong; setting them
   with a prompted model makes the adapter raise ("batch carries no
   `prompt`"), so the mistake cannot go unnoticed.
9. **The adapter drops `batch["prompt"]` for an unprompted policy** in
   `BPP.process_batch_for_training` (`bpp.py`). The plan's arm table
   already described this behavior, but the code used to pass the prompt
   through and move it to the GPU before ignoring it. Now the tensors never
   leave the host. Covered by `test_noprompt_policy_drops_episode_prompt_batch`.
10. **Total-mode tagging fix** (`prompt_dataset.py`): `heldout_groups` is
    resolved in every mode; only the split modes move episodes. One
    further choice: in `total` mode, holding out *every* group is allowed
    (everything is tagged unseen) instead of raising, since nothing is
    dropped there. The split modes still raise. Tests:
    `test_total_mode_tags_heldout_without_moving`.
11. **Per-group paired MSE added now** (`eval_hpt.py`), as
    `Valid/<emb>_<ac_key>_paired_mse_group_<idx>`. Section 8 lists it as a
    needed change, section 7 step 1 does not; it is 12 lines and the smoke
    run verifies the logging, so it went in. Not done: the "group_idx to
    operator id table written once to the run dir". The evaluator has no
    handle on the dataset; the mapping is already in `trainHydra.log` as the
    dataset's `group i = ('folding_clothes', '<operator>')` lines. For the
    valid dataset of this experiment (groups sorted by string):

    | idx | operator | population |
    |---|---|---|
    | 0 | `68e0b875c33a1abcb8fc55b9` | seen |
    | 1 | `6905a4e79adc5c8f26f52ca2` | unseen |
    | 2 | `693cbbbbc67c9e4814b2ce46` | seen |
    | 3 | `69439ad5a2a8b5aee76602f5` | unseen |
    | 4 | `695d09fc83a9fdf2d84d9c11` | seen |
    | 5 | `695d0ba283a9fdf2d84da234` | unseen |

    The train dataset numbers its three groups separately (0, 1, 2); only
    the valid indices appear in `Valid/` metrics.
12. **Pre-existing test fixed**: `test_sample_weights_balance_groups`
    asserted two groups in total mode, but the working-tree fixture (opD
    added in an earlier session) has three. Now asserts every group has
    equal weight. Unrelated to the arms, but it blocked `-x`.
13. **`heldout_groups` accepts an omegaconf `ListConfig`**
    (`prompt_dataset.py`). The old check was `isinstance(x, (list, tuple))`,
    which hydra-instantiated configs fail (lists arrive as `ListConfig`),
    so any data config with a non-empty `heldout_groups` raised at build.
    The cup smoke config would have hit the same error. Now any non-string
    sequence is accepted; tested with `OmegaConf.create([...])`.
14. **Step time and peak GPU memory are now logged**
    (`egomimic/utils/timing_callback.py`, the `WandbProfilerLogger` callback
    every config already uses). It used to log only Lightning profiler
    durations, and no profiler is configured, so it logged nothing. It now
    also logs `perf/step_time_sec` (mean wall-clock per training step since
    the previous log point), `perf/steps_averaged`, and
    `perf/gpu_max_mem_allocated_gib` / `perf/gpu_max_mem_reserved_gib`
    (`torch.cuda.max_memory_*`, never reset, so it is the run's peak) every
    `log_every_n_steps` (100). These are the "s/step" and "peak mem" columns
    of the plan's results table. Applies to every algo, not only BPP.

## Train and model configs

15. **`norm_stats.norm_mode: minmax` folded into
    `train_zarr_cartesian_bpp_folding.yaml`** (section 8 offered "fold in or
    keep as override"). Every arm inherits it, so no launch can forget it.
    (The scheduler override this item originally left per-run is gone; see
    17c.)
16. **`model/bpp_prompt_dit_episode_folding.yaml` only overrides the two
    `shape_meta` values** (60, 2700); everything else inherits from
    `bpp_prompt_dit_episode.yaml` (`p_drop_prompt: 0`, jitter on, LR 5e-5).
17. **`train_zarr_cartesian_bpp_folding.yaml` inherits
    `train_zarr_cartesian_bpp_episode`** (so the null `annotation_key` comes
    along) and adds `_self_` to its defaults to silence hydra's
    composition-order warning.
17a. **`reject_outliers: true`** (user request, 2026-09-03, after the smoke
    runs which had it off). The BPP base configs disable the per-sample
    bounds check; the folding config re-enables it, so a frame with any
    dimension outside the training set's 0.01th / 99.99th percentile (per
    key, per dim) is substituted and masked out of the eval metrics. The
    cached norm stats from smoke A already hold those percentiles, so the
    cache stays reusable. Not re-smoked.
17b. **`norm_stats.norm_mode: quantile_0_01`** (user request, 2026-09-03)
    replaces `minmax`. New mode in `zarr_dataset_multi.py`: the same
    `[lo, hi] -> [-1, 1]` map as `minmax` and `quantile`, with `lo`/`hi` the
    0.01th / 99.99th percentiles, i.e. the `reject_outliers` bounds. Every
    kept frame therefore lands inside `[-1, 1]`; under true min/max the
    rejected outliers defined the range (1.4x wider on the worst dims of the
    36-episode stats). The three range modes now share one code path
    (`RANGE_NORM_MODES`); `minmax` and `quantile` behave exactly as before.
    Decision 15's minmax rationale (DDIM clip window) carries over. Test:
    `egomimic/rldb/zarr/test_norm_modes.py`. Not re-smoked.
17c. **Flat LR after a linear warmup replaces cosine decay** (user request,
    2026-09-03) in the three BPP model configs (`bpp_dit`, `bpp_prompt_dit`,
    `bpp_prompt_dit_episode`): `torch.optim.lr_scheduler.LinearLR`,
    `start_factor 0.01 -> end_factor 1.0` over `total_iters 1500`, constant
    5e-5 afterwards (stepped per training step, `scheduler_interval: step`).
    1500 is 5 % of the 30k-step runs, matching BPP's own
    `lr_warmup_proportion: 0.05`; the two overfit model configs override
    `total_iters: 500` for their 10k-step runs. The `model.scheduler.T_max`
    launch override in the plan is dropped (`LinearLR` has no `T_max`; it
    would fail). The smokes ran under cosine with `T_max=1400`, which decayed
    to 1e-6 within their 400 steps; not re-smoked.

## Smoke run

18. **Partition.** Chosen per `gpu_usage -l` at launch time; the launcher
    default (`rl2-lab`, a40) was nearly full, so the smoke jobs pass
    `hydra.launcher.partition=hoffman-lab hydra.launcher.account=hoffman-lab`
    when needed. This does not change anything about the run.
19. **Norm stats are recomputed by each smoke run** (no
    `precomputed_norm_path`); pinning starts with the first main run, as
    the plan says.

## Not done, flagged

- The 925-frame unseen episode (`69b272c97ba3e5cddc08eff7`) has not been
  watched; the plan asks for that once before trusting its numbers.
- Arm B, C-vid, C-act, the diagnostic script (section 6.3) and gradient
  checkpointing (only if C does not fit at batch 32) are outside this
  step.
- The prompt images travel as float32 through the dataloader (the dataset
  converts uint8 to float in `_normalize_prompt`). At 45 chunks that is
  about 870 MB per batch of 32. If arm C's step time is dataloader-bound in
  the smoke run, keeping uint8 until the adapter is the first thing to
  change.

20. **Gradient checkpointing on the ViT for the prompted folding model**
    (`BPP(grad_checkpointing=True)` in `bpp.py`, switched on in
    `model/bpp_prompt_dit_episode_folding.yaml`). Arm C at batch 32 ran out
    of memory on the A40 in the first smoke run (43.8 GiB in use when an
    832 MiB allocation failed), so the plan's first fallback applies. The
    flag calls `set_grad_checkpointing(True)` on every timm module in the
    policy and forces timm's non-reentrant checkpoint mode, which is what
    makes it safe under DDP when the shared encoder runs twice per step
    (receding frame and prompt frames). Because the prompt and receding
    paths share one ViT, the receding frame is checkpointed too. The math is
    unchanged; the cost is one extra ViT forward per step. Arm A does not
    use it (its model config has no such flag and it fits in 6 GiB), so A
    and C differ in this compute detail only.

## Rotation representation (added after the main runs started)

21. **Continuous 6D rotation representation** (user request, 2026-09-03).
    The bounds check showed `actions_cartesian` chunks with yaw/pitch/roll
    values at +-3.14: the Euler layout wraps at +-pi, so 2.7 % of training
    and 3.6 % of validation frames were rejected (about 40 % of them pure
    wraps, the rest genuine tail values on z, pitch and roll;
    `logs/claude_scratch/reject_rate_check.log`). Even without rejection the
    wrap is a discontinuity the policy has to regress. Added:
    - `pose_utils.py`: `_matrix_to_xyzrot6d` / `_xyzrot6d_to_matrix`
      (first two rotation-matrix rows, the pytorch3d / behavior_prompting
      convention; Gram-Schmidt inverse), and `_split_action_pose` accepts
      the 18-dim layout so the val-video overlay keeps working.
    - `action_chunk_transforms.py`: `XYZWXYZ_to_XYZRot6D` and the inverse
      `XYZRot6D_to_XYZYPR` (for viz / rollout of rot6d policies).
    - `human.py`: transform mode `cartesian_rot6d` (builder arg
      `rotation="rot6d"`); actions and ee state become 2 x (xyz + 6) = 18.
    - Configs: rot6d is now the only representation. On 2026-09-08 the
      `*_rot6d.yaml` overlays were folded into `data/bpp_folding_clothes.yaml`
      and the `model/bpp_*.yaml` shape_meta (18 dims), and the YPR variants
      were deleted; the first YPR main runs are no longer reproducible from
      the configs.
    - Tests: `egomimic/rldb/zarr/test_rot6d_transforms.py` (round trip,
      Gram-Schmidt tolerance, continuity across the wrap where ypr jumps by
      2 pi, transform-list shapes, viz split agreement with the ypr pipeline).
    Norm stats for the 18-dim layout are computed once
    (`logs/claude_scratch/compute_norm_stats_rot6d.py`) and cached under
    `logs/bpp_video_ctx/norm_stats_rot6d_2026-09-03/`. Rotation MSE changes
    units, so A vs. C comparisons must stay within one representation.
    Rollout code (`robot/rollout.py`) still consumes ypr and needs the
    inverse transform before deploying a rot6d policy; not done.

22. **Bounds check on position dims only** (user request, 2026-09-03, after
    the rot6d rejection check: 4.8 % of train and 8.9 % of val frames
    rejected, one unseen ironing episode losing about a quarter of its
    frames on a single right-hand rotation component;
    `logs/claude_scratch/reject_rate_check_rot6d.log`). Not a config
    option: `MultiDataset._check_bounds` always restricts pose keys
    (`ee_pose` / `cartesian`) to their xyz dims, using
    `pose_utils.pose_position_dims` to locate them from the vector size
    (6/7/9 per arm, one or two arms). Unknown sizes fall back to all dims
    with a warning. Rotation is still 6D for the model (decision 21); only
    the rejection rule changed. Test in
    `egomimic/rldb/zarr/test_norm_modes.py`. The first YPR main runs
    (12:32) started before this and reject on all 12 dims.

23. **Sequential validation-video loader** (user request, 2026-09-04). With
    `shuffle: true` on the metrics loader (decision 3) the validation video
    was a slideshow of random frames. `MultiDataModuleWrapper` now takes
    `valid_viz_params`; when set, `val_dataloader` returns
    `[metrics_loader, viz_loader]` where the viz loader is a sequential
    `Subset` over contiguous frames (`episodes: auto` = first seen + first
    unseen val episode, `all`, or a list; `frames_per_episode`, `start_frac`).
    Indices are repeated `world_size` times so every DDP rank sees the whole
    sequence despite Lightning's DistributedSampler sharding. The evaluator
    (`eval_video.py`) logs metrics only from loader 0 (with
    `add_dataloader_idx=False`, so metric names are unchanged) and buffers
    frames only from loader 1, grouped by the new per-sample `episode_idx`
    (`prompt_dataset.py`), writing one video per episode on rank 0:
    `videos/epoch_<N>/HUMAN_BIMANUAL/ep<idx>_<hash8>_<seen|unseen>.mp4`.
    Folding config: 300 frames (10 s) from 30 % into each of the two
    episodes, batch 16. Configs without `valid_viz_params` behave as before.
    Full-validation videos over all 12 episodes: eval mode with
    `episodes=all frames_per_episode=null trainer.limit_val_batches=2000`.
    Tests: `egomimic/pl_utils/test_valid_viz_loader.py`.

24. **Strict resume drops the policy's stray state-dict key** (2026-09-04).
    behavior_prompting's `BasePolicy.state_dict` appends a bare
    `_extra_training_split_info` entry; its own `load_state_dict` pops it,
    but a Lightning `ckpt_path` resume loads through `ModelWrapper` with
    `strict=True` and failed with "Unexpected key(s)". Eval mode loads with
    `strict=False`, which is why evals never hit it. `ModelWrapper.
    on_load_checkpoint` now removes the key before the state dict is
    applied. Also: hydra rejects `=` inside override values, so resume from
    `last.ckpt` rather than `epoch_epoch=299.ckpt` (identical for a finished
    run), and eval mode caps validation batches through
    `+evaluator.limit_val_batches`, not `trainer.limit_val_batches`.

25. **Viz loader under DDP: plain loaders, not nested CombinedLoaders**
    (2026-09-05). The continuation runs' eval clips played at 1/4 speed:
    each 4-GPU rank iterated the whole viz sequence (the seen clip had 1200
    frames for a 300-frame window, each source frame four times with a
    different DDIM sample) and the 80-batch validation cap then truncated the
    unseen clip to 80 frames. Lightning only shards (DistributedSampler) and
    `set_epoch()`s DataLoaders it can see; a CombinedLoader nested inside the
    `[metrics, viz]` list is opaque to it, so the index repetition of
    decision 23 was never divided by the world size. `val_dataloader` now
    returns two plain DataLoaders whose collate wraps batches as
    `{dataset_name: batch}` (`_NamedCollate`); viz is limited to a single
    valid dataset. Single-GPU eval runs (the 30k full validations) were
    correct. The continuation metrics loader was nested the same way, so its
    4 ranks likely scored the same 1280-frame draw rather than 4 shards;
    that adds eval noise but is identical across arms. Test simulates a
    2-rank world; verified 2026-09-08 with a 2-GPU training smoke
    (`logs/bpp_ctx_smoke/vizsmoke2gpu_A_2026-09-08_13-09-42`): all four
    clips are exactly 300 frames and each rank ran 38 viz batches per
    validation (600 repeated indices / 2 ranks / batch 16). An earlier
    relaunch never ran because Hydra treats an unquoted comma in
    `hydra.launcher.exclude=a,b` as a sweep; quote the list.
    Note: eval mode (`mode=eval`) forces `trainer.devices=1` in
    `trainHydra.py`, so eval runs are single-GPU regardless of
    `launch_params.gpus_per_node`; the clean 60k clips come from
    `clips_{A,C}_60k` single-GPU evals.

26. **One video per operator for full validation: `episodes: per_group`**
    (2026-09-08). Twelve whole-episode videos per arm is more than needed
    to eyeball behaviour, so `viz_indices` gained `episodes="per_group"`:
    the first episode by name of every prompt group (task, operator) from
    the dataset's `_episodes_by_group`, giving six whole episodes per arm
    (three seen, three unseen operators). Because only the clips were
    wanted, `valid_viz_params.<ds>.viz_only: true` drops the metrics loader
    (the all-frame pass over 12 episodes was most of the eval time), so
    validation runs the viz loader alone at `dataloader_idx` 0 and logs no
    numbers. The 60k video evals (`vids_{A,C}_60k`) use
    `episodes=per_group frames_per_episode=null viz_only=true`.

## Smoke results

Section 7 step 3 of the plan. 4 epochs of 100 steps at batch 32, val every
2 epochs (80 batches), one A40 each, hoffman-lab.

| Arm | Run | s/step | Peak mem (alloc / reserved) | Seen paired MSE | Unseen paired MSE | Seen loss | Unseen loss | Notes |
|---|---|---|---|---|---|---|---|---|
| A | `smoke_A_2026-09-03_09-13-20` | 0.159 | 6.2 / 7.3 GiB | 0.494 | 0.544 | 0.085 | 0.096 | 400 steps in 4 min; all six `paired_mse_group_<i>` logged |
| C (no ckpt) | `smoke_C_2026-09-03_09-13-20` | n/a | OOM at 43.8 GiB | n/a | n/a | n/a | n/a | first training step; batch 32 x up to 46 images |
| C (ViT ckpt) | `smoke_C_2026-09-03_09-25-11` | 3.48 | 25.3 / 29.6 GiB | 0.441 | 0.478 | 0.091 | 0.093 | 400 steps + 2 evals in 38 min (evals about 5.6 s per val batch); all six `paired_mse_group_<i>` logged |

The MSE values after 400 steps are not results, only proof that the
metrics are logged for both populations. Both arms pass the plan's smoke
criteria: they finish, log the seen and unseen metrics, and the step time
and peak memory at batch 32 with the 45-chunk prompts are measured.

**Budget implication for the main runs (section 5).** At 3.5 s/step a
30 000-step prompted run is about 29 h of training plus about 1.5 h of
evaluation (12 evals of 80 batches at 5.6 s), so about 31 h on one A40,
inside the launcher's 48 h limit. Arm A is about 1.5 h. The plan's 8 runs
(6 prompted) are therefore about 190 GPU-hours. The step time is GPU-bound
(100 % utilization, 31 GiB in use when checked), not dataloader-bound, so
the uint8-through-collate change listed above would not shorten it; the
options are the plan's own (fewer seeds on B first, then fewer steps) or a
coarser prompt grid, which the plan rules out. Adding training data does
not change the step time; raising `max_sequence_length` to about 3700 for
longer episodes would (62 chunks instead of 45, about 1.4x the ViT work
per step and about 35 GiB peak by linear scaling). Arm A's norm-stat cache is at
`logs/bpp_ctx_smoke/smoke_A_2026-09-03_09-13-20/0/norm_stats/norm_stats.json`
(minmax, the 36 training episodes) and can be passed as
`norm_stats.precomputed_norm_path` to any run on the same training set.

### rot6d smokes (2026-09-03, 4 epochs, batch 32, one A40 each, position-only rejection)

| Arm | Run | s/step | peak alloc / reserved | seen MSE | unseen MSE |
|---|---|---|---|---|---|
| A rot6d | smoke_A_rot6d_2026-09-03_12-45-23 | 0.158 | 6.2 GiB | 0.124 | 0.131 |
| C rot6d | smoke_C_rot6d_2026-09-03_12-45-23 | 3.55 | 25.3 / 29.6 GiB | 0.103 | 0.110 |

MSE is in rot6d units (m^2 for xyz, unit-vector components for rotation),
not comparable to the YPR smoke table above. Both arms log every seen /
unseen / per-group metric, save checkpoints and val videos.

## Main run results (rot6d, 4 A40s each, batch 16/GPU, 30k steps, seed 42)

**Arm A** (`A_noprompt_rot6d_4gpu_2026-09-03_13-13-43`, job 3748952): finished
2026-09-03 21:12, 12 evals, checkpoints every 50 epochs + last. Paired MSE in
rot6d units, each eval a fresh 1280-frame draw (about 10 % eval-to-eval noise):

| Step | seen | unseen | loss seen | loss unseen |
|---|---|---|---|---|
| 10000 | 0.108 | 0.113 | 0.030 | 0.034 |
| 20000 | 0.082 | 0.087 | 0.027 | 0.028 |
| 25000 | 0.069 | 0.079 | 0.027 | 0.030 |
| 30000 | 0.063 | 0.073 | 0.027 | 0.030 |
| mean of last 4 | 0.069 | 0.082 | | |

Final per-operator (valid group idx -> operator table in decision 11):
seen 0.068 / 0.067 / 0.056, unseen 0.072 / 0.079 / 0.056. The unseen gap
appears after ~17.5k steps and is carried by the two ironing / assorted
operators; the unseen checkered-cloth operator matches the best seen one.

**Arm C** (`C_prompt_rot6d_4gpu_2026-09-03_13-13-43`, job 3748953): finished
2026-09-04 11:35 (22.3 h, 2.2 s/step), 12 evals, checkpoints every 50
epochs + last.

| Step | A seen / unseen | C seen / unseen |
|---|---|---|
| 5000 | 0.136 / 0.134 | 0.124 / 0.124 |
| 10000 | 0.108 / 0.113 | 0.111 / 0.118 |
| 15000 | 0.096 / 0.108 | 0.085 / 0.092 |
| 17500 | 0.078 / 0.086 | 0.079 / 0.085 |
| 20000 | 0.082 / 0.087 | 0.067 / 0.073 |
| 22500 | 0.073 / 0.085 | 0.067 / 0.072 |
| 25000 | 0.069 / 0.079 | 0.070 / 0.076 |
| 27500 | 0.070 / 0.090 | 0.059 / 0.068 |
| 30000 | 0.063 / 0.073 | 0.058 / 0.065 |
| **mean of last 4** | **0.069 / 0.082** | **0.064 / 0.070** |

C vs. A over the last four evals: -8 % seen, -14 % unseen. The arms are
tied through 17.5k steps; C pulls ahead from 20k on and leads at every
later eval on both splits. Per operator at 30k, C is better on all six,
most on the two A finds hardest (unseen ironing 0.072 -> 0.063, unseen
assorted 0.079 -> 0.074) and least on the checkered-cloth pair. Val
diffusion loss plateaued for both at ~17.5k (about 0.027 seen / 0.030
unseen, indistinguishable between arms) while paired MSE kept improving
to 30k for both, so the runs were cut off while still improving; train
loss was still falling (0.030 in the last window). Caveats: one seed,
~10 % eval-to-eval noise (fresh 1280-frame draw per eval), gain present
on seen operators too, so arm B (prompt zeroed) and the prompt-swap
diagnostic are needed to attribute it to prompt content.

**Full-set validation of the 30k checkpoints** (eval mode, every val
frame once, 2026-09-04; `fullval_{A,C}_30k_2026-09-04_11-54-54`, videos for
all 12 episodes under `0/videos/epoch_0/HUMAN_BIMANUAL/`). Unlike the
training-time evals this is one deterministic pass over all 25 096 frames:

| paired MSE | A | C | C vs A |
|---|---|---|---|
| seen | 0.0632 | 0.0584 | -8 % |
| unseen | 0.0753 | 0.0664 | -12 % |
| seen ironing (68e0) | 0.070 | 0.059 | -15 % |
| unseen ironing (6905) | 0.077 | 0.065 | -16 % |
| seen lint (693c) | 0.068 | 0.065 | -4 % |
| unseen assorted (6943) | 0.079 | 0.073 | -8 % |
| seen cloth (695d09) | 0.050 | 0.050 | 0 % |
| unseen cloth (695d0b) | 0.068 | 0.060 | -12 % |

Diffusion loss: A 0.0270 / 0.0301, C 0.0267 / 0.0303 (seen / unseen), i.e.
identical. Final-step MSE 0.108 vs 0.103; Frechet-over-time mean 0.95 vs
0.90. These full-set numbers agree with the mean of the last four
training-time evals (A 0.069 / 0.082, C 0.064 / 0.070).

Both runs are continued from `last.ckpt` to 60k steps (epochs 300 -> 600)
with the same flat 5e-5 LR (`logs/claude_scratch/launch_cont_rot6d.sh`).

**Arm A continuation 30k -> 60k** (`A_noprompt_rot6d_4gpu_cont_2026-09-04_11-59-05`,
job 3757047, finished 15:35, 0.33 s/step):

| Step | seen | unseen | loss seen | loss unseen |
|---|---|---|---|---|
| 30000 | 0.063 | 0.073 | 0.027 | 0.030 |
| 40000 | 0.055 | 0.071 | 0.028 | 0.032 |
| 50000 | 0.055 | 0.070 | 0.033 | 0.034 |
| 60000 | 0.055 | 0.062 | 0.032 | 0.039 |
| mean of last 4 (52.5k-60k) | 0.052 | 0.064 | | |

Paired MSE kept improving to about 47.5k (seen 0.049) and is flat with
noise after that; val diffusion loss rose steadily from 0.027 / 0.030 to
0.032 / 0.039, i.e. the unprompted model is overfitting the 36 episodes in
loss while sample quality plateaus. Final per-operator: seen iron 0.060,
unseen iron 0.057, seen lint 0.058, unseen assorted 0.067, seen cloth
0.047, unseen cloth 0.060.

**Arm C continuation 30k -> 60k** (`C_prompt_rot6d_4gpu_cont_2026-09-04_11-59-08`,
job 3757048, finished 2026-09-05 10:40, 2.2 s/step). Matched-step
comparison over the continuation (paired MSE, rot6d units):

| Step | A seen / unseen | C seen / unseen |
|---|---|---|
| 40000 | 0.055 / 0.071 | 0.049 / 0.056 |
| 50000 | 0.055 / 0.070 | 0.049 / 0.057 |
| 60000 | 0.055 / 0.062 | 0.047 / 0.054 |
| mean of last 4 (52.5k-60k) | 0.052 / 0.064 | 0.047 / 0.057 |
| mean of all 12 (32.5k-60k) | 0.055 / 0.068 | 0.049 / 0.058 |

C vs. A over the last four evals: -10 % seen, -11 % unseen; over all
twelve continuation evals -12 % / -14 %. C leads at every one of the 12
matched evals on both splits. Per operator (last-4 means): seen iron
-12 %, unseen iron -11 %, seen lint -12 %, unseen assorted -14 %, seen
cloth -8 %, unseen cloth -6 %. C's paired MSE is roughly flat from ~40k
(0.047-0.050 seen, 0.054-0.061 unseen); its val diffusion loss rises like
A's (0.027/0.031 -> 0.037/0.042), so both arms are past the useful step
budget for 36 episodes. Final 60k checkpoints: `.../0/checkpoints/last.ckpt`
in each continuation dir; per-episode eval clips under `0/videos/epoch_<N>/`.


