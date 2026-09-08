# Experiment Plan: BPP With vs. Without Video Context

Context: the BPP algo (`egomimic/algo/bpp.py`) can train with a whole-episode
prompt (`prompt.mode: episode_pair`, `docs/2026-09-02_bpp_episode_prompting.md`) or as a
plain diffusion transformer with no prompt (`model/bpp_dit.yaml`). The
episode-pair overfit check passed (`docs/2026-09-02_bpp_episode_overfit_debugging.md`),
so the pipeline is trusted. This doc plans the first real comparison: does
conditioning on a video prompt of another demonstration by the same operator
improve action prediction, and if so, on which population and through which
part of the prompt.

"Video context" here means the prompt the dataset attaches to every sample:
one front-camera frame, one 12-dim proprio pose and two seconds of actions
per chunk, covering a second `folding_clothes` episode of the same operator
end to end. The plan separates the prompt's frames from its actions so the
video part is measured on its own, not only bundled with the action part.

## 1. Questions and hypotheses

| # | Question | Hypothesis | Decided by |
|---|---|---|---|
| Q1 | Does the prompt lower prediction error on held-out episodes of training operators (seen)? | Small gain at most; the operator's style is already learnable from the training set. | seen-operator paired MSE, arm A vs. C |
| Q2 | Does the prompt lower error on operators never seen in training (unseen)? | This is where the prompt should help: one demo of a new person tells the policy their pace and folding strategy. | unseen-operator paired MSE, A vs. C |
| Q3 | Is the gain from the video frames, the prompt actions, or both? | The action part carries most of the style signal; frames add garment state and layout. | C vs. C-video-only vs. C-actions-only |
| Q4 | Does the prompted model actually read the prompt? | Measured, not assumed: swapping the prompt for another operator's demo must hurt. | prompt-swap diagnostic on C's checkpoint |
| Q5 | Is the difference an architecture effect? | The prompted encoder has extra layers and parameters; arm B controls for that. | B vs. A and B vs. C |

Q2 and Q4 are the primary results. Q1, Q3 and Q5 are supporting.

## 2. Arms

All arms share the same data config, the same split, the same norm stats,
the same sampler, the same optimizer, schedule, augmentation and number of
gradient steps. Only the model differs.

| Arm | Name | Model | Override on top of `train_zarr_cartesian_bpp_folding` | Prompt seen by the policy |
|---|---|---|---|---|
| A | no-prompt | `bpp_dit` (plain `TransformerObsEncoder`) | `model=bpp_dit` | none; the adapter drops `batch["prompt"]` because `supports_prompting()` is False |
| B | prompt-ignored | `bpp_prompt_dit_episode_folding` with the encoder told to zero the prompt | `model.robomimic_model.policy.obs_encoder.ignore_prompt=true` | encoded then multiplied by zero (`prompt_obs_encoder.py:813`), so same parameters and compute as C, no information |
| C | full prompt | `bpp_prompt_dit_episode_folding` as is | none | frames + proprio + actions |
| C-vid | video-only prompt | C with prompt actions and proprio ignored | `...obs_encoder.obs_encoder.ignore_prompt_action=true ...ignore_prompt_proprio=true` | frames only |
| C-act | actions-only prompt | C with prompt frames ignored | `...obs_encoder.obs_encoder.ignore_prompt_obs=true` | proprio + actions, no video |

Notes on the arms:

- Arm A uses the same `EpisodePromptMultiDataset` data config as the others,
  so the episode lists, held-out operators, `balance_by: group` sampler and
  the `operator_seen` tags are identical, and the seen/unseen metrics in
  `eval_hpt.py` are logged for it too. The adapter already passes the
  nested `prompt` dict through and omits it from the obs dict when the
  policy cannot prompt. This combination has not been run yet; the smoke
  run in section 7 covers it. The prompt is still read and collated for
  arm A, which wastes IO; section 8 lists a config knob to skip it.
- Arm B is the cleanest "same model, no information" control. The
  prompt-side ViT still runs, so its step time matches C.
- The `ignore_prompt_*` flags live on `PairPromptTransformerTokenizer`, whose
  config path is `model.robomimic_model.policy.obs_encoder.obs_encoder`. At
  least one of the three must stay on, which both C-vid and C-act satisfy.
  `include_in_prompt_current_obs` in `shape_meta` stays as is for both.
- `p_drop_prompt` stays at 0 in every arm. A dropout arm (`p_drop_prompt:
  0.1`, the classifier-free style used for deployment robustness) is a
  follow-up, not part of this comparison, because it changes what the model
  is trained to do without the prompt.

## 3. Prompt resolution: 2 s chunks

Folding demos run 30 to 90 s here. At the 1 Hz chunk grid used for the
short-task work a prompt is up to 90 frames, and with the ~900 ViT-B
images per step that a 48 GB A40 holds in bf16 (rule from the prompting
doc, section 2) the batch would have to drop to about 10. The tokenizer
builds one token per chunk from that chunk's image, proprio and actions
together (`prompt_obs_encoder.py:322` to `340`, then the attention pool
under `merge_prompt_tokens: obs_and_action`), so images cannot be put on a
coarser grid than the actions without a tokenizer change.

Decision: one chunk every 2 s with every raw action step kept.

| Setting | Value | Effect |
|---|---|---|
| `shape_meta.prompt_chunk_n_actions` | 60 | 60 action steps per chunk at 30 Hz = 2 s; the action projection sizes itself from this |
| `prompt.prompt_stride` | 1 | no action subsampling |
| `shape_meta.max_sequence_length` | 2700 | the longest episode in the experiment is 2699 frames (90 s, a training episode of `68e0b875...`); the dataset lists any longer episode at build time; 45-row positional table |
| Prompt per sample | up to 45 chunks, 45 images + 1 current frame | |
| Batch size | 32 (target) | 32 x 46 = 1472 ViT images per step in the worst case, above the ~900 estimate, so the smoke run measures it; 16 would be 736 |

Nothing changes on the model side except the two `shape_meta` values. The
prompt transform list takes `chunk_length` from `prompt_chunk_n_actions`, so
each 2 s chunk is expressed in the head frame at the chunk start; head
drift within a chunk is twice what a 1 s chunk had.

The 900-image figure is a rule of thumb scaled from the original's UMI
config, not a measurement, so batch 32 is tried first. If it does not
fit, the order is: gradient checkpointing around the prompt-side ViT
(the prompt frames' activations are what fills memory, and checkpointing
trades them for a second forward pass), then batch 16 with gradient
accumulation of 2 to keep the effective batch at 32. Coarsening the chunk
grid is not on the list. Every arm moves together whenever the batch
setting changes, including arm A, which would fit 32 without any of this.

The two local folding episodes (54 s and 95 s) are `folding_clothes` and
can drive the CPU-only checks in section 7 before any S3 sync.

## 4. Data and split

Task: Mecka `human_bimanual` `folding_clothes` (freeform folding,
`s3://rldb/processed_v3/mecka/freeform/`), six operators, three for
training and three held out entirely. Episode hashes come from the SQL
episode table (`egomimic/scripts/tutorials/sql_tutorial.ipynb` shows the
access path) and are pinned below so the split does not depend on row
order or seeds.

| Role | Operator id | Folding episodes in table | Train | Val | Longest used (frames) |
|---|---|---|---|---|---|
| train | `68e0b875c33a1abcb8fc55b9` | 21 | 12 | 2 | 2699 |
| train | `695d09fc83a9fdf2d84d9c11` | 21 | 12 | 2 | 2698 |
| train | `693cbbbbc67c9e4814b2ce46` | 14 | 12 | 2 | 2695 |
| val (unseen) | `69439ad5a2a8b5aee76602f5` | 14 | 0 | 2 | 2122 |
| val (unseen) | `6905a4e79adc5c8f26f52ca2` | 13 | 0 | 2 | 2155 |
| val (unseen) | `695d0ba283a9fdf2d84da234` | 8 | 0 | 2 | 2099 |

Training: 36 episodes, 83 918 frames, about 0.78 h. Seen-operator
validation: 6 episodes, 14 205 frames. Unseen-operator validation: 6
episodes, 10 891 frames. Every other folding episode of these six
operators stays out of both splits.

**Pinned episodes.** Train operators: a seeded shuffle (`random.Random(0)`
over the hash-sorted list per operator: first 2 val, next 12 train). Val
operators: the two shortest folding episodes each. Frozen here; the data
config copies these lists verbatim.

Train, `68e0b875c33a1abcb8fc55b9` (12):
`69b4d5ea2c83b2c34dfe560d 69b4b4ed999caf0c82b98be6 69b1c9e184028c82aba1269e
69b4fe13d309464df7115850 69b494f9690ce5eb0225ca23 69b1cbcd3fa257258bdafc7d
69b1cd61850f4b031baebdfe 69b36456806f8dde7843c8cd 69b1e127bbf6e05cf50b8579
69b337fe7aaac69b6ecbeeee 69b1d90aeaaec6169402ae02 69b547d71666a9ba0092e8f4`
Val: `69b493642127166166f189f2 69b53baf3dd17eb53d7ddc35`

Train, `695d09fc83a9fdf2d84d9c11` (12):
`69b49d6def58e30ec42f86eb 69b31fb3122e58b0d29ad6b1 69b52abf4963ef0ac28d5beb
69b43f198a313c3e6edaab44 69b49d536325cbcd4232ac72 69b49c96d5bde809a4eebb79
69b49d266a0a108e820b1073 69b504f361c85d2534e8f88a 69b529589f66a2746698fbe1
69b54ff560172888461a8600 69b4ae0a7b8718172195a07c 69b496c930ac9bf923d7f170`
Val: `69b3250b098c263fb8c36782 69b4acf23282e47c8735365c`

Train, `693cbbbbc67c9e4814b2ce46` (12):
`69b3a406daef557c66e76aeb 69b4f843758deefb5ec01244 69b3699cd04bd6fd45e77d76
69b4e89e680bedd399fcbb5c 69b9c8f8a6f41015b5203409 69b4953c60b9979b587b0c27
69b4b82da85a9038236c3e9e 69b238d3f6f2b3966869f7c7 69b525b01b5d00f6e2fca79a
69b4ae6d1279cedd289bc325 69b553ebf629796beb9845d4 69b32ea71fea872861dae3f2`
Val: `69b5787761c85d2534e8fb9a 69b485709b2ecfe4a474b8a9`

Unseen val, `69439ad5a2a8b5aee76602f5`:
`69b1ca5b179c113f82bd4614` (1795 frames), `69b33c636d8e889d7aa560a6` (2122)

Unseen val, `6905a4e79adc5c8f26f52ca2`:
`69b1c4dc1b7bf3cf07b4c1e3` (1795), `69b3496f13390784beeac7ad` (2155)

Unseen val, `695d0ba283a9fdf2d84da234`:
`69b272c97ba3e5cddc08eff7` (925), `69b330bf13e59a723bbc6fec` (2099)

The 925-frame episode (31 s) is the only one this short in the whole
set; check its video once before trusting its numbers, since a truncated
or aborted recording would distort that operator's unseen metric.

**Data config.** `data/bpp_folding_clothes.yaml`, built from
`bpp_cup_on_saucer.yaml`. Both datasets run in `mode: total`, so the
dataset does no splitting of its own; the filter lambdas are the split.

```yaml
train_datasets:
  human_bimanual:
    filters:
      filter_lambdas:
        - "lambda row: row['episode_hash'] in [<36 train hashes>]"
    mode: total
    prompt:
      chunk_n_actions: 60          # literal, not interpolated: arm A's model has no prompt shape_meta
      max_sequence_length: 2700
      heldout_groups: []
      balance_by: group

valid_datasets:
  human_bimanual:
    filters:
      filter_lambdas:
        - "lambda row: row['episode_hash'] in [<6 seen val hashes + 6 unseen val hashes>]"
    mode: total
    prompt:
      <same as train, plus>
      heldout_groups: ['69439ad5a2a8b5aee76602f5', '6905a4e79adc5c8f26f52ca2', '695d0ba283a9fdf2d84da234']

train_dataloader_params / valid_dataloader_params: batch_size 16
```

`chunk_n_actions` and `max_sequence_length` must be literals here rather
than `${model.robomimic_model.shape_meta...}` interpolations, because arm
A's `bpp_dit` has no prompt fields in its `shape_meta` and the
interpolation would fail at compose time. The adapter still asserts that a
prompted model's `shape_meta` matches the data, so the two places cannot
drift silently.

**Seen/unseen tagging in `total` mode needs a small fix.**
`EpisodePromptMultiDataset` ignores `heldout_groups` entirely when
`mode == "total"` (`prompt_dataset.py:333` and `:363`), so the valid
dataset above would tag every sample `operator_seen=1` and the seen and
unseen metrics would collapse into one number. The change: in `total`
mode, still resolve `heldout_groups` for `_operator_seen` but do not move
or drop any episode (total keeps everything anyway). About five lines,
with a test that a total-mode dataset with a held-out operator tags its
samples 0 and the rest 1.

**Seen-operator validation is a swap.** Each training operator has
exactly two val episodes. The valid dataset's prompt pool for a group is
that group's episodes in the valid dataset (`prompt_dataset.py:587` draws
from `_episodes_by_group` minus the sample's own episode), so every frame
of val episode 1 is prompted by val episode 2 and every frame of episode 2
by episode 1. Both directions are validated on every eval, and the pairing
is deterministic. The unseen operators work the same way: two episodes
each, prompting each other, so every val sample in either population has
exactly one possible prompt.

**Training prompts.** A training sample is prompted by one of the other 11
training episodes of its operator, drawn per call. Val episodes are never
seen by the training dataset in any role.

**Determinism.** The split is fixed by the hash lists. Prompt draws are
seeded by `prompt.seed` (default 0) and `seed: 42` at the top level; keep
both identical across arms.

**Norm stats.** Computed from the 36 training episodes. Compute once and
pin: the first run writes its stats to `norm_stats.save_cache_dir`; every
other arm gets `norm_stats.precomputed_norm_path=<that file>` so the
normalization is byte-identical. Use `norm_stats.norm_mode=minmax`, the
recommendation from the overfit debugging (quantile mode put targets
outside the DDIM clip window).

**Outliers.** `reject_outliers: false`, as in every BPP config.

## 5. Training recipe (identical across arms)

Carried over from the overfit debugging conclusions and the original UMI
config. Everything below is a per-run override unless it is folded into the
configs first (section 8).

| Setting | Value | Reason |
|---|---|---|
| Batch size | 32 | section 3; the smoke run confirms it fits, with checkpointing or accumulation as fallbacks; arm A stays matched |
| Steps per epoch | 100 (`trainer.limit_train_batches`, the trainer default) | an epoch is a fixed 100 steps, so epochs are a step budget, not a data pass |
| Total steps | 30 000 (`trainer.max_epochs=300 trainer.min_epochs=300`) | about 11 passes over 83 918 training frames at batch 32; the two-episode overfit converged by 10 k steps, and a 36-episode set with color jitter needs more but not 60 k |
| LR | 5e-5 (config default), backbone 0.1x with no weight decay via `optimizer_param_groups` | the original's values; 1e-4 was an overfit-only choice |
| Schedule | linear warmup to 5e-5 over 1500 steps, then flat (`LinearLR` in the model configs; decisions doc 17c) | replaces the cosine decay first planned here |
| Color jitter | on (config default) | the original trains with it; only the overfit variant turns it off |
| Sampler | `balance_by: group` | equal total weight per operator; nearly a no-op with 12 episodes each but keeps the code path identical to later runs |
| Validation | `trainer.check_val_every_n_epoch=25` (every 2.5 k steps), `limit_val_batches: 80` | 80 batches of 32 = 2560 samples per eval over 6 seen and 6 unseen episodes (25 096 frames); with `shuffle` on the valid loader every eval sees a different draw, so compare averages over several evals |
| Checkpoints | `callbacks.model_checkpoint.every_n_epochs=50` (every 5 k steps) plus `last` | diagnostics in section 6.3 run on `last.ckpt` and on the mid-run checkpoints |
| Seeds | 42 and 43 for A, B, C; one seed (42) for C-vid and C-act | the val metrics have visible seed noise (the overfit doc saw 0.10 vs. 0.084 on identical data), so a single-seed difference is not a result |
| Precision | bf16 (trainer default) | |

Compute matching: arms are matched on gradient steps, which means equal
samples seen. Arm A takes fewer seconds per step because it has no prompt
ViT forwards; record the wall-clock per step for each arm in the results
table rather than trying to equalize it.

Run count: 3 arms x 2 seeds + 2 ablation arms = 8 runs of 30 k steps each,
one GPU each. Measure the step time in the smoke run first (section 7) and
scale the step budget down if 8 runs do not fit the GPU allocation; cut
seeds on B before cutting steps.

## 6. Metrics

### 6.1 Logged during training (existing)

All from `egomimic/eval/eval_hpt.py` and `BPP.forward_eval`, per embodiment
`human_bimanual` and action key `actions_cartesian`:

| Metric | What it measures | Role |
|---|---|---|
| `Valid/human_bimanual_actions_cartesian_paired_mse_unseen_operator` | DDIM-sampled 100-step trajectory vs. ground truth, real units, unseen operators | **primary** (Q2) |
| `Valid/human_bimanual_actions_cartesian_paired_mse_seen_operator` | same, the 6 swapped val episodes of training operators | primary (Q1) |
| `Valid/human_bimanual_loss_unseen_operator`, `_seen_operator` | diffusion loss on the split | secondary; averaged over 50 noise levels, dominated by high-noise steps, moves earlier than sample quality |
| `Valid/human_bimanual_actions_cartesian_final_mse_avg`, `frechet_gauss_*` | end-point error and trajectory-shape distance, whole valid set | secondary |
| `Train/action_loss`, `Train/policy_grad_norms_mad_flag` | training health | a `mad_flag` rate above ~5 % means the median-norm clip is active |

Compare arms at the same step, using the last three evals (25 k, 27.5 k,
30 k) averaged, not the single best eval, so the pick is not a noise
maximum.

### 6.2 Per-operator breakdown (small addition)

With three unseen operators, the unseen average can be carried by one
person, and the seen average by one of the six swapped episodes. Extend
`eval_hpt.py` to also log `paired_mse` per `group_idx` (the batch already
carries it) as `Valid/<emb>_<ac_key>_paired_mse_group_<idx>`, with the
`group_idx` to operator id table written once to the run dir. This is
needed to interpret a positive Q2, not only optional.

### 6.3 Post-hoc checkpoint diagnostics (new script)

Extend `logs/claude_scratch/ckpt_diag.py` into `egomimic/eval/bpp_prompt_diag.py`
so it runs on any BPP checkpoint and a data config. It builds a fixed
diagnostic batch (192 valid frames, half from the 6 seen val episodes and
half from the unseen operators, drawn once with a seed and stored) and
reports DDIM-16 paired MSE under these prompt conditions:

| Condition | How it is built | What it tests |
|---|---|---|
| matched | the dataset's own same-operator prompt (for seen episodes, the swapped partner) | reference |
| swapped operator | `build_prompt_for_episode` on an episode of a different operator | Q4: if error does not rise, the prompt is not read |
| zeroed | prompt content set to zero, mask unchanged (the `p_drop_prompt` mechanism) | how much the model depends on any prompt at all |
| self | prompt built from the sample's own episode | upper bound on what a perfect prompt gives |
| shuffled current obs | current frame shuffled across the batch | sanity: the receding path is used |
| batch mean | constant predictor | floor for reading the other numbers |

The dataset never constructs a cross-operator prompt, so the script must
build it by hand by calling `build_prompt_for_episode` on an episode name
from another group. The helper exists; no dataset change is needed. With
45-chunk prompts, run the diagnostic batch in slices of the training
batch size to stay inside the same memory envelope as training.

Run it on `last.ckpt` of every C, C-vid and C-act run. For A and B only the
shuffled-obs and batch-mean rows apply.

### 6.4 Decision rules, fixed in advance

Let d = mean over seeds of (A minus C) unseen-operator paired MSE, and s =
the larger of the two within-arm seed spreads.

- **Prompt helps:** d > 2 s on the unseen population, and the swapped-operator
  row in the diagnostic is worse than the matched row by a similar margin.
  Then Q3 says which part carries it: C-act within noise of C means the
  frames add nothing; C-vid within noise of C means the actions add nothing.
- **Prompt not read:** C within noise of A and B, and swapped equals matched.
  Follow-ups in order: `p_drop_prompt: 0.1` (forces the model to use the
  prompt when it is present, per the original), and the encoder-output
  check from item 6 of the overfit doc (feed the same prompt with and
  without padded chunks).
- **Prompt hurts:** C worse than A by more than 2 s. Then B tells whether
  the extra architecture or the prompt information is the cause.
- Arm B within noise of A means parameter count is not a confound and B can
  be dropped from later sweeps.

## 7. Execution order

Each step names the command shape. Launches go through the hydra submitit
launcher (`python egomimic/trainHydra.py -m ...`), which is how the smoke
and overfit runs were started; check `gpu_usage -l` first and request GPUs
per CLAUDE.md.

1. **Code and configs.** The total-mode tagging fix from section 4 with its
   test; `data/bpp_folding_clothes.yaml` (section 4);
   `model/bpp_prompt_dit_episode_folding.yaml` (inherits
   `bpp_prompt_dit_episode`, sets `prompt_chunk_n_actions: 60` and
   `max_sequence_length: 2700`); `train_zarr_cartesian_bpp_folding.yaml`
   (inherits `train_zarr_cartesian_bpp_episode`, overrides model and data).
2. **CPU checks** on a CPU `salloc`, no GPU: run
   `egomimic/rldb/zarr/test_prompt_dataset.py` and `egomimic/algo/test_bpp.py`
   with chunk 60, then build both folding datasets with a
   `data_check.py`-style script and confirm: the train dataset has 3 groups
   of 12 and 83 918 frames; the valid dataset has 6 groups of 2 episodes,
   the 3 seen ones tagged `operator_seen=1` and the 3 unseen ones tagged
   0; the longest prompt is 45 chunks; the two val episodes of every
   operator prompt each other; cold and warm sample times. The two
   local folding episodes are enough to exercise `read_prompt_chunks` at
   chunk 60 before the S3 episodes are synced.
3. **Smoke every arm** on the full folding config, 4 epochs, val every 2,
   one GPU:

   ```
   python egomimic/trainHydra.py -m --config-name train_zarr_cartesian_bpp_folding \
     model=bpp_dit,bpp_prompt_dit_episode_folding \
     name=bpp_ctx_smoke description=smoke_A_C \
     launch_params.gpus_per_node=1 trainer.max_epochs=4 trainer.min_epochs=4 \
     trainer.check_val_every_n_epoch=2 norm_stats.norm_mode=minmax
   ```

   plus one launch each for B, C-vid and C-act with their overrides from
   section 2. Pass criteria: every arm finishes, the seen and unseen metrics
   are logged for all five, and the peak memory and seconds per step at
   batch 32 with the longest prompt in the data are written into the
   results table. Arm A with prompt-carrying batches is the untested
   combination; if it fails, the fix is in `BPP._build_obs_dict` or
   `process_batch_for_training`, and a unit test in `test_bpp.py` that feeds
   a prompt batch to an unprompted policy should go in with it. If C runs
   out of memory at 32, add prompt-ViT gradient checkpointing and retry
   32; if that still fails, batch 16 with `trainer.accumulate_grad_batches=2`.
   Whatever setting C ends up with applies to every arm.
4. **Norm stats.** Launch arm C seed 42 first; once it has written its
   norm-stat cache, pass that path as `norm_stats.precomputed_norm_path` to
   every other launch.
5. **Main arms.** A, B, C at seeds 42 and 43:

   ```
   python egomimic/trainHydra.py -m --config-name train_zarr_cartesian_bpp_folding \
     model=bpp_dit,bpp_prompt_dit_episode_folding seed=42,43 \
     name=bpp_video_ctx description=main \
     launch_params.gpus_per_node=1 trainer.max_epochs=300 trainer.min_epochs=300 \
     trainer.check_val_every_n_epoch=25 callbacks.model_checkpoint.every_n_epochs=50 \
     norm_stats.norm_mode=minmax norm_stats.precomputed_norm_path=<path>
   ```

   and the same with `model=bpp_prompt_dit_episode_folding
   model.robomimic_model.policy.obs_encoder.ignore_prompt=true` for B.
   Name each run by arm and seed in `description` so wandb groups them.
6. **Ablation arms** C-vid and C-act, seed 42, same command with the
   tokenizer overrides.
7. **Split check** after the first two runs start: diff the episode lists
   in their `trainHydra.log` "Paths:" lines against section 4; abort and
   fix the config if they differ.
8. **Diagnostics** with `bpp_prompt_diag.py` on every finished run's
   `last.ckpt` and on the 15 k-step checkpoint of C, to see whether prompt
   reliance grows with training.
9. **Write up** in this doc: the results table in section 9 and the
   decision from 6.4.

## 8. Code and config changes this plan needs

Small, and listed so they can be done before step 3. None changes model
behavior in arm C.

| Change | Where | Needed by |
|---|---|---|
| Honor `heldout_groups` for `operator_seen` tagging in `total` mode, without moving or dropping episodes | `prompt_dataset.py` + test | seen/unseen metrics in every arm |
| `data/bpp_folding_clothes.yaml` with the pinned hash lists, literal chunk/length values, batch 32 | new file | all arms |
| `model/bpp_prompt_dit_episode_folding.yaml` (chunk 60, length 2700) and `train_zarr_cartesian_bpp_folding.yaml` | new files | B, C, C-vid, C-act |
| Fold `norm_mode: minmax` and a run-spanning scheduler into the BPP train configs, or keep passing them as overrides | `train_zarr_cartesian_bpp_*.yaml`, `model/bpp_*.yaml` | all arms; recommended by the overfit doc |
| Unit test: unprompted policy with a prompt-carrying batch | `egomimic/algo/test_bpp.py` | arm A |
| Optional `prompt.ignore: true` data knob that calls `set_ignore_prompt(True)` so arm A skips prompt IO | `prompt_dataset.py`, data config | arm A speed only; tags and episode lists must stay identical |
| Per-group paired MSE | `eval_hpt.py` | section 6.2 |
| `egomimic/eval/bpp_prompt_diag.py` | new file from `logs/claude_scratch/ckpt_diag.py` | section 6.3 |
| Gradient checkpointing around the prompt-side ViT | `bpp.py` or a wrapper on `TransformerObsEncoder` | only if batch 32 does not fit in the smoke run |

## 9. Results table (to fill in)

Averages of the last three evals; real units for MSE.

| Arm | Seed | s/step | Peak mem | Seen paired MSE | Unseen paired MSE | Seen loss | Unseen loss | Diag: matched | Diag: swapped | Diag: zeroed |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 42 | | | | | | | n/a | n/a | n/a |
| A | 43 | | | | | | | n/a | n/a | n/a |
| B | 42 | | | | | | | n/a | n/a | n/a |
| B | 43 | | | | | | | n/a | n/a | n/a |
| C | 42 | | | | | | | | | |
| C | 43 | | | | | | | | | |
| C-vid | 42 | | | | | | | | | |
| C-act | 42 | | | | | | | | | |

Per-operator paired MSE (section 6.2), one row per arm and seed, one
column per operator (3 seen, 3 unseen), goes below this table.

## 10. Known limits of this comparison

- **Offline metrics only.** Paired MSE on held-out frames rewards matching
  the demonstrator's trajectory; it does not measure whether the garment
  gets folded. A policy that copies the prompt operator's pace scores well
  here even if a different pace would also succeed. This is the right
  metric for the style-conditioning claim, not for a deployment claim.
- **Small data.** Three training operators and 0.78 h of frames. With
  color jitter and 30 k steps the models will partly memorize the 36
  training episodes; the six seen val episodes are the check on that, and
  a seen paired MSE well below the unseen one for every arm is expected.
- **Twelve val episodes** (six seen, six unseen, two per operator) is a
  small population, and the unseen pairs are the shortest episodes of
  their operators rather than typical ones. A positive result should be
  confirmed by re-drawing the val pairs (a different pair per operator)
  and by swapping one train and one val operator.
- **Freeform folding has high per-episode variance.** Different garments
  and strategies within one operator mean the same-operator prompt is a
  noisier signal than on a short repetitive task; the `self` row of the
  diagnostic bounds what any prompt could give.
- **The prompt is the same modality as training data.** Nothing here tests
  a prompt recorded under different conditions than the rollout.
