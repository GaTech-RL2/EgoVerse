# Whole-Demo Prompt Sampling for the BPP Algo

This doc describes replacing the `batch_roll` prompt shortcut in
`egomimic/algo/bpp.py` with dataset-side, whole-demonstration prompt sampling
that matches `prompt_sample_mode: pair` in the original
behavior_prompting sampler
(`external/behavior_prompting/behavior_prompting/train_network/common/sampler.py:502`).
It is the follow-up promised in `docs/2026-09-02_bpp_algo_option.md` ("dataset-side
same-task episode sampling for real use").

**Status (2026-09-02): implemented.** Code lives in
`egomimic/rldb/zarr/prompt_dataset.py` (dataset, reader, deployment helper),
`egomimic/pl_utils/pl_data_utils.py` (`prompt_collate`, weighted sampler),
`egomimic/algo/bpp.py` (`prompt.mode: episode_pair`, seen/unseen-operator
losses, optimizer groups), `egomimic/eval/eval_hpt.py` (seen/unseen metrics),
and the configs `model/bpp_prompt_dit_episode.yaml`,
`data/bpp_folding_clothes.yaml`, `train_zarr_cartesian_bpp_folding.yaml`
(the original cup_on_saucer configs were removed on 2026-09-08). Tests:
`egomimic/rldb/zarr/test_prompt_dataset.py` and the episode-pair block in
`egomimic/algo/test_bpp.py` (29 tests, CPU). Sections 3 to 5 describe the
design; where the implementation deviates it is noted inline as "As built".

## 1. What we have vs. what BPP does

| | Original `pair` mode | Our `batch_roll` |
|---|---|---|
| Prompt source | one whole demo of the same task, drawn from the training split | the neighboring row of the minibatch (`torch.roll`) |
| Prompt frames | one real frame per action chunk, spanning the demo | one frame repeated for every chunk |
| Prompt length | variable, one chunk per second of demo, padded to the longest in the batch + masked | fixed: `action_horizon / chunk_n_actions` = 5 chunks |
| Prompt pool | same task (pairing keyed on task name) | the whole batch |
| Ours, proposed | same task **and same operator**; never across operators | |
| Re-pairing | new prompt per rollout sample each epoch | implicit, via dataloader shuffle |
| Loss | rollout actions only | rollout actions only |

Both use `PromptActionChunker` and the same `{"obs", "action", "metadata.mask"}`
prompt dict, so the policy side does not change. Everything below is data-side
plus a small adapter switch.

## 2. Facts about our pipeline that constrain the design

Established by reading the code; the doc assumes these.

- **Samples are per-timestep.** `MultiDataset.index_map` is a flat list of
  `(episode_name, frame_idx)`; `ZarrDataset.__getitem__`
  (`egomimic/rldb/zarr/zarr_dataset_multi.py:1657`) reads each `key_map` entry
  at `frame_idx` with an optional `horizon`, runs `transform_list`, and
  `MultiDataset.__getitem__` bounds-checks and normalizes. There is no episode
  level read path.
- **Cartesian actions are head-frame relative and resampled.** For `Human`
  cartesian mode (`egomimic/rldb/embodiment/human.py:1044`) a sample's
  `actions_cartesian` is: the next `ACTION_HORIZON=30` raw frames of both
  ee poses (stride from the data config, 1 for Mecka), expressed in the head
  frame at the current frame, interpolated to `chunk_length=100` steps, then
  converted to xyz+ypr and concatenated to 12 dims.
  `observations.state.ee_pose` is the current-frame 12-dim pose in the same head
  frame. A prompt chunk must be built with the same convention or the policy's
  prompt actions and rollout actions live in different spaces.
- **Task identity is not in the zarr.** Local Mecka episodes carry
  `task_name: "debug"` in their zarr attrs; the reliable task label is the
  `task` column of the SQL episode table that `S3EpisodeResolver` filters on.
  `_get_filtered_paths` (`zarr_dataset_multi.py:311`) currently keeps only
  `(zarr_processed_path, episode_hash)`, so the task has to be plumbed through.
- **Prompt resolution and cap.** The original uses one prompt chunk per
  second of demonstration in every task config (UMI: 20 Hz data, chunk 20;
  LIBERO: 20 Hz, chunk 20; drawing: 10 Hz, chunk 10) and sets
  `max_sequence_length` to the longest demo in the dataset (1200, 1000 and 512
  steps). `max_sequence_length` only sizes the learned prompt positional table
  (`prompt_obs_encoder.py:703`); it does not cost memory per sample. Prompts are
  padded to the longest demo in the batch by `collate_prompts`
  (`train_network/utils/prompt_util.py:141`), so compute and memory follow the
  actual demo lengths. Decision for this port: match the 1 Hz convention and
  train on tasks with short demos. Mecka data is 30 fps at stride 1, so
  `prompt_chunk_n_actions: 30` with `prompt_stride: 1` gives 1 Hz chunks with
  no resampling. `cup_on_saucer` needs `max_sequence_length: 450` (15 chunks)
  and `fold_clothes` 750 (25 chunks). The two local folding episodes (54 s and
  95 s) are too long for this budget and are not the target.
- **Operator identity.** The SQL episode table has an `operator` column (an
  id string) and `num_frames`; the zarr attrs carry the same id as `user_id`.
  Pools are keyed on `(task, operator)`; a prompt never crosses operators.
- **What the data supports.** Mecka `human_bimanual` episodes with a processed
  zarr, by task, as of 2026-09-02:

  | task | episodes | operators | mean s | max s | episodes per operator (min / median / max) |
  |---|---|---|---|---|---|
  | cup_on_saucer | 5409 | 22 | 5.6 | 14.8 | 11 / 206 / 1019 |
  | fold_clothes | 1607 | 20 | 6.6 | 24.2 | 1 / 32 / 405 |
  | cleaning_shoes | 1095 | 178 | 88 | 120 | 1 / 2 / 62 |
  | dishwashing | 959 | 502 | 83 | 120 | 1 / 1 / 14 |
  | folding_clothes and the rest | < 720 each | | 80 to 96 | 120 to 130 | mostly 1 |

  Only `cup_on_saucer` and `fold_clothes` work for whole-demo 1 Hz prompting
  with per-operator pools: short demos and many episodes per operator. The
  long tasks have 80 to 120 s demos and one or two episodes per operator, so
  most operators could not even supply a prompt. (`folding_clothes`, the task
  of the two local episodes, is one of the long ones and differs from
  `fold_clothes`.)
- **Memory budget.** Every prompt frame goes through the shared, unfrozen ViT-B
  with gradients, and the original does not checkpoint that path. Their UMI
  prompted config runs batch 70 in bf16 on a 46 GB GPU. Our trainer is bf16 on
  a 48 GB A40, so the working rule is `batch_size * (chunks_per_prompt + 1)`
  of roughly 900 ViT-B images per step. At batch 32 that is demos of up to
  about 25 s; at batch 16, about 50 s. Verify on the first run and drop the
  batch size, not the prompt resolution, if it does not fit.
- **Mask polarity.** `metadata.mask` is `(B, P)` bool, `True` = padded, fed as
  `memory_key_padding_mask`. An all-`True` row produces NaNs, which is why
  `p_drop_prompt` zeroes content instead of masking.
- **Collate.** `annotation_collate` (`egomimic/pl_utils/pl_data_utils.py:96`)
  pops list-valued keys and hands the rest to `default_collate`, which requires
  identical shapes per sample. Prompts vary in length, so the collate needs a
  padding branch like the original's `collate_prompts`.
- **Adapter device handling.** `BPP.process_batch_for_training` only moves
  top-level tensors to the device; a nested `prompt` dict needs its own pass.
- **No epoch hook exists.** `ModelWrapper.on_train_epoch_start` only logs the
  learning rate. If prompts are drawn fresh in `__getitem__` no hook is needed.

## 3. Design

### 3.1 Prompt definition

For a rollout sample from episode `e` at frame `t`, the prompt is a second
episode `e'` from the same group, where a group is `(task, operator)`, covered
end to end by `P = ceil(len(e') / chunk_n_actions)`
chunks of one second each. Chunk `p` starts at raw frame
`s_p = p * chunk_n_actions * prompt_stride` (with `chunk_n_actions: 30`,
`prompt_stride: 1` on 30 fps data) and contains:

| Field | Content | Shape after chunking |
|---|---|---|
| `obs.front_img_1` | front image at `s_p`, eval transform (resize only) | `(P, 1, 3, 224, 224)` |
| `obs.state_ee_pose` | 12-dim pose at `s_p`, in the head frame at `s_p` | `(P, 12)` |
| `action` | the `chunk_n_actions` poses at `s_p, s_p+stride, ...`, in the head frame at `s_p`, xyz+ypr, left+right | `(P, chunk_n_actions, 12)` |
| `metadata.mask` | `True` for padded chunks | `(P,)` |

`P` varies per demo. The last partial chunk is kept with its actions padded by
`pad_end_prompt_actions` (the original UMI config uses `repeat`; ours currently
says `"no"`, switch it to `repeat` so the tail of a demo is not dropped). The
dataset asserts `len(e') <= max_sequence_length` at construction and lists the
offending episodes, so a too-long demo fails loudly instead of being silently
truncated. Set `max_sequence_length` in the model config to the longest demo
of the chosen task in raw frames, as the original does.

`prompt_stride` stays as a knob (raw frames between prompt action steps) for
vendors whose data is not 30 fps, but the 1 Hz rule is
`chunk_n_actions * prompt_stride == fps`. Each chunk's actions are expressed in
the head frame at the chunk's start, matching BPP's `prompt_relative_to: start`.

Prompt actions and proprio are normalized with the same `norm_stats` entries as
the rollout (`actions_cartesian`, `observations.state.ee_pose`), and the
adapter keeps the policy's internal normalizer at identity as it does today.

### 3.2 Where the prompt is built

A `MultiDataset` subclass, so the existing resolver, split, norm-stat and
bounds-check machinery is reused unchanged:

```
egomimic/rldb/zarr/prompt_dataset.py
    class EpisodePromptMultiDataset(MultiDataset)
```

`__getitem__` calls `super().__getitem__(idx)` for the rollout sample, then
draws a prompt episode from the sample's own group and attaches `data["prompt"]`. Drawing happens per call
with a per-worker RNG. This differs from BPP, which fixes pairings once per
epoch and reshuffles; per-call sampling is strictly more random and removes the
need for an epoch hook. Determinism for tests comes from seeding the RNG.

Prompt drawing is disabled while norm stats are inferred and while the
`MultiDataset` is in state-only (deploy) mode, mirroring BPP's
`set_ignore_prompt`.

### 3.3 Group plumbing

1. `S3EpisodeResolver._get_filtered_paths` returns `(path, hash, group)`,
   where `group` is the result of a `group_key` expression evaluated on the SQL
   row. Default: `lambda row: (row['task'], row['operator'])`. `sync_from_filters`
   and `resolve` pass it through and set `ds_obj.group = group` on each
   `ZarrDataset`. `LocalEpisodeResolver` falls back to
   `(metadata['task_name'], metadata['user_id'])` from the zarr attrs and logs a
   warning, since local `task_name` can be a placeholder.
2. `EpisodePromptMultiDataset.__init__` builds
   `self._episodes_by_group: dict[group, list[str]]` from `self.datasets` after
   the train/valid split is applied, so prompts always come from the same split
   as the rollout, as in BPP. A prompt for a sample in group `g` is drawn
   uniformly from `_episodes_by_group[g]` minus the sample's own episode, so
   it never crosses operators and never self-prompts.
3. Groups too small to prompt are dropped at construction and logged: a
   group needs two episodes in every split it appears in (two train and two
   valid in the split modes, two overall in `total` mode or when held out).
   `min_episodes_per_group` is an additional floor.

The `group_key` expression is what makes this general: `lambda row: row['task']`
recovers the original's task-keyed pools if that is ever wanted.

As built: the resolver does not evaluate `group_key`. `S3EpisodeResolver`
keeps the full SQL row per episode (`_get_filtered_rows`) and sets
`ds.metadata_row` on each `ZarrDataset`; `LocalEpisodeResolver` sets it from
the zarr attrs with `task <- task_name` and `operator <- user_id`. The
dataset evaluates `group_key` on `metadata_row`, so the resolver stays
generic.

### 3.4 Reading a prompt from a `ZarrDataset`

Add `ZarrDataset.read_prompt_frames(starts, chunk_n, stride)` that, for chunk
start indices `starts`, reads:

- `images.front_1` at each start (single-frame reads, decoded as today),
- `obs_head_pose` at each start,
- `left.obs_ee_pose` and `right.obs_ee_pose` over
  `slice(s, s + chunk_n * stride, stride)`, padded by repeating the last frame
  when the slice runs off the end (same rule as `_pad_sequences`).

Each chunk is then run through a prompt transform list built by the existing
factory with different arguments, so the frame convention is identical to the
rollout by construction:

```python
_build_human_cartesian_bimanual_transform_list(
    chunk_length=chunk_n_actions,   # 30, not 100
    stride=1,                       # already strided at read time
)
```

`ActionChunkCoordinateFrameTransform` and `PoseCoordinateFrameTransform` accept
any chunk length, `InterpolatePose` with `new_chunk_length == T` is the
identity. Output keys are `actions_cartesian` `(chunk_n, 12)` and
`observations.state.ee_pose` `(12,)` per chunk, which the sampler stacks to
`(P, chunk_n, 12)` and `(P, 12)` and normalizes with
`MultiDataset._apply_norm_one` using the rollout's stats. The stats are
per-dimension vectors of length 12, so they broadcast over the leading
`(P, chunk_n)` axes without reshaping.

The embodiment's keymap and transform list are already on the leaf
(`ds.key_map`, `ds.transform`); the prompt reader derives the raw zarr keys from
`key_map` rather than hardcoding them, so a `_pi` keymap (front key
`base_0_rgb`) works too. Keypoint modes are out of scope for the first version;
the reader raises if the action key is not `actions_cartesian`.

As built: `PromptActionChunker` is not used. `read_prompt_chunks` in
`prompt_dataset.py` reads the whole episode's numeric keys once, decodes one
frame per chunk start, reads each windowed key_map entry at
`s_p + stride * arange(chunk_n)` clipped to the last frame (repeat-padding,
same rule as `_pad_sequences`), and runs the transform list per chunk. The
output is already `(P, chunk_n, D)`, so no chunker and no
`pad_end_prompt_actions` are involved. Frames are resized to `image_size`
and cached as uint8. The sample carries its own `P`; padding and the mask
are produced at collate time.

### 3.5 Batch layout and the adapter

Per sample, `data["prompt"]` is:

```
prompt:
  obs:
    images.front_1:            (P, 3, 224, 224) float32, eval-resized, unnormalized RGB in [0,1]
    observations.state.ee_pose:(P, 12)          normalized
  action:                      (P, chunk_n, 12) normalized
  metadata:
    episode_idx:               int   (index into a table the dataset owns)
```

`annotation_collate` in `pl_data_utils.py` pops the prompt dicts and hands
them to `prompt_collate`, which pads every prompt tensor to the batch's
longest `P` with `torch.nn.utils.rnn.pad_sequence`, builds `metadata.mask`
`(B, P_batch)` with `True` past each sample's length, and keeps
`metadata.length`. The rest goes through `default_collate`, so the sample's
`group_idx` and `prompt_episode_idx` ints become `(B,)` tensors. This is a
port of the original's `collate_prompts` minus its labels branch. Padded
frames are zeros and masked, so they still cost a ViT forward; sorting the
sampler by prompt length is not worth it at these batch sizes.

Adapter changes in `egomimic/algo/bpp.py`:

- `prompt.mode` accepts `episode_pair` alongside `batch_roll`. In
  `episode_pair` mode `_build_obs_dict` takes `_batch["prompt"]`, renames its
  obs keys through `self.obs_key_map` (the same `front_img_1` /
  `state_ee_pose` mapping used for the current obs), unsqueezes the image to
  `(B, P, 1, 3, H, W)` if the encoder expects a camera axis (check against what
  `_batch_roll_prompt` produces today), and applies `p_drop_prompt` content
  dropout exactly as now. `_batch_roll_prompt` stays for the overfit config,
  but note it requires the 100-step action horizon to be divisible by
  `prompt_chunk_n_actions`, which 30 is not. The overfit config keeps chunk 20;
  the episode-pair model config sets 30.
- `process_batch_for_training` recurses into `prompt` to move tensors to the
  device and cast floats; it must not run `zarr_key_to_keyname` on the
  `prompt` key itself.
- Assert at construction that the data-side `chunk_n_actions` and
  `max_sequence_length` match `shape_meta`. Keep one source of truth by
  interpolating the data config from the model config:

  ```yaml
  prompt:
    chunk_n_actions: ${model.robomimic_model.shape_meta.prompt_chunk_n_actions}
    max_sequence_length: ${model.robomimic_model.shape_meta.max_sequence_length}
  ```

### 3.6 Deployment

`BPP.prompt(prompt_dict)` already exists. Add a helper next to the sampler:

```python
build_episode_prompt(episode_path, norm_stats, prompt_cfg) -> dict
```

that runs the same read, transform, normalize, chunk and pad code on one
episode and returns a batch-size-1 prompt dict. Deployment then prompts with
the same function training used, which is the property BPP relies on.

### 3.7 IO cost and caching

Each rollout sample now also decodes `P` JPEGs, one per second of the prompt
demo, instead of 1. Two mitigations, in order:

1. **Fixed chunk grid plus per-worker cache.** Chunk starts are a deterministic
   function of the episode, `chunk_n_actions` and `prompt_stride`, so the
   decoded, resized prompt frames of an episode are a fixed tensor. Cache them
   per worker in an LRU keyed by episode hash, with a configurable byte cap.
   At 224x224 uint8 a 30-chunk prompt is 4.5 MB, so a few hundred episodes per
   worker fit in memory.
2. **Prompt-only image size.** Resize prompt frames to the policy's 224 at
   read time inside the dataset rather than in the adapter, so the cache holds
   small tensors and the collate moves less data.

Do not pre-render prompts to disk in the first version; the cache makes steady
state cost close to one decode per sample.

### 3.8 Operators within a task

The original's prompt tells the policy *which task* to do. Here the task is
fixed and the prompt tells the policy *whose style* to reproduce: every prompt
comes from the same operator as the rollout target, and the policy is judged
on whether it actually conditions on that. Four things follow.

**Group is `(task, operator)`.** Set through `group_key` in 3.3. Multi-task on
top of this is a filter that admits several tasks; groups stay disjoint across
tasks automatically because the task is part of the key.

**Per-group train/valid split.** `split_dataset_names` splits episode names
uniformly, so a small operator can land entirely in one split. Then the valid
loader cannot draw a same-operator prompt, or the train loader has nothing to
prompt with. Mirror `get_val_mask` in the original sampler (`sampler.py:25`),
which applies `valid_ratio` per group and asserts at least one episode per
group on each side. `EpisodePromptMultiDataset` overrides the split
accordingly and drops, with a log line, any group that cannot supply two
train and two valid episodes. On `cup_on_saucer` the smallest operator has
11 episodes, so this holds for every operator; on `fold_clothes` a few
operators have one episode and are dropped.

**Balancing.** Operators are very unbalanced (11 to 1019 episodes on
`cup_on_saucer`). Frame-proportional sampling lets the largest operators
dominate and lets the policy get most of the loss down without reading the
prompt. `balance_by: group` makes `sample_weights()` return
`1 / frames_in_group` per index and the data module pass a
`WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)`
with `shuffle=False`. Lightning wraps custom samplers in its
`DistributedSamplerWrapper` under DDP when `use_distributed_sampler` is on
(the default), so multi-GPU should work; verify the per-rank draws differ
before relying on it. `balance_by: none` keeps frame-proportional sampling;
`balance_by: task` is the original's task-balancing.

**Validation splits and metrics.** Two validation populations matter:
held-out episodes of training operators (seen operator) and every episode of
operators held out of training entirely (unseen operator). As built:
`prompt.heldout_groups` is an explicit list of operator ids written in the
data config per experiment (no automatic selection). Those operators' episodes
all go to the `valid` split; the other operators are split per group as
above. Every sample carries `operator_seen` (1/0) and `group_idx`. Prompts
never cross operators in either population: a held-out operator's samples are
prompted by that operator's other episodes, which is the deployment setting
of "one demo from a new person". `BPP.forward_eval` computes the diffusion
loss on the seen and unseen sub-batches and `eval_hpt.py` logs
`Valid/<emb>_loss_seen_operator`, `Valid/<emb>_loss_unseen_operator`, and the
paired action MSE `Valid/<emb>_<ac_key>_paired_mse_{seen,unseen}_operator`.
Prompts from a different operator are never constructed anywhere.

**Unseen prompts.** With the per-group split, validation prompts each seen
operator's valid rollout with a held-out episode of that operator, matching
the original's evaluation protocol of unseen prompts for known tasks.

## 4. Config

`egomimic/hydra_configs/data/bpp_episode_prompt.yaml` (example for Mecka
folding; copy the train dataset block from `mecka.yaml`):

```yaml
_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper

train_datasets:
  human_bimanual:
    _target_: egomimic.rldb.zarr.prompt_dataset.EpisodePromptMultiDataset._from_resolver
    resolver: {...same as mecka.yaml...}
    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
        - "lambda row: row['lab'] == 'mecka' and row['embodiment'] == 'human_bimanual' and row['task'] == 'cup_on_saucer'"
    mode: train
    prompt:
      chunk_n_actions: ${model.robomimic_model.shape_meta.prompt_chunk_n_actions}  # 30
      max_sequence_length: ${model.robomimic_model.shape_meta.max_sequence_length}
      prompt_stride: 1          # raw frames between prompt action steps; chunk_n * stride == fps
      group_key: "lambda row: (row['task'], row['operator'])"  # prompt pool key; never crosses operators
      min_episodes_per_group: 2
      balance_by: group         # none | group | task; WeightedRandomSampler
      heldout_groups: []        # operator ids held out of training entirely (unseen-operator validation)
      image_size: [224, 224]
      cache_bytes: 2.0e9
      seed: 0

valid_datasets:
  human_bimanual:
    <<same block>>
    mode: valid
```

Model config: copy `bpp_prompt_dit.yaml` to `bpp_prompt_dit_episode.yaml` with

```yaml
prompt:
  mode: episode_pair
  p_drop_prompt: 0.0
shape_meta:
  prompt_chunk_n_actions: 30      # 1 Hz at 30 fps
  max_sequence_length: 450        # longest demo of the chosen task in raw frames: cup_on_saucer 450, fold_clothes 750
  pad_end_prompt_actions: repeat  # keep the partial last chunk, as the original UMI config does
```

`max_sequence_length` is per task and must be raised if a longer task is added;
the dataset assert in 3.1 catches it.

`train_zarr_cartesian_bpp.yaml` keeps `batch_roll` with `bpp_overfit_test` so
the port smoke test is untouched; add
`train_zarr_cartesian_bpp_episode.yaml` that overrides `data` and the prompt
mode.

## 5. Implementation order

Each step is testable on CPU with two local episodes (`salloc` without
`--gres`, per CLAUDE.md).

1. **Group plumbing** in `zarr_dataset_multi.py`: `_get_filtered_paths`
   evaluates `group_key` and returns it, `resolve` sets `ds.group`. Test:
   resolver on the two local hashes yields the SQL `(task, operator)`, not
   `("debug", ...)`.
2. **`ZarrDataset.read_prompt_frames`** and the prompt transform list.
   Test: for `prompt_stride=1` and `chunk_n=100`, the chunk built at frame `t`
   equals the rollout sample's `actions_cartesian` and
   `observations.state.ee_pose` at `t` to float tolerance. This pins the frame
   convention.
3. **`EpisodePromptMultiDataset`**: same-group draw, chunking, padding, mask,
   normalization, cache. Tests: `P` equals `ceil(len / chunk_n)`; the prompt
   episode's group equals the rollout's, so the operator always matches;
   the prompt never is the rollout episode; the last chunk's padded actions follow
   `pad_end_prompt_actions`; norm inference on this dataset produces the same
   stats as on a plain `MultiDataset`; an episode longer than
   `max_sequence_length` raises at construction.
4. **Collate**: a batch of 4 samples with different `P` stacks to
   `(4, max_P, ...)`, `mask` is `True` exactly past each sample's length, and
   the episode index survives.
5. **Adapter** `episode_pair` mode in `bpp.py` plus the device recursion and
   the shape assert. Extend `egomimic/algo/test_bpp.py` with a fixture that
   feeds a synthetic `prompt` dict of the batch layout above and checks
   `forward_training` backprops and `forward_eval` returns `(B, 100, 12)`.
   Keep the existing `batch_roll` tests green.
   Also a memory check on a GPU: one training step at the target batch size
   with prompts padded to `max_sequence_length / chunk_n` chunks, which is the
   worst case a batch can reach.
6. **Deployment helper** `build_episode_prompt`, tested by asserting it returns
   the same tensors the dataset attaches for the same episode.
6b. **Operator groups**: `group_key`, `min_episodes_per_group`, per-group
   split, `sample_weights` plus the sampler branch in `MultiDataModuleWrapper`,
   `group_idx` and `operator_seen` in the batch, seen/unseen val losses.
   Tests: a synthetic table with three operators splits with at least one
   episode per operator per side; an operator with one episode is dropped and
   logged; a listed operator is held out entirely and tagged unseen; sampler
   weights give every operator equal total weight.
7. **Overfit run** with `episode_pair` on one `cup_on_saucer` episode
   (train == valid == one episode, so the prompt is the episode itself). Loss
   should reach the same floor as `batch_roll`. Then a two-episode run from
   the same operator where the prompt is the other episode. The two local
   folding episodes are too long for the 1 Hz budget at batch 32 and are the
   wrong task; sync a few `cup_on_saucer` episodes first.
8. **Operator run** on `cup_on_saucer` with the training operators and a few
   listed in `heldout_groups`. Success criteria: the seen-operator val loss
   falls, and the unseen-operator val loss tracks it.

## 6. Open questions to settle during implementation

- **Prompt resolution.** Decided: 1 Hz chunks like the original, on tasks with
  short demos, padding to the batch's longest prompt. The transformer-side cost
  of a long prompt is small (one positional row per chunk, a longer
  cross-attention memory for two receding tokens, no prompt self-attention).
  The vision-side cost is one unfrozen ViT-B forward with gradients per prompt
  frame; see the memory budget in section 2. If a longer task is needed later,
  the options are, in order: smaller batch; gradient checkpointing around the
  prompt-side ViT (the original only has this for its timm UNet encoder,
  `timm_obs_encoder.py:339`); or `separate_prompt_and_receding_obs_encoders: true`
  with a frozen prompt-side ViT and cached per-episode features, which departs
  from the paper.
- **Head-frame drift within a chunk.** A one-second chunk expressed in the
  head frame at its start matches the rollout's own one-second horizon, so
  drift is no worse than what the policy already predicts over. If prompt
  actions look noisy anyway, `prompt_relative_to: last` semantics (express the
  chunk in the head frame at its end) is a one-line change in the transform
  arguments.
- **Backbone learning rate.** The ViT is unfrozen in both ports and in the
  original. The original gives it `lr * pretrained_lr_scale` (0.1) and zero
  weight decay through `get_optim_groups`; our `ModelWrapper.configure_optimizers`
  passed a single parameter group, so the ViT trained at the full 5e-5. As
  built: `BPP.optimizer_param_groups(lr, weight_decay)` returns the policy's
  own groups (decoder decay/no-decay, backbone at 0.1x with no decay) and
  `ModelWrapper.configure_optimizers` uses them when an algo defines that
  method. Applies to every BPP config, including the batch-roll ones.
- **Operator generalization.** Implemented through `heldout_groups` (3.8):
  whole operators held out, prompted by their own episodes at validation.
- **Outlier rejection.** The BPP training configs set `reject_outliers: false`
  so no frame is substituted; the base `train_zarr_cartesian.yaml` default is
  true.
- **Multi-embodiment.** `BPP` is single-embodiment today. The sampler is
  written per leaf `MultiDataset`, so a second embodiment gets its own prompt
  pool automatically once the adapter supports it.
- **Annotations.** `viz/cartesian.yaml` expects `sampled_prompt` for the video
  overlay. The prompt episode index is enough to overlay "prompted by episode
  X" once the evaluator is taught to read it; not needed for training.
