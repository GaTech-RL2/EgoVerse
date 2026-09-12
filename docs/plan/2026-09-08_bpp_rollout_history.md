# BPP rollout history: own-episode history as prompt chunks

## Context

The BPP policy (`egomimic/algo/bpp.py`, wrapping the vendored `behavior_prompting` diffusion transformer) conditions each action chunk on one current frame plus a whole demo episode of another operator, encoded as (frame, 60-action chunk) tokens that the current-obs tokens cross-attend to. It has no memory of the rollout so far, which matters for the 90 s folding task where a single frame cannot tell which fold step is done.

Decision (with the user): add "own history" as a second prompt-shaped payload. At training sample (episode e, frame t), history = the completed chunks of the same episode before t, on the same chunk grid as the demo prompt (grid anchored at the episode start, so the cached whole-episode chunks are reused; the newest chunk ends 0-59 frames before t), with images, proprio state and actions. History frames go through the eval image transform, like the demo prompt. At rollout the robot buffers what it executed and pushes chunks incrementally.

Settled decisions (2026-09-08): episode-anchored chunk grid; eval image transform for history frames; per-modality history dropout knobs; no warm start from existing checkpoints and no folding runs in this work, verification is the overfit protocol only; implement on a new graphite branch `ryanco/behavior-prompt-history` stacked on `ryanco/behavior-prompt-train`. The user commits; do not run `gt modify`, `gt create` with changes, or `git commit`.

History is variable-length, like an LLM context: H runs from 0 (episode start) up to a cap `max_chunks`, padded and masked within a batch. The cap exists only to bound compute (each chunk is one more ViT image per sample); within it the model sees everything completed so far. At deploy the history grows one chunk per completed chunk until the cap, then slides (keep the last `max_chunks`). Positions are age-indexed (newest chunk = 0) with a sinusoidal embedding, so the cap can be changed at eval or deploy without retraining. The user wants to run caps of 5 and 15 chunks (10 s and 30 s on the folding config); one model trained at 15 can also be evaluated at 0, 5 and 15 via an eval-time slice. Robot rollout wiring in `rollout.py` is out of scope; this delivers the buffer/push API only.

Constraints found during exploration:
- `external/behavior_prompting` is a clean upstream submodule (real-stanford), not a fork. Do not modify it. The network change is a subclass in egomimic selected by hydra `_target_`.
- The vendored `Normalizer._normalize_impl` indexes `params_dict[key]` for every top-level obs key, and `normalize_obs_with_optional_prompt` (`train_network/utils/prompt_util.py:235`) passes `prompt['metadata']` through untouched. So the history payload must travel as `obs_dict["prompt"]["metadata"]["history"]`, not as a top-level key.
- `PairPromptObsEncoder.get_max_token_count` returns `num_current_obs_tokens + num_prompt_current_obs_tokens` in decoder mode, so the diffusion head's conditioning size is independent of memory length. Only the prompt decoder's memory grows.
- `prompt_encoder_enabled: False` in all our configs, so per-chunk prompt tokens are independent and can be cached incrementally at deploy time.
- `MultiDataset.index_map[idx] == (episode_name, local_idx)` and `local_idx` is the raw frame index within the episode (`zarr_dataset_multi.py:872-877`, `ZarrDataset.__getitem__` at 1755). The own episode is normally already in the dataset's LRU (`_raw_prompt`), so history is a slice of it with no extra IO.
- Substitution: on a bounds/NaN failure `MultiDataset.__getitem__` (line 999) serves a random other frame of the same episode and only exports a `substituted` bool; the leaf's decode-failure retry (line 1773, flag set at 1869) does the same. Neither returns the served frame index, so history must not be computed from the requested index. Decision (with the user): the served-frame refactor below, so substituted samples get an exact history.
- `pad_sequence` on all-zero-length tensors returns `(B, 0, ...)` (verified in the venv), so a batch with no history collates cleanly.

## Payload layout

Batch (dataset -> `annotation_collate` -> `process_batch_for_training`), same schema as `prompt`:
```
batch[emb]["history"] = {
  "obs": {"observations.images.front_img_1": (B, Hmax, 3, 224, 224) float [0,1],
          "observations.state.ee_pose":      (B, Hmax, 18) normalized},
  "action":   (B, Hmax, chunk_n, 18) normalized,
  "metadata": {"mask": (B, Hmax) bool (True = pad), "length": (B,) long},
}
```
Policy obs_dict built by `BPP._build_obs_dict` (shape_meta names, eval image transform):
```
obs_dict["prompt"]["metadata"]["history"] = {
  "obs": {"front_img_1": (B, H, 3, 224, 224), "state_ee_pose": (B, H, 18)},
  "action": (B, H, chunk_n, 18), "metadata": {"mask": (B, H)}}
```
Hmax / H may be 0. Memory for the prompt decoder = `cat([sink?, demo (B,P,D), history (B,H,D)], 1)` with the concatenated key-padding mask. The demo always has at least one valid chunk, so no row is ever fully masked even when history is empty or dropped.

## Steps

### 0. Served-frame index: `egomimic/rldb/zarr/zarr_dataset_multi.py`
- `ZarrDataset.__getitem__` (line 1755): next to `data["substituted"] = idx != origin` (line 1869) write `data["frame_idx"] = int(idx)`, the frame actually served after any decode-failure retry. Collates to a `(B,)` long tensor; same category as `substituted` / `episode_hash`, which already pass through every algo's batch untouched.
- `MultiDataset.__getitem__` (line 999): move the body into `_getitem_with_index(self, idx, _attempts=None) -> tuple[dict, int]` returning `(data, served_global_idx)`; `__getitem__` becomes `return self._getitem_with_index(idx, _attempts)[0]`. For a nested `MultiDataset` leaf the returned index is the parent-level served `idx` (the leaf's own index is leaf-local and not an index into the parent's `index_map`); the served frame always comes from the ZarrDataset's `frame_idx`. Behaviour of `__getitem__` is unchanged.

### 1. Dataset: `egomimic/rldb/zarr/prompt_dataset.py`
- Refactor `read_prompt_chunks` (line 53) into two helpers without changing behaviour: `_read_raw_chunk(leaf, s, full, chunk_n_actions, prompt_stride)` (loop body lines 110-123) and `assemble_chunk(raw, transform_list, camera_names, state_key, action_key, image_size, action_steps)` (lines 124-140). `read_prompt_chunks` calls both. `test_prompt_chunk_matches_rollout_frame_convention` pins this.
- New public `build_history_chunk(raw_frames, *, key_map, image_keys, chunk_n_actions, prompt_stride, transform_list, image_size, action_key, state_key, action_steps=None)` for rollout: takes raw per-key arrays for one chunk (image and scalar keys at chunk start, windowed keys over the chunk) and returns `{"obs": {k: (1, ...)}, "action": (1, chunk_n, D), "length": 1}` unnormalized, via `assemble_chunk`.
- Config in `EpisodePromptMultiDataset.__init__` (after line 305): `history: {max_chunks: int (0 = off), gap_frames: int}` -> `self.history_max_chunks`, `self.history_gap_frames`, `self.use_history`. The `state is not None` early-return branch (line 249) must set `self.use_history = False`; the `__new__` helper in `build_episode_prompt` (line 648) does not touch history. Document in the class docstring.
- New `_history_for(episode_name, t, embodiment_id)`: `own = self._raw_prompt(episode_name)`; `window = chunk_n_actions * prompt_stride`; `n_done = min(max(0, (t - gap) // window), own["length"])`; `start = max(0, n_done - max_chunks)` (a negative start would wrap in Python and silently return an empty slice for the first `max_chunks` chunks of every episode); slice chunks `[start:n_done]` of obs/action, set `length = n_done - start`, then `_normalize_prompt`.
- `__getitem__` (line 590): call `data, served = self._getitem_with_index(idx, _attempts=_attempts)` instead of `super().__getitem__`; `dataset_name, _ = self.index_map[served]` (same episode as the requested index, since substitution never leaves the episode); `t = int(data["frame_idx"])` (falls back to `self.index_map[served][1]` if the leaf did not set it). After the prompt is attached, if `self.use_history`: `data["history"] = self._history_for(dataset_name, t, emb)`. Same guards as the prompt (`_ignore_prompt`, norm stats present). No empty-history special case for substituted rows.
- Fix the `build_episode_prompt` docstring (line 632) to name `BPP.episode_prompt_to_policy` (added in step 4).

### 2. Collate: `egomimic/pl_utils/pl_data_utils.py`
- `annotation_collate` (line 291): pop and pad `history` exactly like `prompt` using `prompt_collate` (all-or-none check). Note in the docstring that `P_max` may be 0.

### 3. Encoder subclass: `egomimic/algo/bpp_history_encoder.py` (new)
`class HistoryPairPromptObsEncoder(PairPromptObsEncoder)`, `__init__(*, history_max_chunks: int = 15, **kwargs)`.
- Asserts: decoder enabled, `not prompt_encoder_enabled`, `obs_encoder.num_output_modalities_prompt == 1`, `not only_use_last_step_of_prompt`, `cut_first_n_steps_of_prompt == 0`, `history_max_chunks >= 1`.
- Params: `history_pos_proj = nn.Linear(D, D)` applied to a fixed sinusoidal embedding of the chunk age (0 = most recent chunk; computed on the fly for any H, so the cap is not baked into the weights and the newest chunk has the same position at train and deploy regardless of H); `segment_emb (2, 1, 1, D)` zero-init (row 0 demo, row 1 history), so at init the streams are told apart only by their positional tables and the model learns the separation. `history_max_chunks` is stored only as the cap used by the incremental cache and the input check.
- `_encode_history_raw(history) -> (raw (B,H,D), mask (B,H))`: empty tensors when H == 0; else `self.obs_encoder({"prompt": history})` (tokenizer prompt-only branch, `prompt_obs_encoder.py:296-483`), `drop_tokens`, zeros mask if the tokenizer returns None, assert `H <= history_max_chunks`.
- `_embed_history(raw, mask)`: `n_valid = (~mask).sum(1)`; `age = (n_valid[:, None] - 1 - arange(H)[None]).clamp(min=0)`; `raw + history_pos_proj(sinusoid(age)) + segment_emb[1]`. Padded chunks get an arbitrary, masked position.
- `_prepare_demo_tokens(tokens)`: base closure at lines 746-754 plus `segment_emb[0]`.
- `forward(obs_dict, need_weights=False, average_attn_weights=False, ...)`: must handle `history` absent from the metadata (treat as H == 0) and a demo `prompt_mask` of `None` (the tokenizer returns None when the payload has no `mask`, e.g. B == 1 deploy prompts and the base `output_shape()` probe at `prompt_obs_encoder.py:897-931`): materialize `zeros(prompt_B, P, bool)` before concatenating with the history mask. Reimplements the decoder branch of the base (lines 729-836) with three cases: prompt+receding (train/eval: history taken from `prompt.metadata.history`), prompt only (used by `prompt()`, returns demo tokens and mask), receding only (deploy: demo from `prompt_tokens_cache`, history from `history_raw_cache`). Pop `history` from the metadata before handing the dict to the tokenizer. Build memory and mask by concatenation, then sink prepend, batch expand, `prompt_with_obs_decoder(...)`, `ignore_prompt`, concat with receding tokens, `flatten_output`, metadata exactly as the base. When H == 0 for the whole call, add `0 * (sum of history_pos_proj params + segment_emb[1].sum())` so the new params always receive a grad (the base uses the same trick at line 815; the grad tests require it).
- `prompt(prompt_dict)`: read `history = prompt_dict["metadata"].get("history")` FIRST, then `super().prompt(prompt_dict)` (which calls this class's `forward`), then `clear_history()` and seed `history_raw_cache` from `history` if H > 0. Order matters: `normalize_obs_with_optional_prompt` reuses the same metadata dict object, so a forward that pops `history` in place would empty it before the seeding line reads it. In `forward`, never mutate the caller's metadata: `metadata = dict(prompt["metadata"]); history = metadata.pop("history", None)` and hand the tokenizer a shallow copy of the prompt with that metadata (the tokenizer only reads `metadata["mask"]`, so leaving `history` in place is also harmless).
- `push_history_chunk(chunk)` under `torch.inference_mode`: assert prompted; encode one B=1, H=1 chunk with `_encode_history_raw`; append; trim to the last `history_max_chunks`; return the new length. `clear_history()`, `history_len` property, `reset()` = super + clear.
- `get_optim_groups(lr, weight_decay)`: super + `{"params": [segment_emb, history_pos_proj.bias], "weight_decay": 0.0, "lr": lr}` and `history_pos_proj.weight` with `weight_decay`.
- `get_max_token_count`, `output_shape`, token names inherited unchanged.

### 4. Adapter: `egomimic/algo/bpp.py`
- `__init__` (after line 111): parse `prompt.history` -> `use_history`, `p_drop_history` (default 0.2, whole-history dropout), `p_drop_history_state` and `p_drop_history_action` (default 0.0, per-modality: zero that modality's history content for the row, mask unchanged), `action_noise_std` (default 0.0), `history_only` (default false; masks every demo prompt token so the policy sees current obs + history only, used by the overfit protocol). `history_only` requires the encoder's attention sinks (`policy.obs_encoder.attention_sink_enabled` is the attribute name; the yaml key is `use_attention_sink`), otherwise a row with H == 0 would have an all-masked memory and NaN the cross-attention softmax; assert this at init. When enabled, require `prompt_mode == "episode_pair"` and that `policy.obs_encoder` has `history_max_chunks`; store it as `self.history_max_chunks`. Raise when the encoder is the history subclass but `enabled` is false.
- `process_batch_for_training` (line 251): pass `history` through when `use_history`, drop it otherwise.
- Factor `_episode_prompt` (515-552) into `_adapt_chunks(payload, training, *, max_len, name)` (rename keys, eval image transform with an H == 0 guard, chunk-size check, `max_len` check, bool mask). `_episode_prompt` = `_adapt_chunks(..., max_len=self.max_prompt_len, name="prompt")` + `_drop_prompt_content`.
- New `_episode_history(history, training)`: `_adapt_chunks(..., max_len=self.history_max_chunks, name="history")`; in training add gaussian noise to actions if `action_noise_std > 0`, apply the per-modality drops (`p_drop_history_state` zeroes `state_ee_pose`, `p_drop_history_action` zeroes `action`, independently per row), then `_drop_history` (zero all content and set the whole mask row True for dropped samples, safe because demo tokens remain). In eval, if `eval_history_chunks` is set (int, default None), keep only the last `eval_history_chunks` valid chunks per row (re-slice by `length`, then re-pad and re-mask) so one checkpoint can be scored at 0, 5, 15 and full history without retraining.
- `_build_obs_dict` (505-512): when `use_history`, require `"history" in _batch` and set `obs["prompt"]["metadata"]["history"] = self._episode_history(_batch["history"], training)`. When `history_only`, replace the demo prompt by a single fully-masked zero chunk (P = 1; implemented as `_mask_demo_prompt`, never in place). P = 1 keeps every shape and the prompt-cache path valid and skips the ViT work for the real demo frames, so stage A costs 1 + 1 + H images per sample instead of 1 + 30 + H. Update the layout docstring.
- Deployment API: `episode_prompt_to_policy(prompt)` (= `_episode_prompt(prompt, training=False)`, the method the dataset docstring already names), `history_chunk_to_policy(chunk)`, `push_history_chunk(chunk) -> int` (adapt, move to device, forward to the encoder), `clear_history()`, `history_len`, and `predict_action_deployed(_batch, embodiment_id)` which builds only the receding obs (a `deployed=True` flag on `_build_obs_dict` skips prompt/history attachment, since a prompted encoder asserts `'prompt' not in obs_dict` at line 730), calls `policy.predict_action`, and unnormalizes like `forward_eval`.

### 5. Configs (new files; existing ablation arms untouched)
- `egomimic/hydra_configs/model/bpp_prompt_dit_episode_history.yaml`: defaults `bpp_prompt_dit_episode_folding`; `robomimic_model.prompt.history: {enabled: true, max_chunks: 15, p_drop_history: 0.2, p_drop_history_state: 0.0, p_drop_history_action: 0.0, action_noise_std: 0.0, history_only: false, eval_history_chunks: null}`; `policy.obs_encoder._target_: egomimic.algo.bpp_history_encoder.HistoryPairPromptObsEncoder` with `history_max_chunks: ${model.robomimic_model.prompt.history.max_chunks}`.
- `egomimic/hydra_configs/data/bpp_folding_clothes_history.yaml`: defaults `bpp_folding_clothes`; add `history: {max_chunks: ${model.robomimic_model.prompt.history.max_chunks}, gap_frames: 0}` under both the train and valid `prompt` blocks (both are literals, so both need it). Interpolation is safe here because this data yaml is only composed with the history model.
- `egomimic/hydra_configs/train_zarr_cartesian_bpp_folding_history.yaml`: defaults `train_zarr_cartesian_bpp_folding` with model/data overridden to the two files above.
- Overfit configs for the verification protocol below:
  - `egomimic/hydra_configs/data/bpp_overfit_history.yaml`: `EpisodePromptMultiDataset._from_resolver` (copy the folding data yaml's resolver / prompt / prompt_transform_list blocks), filtered to two `folding_clothes` episodes of one training operator (pick two hashes from the folding train list whose `operator` matches via `episode_hash_to_table_row`), `mode: total`, valid = the same two episodes, `history: {max_chunks: 15, gap_frames: 0}`, `balance_by: none`. The two episodes prompt each other, which satisfies the dataset's two-per-group rule.
  - `egomimic/hydra_configs/train_zarr_cartesian_bpp_history_overfit.yaml`: defaults `train_zarr_cartesian_bpp_folding_history`, `override data: bpp_overfit_history`, `reject_outliers: false`, `evaluator.viz_func.human_bimanual.annotation_key: null`, `model.robomimic_model.prompt.history.p_drop_history: 0.0`, `...policy.obs_encoder.use_attention_sink: true`. Stage A adds `model.robomimic_model.prompt.history.history_only=true`.
- The 5 vs 15 arms are then one override: `model.robomimic_model.prompt.history.max_chunks=5`. A single 15-chunk checkpoint can additionally be evaluated at `eval_history_chunks=0|5|15` to see how much history it uses.

### 6. Tests
`egomimic/algo/test_bpp.py` (reuse `_load_small_cfg`, `_build_episode_batch` at line 255, `prompt_collate`):
- `_load_small_cfg` (`test_bpp.py:63-75`) uses `OmegaConf.load`, which does not merge a hydra `defaults:` list; the new history model yaml has one. Add `_load_yaml_with_defaults(path)` to the test: read the yaml, for each entry of `defaults` that is a bare name (skip `_self_`), recursively load `<same dir>/<name>.yaml` and `OmegaConf.merge` in order, then merge the file's own keys on top; drop the `defaults` key. `_load_small_cfg` calls it. Existing fixtures are unaffected (their yamls have no defaults).
- fixture `algo_history` from the new model yaml with `prompt_chunk_n_actions=30`, `max_sequence_length=450`, `grad_checkpointing=false`; helper `_build_history_batch(lengths, hist_lengths)`.
- `test_history_config_and_encoder`: flags, subclass, `get_max_token_count()` equals the episode model's.
- `test_history_obs_dict_structure`: nested payload shapes, mask sums; `H > max_chunks` and wrong chunk size raise.
- `test_history_training_path_and_grads` (mixed lengths incl. 0) and `test_zero_history_batch_trains_without_nan` (all 0): finite loss, every trainable param has a grad.
- `test_history_dropout_masks_and_zeroes`: `p_drop_history=1.0` zeroes content and masks all history, loss finite; `p_drop_history_state=1.0` zeroes only the state entry and `p_drop_history_action=1.0` only the actions, mask unchanged; eval leaves content untouched; noise only in training.
- `test_eval_history_slice`: `eval_history_chunks=1` on a batch with lengths (0,2,3) keeps exactly the newest valid chunk per row; training is untouched. `test_variable_lengths_share_positions`: the newest chunk's embedded token is identical whether H is 1 or `max_chunks` (age indexing).
- `test_history_deploy_cycle`: `prompt()` with 1 history chunk -> `history_len == 1`; two `push_history_chunk` -> 3; `predict_action` finite `(1, T, D)`; `reset()` clears both; push before prompt raises.
- `test_incremental_cache_matches_batch_encoding`: batch encoding of H=3 vs. three pushes, allclose; pushing `max_chunks + 1` keeps the last `max_chunks`.
- `test_history_seen_unseen_losses`: exercises `_select_rows` on the nested history.
- Extend `test_optimizer_param_groups_cover_all_and_scale_backbone` to the history model; new params in a weight-decay-0 group.
- `test_history_disabled_drops_history_batch`: episode model drops `history`; history model raises without it.
- `test_history_only_masks_demo`: `history_only=true` with sink enabled -> demo mask all True and demo content zero in the obs_dict, loss finite on a batch that includes an H == 0 row; `history_only=true` without sink raises at init.

`egomimic/rldb/zarr/test_prompt_dataset.py` (reuse `episode_root`, `_dataset(**over)`):
- `test_history_equals_own_episode_prompt_chunks`: for sampled idx, with `n_done = t // CHUNK` and `start = max(0, n_done - max_chunks)`, history equals `build_prompt_for_episode(name)` chunks `[start:n_done]` exactly, including frames inside the first `max_chunks` chunks of the episode.
- `test_history_gap_frames`, `test_history_zero_at_episode_start`, `test_history_collate_zero_lengths` (mask rows and `(B, 0, ...)` shapes), `test_history_absent_without_config_or_when_ignored`.
- `test_frame_idx_matches_index_map`: for sampled idx, `sample["frame_idx"] == ds.index_map[idx][1]` when not substituted.
- `test_substituted_sample_history_uses_served_frame`: monkeypatch `MultiDataset._check_bounds` to reject the requested index once; assert `substituted` is True, `frame_idx != requested t`, and `history` equals `_history_for(name, frame_idx)` (not the requested frame's history).
- `test_build_history_chunk_matches_read_prompt_chunks`: raw window read through `leaf.episode_reader` -> `build_history_chunk` equals `read_prompt_chunks(chunk_starts=[s])`.

### 7. Rollout (design only, deferred)
`egomimic/robot/rollout.py` is Eva-only, unprompted, and calls `forward_eval`. Future wiring: a `HistoryBuffer` that stores raw per-frame obs and executed actions, assembles a chunk every `chunk_n_actions * prompt_stride` frames with `build_history_chunk`, normalizes with the `build_episode_prompt` pattern, and calls `BPP.push_history_chunk`; inference switches to `predict_action_deployed`; `reset()` at episode end. The human-data model vs. Eva key mapping is a separate problem.

## Verification
Branch first, before any edit: `gt create ryanco/behavior-prompt-history` on a clean tree (no staged changes, so it creates an empty branch stacked on `ryanco/behavior-prompt-train`). No commits, no `gt modify`; the user commits.

Implemented alongside (2026-09-08): `trainHydra.py` now passes `model.enable_grad_norm` (default true) to `ModelWrapper`, so the overfit yaml can turn the MAD gradient clip off; the overfit yaml also sets `train_image_augs` to resize only (both per docs/2026-09-02_bpp_episode_overfit_debugging.md). Launch scripts: `logs/claude_scratch/history/launch_overfit.sh <A|B> <desc>` (SMOKE=1 for a pipeline check), `launch_eval_history.sh <A|B> <ckpt> <desc> <k>` for the eval-time history slice, `status.sh` for job status.

Note: there is no weights-only training resume in `trainHydra.py` (commit e17cf98fe only added the `weights_only` flag of `torch.load` and a memory-mapped reader; `trainer.fit(ckpt_path=...)` is a strict full resume). Not needed here since every run trains from scratch.
```
source /coc/flash7/rco3/EgoVerse/emimic/bin/activate
salloc -p rl2-lab -A rl2-lab -c 12 --mem=30G      # CPU only, no --gres
pytest egomimic/rldb/zarr/test_prompt_dataset.py -x -q
pytest egomimic/algo/test_bpp.py -x -q             # needs the HF cache for vit_tiny
ruff check egomimic/algo egomimic/rldb/zarr/prompt_dataset.py egomimic/pl_utils/pl_data_utils.py
python egomimic/trainHydra.py --config-name train_zarr_cartesian_bpp_folding_history --cfg job --resolve | head -80   # --resolve exercises the model->data interpolation
```
Overfit protocol (GPU; check `gpu_usage -l`, then `salloc ... --gres=gpu:a40:1` on a fresh node). Two folding episodes of one operator, train = valid, so a working model drives the loss toward zero:
```
# Stage A: history only (demo prompt masked). Proves the history path carries signal on its own.
python egomimic/trainHydra.py -m --config-name train_zarr_cartesian_bpp_history_overfit \
  name=bpp_hist_overfit description=A_history_only \
  model.robomimic_model.prompt.history.history_only=true
# Stage B: demo prompt + history, same data and model.
python egomimic/trainHydra.py -m --config-name train_zarr_cartesian_bpp_history_overfit \
  name=bpp_hist_overfit description=B_prompt_and_history
```
Smoke (2026-09-08 16:23, 6 steps, batch 32, one A40 each, both COMPLETED): stage A 2.6 s/step, 10.4 GiB peak; stage B 4.7 s/step, 18.0 GiB peak; first val loss 1.07 for both, one video per episode written. Full runs launched 16:31: A job 3788391 `logs/bpp_hist_overfit/A_history_only_2026-09-08_16-31-24`, B job 3788392 `logs/bpp_hist_overfit/B_prompt_and_history_2026-09-08_16-31-29` (100 epochs x 100 steps, val every 10 epochs, ckpt every 25; progress via `python logs/claude_scratch/history/progress.py`).

Stage A result (job 3788391, COMPLETED 22:27 after 5 h 56 min, 10k steps): val loss 0.0845 -> 0.0237, sampled-action paired MSE 0.133 -> 0.027 (final-step MSE 0.032), train loss 0.034. Below the batch-roll overfit reference (0.036 after 13k steps) and the folding runs' best paired MSE (0.047). Sampled-action MSE only started falling after step 4000 while the loss fell from the start, as the overfit note predicts. Scoring evals of `last.ckpt` at `eval_history_chunks` 0 / 5 / 15: jobs 3794201 / 3794202 / 3794203 (`logs/bpp_hist_overfit/evalA_k{0,5,15}_2026-09-08_22-30-*`).

Stage A scoring (22:30, all three COMPLETED in ~5 min): val loss with `eval_history_chunks` 0 / 5 / 15 = 0.0249 / 0.0247 / 0.0243. Inconclusive by design: in a two-episode overfit the current frame alone determines the target, so history is redundant. Direct diagnostic on `last.ckpt` (`logs/claude_scratch/history/attn_diag.py`, 16 frames from the second halves of both episodes, history 10-15 chunks): prompt-decoder cross-attention mass 0.81 on the sinks, 0.00 on the masked demo chunk, 0.19 on history tokens (mass decreasing with age, so the path is live and read); loss on identical noise draws with real / no / zeroed / other-episode history = 0.0273 / 0.0280 / 0.0276 / 0.0280, and 0.0513 vs 0.0525 with the current frame zeroed. Conclusion: the implementation trains, overfits and attends to history; whether history carries task signal is not testable on a two-episode overfit and needs the folding runs (or a variant of stage A with the current image withheld from the policy: `model.robomimic_model.shape_meta.obs.front_img_1.include_in_receding_obs=false ...include_in_prompt_current_obs=false`, so only history images carry the garment state).

Stage B mid-training diagnostic (epoch-49 checkpoint, step 5000, same script): memory = 4 sinks + 30 demo + 15 history columns; cross-attention is near-uniform (about 0.02 per column: sinks 0.12, demo 0.60, history 0.27) and the loss is 0.0361 under every history perturbation (real / none / zeroed / other-episode). Same reading as stage A: on a two-episode overfit the current frame suffices and the memory is treated as background.

Stage A2 (history only + current image withheld from the policy: `shape_meta.obs.front_img_1.include_in_receding_obs=false include_in_prompt_current_obs=false`, so the garment state is visible only through history frames; job 3794317, `logs/bpp_hist_overfit/A2_history_only_noimg_2026-09-08_22-48-48`, COMPLETED 2026-09-09 04:33 after 5 h 45 min, 1.24 s/step, 11.8 GiB): val loss 0.0235, paired MSE 0.029 at 10k steps, same as stage A. Diagnostic on `last.ckpt`: attention mass on history 0.41 (newest chunk 0.11, oldest 0.06), sinks 0.59; loss with real / no / zeroed / other-episode history = 0.0269 / 0.0343 / 0.0327 / 0.0356; history images zeroed 0.0324, history actions zeroed 0.0269; current frame zeroed 0.0269 (confirms the image is out of the policy path). So the history path carries task signal end to end: removing it costs +28 %, the wrong episode's history +32 %, and the signal is in the history frames. This is the "history carries signal" check the two-episode overfit could not give with the current frame present.

Stage B result (job 3788392, COMPLETED 2026-09-09 04:55 after 12 h 24 min, 10k steps at 2.58 s/step, 19.6 GiB): val loss 0.0836 -> 0.0236, paired MSE 0.137 -> 0.0265 (final-step MSE 0.031), train loss 0.034; matches stage A at every validation point. Videos in `0/videos/epoch_*/HUMAN_BIMANUAL/`.

Stage B final diagnostic (`last.ckpt`): attention mass demo 0.75 / history 0.11 / sinks 0.14; loss 0.0270 with real, zeroed or other-episode history and 0.0274 with none. With the current frame present, neither memory stream is needed on two episodes; the demo gets most of the attention.

Verdict (2026-09-09): both requested overfit tests pass (A: history only; B: demo prompt + history), and stage A2 shows the history path carries task signal when the current frame cannot (+23 % val loss without history on the whole set, +28 % on the diagnostic batch, +32 % with the wrong episode's history).

Pass criteria: train and val diffusion loss fall toward the arm-C overfit level within the same number of steps, and the val videos reproduce the episodes' action chunks. For stage A additionally score the checkpoint with `eval_history_chunks=0` and `=15`: the loss at 0 must be clearly higher than at 15, which shows the policy reads the history rather than only the current frame. If stage A does not overfit, the bug is in the history path (dataset slice, adapter, or encoder); if A passes and B does not, it is in the demo/history concatenation or masking. Both stages train from scratch; record peak GPU memory at batch 32 during stage B, since it is the same per-sample image count the folding runs will see.

Folding training (the 5 vs 15 arms via `train_zarr_cartesian_bpp_folding_history` and `max_chunks=5`) is out of scope for this work and is the user's call after the overfit tests pass.

## Risks
- Memory: up to `max_chunks` extra ViT images per sample on top of 46 (about a third more ViT work per step at 15; the ViT is already checkpointed). Batches pad to the longest history present, so most batches pay close to the cap. Stage B's peak memory tells whether folding needs `batch_size=16 trainer.accumulate_grad_batches=2`. If it still does not fit, encode history images under `no_grad` (activations not stored; the ViT keeps training from current obs and demo prompt).
- Host memory: each history chunk is a float32 224x224 frame in the DataLoader (about 9 MB per sample at 15 chunks, about 290 MB per batch of 32, times 8 workers and prefetch) on top of the demo prompt's roughly 27 MB per sample. Not a correctness issue; lower `num_workers` or `prefetch_factor` if the node runs short.
- Variable length within the cap: H is roughly uniform over 0..cap across random frames (more mass at the cap for long episodes), so the model trains on every length it will see at deploy, including 0.
- Train/test mismatch: training history is the expert's own trajectory, which predicts the next chunk well. `p_drop_history=0.2` and `gap_frames` are the mitigations; offline val uses ground-truth history and will overstate the benefit.
- The served-frame refactor adds a `frame_idx` key to every algo's batch and splits `MultiDataset.__getitem__`. Zero training cost; run the existing dataset tests (`egomimic/rldb/zarr/test_*.py`) to confirm nothing keys on the exact set of batch keys.
- History images of dropped rows still pass through the ViT (wasted compute, no correctness issue).
- `segment_emb[0]` is added to demo tokens too (zero-init, so the demo path is unchanged at init).
- The plan copy in `docs/plan/2026-09-08_bpp_rollout_history.md` must be synced from this file as the first action after approval (plan mode only allows editing this file).
