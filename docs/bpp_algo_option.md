# Adding the Behavior-Prompting Diffusion Transformer as an EgoMimic Algo

This doc describes exposing the trained BPP policy itself —
`DiffusionTransformerPolicy` + `PairPromptObsEncoder` from
`external/behavior_prompting` — as a first-class algo option in
`egomimic/algo`, selectable from a hydra model config like `act.py`, `hpt.py`,
and `pi.py` are today.

## 1. What the Algo contract requires

Everything in `egomimic/algo` implements the small interface in
`egomimic/algo/algo.py`, driven by the Lightning `ModelWrapper`
(`egomimic/pl_utils/pl_model.py`):

| Method | Role |
|---|---|
| `__init__(norm_stats, camera_transforms, train/eval_image_augs, ...)` | build `self.nets = nn.ModuleDict({"policy": ...})` |
| `process_batch_for_training(batch)` | zarr-key → model-key translation, device/dtype, pad masks |
| `forward_training(batch)` | forward + loss into a `predictions` dict |
| `forward_eval(batch)` | predictions, unnormalized via `self.norm_stats` |
| `compute_losses(predictions, batch)` | scalar loss dict (`action_loss` is the optimized key) |
| `log_info(info)` | flatten losses for the logger |

Algos are instantiated by hydra (`model/*.yaml` sets
`robomimic_model._target_`), and registered by import in
`egomimic/algo/__init__.py`. So "adding an option" = one new algo file, one new
model config, one `__init__.py` line.

## 2. Two implementation strategies

### Option A — adapter around the vendored package (recommended)

`external/behavior_prompting` is already a submodule of this repo and an
installable package (`setup.py`). Rather than porting ~125M params of
architecture code, write a thin adapter:

```
egomimic/algo/bpp.py
    class BPP(Algo):
        # wraps behavior_prompting.train_network.policy.
        #   diffusion_transformer_policy.DiffusionTransformerPolicy
```

The whole BPP model stack (`TransformerForActionDiffusion`,
`PairPromptObsEncoder`, `PairPromptTransformerTokenizer`,
`TransformerObsEncoder`) is hydra-instantiable already — BPP's own configs
(`train_network/config/model/prompt_diffusion_transformer.yaml`,
`obs_encoder/pair_prompt_encoder.yaml`) are hydra files. We can translate them
nearly verbatim into `egomimic/hydra_configs/model/bpp_prompt_dit.yaml` and let
our hydra tree instantiate the BPP policy directly as a nested `_target_`.

Pros: exact parity with the trained/published model, their checkpoints load
directly, upstream fixes flow in via the submodule. Cons: pulls BPP's
`shape_meta` convention and `Normalizer` into our config surface (handled
below), and adds `external/behavior_prompting` as a hard runtime dependency of
one algo.

### Option B — native re-implementation

Port the three model files into `egomimic/models/bpp_nets.py` and own them.
Only worth it if we start diverging architecturally (e.g. swapping the ViT for
our shared encoders, multi-embodiment heads). Start with Option A; the adapter
layer is identical either way, so B remains a drop-in swap later.

## 3. The adapter in detail (`egomimic/algo/bpp.py`)

### 3.1 Construction

```python
class BPP(Algo):
    def __init__(self, norm_stats, camera_transforms,
                 train_image_augs, eval_image_augs,
                 policy,                 # hydra-instantiated DiffusionTransformerPolicy
                 prompt: dict = None,    # chunk_n_actions, max_prompt_steps, p_drop_prompt, sampling mode
                 **kwargs):
        self.nets = nn.ModuleDict({"policy": policy})
        ...
```

Two impedance mismatches to resolve at construction time:

1. **`shape_meta`** — BPP configures everything (obs keys, `prompt_type` per
   key, horizons, action dim, `prompt_chunk_n_actions`) through a `shape_meta`
   dict interpolated across its configs. We generate it in the model yaml as a
   plain dict, mapping our zarr key names (`front_img_1`, `state_ee_pose`, …)
   to BPP entries (`type: rgb|low_dim`, `prompt_type:
   observation|proprioception|ignore`, `horizon`, `shape`). Keep our key names;
   BPP is name-agnostic as long as `shape_meta` and the batch agree.
2. **Normalization** — `MultiDataset.__getitem__` already normalizes samples,
   while BPP's policy calls `normalize_obs_with_optional_prompt` internally.
   Cleanest fix: construct the BPP `Normalizer` as **identity** (unit
   scale/zero offset for every key) so the policy's internal calls are no-ops,
   and keep all real normalization on the EgoMimic side (`self.norm_stats`),
   including `forward_eval` unnormalization. This avoids maintaining two
   normalizer stacks that must agree.

### 3.2 `process_batch_for_training`

Mirror `HPT.process_batch_for_training` (zarr-key translation via
`norm_stats.zarr_key_to_keyname`, `.to(device).float()`), then add the prompt:

- **Prompt sampling** — batch-roll (each sample prompted by a same-task batch
  neighbor) for the single-task first cut; dataset-side same-task episode
  sampling for real use.
- **Chunking** — reuse BPP's own `PromptActionChunker`
  (`train_network/utils/prompt_util.py:7`) directly rather than porting it: it
  consumes `shape_meta` (which we now build anyway) and produces exactly the
  nested dict the policy expects.
- Assemble BPP's expected obs_dict:

```python
obs_dict = {
    <current obs keys>: (B, To, ...),
    "prompt": {
        "obs":      {<prompt obs keys>: (B, P, ...)},
        "action":   (B, P, chunk_n_actions, action_dim),
        "metadata": {"mask": (B, P)},
    },
}
```

Note BPP's shipped config runs `concat_prompt_and_receding: False` with a
6-layer prompt-with-obs cross-attention decoder — the prompt path is entirely
inside the policy; the adapter only has to deliver this dict.

### 3.3 `forward_training` / `compute_losses`

`DiffusionTransformerPolicy.compute_loss(batch)` returns the diffusion loss
(epsilon prediction, DDIM 50 train steps, `input_pertub=0.1`). So:

```python
predictions["bpp_loss"] = self.nets["policy"].compute_loss(
    {"obs": obs_dict, "action": actions})
```

and `compute_losses` just relabels it `action_loss`. Unlike HPT there are no
per-domain heads, shared heads, or OT terms — BPP is single-embodiment. If a
multi-domain batch arrives, either assert single-domain or loop domains through
separate policy instances (not recommended).

### 3.4 `forward_eval` and deployment

Training-time eval: call `policy.predict_action(obs_dict)` (16 DDIM inference
steps), slice to the reference horizon, `norm_stats.unnormalize`, and also log
the diffusion val loss — mirroring the two-pass structure in
`HPT.forward_eval`.

Deployment: surface the BPP prompting API on the algo —

```python
def supports_prompting(self):  return self.nets["policy"].supports_prompting()
def prompt(self, prompt_dict): self.nets["policy"].prompt(prompt_dict)
```

`DiffusionTransformerPolicy.prompt()` normalizes (identity for us) and caches
the encoded prompt inside `PairPromptObsEncoder`, so rollout code prompts once
per episode and then calls `predict_action` per step at unprompted cost. This
is the hook the eval/env-runner needs for "demonstrate once, then roll out".

## 4. Config and registration

- `egomimic/hydra_configs/model/bpp_prompt_dit.yaml`:
  `robomimic_model._target_: egomimic.algo.bpp.BPP`, with the nested `policy:`
  block translated from BPP's `prompt_diffusion_transformer.yaml` — `n_emb:
  768`, 7-layer/8-head diffusion decoder, encoder disabled, 6-layer
  prompt-with-obs decoder, `merge_prompt_tokens: obs_and_action` +
  `AttentionPoolLatent`, `vit_base_patch16_clip_224.openai` backbone
  (pretrained, unfrozen, one per camera). Total ≈ 210M params single-camera.
- `shape_meta:` generated in the same yaml from our data config; `prompt:`
  block with `chunk_n_actions`, `max_prompt_steps`, `p_drop_prompt`.
- Register: `from egomimic.algo.bpp import BPP as BPP` in
  `egomimic/algo/__init__.py`; add a `train_zarr_*_bpp.yaml` top-level config
  pointing `model: bpp_prompt_dit`.

## 4.5 Training without a prompt

The adapter also wraps BPP's *unprompted* diffusion transformer
(`model/bpp_dit.yaml`): same head, backbone, optimizer, and augmentations, but
the obs encoder is BPP's plain `TransformerObsEncoder` instead of the
`PairPromptObsEncoder` stack. The switch is the policy's own
`supports_prompting()`, which the base encoder returns False for. When it is
False the adapter never builds `PromptActionChunker`, never attaches a
`"prompt"` entry to the obs dict, and `prompt()` raises; a stray `prompt:`
block in the config is rejected at construction. Select it with
`model=bpp_dit` on any BPP train config.

`feature_aggregation` is set to `cls` there so the receding path has one token
per camera, identical to the prompted model. BPP's own unprompted config uses
`null` (all ViT patch tokens per camera); override it to reproduce that.

This is the honest "same model without prompting" baseline. The other two
ways to remove the prompt are weaker: `prompt.p_drop_prompt: 1.0` zeros the
prompt content but still trains and attends to the prompt path, and BPP's
`ignore_prompt: True` on `PairPromptObsEncoder` currently fails with our
config (upstream adds receding and encoded-prompt tokens elementwise and the
token counts differ; there is a TODO on that line).

## 5. Gotchas

- **Image geometry**: the CLIP ViT expects 224×224. BPP applies its own
  `train/eval_image_transforms` *inside* `TransformerObsEncoder` (they're
  `shape_meta`-referenced config entries). Decide one owner: pass ours as those
  transforms and make the algo-level `train_image_augs` a no-op, or vice versa
  — but never both, or prompt and current frames get doubly/differently
  augmented.
- **Prompt frames vs. augmentation**: BPP distinguishes
  `non_prompt_train_image_transforms`; if we augment, keep prompt frames on the
  eval transform so the demonstration stays in-distribution.
- **Horizons**: BPP is receding-horizon (`obs horizon` per key in `shape_meta`)
  vs. our mostly `To=1` batches — set horizons to match what `MultiDataset`
  actually emits, don't inherit BPP's UMI defaults.
- **Action space**: `shape_meta['action']['shape']` must equal the normalized
  ac-key dim; BPP has no notion of our auxiliary action keys — drop them.
- **Optimizer/EMA**: BPP's workspace trains with EMA of policy weights; our
  `ModelWrapper` doesn't. Fine to skip initially, but expect slightly worse
  eval than their reported numbers until an EMA callback is added.
- **Dataset shuffling**: prompt sampling needs per-epoch episode reshuffling
  (BPP's `base_dataset.py:43` hook) once we move past batch-roll sampling.
