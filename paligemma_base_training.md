# Training pi0.5 from PaliGemma base weights

## TL;DR
Instead of starting from the `pi05_base` checkpoint (which already includes Physical Intelligence's action expert + flow-matching heads pretrained on robot data), we start from Google's raw PaliGemma pretrained weights (`pt_224.npz`) and randomly initialize the action expert. This mirrors the initialization Physical Intelligence used to pretrain pi0.5 from scratch, and is useful when you want to pretrain your own pi0.5 without depending on PI's robot pretraining.

## What "PaliGemma base weights" means here

Two very different checkpoints are floating around and it's easy to confuse them.

| checkpoint | contents | source | current usage in this repo |
|---|---|---|---|
| `pi05_base` | PaliGemma (SigLIP + Gemma 3B) **+** action expert (Gemma 300M) **+** flow-matching projection heads, all pretrained on robot data | `gs://openpi-assets/checkpoints/pi05_base` | `pi0.5_base.yaml`, `pi0.5_mecka.yaml`, etc. |
| **PaliGemma base** | Only PaliGemma (SigLIP + Gemma 3B). No action expert, no projections. | HuggingFace: `google/paligemma-3b-pt-224` (gated, license accept required) | **not currently supported** |

The JAX side of openpi has a `PaliGemmaWeightLoader` for the second case (see `external/openpi/src/openpi/training/weight_loaders.py:57`) that points at `gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz`. That bucket no longer allows anonymous access as of 2026, so we take the HuggingFace route instead. The PyTorch side has no equivalent loader, so this plan builds one.

## Why this initialization

- Pretrain pi0.5 from scratch on your own robot data without depending on PI's pretraining.
- Ablations on what PI's robot pretraining actually contributes.
- Fine-tuning to robot embodiments very different from PI's training mix, where the pretrained action expert might be a bad prior.

## Prerequisites (already true in this repo)

- `external/openpi` is installed in the uv env (see `pi05.md`).
- The custom transformers replacement is copied into the venv (see `pi05.md`).
- `transformers`, `safetensors`, `torch`, `tyro`, `huggingface_hub` — all already deps.
- HuggingFace login: `huggingface-cli login` (or `HF_TOKEN` env var), and the license for `google/paligemma-3b-pt-224` accepted at https://huggingface.co/google/paligemma-3b-pt-224.

## Plan

### Step 1 — new conversion script
New file: `external/openpi/examples/convert_paligemma_to_pytorch.py`

Behavior:
1. `PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224", torch_dtype=torch.float32)` — HF caches under `HF_HOME` (~5 GB across 3 safetensors shards).
2. Get its `state_dict()` — keys already in PyTorch format (`model.vision_tower...`, `model.language_model...`).
3. Prefix every key with `paligemma_with_expert.paligemma.` to match `PI0Pytorch`'s naming.
4. Instantiate `PI0Pytorch(Pi0Config(pi05=True, ...))`.
5. `model.load_state_dict(prefixed_state, strict=False)` — action expert (Gemma 300M) and projection heads (`action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`) are left at random init.
6. Cast to `bfloat16` and save via `safetensors.torch.save_model(model, "model.safetensors")`.
7. Write a small `config.json` next to it recording `init: "paligemma_base"` and the HF source.

Command surface (via `tyro`):
```
uv run examples/convert_paligemma_to_pytorch.py \
  --output_path ../../egomimic/algo/pi_checkpoints/paligemma_base_pytorch
```

### Step 2 — new Hydra model config
New file: `egomimic/hydra_configs/model/pi0.5_paligemma_init.yaml`

Copy of `pi0.5_base.yaml` with:
- `pytorch_weight_path` pointing at `.../paligemma_base_pytorch`
- Higher LR and longer warmup, because the action expert and projection heads are random:
  - `lr: 1e-4` (vs `3e-5` in `pi0.5_base.yaml`)
  - `num_warmup_steps: 5000` (vs `2000`)
  - `num_training_steps: 200000` (vs `60000`)
  - `weight_decay: 1e-4` (vs `0.0`) — small amount of regularization for the random expert
- All prompt / tokenizer / augmentation fields stay identical.

### Step 3 — verify PI algo tolerates a partial checkpoint
Currently `egomimic/algo/pi.py:173-193` does:
```python
safetensors.torch.load_model(target, model_path)
```
`safetensors.torch.load_model` uses `strict=True` by default. The new checkpoint is a complete model on disk (we saved it via `safetensors.torch.save_model` on a fully-instantiated `PI0Pytorch`), so every key is present, just some carry random values. This should load cleanly with no change needed. Verify by dry-running one training step before spinning up a real run.

### Step 4 — one-time setup command sequence

Do NOT run this on the PACE login node — the conversion loads ~5 GB of weights into RAM and gets SIGKILL'd by login-node resource limits. Use a compute node (interactive `salloc` or a slurm batch job).

```bash
# Interactive compute node, e.g.:
salloc -N1 --cpus-per-task=8 --mem=32G --time=1:00:00

# Inside the compute node:
cd /storage/project/r-dxu345-0/acheluva3/EgoVerse/external/openpi
huggingface-cli login   # first time only; needs a token with read access
uv run examples/convert_paligemma_to_pytorch.py \
  --output_path /storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/egomimic/algo/pi_checkpoints/paligemma_base_pytorch
```
Downloads ~5 GB from HF into `$HF_HOME/hub/models--google--paligemma-3b-pt-224/`, then produces ~6-7 GB (bf16) `model.safetensors` in `egomimic/algo/pi_checkpoints/paligemma_base_pytorch/`.

### Step 5 — kick off training
```bash
python egomimic/trainHydra.py \
  --config-name train_zarr_cartesian_pi \
  model=pi0.5_paligemma_init \
  data=<your data config>
```
Point `model.robomimic_model.config.pytorch_weight_path` at wherever you saved the converted checkpoint (or hard-code it in the new YAML for your machine, matching how `pi0.5_base.yaml` and `pi0.5_mecka.yaml` do it today).

## Training considerations

- **Higher LR, longer warmup**: action expert and flow-matching heads are cold. Loss will be higher for the first few thousand steps and can look worrying compared to a `pi05_base` fine-tune. This is expected.
- **Precision**: keep `pytorch_training_precision: bfloat16`. The random-init tensors will be created in fp32 by default and cast down; that's fine.
- **Grad clipping**: consider setting `enable_grad_norm: true` and clipping to 1.0 during the warmup, because the random projection heads can produce large early gradients. Not strictly required.
- **Data scale**: from-scratch pretraining wants meaningfully more data than a fine-tune. If you have <10k trajectories, `pi05_base` is probably still the right starting point.
- **Wall-clock**: expect 3-5x more training steps to reach comparable eval performance vs. starting from `pi05_base` on the same data.

## Verification checklist

Once the conversion script runs:

- [ ] `model.safetensors` exists and is ~7 GB (bf16).
- [ ] `config.json` in the output dir records `init: "paligemma_base"`.
- [ ] `safetensors.torch.load_model` on the file (via `pi.py` init) completes with no missing-key errors.
- [ ] A single training step runs and produces a finite loss.
- [ ] After ~1000 steps the loss is decreasing (not diverging).

## Open questions / risks

1. **Custom transformers replacement**: `external/openpi/src/openpi/models_pytorch/transformers_replace/` overrides HF's Gemma/PaliGemma/SigLIP modules. Loading the HF `google/paligemma-3b-pt-224` weights into `PaliGemmaForConditionalGeneration` still uses HF's own class signature, so this should work as long as the replacement is applied before `PI0Pytorch` is constructed. If `pi05.md`'s copy-into-venv step was skipped or is stale, expect either extra keys reported as `unexpected` during load, or shape mismatches — both signals to re-apply the replacement.
2. **Vocab size**: PaliGemma vocab is 257152 tokens; `PI0Pytorch` inherits this from `Pi0Config`. No action needed, just noting it.
3. **HF gated access**: `google/paligemma-3b-pt-224` is gated. If someone else on the team runs this and hits a 401, they need to accept the license on the model card and be logged in via `huggingface-cli login`.
4. **Gemma 300M pretrained weights**: we deliberately leave the action expert random. If we later want a warm-start for it, Google's Gemma 300M weights would need a separate conversion path — out of scope for this plan.

## File map

New files:
- `external/openpi/examples/convert_paligemma_to_pytorch.py`
- `egomimic/hydra_configs/model/pi0.5_paligemma_init.yaml`
- `paligemma_base_training.md` (this doc)

Reused, unchanged:
- `external/openpi/src/openpi/models_pytorch/pi0_pytorch.py` — model definition
- `external/openpi/src/openpi/shared/download.py` — GCS fetch + cache
- `egomimic/algo/pi.py` — weight loading path
- `external/openpi/examples/convert_jax_model_to_pytorch.py` — reference for the tensor key mappings
