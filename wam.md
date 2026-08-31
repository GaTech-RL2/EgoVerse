## Installation
Usable with the everse uv environment. The Wan2.1 backbone (DiT + VAE) is
vendored in-repo under `egomimic/models/wan/`, so no extra package install is
needed beyond the everse env.

## Getting pretrained checkpoint
WAM builds on the **Wan2.1-T2V-1.3B** backbone. It only needs the DiT and VAE
weights (no text encoder — WAM does joint video+action, not text conditioning).

From the EgoVerse directory:
```
uv pip install "huggingface_hub[cli]"
hf download Wan-AI/Wan2.1-T2V-1.3B \
  diffusion_pytorch_model.safetensors \
  Wan2.1_VAE.pth \
  config.json \
  --local-dir egomimic/algo/wan_checkpoints/Wan2.1-T2V-1.3B
```

This pulls only the two weight files WAM uses:
- `diffusion_pytorch_model.safetensors` (~5.6G) — the 1.3B T2V DiT
- `Wan2.1_VAE.pth` (~0.5G) — the frame <-> latent VAE

(To grab the whole repo instead, drop the per-file args:
`hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir <same dir>`.)

## Optional: text encoder setup
The default WAM does **not** use text conditioning — the DiT + VAE above are all
the training/eval path loads. The T5 (UMT5-XXL) text encoder is only needed if
you want text-conditioned generation. It is optional and not wired into
`wam_bc_human_wan21_1_3b.yaml`; the `WanTextEncoder` class and its `from_civitai`
converter are vendored in-repo for that future use.

Download the encoder weights (~11G) into the same checkpoint dir:
```
hf download Wan-AI/Wan2.1-T2V-1.3B \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir egomimic/algo/wan_checkpoints/Wan2.1-T2V-1.3B
```

Load it the same way the DiT/VAE are loaded (see `build_wan_vae` in
`egomimic/models/wam_nets.py`):
```python
import torch
from egomimic.models.wan.wan_video_text_encoder import WanTextEncoder

text_encoder = WanTextEncoder()
sd = torch.load(
    "egomimic/algo/wan_checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    map_location="cpu",
)
sd = WanTextEncoder.state_dict_converter().from_civitai(sd)
text_encoder.load_state_dict(sd, strict=False)
text_encoder.eval().requires_grad_(False)
```

## No conversion step
Unlike pi0.5, there is no separate JAX->PyTorch conversion. `build_wam_dit` and
`build_wan_vae` (in `egomimic/models/wam_nets.py`) load the official Wan
safetensors/pth and convert them to the WanModel naming at runtime via the Wan
`from_civitai` converter.

## Config paths
The model config `egomimic/hydra_configs/model/wam_bc_human_wan21_1_3b.yaml`
points `dit.checkpoint_path` and `vae.checkpoint_path` at:
```
egomimic/algo/wan_checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
egomimic/algo/wan_checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```
These are currently absolute paths for this checkout. If you download to a
different location, update those two `checkpoint_path` fields to match.

`egomimic/algo/wan_checkpoints/` is gitignored, so the downloaded weights are
never committed.
