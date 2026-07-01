"""
Network building blocks for WAM (World-Action Model) — the dreamzero port.

The model is the **CausalWanModel** (a Wan video DiT extended with action+state
register tokens): one joint token sequence ``[video latent | action | state]``,
flow-matched jointly so a single forward predicts BOTH the next video-frame
velocity and the action-chunk velocity. The whole WAN stack is vendored under
``egomimic/models/wan/``; this module is the builder/import surface the WAM algo
(``egomimic/algo/wam.py``) and Hydra configs use.

Builders (used as Hydra ``_target_``):
  - ``build_wam_dit``  -> CausalWanModel (loads the 1.3B Wan2.1 T2V DiT weights,
                          freezes the pretrained video blocks / optional LoRA,
                          leaves state/action/decoder modules trainable)
  - ``build_wan_vae``  -> frozen WanVideoVAE (frame <-> latent)

Re-exports: CausalWanModel, WANPolicyHead, FlowMatchScheduler,
FlowUniPCMultistepScheduler, WanVideoVAE, WanTextEncoder.
"""

import torch

# --- WAN backbone (dreamzero) ----------------------------------------------
from egomimic.models.wan.wan_video_dit import WanModel  # legacy (build_wan21_1_3b)
from egomimic.models.wan.wan_video_dit_action_casual_chunk import CausalWanModel
from egomimic.models.wan.wan_video_vae import WanVideoVAE, WanVideoVAE38

# ===========================================================================
# CausalWanModel (the WAM DiT) — 1.3B Wan2.1 T2V architecture preset
# ===========================================================================
# Canonical Wan2.1-T2V-1.3B DiT dims (dreamzero has no 1.3B config; these are
# the upstream 1.3B numbers, and they match the downloaded T2V-1.3B checkpoint).
WAM_DIT_1_3B = dict(
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12,
    num_layers=30,
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    num_frame_per_block=1,
    concat_first_frame_latent=False,  # T2V: no first-frame channel concat
    hidden_size=1024,  # state/action encoder hidden width
    max_num_embodiments=1,  # single (human) embodiment
)

# ===========================================================================
# Wan2.2 TI2V-5B preset — the actual dreamzero backbone (video DiT + VAE38).
# Config values from Wan-AI/Wan2.2-TI2V-5B/config.json. `in_dim=out_dim=48` is
# the VAE38 latent channel count; `model_type="ti2v"` selects the i2v cross-
# attention path (first-frame conditioning); `concat_first_frame_latent=False`
# because 5B does NOT do channel-concat (uses clean_x teacher forcing instead).
# ===========================================================================
WAM_DIT_5B = dict(
    model_type="ti2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=48,
    dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    out_dim=48,
    num_heads=24,
    num_layers=30,
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    num_frame_per_block=1,
    concat_first_frame_latent=False,  # 5B pretrained style
    hidden_size=1024,
    max_num_embodiments=1,
)


def _load_sharded_or_single(checkpoint_path: str):
    """Load a safetensors file OR a directory of sharded safetensors + index."""
    import json
    import os

    if os.path.isdir(checkpoint_path):
        index_path = os.path.join(
            checkpoint_path, "diffusion_pytorch_model.safetensors.index.json"
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No safetensors index in {checkpoint_path}")
        from safetensors.torch import load_file

        with open(index_path) as f:
            index = json.load(f)
        sd = {}
        for shard in set(index["weight_map"].values()):
            sd.update(load_file(os.path.join(checkpoint_path, shard)))
        return sd
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint_path)
    return torch.load(checkpoint_path, map_location="cpu")


def build_wam_dit(
    checkpoint_path: str = None,
    action_dim: int = 12,
    max_state_dim: int = 12,
    num_action_per_block: int = 100,
    num_state_per_block: int = 1,
    frame_seqlen: int = 1560,
    freeze_video: bool = True,
    lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    preset: str = "1.3B",  # "1.3B" (Wan2.1 T2V-1.3B) or "5B" (Wan2.2 TI2V-5B)
    **overrides,
):
    """Build the CausalWanModel WAM DiT at a Wan preset (1.3B or 5B).

    - Loads the pretrained video DiT weights from ``checkpoint_path``. For 1.3B
      this is the T2V-1.3B safetensors, converted via the Wan civitai converter;
      for 5B this is the Wan2.2-TI2V-5B sharded safetensors dir. The
      state/action/decoder register modules stay randomly initialized either way.
    - Trainability (memory): ``freeze_video`` freezes the pretrained blocks and
      trains only state/action/decoder (dreamzero keeps those trainable too);
      ``lora`` additionally injects LoRA adapters on the blocks (dreamzero's
      setup). ``num_action_per_block`` must equal the action horizon and
      ``num_state_per_block`` the state horizon (register-length invariant).
    """
    if preset == "1.3B":
        cfg = dict(WAM_DIT_1_3B)
    elif preset == "5B":
        cfg = dict(WAM_DIT_5B)
    else:
        raise ValueError(f"Unknown preset {preset!r} (expected '1.3B' or '5B')")
    cfg.update(
        action_dim=action_dim,
        max_state_dim=max_state_dim,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
        frame_seqlen=frame_seqlen,
    )
    cfg.update(overrides)
    model = CausalWanModel(**cfg)

    if checkpoint_path:
        sd = _load_sharded_or_single(checkpoint_path)
        if preset == "1.3B":
            # Wan2.1 checkpoint needs the civitai key converter (filters vace, matches hash).
            try:
                sd, _ = WanModel.state_dict_converter().from_civitai(sd)
            except Exception as e:  # noqa: BLE001
                print(f"[build_wam_dit] from_civitai skipped ({e}); loading raw")
        # Wan2.2 TI2V-5B keys already match CausalWanModel layout (bare
        # `blocks.<i>.<...>`, `patch_embedding.*`, etc.), so no conversion.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[build_wam_dit] preset={preset} loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )

    # --- trainability ------------------------------------------------------
    if lora:
        _inject_lora(model, lora_rank, lora_alpha)
        freeze_video = True  # lora handles the blocks; freeze the rest of them
    if freeze_video:
        model.requires_grad_(False)
        # The 3 robot modules are always trained (fresh, dreamzero-style).
        for mod in (model.state_encoder, model.action_encoder, model.action_decoder):
            mod.requires_grad_(True)
        if lora:  # re-enable LoRA adapter params
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[build_wam_dit] trainable params: {n_train/1e6:.1f}M "
        f"(lora={lora}, freeze_video={freeze_video})"
    )
    return model


def _inject_lora(model, rank, alpha, targets=("q", "k", "v", "o", "ffn.0", "ffn.2")):
    """Inject LoRA adapters on the DiT blocks (dreamzero: rank 4 / alpha 4 on
    attn q,k,v,o + the two FFN linears)."""
    from peft import LoraConfig, inject_adapter_in_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(targets),
    )
    inject_adapter_in_model(cfg, model)


def build_wan_vae(checkpoint_path: str = None, z_dim: int = 16):
    """Build the vendored Wan 3D causal VAE (frozen) for the frame<->latent path.
    encode(video in [-1,1]) -> latent ; decode(latent) -> frames in [-1,1].
    """
    vae = WanVideoVAE(z_dim=z_dim)
    if checkpoint_path:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if "model_state" in sd:
            sd = sd["model_state"]
        # Load into the inner nn.Module (vae.model) so keys stay bare
        # ("encoder.conv1.weight", etc.) — the top-level WanVideoVAE wraps
        # VideoVAE_ under the attribute "model", so calling `from_civitai`
        # (which prepends "model.") here would double-prefix and mis-load.
        missing, unexpected = vae.model.load_state_dict(sd, strict=False)
        print(
            f"[build_wan_vae] loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    vae.eval().requires_grad_(False)
    return vae


def build_wan_vae_38(checkpoint_path: str = None, z_dim: int = 48, dim: int = 160):
    """Build the Wan2.2 VideoVAE38 (frozen). Same encode/decode interface as
    build_wan_vae but 16× spatial downsampling and z_dim=48 (vs 8× / z_dim=16 in
    the 2.1 VAE). Pairs with the Wan2.2 5B DiT preset."""
    vae = WanVideoVAE38(z_dim=z_dim, dim=dim)
    if checkpoint_path:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if "model_state" in sd:
            sd = sd["model_state"]
        missing, unexpected = vae.model.load_state_dict(sd, strict=False)
        print(
            f"[build_wan_vae_38] loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    vae.eval().requires_grad_(False)
    return vae


# ===========================================================================
# Legacy: plain (non-causal) WanModel builder for the simple action-only path.
# Kept for back-compat; the dreamzero WAM uses build_wam_dit above.
# ===========================================================================
WAN21_T2V_1_3B_CONFIG = dict(
    dim=1536,
    in_dim=16,
    ffn_dim=8960,
    out_dim=16,
    freq_dim=256,
    eps=1e-6,
    num_heads=12,
    num_layers=30,
    text_dim=4096,
    patch_size=[1, 2, 2],
    has_image_input=False,
)


def build_wan21_1_3b(checkpoint_path: str = None, **overrides):
    """(Legacy) plain Wan2.1 T2V-1.3B DiT (no action register tokens)."""
    model = WanModel(**{**WAN21_T2V_1_3B_CONFIG, **overrides})
    if checkpoint_path:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(checkpoint_path)
        else:
            sd = torch.load(checkpoint_path, map_location="cpu")
        try:
            sd, _ = WanModel.state_dict_converter().from_civitai(sd)
        except Exception as e:  # noqa: BLE001
            print(f"[build_wan21_1_3b] from_civitai skipped ({e}); loading raw sd")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[build_wan21_1_3b] loaded {checkpoint_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    return model
