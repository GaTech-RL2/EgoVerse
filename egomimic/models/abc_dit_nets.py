"""ABC-DiT: Amazon FAR's ABC diffusion-transformer policy, ported to EgoVerse.

Port of ``abc_minimal/dit.py`` from https://github.com/amazon-far/abc
(Apache-2.0; "ABC: Scalable Behavior Cloning with Open Data, Training, and
Evaluation"). The module names, tensor shapes and forward algebra follow that
reference so the architecture is recognisably ABC's: a DINOv3 tower whose
patch tokens are attention-pooled to a handful of query tokens per camera,
one 512-d CLIP task vector, and a DiT over the action chunk whose blocks
cross-attend to the vision tokens while a 9-way AdaLN modulation carries
``[state, task, flow time]``.

Three things differ from the reference, all of them plumbing:

* Vision weights come from HF ``facebook/dinov3-vitb16-pretrain-lvd1689m``
  instead of ABC's vendored ``DinoVisionTransformer`` reading Meta's ``.pth``
  (which ABC leaves to the user to obtain, falling back to random init). Same
  ViT-B/16, same 196 patch tokens per 224x224 frame -- CLS and the four
  register tokens are dropped exactly as ``dinov3_embedding.py`` does.
* The task vector comes from HF ``openai/clip-vit-base-patch32``'s text tower
  rather than ABC's vendored BPE + TorchScript loader. ``text_embeds`` is the
  same EOT-token projection to 512-d, and it is L2-normalised as ABC does.
* Actions and state are EgoVerse-shaped: a padded shared action width with a
  loss mask (so cotrain configs work), per-embodiment state embedders, and
  proprio history flattened into the state vector.

Sizes: the paper sweeps four (Figure 21). S/B/L are the standard DiT ladder
and xL is ABC's own shape (paper S3.1: "32 layers, 24 attention heads, and a
hidden dimension of 1536, which is larger than the DiT-XL architecture").
``tests/unit/test_abc_dit_nets.py`` asserts the parameter counts those shapes
produce against the paper's published numbers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hpt_nets import PretrainedWeights, _local_snapshot_if_offline


@dataclass(frozen=True)
class DiTSize:
    """One row of the ABC-DiT size ladder."""

    depth: int
    hidden: int
    heads: int


# The four sizes ABC sweeps. S/B/L are the DiT ladder of Peebles & Xie; xL is
# ABC's own. Cross-checked against the paper's Figure 21 counts by building
# each shape: with ABC's 85.7M DINOv3 ViT-B they come to 153.2M / 290.1M /
# 746.4M / 2015.7M, i.e. its "153M, 290M, 746M" for S/B/L, and for xL the
# 1.93B it quotes for the action head alone (Table 2).
ABC_DIT_SIZES: Dict[str, DiTSize] = {
    "s": DiTSize(depth=12, hidden=384, heads=6),
    "b": DiTSize(depth=12, hidden=768, heads=12),
    "l": DiTSize(depth=24, hidden=1024, heads=16),
    "xl": DiTSize(depth=32, hidden=1536, heads=24),
}


def resolve_size(
    size: Optional[str],
    hidden_size: Optional[int] = None,
    depth: Optional[int] = None,
    num_heads: Optional[int] = None,
) -> DiTSize:
    """``size`` name from ``ABC_DIT_SIZES``, with per-field overrides."""
    if size is None and (hidden_size is None or depth is None or num_heads is None):
        raise ValueError(
            "pass size=<s|b|l|xl>, or all of hidden_size / depth / num_heads"
        )
    base = ABC_DIT_SIZES.get(str(size).lower()) if size is not None else None
    if size is not None and base is None:
        raise KeyError(
            f"unknown ABC-DiT size {size!r}; known sizes: {sorted(ABC_DIT_SIZES)}"
        )
    resolved = DiTSize(
        depth=int(depth if depth is not None else base.depth),
        hidden=int(hidden_size if hidden_size is not None else base.hidden),
        heads=int(num_heads if num_heads is not None else base.heads),
    )
    if resolved.hidden % resolved.heads:
        raise ValueError(
            f"hidden {resolved.hidden} is not divisible by {resolved.heads} heads"
        )
    return resolved


# ---------------------------------------------------------------------------
# DiT pieces (abc_minimal/dit.py)
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate_residual(gate: torch.Tensor, residual: torch.Tensor):
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * residual


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", np.arange(length, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the flow time, then a 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs
        if freqs.device != t.device:
            freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_shape = t.shape
        t_freq = self.timestep_embedding(t.reshape(-1))
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb.reshape(*t_shape, -1)


class DiTAttention(nn.Module):
    """Self-attention over the action tokens (timm-equivalent, qkv_bias=True)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class DiTMlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block with vision cross-attention (9-way modulation)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = DiTMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.norm_xattn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_xattn_kv = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        vision_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modulation = self.adaLN_modulation(c)
        if prefix_mask is not None:
            # c is (B, 2, H): row 0 modulates the conditioned prefix positions,
            # row 1 the positions being denoised.
            modulation = torch.where(
                prefix_mask.unsqueeze(-1), modulation[:, :1], modulation[:, 1:]
            )
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_xattn,
            scale_xattn,
            gate_xattn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(9, dim=-1)

        x = x + gate_residual(
            gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        )
        x_normed = modulate(self.norm_xattn(x), shift_xattn, scale_xattn)
        kv = self.norm_xattn_kv(vision_tokens)
        xattn_out, _ = self.cross_attn(x_normed, kv, kv, need_weights=False)
        x = x + gate_residual(gate_xattn, xattn_out)
        return x + gate_residual(
            gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modulation = self.adaLN_modulation(c)
        if prefix_mask is not None:
            modulation = torch.where(
                prefix_mask.unsqueeze(-1), modulation[:, :1], modulation[:, 1:]
            )
        shift, scale = modulation.chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PoolMlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AttentionPoolBlock(nn.Module):
    """Learnable queries cross-attend to the ViT tokens (per camera)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = PoolMlp(embed_dim, int(mlp_ratio * embed_dim))

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        x_kv = self.ln_1(x)
        x_q = self.ln_1(queries)
        out, _ = self.attention(x_q, x_kv, x_kv, need_weights=False)
        return self.mlp(self.ln_2(out)) + out


# ---------------------------------------------------------------------------
# Pretrained encoders
# ---------------------------------------------------------------------------


def _safetensors_reference(
    snapshot_dir: Optional[str],
    attr: str,
    live: set,
    dtype: torch.dtype,
) -> Optional[dict]:
    """A checkpoint's tensors re-keyed onto ``<attr>.*``, for the hash check.

    ``transformers`` may nest part of a checkpoint under a submodule when it
    builds the live model -- DINOv3's file stores ``layer.0.attention...``
    while ``DINOv3ViTModel`` exposes ``model.layer.0.attention...``, with its
    embeddings and final norm NOT nested -- so each key is resolved against
    the live names through the candidate prefixes. Anything short of a
    complete map returns ``None`` (i.e. "cannot be verified"), because
    ``verify_pretrained_weights`` hashes whatever it is handed and would read
    a naming difference as corrupted weights. Tensors the live module does not
    hold are dropped: CLIP's file carries the vision tower too.
    """
    if not snapshot_dir:
        return None
    path = os.path.join(snapshot_dir, "model.safetensors")
    if not os.path.isfile(path):
        return None
    from safetensors import safe_open

    live = set(live)
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            for prefix in ("", "model."):
                name = f"{attr}.{prefix}{key}"
                if name in live:
                    state[name] = handle.get_tensor(key).to(dtype)
                    break
    return state if set(state) == live else None


class DINOv3Tower(PretrainedWeights, nn.Module):
    """DINOv3 ViT-B/16 patch tokens, ABC's ``DinoVisionBackbone`` over HF.

    Owns its input pipeline like ``SigLIPStem`` does: a squash resize to the
    checkpoint's square resolution and the checkpoint's own mean / std, so the
    config's image augs must NOT normalise. The squash is the repo's
    convention (``SigLIPStem``, pi0 / PaliGemma) rather than ABC's, which
    letterboxes in ``preprocess.resize_pad_normalize``; every arm here is fed
    the same way, which is what makes the comparison a comparison.

    Args:
        model_name: HF repo id (or local path) of the DINOv3 checkpoint.
        freeze: freeze the tower and keep it in eval mode.
        bf16_autocast: run the tower under bf16 autocast on CUDA and return
            fp32 tokens, ABC's ``dino_bf16`` default.
        image_size: override the checkpoint's native resolution.
        pretrained: ``False`` builds the architecture from the config alone
            (random weights) -- what ABC falls back to without Meta's ``.pth``,
            and what the unit tests use to stay offline.
        config_overrides: architecture fields to override when
            ``pretrained=False`` (e.g. a tiny tower for tests).
    """

    DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    # ImageNet stats; DINOv3's preprocessor_config.json carries the same pair.
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)
    _hpt_pretrained = True
    _hpt_pretrained_attrs = ("tower",)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        freeze: bool = False,
        bf16_autocast: bool = True,
        image_size: Optional[int] = None,
        pretrained: bool = True,
        config_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_name = model_name
        self.pretrained = bool(pretrained)
        if self.pretrained:
            self._snapshot_dir = _local_snapshot_if_offline(model_name)
            self.tower = AutoModel.from_pretrained(self._snapshot_dir)
        else:
            # No hub access: build the architecture from its config class.
            self._snapshot_dir = None
            config = AutoConfig.for_model("dinov3_vit", **(config_overrides or {}))
            self.tower = AutoModel.from_config(config)

        config = self.tower.config
        self.hidden_size = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.image_size = int(
            image_size if image_size is not None else config.image_size
        )
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} is not a multiple of the patch "
                f"size {self.patch_size}"
            )
        side = self.image_size // self.patch_size
        self.num_patches = side * side

        mean, std = self._preprocessor_stats()
        self.register_buffer(
            "image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )
        self.bf16_autocast = bool(bf16_autocast)
        self.freeze = bool(freeze)
        if self.freeze:
            for param in self.tower.parameters():
                param.requires_grad = False

    def _preprocessor_stats(self):
        import json

        if not self._snapshot_dir:
            return list(self.DEFAULT_MEAN), list(self.DEFAULT_STD)
        path = os.path.join(self._snapshot_dir, "preprocessor_config.json")
        if not os.path.isfile(path):
            return list(self.DEFAULT_MEAN), list(self.DEFAULT_STD)
        with open(path, encoding="utf-8") as handle:
            preproc = json.load(handle)
        return (
            preproc.get("image_mean", list(self.DEFAULT_MEAN)),
            preproc.get("image_std", list(self.DEFAULT_STD)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.tower.eval()
        return self

    def backbone_parameters(self) -> List[nn.Parameter]:
        """The tower's parameters, for ``ModelWrapper._backbone_param_groups``
        (ABC trains the vision encoder at 0.1x the head LR; paper S B.3)."""
        return list(self.tower.parameters())

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The snapshot's own tensors, re-keyed as ``tower.*``."""
        return _safetensors_reference(
            self._snapshot_dir,
            "tower",
            set(self.pretrained_state_dict()),
            next(self.tower.parameters()).dtype,
        )

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, num_patches, hidden_size)."""
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        images = (images - self.image_mean) / self.image_std

        def run():
            # CLS and the register tokens lead; the patch grid is the tail.
            if self.bf16_autocast and images.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens = self.tower(pixel_values=images).last_hidden_state
                tokens = tokens.to(torch.float32)
            else:
                tokens = self.tower(pixel_values=images).last_hidden_state
            return tokens[:, -self.num_patches :]

        if self.freeze:
            with torch.no_grad():
                return run()
        return run()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_tokens(images)


class CLIPTaskEncoder(PretrainedWeights, nn.Module):
    """Frozen CLIP ViT-B/32 text tower -> one L2-normalised 512-d task vector.

    ABC computes this in its dataloader and hands the policy a ``task_vec_clip``
    column; here it is a frozen submodule so a prompt string is all the algo
    has to pass. Features are cached per unique prompt like ABC's
    ``CLIPTextEmbedder`` (and like the repo's Qwen text-feature cache): the
    encoder is frozen, so the map from string to vector never changes, and a
    flagship run with one constant prompt then pays for a single forward.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"
    _hpt_pretrained = True
    _hpt_pretrained_attrs = ("text_model",)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 77,
        cache_text_features: bool = True,
        text_cache_max_entries: int = 8192,
        pretrained: bool = True,
        config_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__()
        from transformers import (
            AutoTokenizer,
            CLIPTextConfig,
            CLIPTextModelWithProjection,
        )

        self.model_name = model_name
        self.pretrained = bool(pretrained)
        if self.pretrained:
            self._snapshot_dir = _local_snapshot_if_offline(model_name)
            self.text_model = CLIPTextModelWithProjection.from_pretrained(
                self._snapshot_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self._snapshot_dir)
        else:
            self._snapshot_dir = None
            self.text_model = CLIPTextModelWithProjection(
                CLIPTextConfig(**(config_overrides or {}))
            )
            self.tokenizer = None

        self.output_dim = int(self.text_model.config.projection_dim)
        self.max_length = int(
            min(max_length, self.text_model.config.max_position_embeddings)
        )
        self.cache_text_features = bool(cache_text_features)
        self.text_cache_max_entries = int(text_cache_max_entries)
        self._cache: Dict[str, torch.Tensor] = {}
        # Frozen: ABC's task vector is a fixed feature, never fine-tuned.
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_model.eval()
        return self

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The text half of the CLIP snapshot, re-keyed as ``text_model.*``
        (the vision tower's tensors are in the same file and are dropped).

        Usually ``None`` in practice: openai/clip-vit-base-patch32 ships a
        ``pytorch_model.bin`` and no safetensors file, so there is nothing to
        hash against -- the weights still load, they are just unverified.
        """
        return _safetensors_reference(
            self._snapshot_dir,
            "text_model",
            set(self.pretrained_state_dict()),
            next(self.text_model.parameters()).dtype,
        )

    @torch.no_grad()
    def _encode(self, prompts: List[str]) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError(
                "CLIPTaskEncoder was built with pretrained=False and has no "
                "tokenizer; pass encoded prompts or build it pretrained"
            )
        device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        features = self.text_model(**tokens).text_embeds.float()
        return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        """``list[str]`` of length B -> (B, 512) L2-normalised task vectors."""
        prompts = [str(p) for p in prompts]
        if not self.cache_text_features:
            return self._encode(prompts)
        missing = [p for p in dict.fromkeys(prompts) if p not in self._cache]
        if missing:
            features = self._encode(missing)
            if len(self._cache) + len(missing) > self.text_cache_max_entries:
                self._cache.clear()
            for prompt, feature in zip(missing, features):
                self._cache[prompt] = feature
        device = next(self.text_model.parameters()).device
        return torch.stack([self._cache[p] for p in prompts], dim=0).to(device)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def _role_key(role: str) -> str:
    """``nn.ModuleDict`` keys cannot contain '.', and camera roles are dotted."""
    return role.replace(".", "_")


class ABCDiTPolicy(nn.Module):
    """ABC-DiT over an EgoVerse batch.

    Args:
        camera_roles: canonical camera keys, in the order the algo stacks them.
            One attention-pool query set and one camera embedding per role.
        action_width: the padded shared action width the head decodes.
        action_horizon: chunk length (ABC's ``chunk_length``).
        state_dims: ``{embodiment: proprio width}``; one state embedder each.
        size: ``s`` / ``b`` / ``l`` / ``xl`` from ``ABC_DIT_SIZES``, with
            ``hidden_size`` / ``depth`` / ``num_heads`` overriding fields.
        history_len: proprio steps per sample; they are flattened into the
            state vector, so the state embedder is ``history_len * width`` wide.
        vision / task_encoder: the two pretrained encoders.
        mask_state_ratio: per-sample probability of zeroing the state during
            training (ABC's ``flow.mask_state_ratio``, 0.1).
        max_action_prefix: > 0 turns on ABC's action-prefix conditioning, whose
            production default is 8. Off here: the repo's evaluators sample a
            whole chunk from noise, so the prefix branch would train a
            capability nothing measures.
    """

    def __init__(
        self,
        camera_roles: Sequence[str],
        action_width: int,
        action_horizon: int,
        state_dims: Dict[str, int],
        vision: nn.Module,
        task_encoder: nn.Module,
        size: Optional[str] = "xl",
        hidden_size: Optional[int] = None,
        depth: Optional[int] = None,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        history_len: int = 1,
        vision_pool_num_queries: int = 12,
        vision_pool_num_heads: int = 8,
        vision_pool_mlp_ratio: int = 4,
        num_inference_steps: int = 10,
        time_dist: str = "uniform",
        mask_state_ratio: float = 0.1,
        max_action_prefix: int = 0,
        prefix_conditioning_prob: float = 1.0,
        prefix_noise_scale: float = 0.0,
        fuse_camera_batch_limit: int = 32,
    ) -> None:
        super().__init__()
        self.camera_roles = list(camera_roles)
        if not self.camera_roles:
            raise ValueError("camera_roles must name at least one camera")
        if len(set(self.camera_roles)) != len(self.camera_roles):
            raise ValueError(f"camera_roles has duplicates: {self.camera_roles}")
        if not state_dims:
            raise ValueError(
                "ABC-DiT conditions on the state; state_dims must name at least "
                "one embodiment's proprio width"
            )
        if time_dist not in ("uniform", "beta"):
            raise ValueError(f"time_dist must be uniform or beta, got {time_dist!r}")

        self.size = resolve_size(size, hidden_size, depth, num_heads)
        H = self.size.hidden
        self.hidden_size = H
        self.action_width = int(action_width)
        self.action_horizon = int(action_horizon)
        self.history_len = int(history_len)
        self.state_dims = {str(k): int(v) for k, v in state_dims.items()}
        self.num_inference_steps = int(num_inference_steps)
        self.time_dist = time_dist
        self.mask_state_ratio = float(mask_state_ratio)
        self.max_action_prefix = int(max_action_prefix)
        self.prefix_conditioning_prob = float(prefix_conditioning_prob)
        self.prefix_noise_scale = float(prefix_noise_scale)
        self.fuse_camera_batch_limit = int(fuse_camera_batch_limit)

        self.vision = vision
        self.task_encoder = task_encoder
        vit_dim = int(vision.hidden_size)
        task_embed_dim = int(task_encoder.output_dim)

        # One state embedder per embodiment (ABC has a single ``x_embedder``;
        # the repo's algos are cotrain-capable, so the widths may differ).
        self.x_embedder = nn.ModuleDict(
            {
                name: nn.Linear(self.history_len * dim, H)
                for name, dim in self.state_dims.items()
            }
        )
        self.y_embedder = nn.Linear(self.action_width, H)
        self.t_embedder = TimestepEmbedder(H)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.action_horizon, H), requires_grad=False
        )

        self.apool_queries = nn.ParameterDict(
            {
                _role_key(role): nn.Parameter(
                    torch.randn(1, int(vision_pool_num_queries), vit_dim) * 0.02
                )
                for role in self.camera_roles
            }
        )
        self.apool = nn.ModuleDict(
            {
                _role_key(role): AttentionPoolBlock(
                    vit_dim, int(vision_pool_num_heads), int(vision_pool_mlp_ratio)
                )
                for role in self.camera_roles
            }
        )
        self.vision_tokens_proj = nn.Linear(vit_dim, H)
        self.vision_camera_embed = nn.Embedding(len(self.camera_roles), H)

        self.task_to_hidden = nn.Linear(task_embed_dim, H)
        self.blocks = nn.ModuleList(
            DiTBlock(H, self.size.heads, mlp_ratio) for _ in range(self.size.depth)
        )
        self.final_layer = FinalLayer(H, self.action_width)
        # cond = [state, task, timestep] -> hidden; vision goes via cross-attn.
        self.cond_proj = nn.Sequential(
            nn.Linear(3 * H, H), nn.SiLU(), nn.Linear(H, H), nn.LayerNorm(H)
        )

        pos = get_1d_sincos_pos_embed(H, self.action_horizon)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Read by ``ModelWrapper._backbone_param_groups`` for the vision LR
        # group; the task encoder is frozen and never reaches an optimizer.
        self.encoders = {"vision": vision}

    # -- pieces -----------------------------------------------------------

    def build_vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """(B, F, 3, H, W) in [0, 1] -> (B, F * queries, hidden)."""
        if images.ndim != 5:
            raise ValueError(
                f"images must be (B, F, 3, H, W), got {tuple(images.shape)}"
            )
        batch, cameras = images.shape[:2]
        if cameras != len(self.camera_roles):
            raise ValueError(
                f"images carry {cameras} cameras but the policy was built for "
                f"{len(self.camera_roles)}: {self.camera_roles}"
            )
        if batch * cameras <= self.fuse_camera_batch_limit:
            # Fuse small inference batches for GPU utilization.
            flat = images.transpose(0, 1).reshape(batch * cameras, *images.shape[2:])
            all_tokens = self.vision.encode_image_tokens(flat).reshape(
                cameras, batch, -1, self.vision.hidden_size
            )
        else:
            # Split large training batches to limit backbone activation memory.
            all_tokens = torch.stack(
                [
                    self.vision.encode_image_tokens(images[:, index])
                    for index in range(cameras)
                ],
                dim=0,
            )

        pooled = []
        for index, role in enumerate(self.camera_roles):
            key = _role_key(role)
            tokens = all_tokens[index].to(self.apool_queries[key].dtype)
            queries = self.apool_queries[key].expand(tokens.shape[0], -1, -1)
            pooled.append(self.apool[key](tokens, queries))
        tokens_by_camera = torch.stack(pooled, dim=1)  # (B, Nc, K, vit_dim)
        B, Nc, K, D = tokens_by_camera.shape
        vision_tokens = self.vision_tokens_proj(
            tokens_by_camera.reshape(B * Nc * K, D).to(
                self.vision_tokens_proj.weight.dtype
            )
        ).reshape(B, Nc, K, -1)
        cam_emb = self.vision_camera_embed(
            torch.arange(Nc, device=vision_tokens.device)
        )
        vision_tokens = vision_tokens + cam_emb[None, :, None, :]
        return vision_tokens.reshape(B, Nc * K, -1)

    def _flatten_state(self, state: torch.Tensor, embodiment: str) -> torch.Tensor:
        if state.ndim == 2:
            state = state[:, None]
        if state.ndim != 3:
            raise ValueError(f"state must be (B, K, D), got {tuple(state.shape)}")
        if state.shape[1] != self.history_len:
            raise ValueError(
                f"state carries {state.shape[1]} history steps but history_len="
                f"{self.history_len}; set the keymap's proprio_history and the "
                "model's history_len to the same K"
            )
        if state.shape[-1] != self.state_dims[embodiment]:
            raise ValueError(
                f"{embodiment} state is {state.shape[-1]}-wide, expected "
                f"{self.state_dims[embodiment]}"
            )
        return state.reshape(state.shape[0], -1)

    def compute_cond(
        self,
        state: torch.Tensor,
        task_vec: torch.Tensor,
        t_cond: torch.Tensor,
        embodiment: str,
    ) -> torch.Tensor:
        """state (B, K, D); task_vec (B, 512); t_cond (B,) or (B, T) -> (B[, T], H)."""
        if embodiment not in self.x_embedder:
            raise KeyError(
                f"no state embedder for embodiment {embodiment!r}; "
                f"known: {sorted(self.x_embedder)}"
            )
        embedder = self.x_embedder[embodiment]
        model_dtype = embedder.weight.dtype
        cond_dtype = self.cond_proj[0].weight.dtype
        st_vec = embedder(self._flatten_state(state, embodiment).to(model_dtype))
        task_vec_h = self.task_to_hidden(
            task_vec.to(self.task_to_hidden.weight.dtype)
        ).to(model_dtype)
        t_vec = self.t_embedder(t_cond.to(model_dtype))
        cond_parts = [st_vec, task_vec_h, t_vec]
        if t_vec.ndim == 3:
            T = t_vec.shape[1]
            cond_parts = [
                p.unsqueeze(1).expand(-1, T, -1) if p.ndim == 2 else p
                for p in cond_parts
            ]
        cond_concat = torch.cat(cond_parts, dim=-1).to(cond_dtype)
        if cond_dtype == torch.float32 and cond_concat.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self.cond_proj(cond_concat).to(model_dtype)
        return self.cond_proj(cond_concat).to(model_dtype)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        c: torch.Tensor,
        vision_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.y_embedder(x_t) + self.pos_embed.data[:, : x_t.shape[1], :]
        for block in self.blocks:
            z = block(z, c, vision_tokens, prefix_mask)
        return self.final_layer(z, c, prefix_mask)

    def sample_time(self, n: int, device, dtype) -> torch.Tensor:
        if self.time_dist == "beta":
            t = torch.distributions.Beta(1.5, 1.0).sample((n,)).to(device)
        else:
            t = torch.rand(n, device=device)
        return t.to(dtype)

    def task_vectors(self, data: dict) -> torch.Tensor:
        vec = data.get("task_vec_clip")
        if vec is None:
            vec = self.task_encoder(data["prompts"])
        return vec.to(self.y_embedder.weight.device)

    # -- training / inference ---------------------------------------------

    def compute_loss(self, data: dict) -> torch.Tensor:
        """Flow-matching loss, with ABC's state masking and prefix conditioning.

        ``x_t = (1 - t) * a + t * noise`` with target ``noise - a``, the
        convention the repo's other flow heads use (``fm_policy.py``).
        """
        actions = data["action"]
        state = data["state"]
        embodiment = data["embodiment_name"]
        N, T_chunk, _ = actions.shape
        if T_chunk != self.action_horizon:
            raise ValueError(
                f"action chunk has {T_chunk} steps, policy expects "
                f"{self.action_horizon}"
            )
        device = actions.device

        state_is_masked = None
        if self.training and self.mask_state_ratio > 0:
            state_is_masked = torch.rand(N, device=device) < self.mask_state_ratio
            state = torch.where(
                state_is_masked.view(-1, *([1] * (state.ndim - 1))),
                torch.zeros_like(state),
                state,
            )

        noise = torch.randn_like(actions)
        t = self.sample_time(N, device, actions.dtype).view(N, 1, 1)

        prefix_mask_expanded = None
        if self.max_action_prefix > 0:
            apply_prefix = torch.rand(N, device=device) < self.prefix_conditioning_prob
            if state_is_masked is not None:
                apply_prefix = apply_prefix & ~state_is_masked
            delay = torch.randint(0, self.max_action_prefix, (N,), device=device)
            delay = torch.where(apply_prefix, delay, torch.zeros_like(delay))
            prefix_mask = torch.arange(T_chunk, device=device)[None, :] < delay[:, None]
            prefix_mask_expanded = prefix_mask.unsqueeze(-1)
            t_per_pos = torch.where(prefix_mask_expanded, torch.zeros_like(t), t)
        else:
            t_per_pos = t

        x_t = (1 - t_per_pos) * actions + t_per_pos * noise
        if self.prefix_noise_scale > 0.0 and prefix_mask_expanded is not None:
            x_t = x_t + prefix_mask_expanded.to(x_t.dtype) * (
                torch.randn_like(x_t) * self.prefix_noise_scale
            )

        vision_tokens = self.build_vision_tokens(data["images"])
        t_cond = (
            t_per_pos.squeeze(-1) if prefix_mask_expanded is not None else t[:, 0, 0]
        )
        c = self.compute_cond(state, self.task_vectors(data), t_cond, embodiment)
        v_t = self.predict_velocity(x_t, c, vision_tokens)

        u_t = noise - actions
        mask = data["loss_mask"].to(v_t.dtype).expand_as(v_t)
        if prefix_mask_expanded is not None:
            # The conditioned prefix positions are given, not predicted.
            mask = mask * (~prefix_mask_expanded).to(v_t.dtype)
        return ((v_t - u_t) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(
        self,
        data: dict,
        generator: Optional[torch.Generator] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Euler flow integration from noise (t = 1) to actions (t = 0)."""
        state = data["state"]
        embodiment = data["embodiment_name"]
        B = state.shape[0]
        device = state.device
        model_dtype = self.y_embedder.weight.dtype
        n = self.num_inference_steps if num_steps is None else int(num_steps)
        x_t = torch.randn(
            (B, self.action_horizon, self.action_width),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )
        # Vision tokens and the task vector are static across the integration.
        vision_tokens = self.build_vision_tokens(data["images"])
        task_vec = self.task_vectors(data)
        dt = -1.0 / n
        for step in range(n):
            t = torch.full((B,), 1.0 + step * dt, device=device, dtype=model_dtype)
            c = self.compute_cond(state, task_vec, t, embodiment)
            v = self.predict_velocity(x_t, c, vision_tokens)
            x_t = x_t + v * dt
        return x_t

    @torch.no_grad()
    def sample_with_prefix(
        self,
        data: dict,
        action_prefix: torch.Tensor,
        prefix_length: int,
        generator: Optional[torch.Generator] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """ABC's real-time-chunking sampler: the first ``prefix_length`` steps
        are held at ``action_prefix`` and modulated as already-denoised."""
        state = data["state"]
        embodiment = data["embodiment_name"]
        B = state.shape[0]
        device = state.device
        model_dtype = self.y_embedder.weight.dtype
        n = self.num_inference_steps if num_steps is None else int(num_steps)
        x_t = torch.randn(
            (B, self.action_horizon, self.action_width),
            generator=generator,
            device=device,
            dtype=model_dtype,
        )
        action_prefix = action_prefix.to(device=device, dtype=model_dtype)
        prefix_pos = torch.arange(self.action_horizon, device=device) < int(
            prefix_length
        )
        prefix_mask = prefix_pos.view(1, -1, 1).expand_as(x_t)
        prefix_t_mask = prefix_pos.view(1, -1).expand(B, self.action_horizon)
        x_t = torch.where(prefix_mask, action_prefix, x_t)

        vision_tokens = self.build_vision_tokens(data["images"])
        task_vec = self.task_vectors(data)
        dt = -1.0 / n
        for step in range(n):
            t = torch.full((B,), 1.0 + step * dt, device=device, dtype=model_dtype)
            t_pair = torch.stack([torch.zeros_like(t), t], dim=1)
            c_pair = self.compute_cond(state, task_vec, t_pair, embodiment)
            v = self.predict_velocity(x_t, c_pair, vision_tokens, prefix_t_mask)
            x_t = x_t + v * dt
            x_t = torch.where(prefix_mask, action_prefix, x_t)
        return x_t
