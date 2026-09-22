"""RDT (Robotics Diffusion Transformer) on the HPT chassis.

``RDT`` is ``HPT`` with a different policy network: the same batch handling,
stems, encoders, denoising heads, eval and config layout, but no Perceiver
pooling. Every stem hands its full token sequence to the head, whose denoiser
(``rdt_nets.RDTDenoiser``) runs the noisy action chunk through the trunk -- the
DiT blocks (``rdt_nets.RDTBackbone``), shared across embodiments, which
cross-attend to the language and image tokens in alternating blocks.
"""

import torch
import torch.nn as nn

from egomimic.algo.hpt import HPT, HPTModel
from egomimic.models.rdt_nets import (
    RDTBackbone,
    RDTConditions,
    RDTMemory,
    mlp_gelu,
    sincos_2d,
)
from egomimic.rldb.embodiment.embodiment import (
    HISTORY_MASK_SUFFIX,
    IMAGE_HISTORY_SUFFIX,
    IMAGE_MEMORY_SUFFIX,
)
from egomimic.utils.tensor_utils import get_sinusoid_encoding_table

INIT_STD = 0.02


class RDTModel(HPTModel):
    """Stems -> ``RDTConditions`` -> head -> shared ``RDTBackbone``. Streams: a
    modality with an encoder is an image, a stem with ``forward_with_mask`` is
    language, the rest is state.

    ``cond_drop`` (RDT's ``cond_mask_prob``, per stream, training only): the
    state is replaced by a learned null token; the prompt by ``""``, i.e. the
    no-prompt deployment condition; a camera by a learned null token, never
    every camera of a sample, so ``img`` is inert with a single camera.

    ``image_history`` is the number of frames per camera (axis 1 of the image
    input, current frame last); ``image_history_dropout`` replaces the past
    frames by the current one, which is also what an episode start looks like.

    ``memory`` (``RDTMemory`` kwargs, its ``encoder`` a frozen image stem, plus
    ``dropout``) adds long-range memory tokens to the DiT's self-attention
    prefix from the batch's ``memory`` window; padded steps are masked out,
    and ``dropout`` masks a sample's whole memory (training only).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 1.0,
        cond_drop: dict | None = None,
        image_history: int = 1,
        image_history_dropout: float = 0.0,
        memory: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, no_trunk=True, token_postprocessing="no-op"
        )
        self.trunk = nn.ModuleDict(
            {"trunk": RDTBackbone(embed_dim, depth, num_heads, mlp_ratio)}
        )
        self.cond_drop = {"state": 0.0, "lang": 0.0, "img": 0.0, **(cond_drop or {})}
        self.image_history = int(image_history)
        self.image_history_dropout = float(image_history_dropout)
        self.stem_modality = {}
        self.long_memory = None
        self.memory_dropout = 0.0
        if memory is not None:
            memory = dict(memory)
            self.memory_dropout = float(memory.pop("dropout", 0.0))
            memory.setdefault("num_heads", num_heads)
            self.long_memory = RDTMemory(hidden_dim=embed_dim, **memory)

    def _create_policy_trunk(self, *args, **kwargs):
        return nn.ModuleDict()

    def init_domain_stem(self, domain_name, stem_spec):
        # no init_cross_attn: the stems' Perceiver latents would be dead weights
        stem_spec = {k: v for k, v in stem_spec.items() if v is not None}
        self.stem_spec[domain_name] = stem_spec
        self.modalities[domain_name] = list(stem_spec)
        for modality, stem in stem_spec.items():
            self.stems[f"{domain_name}_{modality}"] = stem
            self.stem_modality[f"{domain_name}_{modality}"] = modality

    def _stream(self, modality, stem) -> str:
        if modality in self.encoders:
            return "img"
        return "lang" if hasattr(stem, "forward_with_mask") else "state"

    def finalize_modules(self):
        D = self.embed_dim
        streams = {
            name: self._stream(self.stem_modality[name], stem)
            for name, stem in self.stems.items()
        }
        cameras = [name for name, s in streams.items() if s == "img"]
        # RDT's mlp2x_gelu language adaptor; image / state adaptors are the stems
        self.lang_adaptors = nn.ModuleDict(
            {name: mlp_gelu(D, D, 2) for name, s in streams.items() if s == "lang"}
        )
        super().finalize_modules()
        self.camera_embed = nn.ParameterDict(
            {name: torch.zeros(1, 1, 1, D) for name in cameras}
        )
        self.frame_embed = nn.Parameter(torch.zeros(1, self.image_history, 1, D))
        # Only what can be used: an unused parameter crashes plain DDP.
        if self.cond_drop["state"] > 0 and "state" in streams.values():
            self.null_state = nn.Parameter(torch.randn(1, 1, D) * INIT_STD)
        if self.cond_drop["img"] > 0 and len(cameras) > 1:
            self.null_img = nn.Parameter(torch.randn(1, 1, D) * INIT_STD)
        self.trunk["trunk"].initialize_weights()
        for head in self.heads.values():
            init = getattr(getattr(head, "model", None), "initialize_weights", None)
            if callable(init):
                init()

    def _drop(self, prob: float, batch_size: int, device):
        """(B,) bool: which samples lose a condition; None outside training."""
        if not (self.training and prob > 0):
            return None
        return torch.rand(batch_size, device=device) < prob

    def _image_tokens(self, name, modality, stem, x):
        """(B, T, N, 3, H, W) frames, current last -> (B, T * N * patches, D)."""
        B, T = x.shape[:2]
        if T != self.image_history:
            raise ValueError(
                f"'{modality}' carries {T} frames but trunk.image_history="
                f"{self.image_history}; set it to match the data config's "
                "image_history_gap_s (2 frames when set, else 1)"
            )
        drop = self._drop(self.image_history_dropout, B, x.device) if T > 1 else None
        if drop is not None:
            x = torch.where(drop.view(B, 1, 1, 1, 1, 1), x[:, -1:].expand_as(x), x)
        encoder = self.encoders[modality]
        tokens = stem(encoder(x))
        tokens = tokens.reshape(B, T, -1, tokens.shape[-1])
        # DINOv3 is RoPE-only: its tokens carry no absolute position.
        grid = getattr(encoder, "grid_size", None)
        if grid is not None and grid[0] * grid[1] == tokens.shape[2]:
            pos = sincos_2d(tokens.shape[-1], grid)
        else:
            pos = get_sinusoid_encoding_table(0, tokens.shape[2], tokens.shape[-1])[0]
        tokens = tokens + pos.to(tokens) + self.frame_embed + self.camera_embed[name]
        return tokens.reshape(B, -1, tokens.shape[-1])

    def _drop_cameras(self, img: list) -> list:
        """Per-camera dropout that never blinds a sample completely."""
        if len(img) < 2 or not (self.training and self.cond_drop["img"] > 0):
            return img
        B, device = len(img[0]), img[0].device
        drop = torch.rand(B, len(img), device=device) < self.cond_drop["img"]
        keep = torch.randint(len(img), (B,), device=device)
        blind = drop.all(dim=1)
        drop[blind, keep[blind]] = False
        return [
            torch.where(drop[:, i, None, None], self.null_img.to(tokens), tokens)
            for i, tokens in enumerate(img)
        ]

    def forward_features(self, domain, data):
        if "fps" not in data:
            raise ValueError("RDT needs the per-sample 'fps' (ZarrDataset emits it)")
        img, state, lang, lang_mask = [], [], [], []
        for modality in self.modalities.get(domain, []) + self.shared_keys:
            if modality not in data:
                continue
            owner = "shared" if modality in self.shared_keys else domain
            name = f"{owner}_{modality}"
            stem = self.stems[name]
            stream = self._stream(modality, stem)
            x = data[modality]

            if stream == "lang":
                drop = self._drop(self.cond_drop["lang"], len(x), self.device)
                if drop is not None:
                    x = ["" if d else p for p, d in zip(x, drop.tolist())]
                tokens, mask = stem.forward_with_mask(x)
                if not mask.any(dim=1).all():
                    raise ValueError(f"'{modality}': a prompt encoded to zero tokens")
                lang.append(self.lang_adaptors[name](tokens))
                lang_mask.append(mask)
            elif stream == "img":
                img.append(self._image_tokens(name, modality, stem, x))
            else:
                tokens = stem(x)
                state.append(tokens.reshape(len(tokens), -1, tokens.shape[-1]))

        if not img:
            raise ValueError(f"RDT needs at least one camera for domain '{domain}'")
        if (self.long_memory is None) != ("memory" not in data):
            raise ValueError(
                "trunk.memory and the data config's key_map.image_memory must be "
                "set together (model memory: "
                f"{self.long_memory is not None}, batch memory: {'memory' in data})"
            )
        img = torch.cat(self._drop_cameras(img), dim=1)
        cond = RDTConditions(
            img=img,
            freq=data["fps"].reshape(-1).expand(len(img)),
            backbone=self.trunk["trunk"],
        )
        if state:
            cond.state = torch.cat(state, dim=1)
            drop = self._drop(self.cond_drop["state"], len(img), img.device)
            if drop is not None:
                null = self.null_state.to(cond.state)
                cond.state = torch.where(drop[:, None, None], null, cond.state)
        if lang:
            cond.lang = torch.cat(lang, dim=1)
            cond.lang_mask = torch.cat(lang_mask, dim=1)
        if self.long_memory is not None:
            mask = data["memory_mask"].bool()
            drop = self._drop(self.memory_dropout, len(mask), mask.device)
            if drop is not None:
                mask = mask & ~drop[:, None]
            # after state dropout, so a fused memory loses its proprio with it
            cond.memory, cond.memory_mask = self.long_memory(
                data["memory"], mask, cond.state
            )
        return cond, None


class RDT(HPT):
    def __init__(self, *args, **kwargs):
        for flag in ("ot", "freeze_repr"):
            if kwargs.get(flag):
                raise ValueError(f"RDT has no HPT trunk; '{flag}' is HPT-only")
        super().__init__(*args, **kwargs)

    def _build_policy(self, trunk: dict) -> RDTModel:
        return RDTModel(**trunk)

    def _robomimic_to_hpt_data(
        self,
        batch,
        cam_keys,
        proprio_keys,
        lang_keys,
        ac_key,
        aux_ac_keys=[],
        domain="",
    ):
        """HPT's layout plus ``fps``; per camera with a ``*_hist`` twin, the
        (past, current) frame pair on the image input's time axis; and a
        ``*_mem`` window as ``memory`` / ``memory_mask``, un-augmented (its
        tower is frozen and the stem normalizes)."""
        past = {
            key[: -len(IMAGE_HISTORY_SUFFIX)]: key
            for key in cam_keys
            if key.endswith(IMAGE_HISTORY_SUFFIX) and key in batch
        }
        memory = [k for k in cam_keys if k.endswith(IMAGE_MEMORY_SUFFIX) and k in batch]
        if len(memory) > 1:
            raise ValueError(f"RDT takes one memory camera, got {memory}")
        single = [
            k
            for k in cam_keys
            if k not in past and k not in past.values() and k not in memory
        ]
        data = super()._robomimic_to_hpt_data(
            batch, single, proprio_keys, lang_keys, ac_key, aux_ac_keys, domain=domain
        )
        for key, past_key in past.items():
            short = key.rsplit(".", 1)[-1]
            # (B * 2, 3, H, W), each sample's (past, current) adjacent: one
            # jitter draw per sample covers both frames, and every op -- a
            # contrast mean, a crop, a resize -- still sees one frame.
            pair = torch.stack([batch[past_key], batch[key]], dim=1)
            n = pair.shape[0]
            pair = self._apply_image_augs(pair.flatten(0, 1), short, frames=2)
            data[short] = pair.reshape(n, 2, *pair.shape[1:]).unsqueeze(2)
        for key in memory:
            data["memory"] = batch[key]
            data["memory_mask"] = batch[f"{key}{HISTORY_MASK_SUFFIX}"] > 0.5
        data["fps"] = batch["fps"]
        return data
