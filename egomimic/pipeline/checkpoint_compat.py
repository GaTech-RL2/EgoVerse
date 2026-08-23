"""Narrow rollout compatibility for the frozen Fold DDPM checkpoint.

The Pipeline consolidation intentionally removed the old ``stages_io``,
``stages_seq``, and ``stages_flow`` modules.  Lightning checkpoints serialize
their Hydra targets, though, so the verified 2026-08-21 Fold checkpoint needs
the small subset below to reconstruct its original module tree.  This is a
serialization boundary, not a second active Pipeline stack:

* only the exact known model-config fingerprint is accepted;
* unsupported legacy heads (dual stream, MoE, heterogeneous actions) fail;
* target migration is in-memory, then saved as a stripped rollout artifact;
* module attribute names match the frozen source so strict state loading still
  proves that the compatibility graph is identical.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from egomimic.models.denoising_nets import SinusoidalPosEmb
from egomimic.pipeline import packed
from egomimic.pipeline.core import Stage

COMPAT_VERSION = "fold-dn-single-cart-20260821-v1"
# Canonical JSON SHA256 of model.robomimic_model in both the epoch-3
# checkpoint and its complete .hydra/config.yaml (including both loss domains).
LEGACY_FOLD_MODEL_SHA256 = (
    "d71510d2191da5deaa3d82df03d36650d7968ea3dedc17d49bf7973cd6da958a"
)
LEGACY_EMBODIMENT_IDS = {8: 6, 18: 3}

_LEGACY_TARGETS = (
    "egomimic.pipeline.stages_io.NormalObsExpand",
    "egomimic.pipeline.stages_io.ObsEncoders",
    "egomimic.pipeline.stages_seq.Rename",
    "egomimic.pipeline.stages_seq.ObsStack",
    "egomimic.pipeline.stages_io.NormalObsCollapse",
    "egomimic.pipeline.stages_flow.DiffusionHead",
    "egomimic.pipeline.stages_flow.MaskedActionLoss",
)
_COMPAT_TARGETS = tuple(
    f"egomimic.pipeline.checkpoint_compat.{target.rsplit('.', 1)[-1]}"
    for target in _LEGACY_TARGETS
)
_REMOVED_STAGE_PREFIXES = (
    "egomimic.pipeline.stages_io.",
    "egomimic.pipeline.stages_seq.",
    "egomimic.pipeline.stages_flow.",
)


def _plain_config(tree):
    if isinstance(tree, DictConfig):
        return OmegaConf.to_container(tree, resolve=True)
    return copy.deepcopy(tree)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_config(config: dict) -> dict:
    try:
        return config["model"]["robomimic_model"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Fold compatibility requires model.robomimic_model in config_tree"
        ) from exc


def _legacy_targets(model: dict) -> tuple[str, ...]:
    stages = model.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Fold compatibility requires a concrete stages list")
    return tuple(str(stage.get("_target_", "")) for stage in stages)


def _remap_norm_stats_state(state: dict) -> dict:
    if not isinstance(state, dict):
        raise ValueError("Fold compatibility requires norm_stats_state")
    out = copy.deepcopy(state)
    raw_embodiments = list(out.get("embodiments", []))
    try:
        observed = {int(value) for value in raw_embodiments}
    except (TypeError, ValueError) as exc:
        raise ValueError("Fold norm-stat embodiment IDs must be integers") from exc
    expected = set(LEGACY_EMBODIMENT_IDS)
    if observed != expected:
        raise ValueError(
            "Fold compatibility expected legacy norm-stat embodiment IDs "
            f"{sorted(expected)}, got {sorted(observed)}"
        )
    out["embodiments"] = sorted(LEGACY_EMBODIMENT_IDS[value] for value in observed)

    for field in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        table = out.get(field)
        if not isinstance(table, dict):
            raise ValueError(f"Fold norm_stats_state has no {field!r} mapping")
        remapped = {}
        for raw_key, value in table.items():
            try:
                old_id = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Fold norm_stats_state.{field} has non-integer key {raw_key!r}"
                ) from exc
            if old_id not in LEGACY_EMBODIMENT_IDS:
                raise ValueError(
                    f"Fold norm_stats_state.{field} has unexpected embodiment "
                    f"ID {old_id}"
                )
            new_id = LEGACY_EMBODIMENT_IDS[old_id]
            if new_id in remapped:
                raise ValueError(
                    f"Fold norm_stats_state.{field} remap collides at ID {new_id}"
                )
            remapped[new_id] = value
        if set(remapped) != set(LEGACY_EMBODIMENT_IDS.values()):
            raise ValueError(
                f"Fold norm_stats_state.{field} does not cover both embodiments"
            )
        out[field] = remapped
    return out


def _state_schema(checkpoint: dict) -> list[tuple[str, list[int], str]]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Fold compatibility requires a non-empty state_dict")
    schema = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Checkpoint state_dict value {key!r} is not a tensor")
        schema.append((str(key), list(value.shape), str(value.dtype)))
    return schema


def _artifact_fingerprint(checkpoint_path: str, checkpoint: dict) -> str:
    stat = os.stat(checkpoint_path)
    payload = {
        "compat_version": COMPAT_VERSION,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "state_schema": _state_schema(checkpoint),
    }
    return _canonical_sha256(payload)


def _artifact_paths(checkpoint_path: str, fingerprint: str) -> tuple[Path, Path]:
    source = Path(checkpoint_path)
    suffix = source.suffix or ".ckpt"
    stem = source.name[: -len(suffix)] if source.suffix else source.name
    artifact = source.with_name(
        f"{stem}.rollout-{COMPAT_VERSION}-{fingerprint[:12]}{suffix}"
    )
    return artifact, Path(f"{artifact}.json")


def _valid_existing_artifact(artifact: Path, marker: Path, fingerprint: str) -> bool:
    if not artifact.is_file() or artifact.stat().st_size <= 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    return (
        metadata.get("compat_version") == COMPAT_VERSION
        and metadata.get("fingerprint") == fingerprint
        and metadata.get("artifact_size") == artifact.stat().st_size
    )


def _write_rollout_artifact(
    artifact: Path,
    marker: Path,
    checkpoint: dict,
    fingerprint: str,
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(f".{artifact.name}.{os.getpid()}.partial")
    marker_temporary = marker.with_name(f".{marker.name}.{os.getpid()}.partial")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, artifact)
        metadata = {
            "compat_version": COMPAT_VERSION,
            "fingerprint": fingerprint,
            "artifact_size": artifact.stat().st_size,
        }
        marker_temporary.write_text(json.dumps(metadata, sort_keys=True) + "\n")
        os.replace(marker_temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
        marker_temporary.unlink(missing_ok=True)


def prepare_legacy_fold_rollout_checkpoint(
    checkpoint_path: str,
    checkpoint: dict | None = None,
) -> str:
    """Create and return a strict, stripped compatibility checkpoint.

    The source checkpoint is never modified.  ``checkpoint`` should be the
    dictionary the rollout dispatcher already loaded; accepting it avoids a
    second multi-gigabyte deserialization.
    """

    if checkpoint is None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Fold compatibility requires checkpoint hyper_parameters")
    config = _plain_config(hyper_parameters.get("config_tree"))
    model = _model_config(config)
    targets = _legacy_targets(model)

    if targets == _COMPAT_TARGETS:
        marker = checkpoint.get("egomimic_rollout_compat")
        if not isinstance(marker, dict) or marker.get("version") != COMPAT_VERSION:
            raise ValueError(
                "Refusing an unmarked checkpoint with compatibility targets"
            )
        return checkpoint_path
    if targets != _LEGACY_TARGETS:
        legacy = [
            target for target in targets if target.startswith(_REMOVED_STAGE_PREFIXES)
        ]
        if legacy:
            raise ValueError(
                "Unsupported legacy Pipeline stage graph; compatibility accepts "
                f"only the verified Fold graph, got {targets}"
            )
        return checkpoint_path

    actual_model_sha = _canonical_sha256(model)
    if actual_model_sha != LEGACY_FOLD_MODEL_SHA256:
        raise ValueError(
            "Legacy Fold model config fingerprint mismatch: expected "
            f"{LEGACY_FOLD_MODEL_SHA256}, got {actual_model_sha}. Refusing to "
            "guess at checkpoint compatibility."
        )

    stages = model["stages"]
    for stage, target in zip(stages, _COMPAT_TARGETS):
        stage["_target_"] = target
    chunk_len = int(stages[5]["chunk_len"])
    if chunk_len != 100:
        raise ValueError(f"Verified Fold checkpoint chunk_len is 100, got {chunk_len}")
    model["action_horizon"] = chunk_len

    migrated_hparams = dict(hyper_parameters)
    migrated_hparams["config_tree"] = OmegaConf.create(config)
    migrated_hparams["norm_stats_state"] = _remap_norm_stats_state(
        hyper_parameters.get("norm_stats_state")
    )

    stripped = {
        key: checkpoint[key]
        for key in (
            "epoch",
            "global_step",
            "pytorch-lightning_version",
            "hparams_name",
            "hyper_parameters_name",
        )
        if key in checkpoint
    }
    stripped["state_dict"] = checkpoint["state_dict"]
    stripped["hyper_parameters"] = migrated_hparams
    stripped["egomimic_rollout_compat"] = {
        "version": COMPAT_VERSION,
        "source_model_config_sha256": actual_model_sha,
    }

    fingerprint = _artifact_fingerprint(checkpoint_path, checkpoint)
    artifact, marker = _artifact_paths(checkpoint_path, fingerprint)
    if not _valid_existing_artifact(artifact, marker, fingerprint):
        _write_rollout_artifact(artifact, marker, stripped, fingerprint)
    print(
        f"[rollout] Migrated frozen Fold checkpoint targets and norm IDs to {artifact}"
    )
    return str(artifact)


# ---------------------------------------------------------------------------
# Exact frozen stage implementations used by the accepted checkpoint.
# ---------------------------------------------------------------------------


class ObsEncoders(Stage):
    reads = ["obs/*", "cu_seqlens", "embodiment"]
    writes = ["A", "S", "time_pos"]

    def __init__(
        self,
        agnostic: nn.Module,
        specific: List[nn.Module],
        expose_tokens: bool = False,
    ):
        super().__init__()
        self.agnostic = agnostic
        self.specific = nn.ModuleList(specific)
        self.expose_tokens = bool(expose_tokens)
        if self.expose_tokens:
            self.writes = list(type(self).writes) + ["obs_tokens"]

    def forward(self, batch: dict) -> dict:
        obs_packed = {
            key.split("/", 1)[1]: value
            for key, value in batch.items()
            if key.startswith("obs/")
        }
        ref = next(value for value in obs_packed.values() if torch.is_tensor(value))
        total, device = ref.shape[0], ref.device
        dtype = torch.float32 if not ref.is_floating_point() else ref.dtype
        actions = batch.get("actions")
        if actions is None:
            actions = torch.zeros((total, 1), device=device, dtype=dtype)
        cu = batch["cu_seqlens"].to(device=device, dtype=torch.long)
        embodiment = batch["embodiment"]
        for module in self.modules():
            if getattr(module, "crop_scope", None) == "episode":
                module._episode_cu = cu
        call = dict(
            actions_packed=actions,
            obs_packed=obs_packed,
            cu_seqlens=cu,
            T_total=total,
            device=device,
            dtype=actions.dtype,
            embodiment_id=embodiment,
        )
        if self.expose_tokens:
            token_sets = []
            a_fused, a_tokens = self.agnostic.forward_packed_both(
                obs_packed=obs_packed,
                T_total=total,
                embodiment_id=embodiment,
            )
            batch["A"] = a_fused.to(actions.dtype)
            if a_tokens is not None:
                token_sets.append(a_tokens)
            specific = None
            for module in self.specific:
                s_fused, s_tokens = module.forward_packed_both(
                    obs_packed=obs_packed,
                    T_total=total,
                    embodiment_id=embodiment,
                )
                specific = s_fused if specific is None else specific + s_fused
                if s_tokens is not None:
                    token_sets.append(s_tokens)
            batch["S"] = specific.to(actions.dtype)
            batch["time_pos"] = packed.frame_idx(cu)
            if token_sets:
                batch["obs_tokens"] = torch.cat(token_sets, dim=1).to(actions.dtype)
            return batch
        specific = None
        for module in self.specific:
            encoded = module.forward_packed(**call)
            specific = encoded if specific is None else specific + encoded
        batch["A"] = self.agnostic.forward_packed(**call)
        batch["S"] = specific
        batch["time_pos"] = packed.frame_idx(cu)
        return batch


class NormalObsExpand(Stage):
    reads = ["obs/*"]
    writes = ["cu_seqlens", "max_seq_len", "_normal_target"]

    def __init__(self, n_obs_steps: int = 2):
        super().__init__()
        self.n = int(n_obs_steps)
        self.rollout_obs_steps = self.n

    def forward(self, batch: dict) -> dict:
        n = self.n
        obs_keys = [
            key
            for key in batch
            if key.startswith("obs/") and torch.is_tensor(batch[key])
        ]
        if not obs_keys:
            raise ValueError("NormalObsExpand: no obs/* tensors in batch.")
        ref = batch[obs_keys[0]]
        batch_size = ref.shape[0]
        for key in obs_keys:
            value = batch[key]
            if value.ndim < 2 or value.shape[1] != n:
                raise ValueError(
                    f"NormalObsExpand: {key} has shape {tuple(value.shape)}; "
                    f"expected (B, {n}, ...). Every obs key must be fetched "
                    f"with horizon=n_obs_steps={n} in the keymap."
                )
            batch[key] = value.reshape(batch_size * n, *value.shape[2:])
        device = ref.device
        target = batch.pop("actions", None)
        if target is None and "rollout_t" not in batch:
            raise ValueError(
                "NormalObsExpand: batch has no 'actions' chunk outside rollout."
            )
        batch["_normal_target"] = target
        batch["cu_seqlens"] = torch.arange(
            0,
            batch_size * n + 1,
            n,
            device=device,
            dtype=torch.long,
        )
        batch["max_seq_len"] = n
        return batch


class NormalObsCollapse(Stage):
    writes = ["target", "cu_seqlens", "max_seq_len", "frame_idx", "time_pos"]

    def __init__(self, keys: List[str]):
        super().__init__()
        self.keys = [str(key) for key in keys]
        self.reads = list(self.keys) + ["cu_seqlens", "_normal_target"]

    def forward(self, batch: dict) -> dict:
        cu = batch["cu_seqlens"]
        last = (cu[1:] - 1).to(dtype=torch.long)
        for key in self.keys:
            batch[key] = batch[key].index_select(0, last.to(batch[key].device))
        batch_size, device = last.numel(), last.device
        batch["cu_seqlens"] = torch.arange(
            0, batch_size + 1, device=device, dtype=torch.long
        )
        batch["max_seq_len"] = 1
        batch["frame_idx"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        batch["time_pos"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        target = batch.pop("_normal_target")
        if target is not None:
            batch["target"] = target
        return batch


def _grid_keys(key: str):
    return f"cu_seqlens@{key}", f"max_seq_len@{key}", f"time_pos@{key}"


def _write_grid(batch: dict, key: str, cu, maximum, time_position):
    cu_key, maximum_key, time_key = _grid_keys(key)
    batch[cu_key], batch[maximum_key], batch[time_key] = (
        cu,
        int(maximum),
        time_position,
    )


def _within_episode_time_pos(cu: torch.Tensor, total: int, device) -> torch.Tensor:
    cu = cu.to(device=device, dtype=torch.long)
    segment = packed.episode_ids(cu)
    return torch.arange(total, device=device, dtype=torch.long) - cu[:-1][segment]


class Rename(Stage):
    def __init__(self, mapping: dict):
        super().__init__()
        self.mapping = {str(key): str(value) for key, value in dict(mapping).items()}
        self.reads = list(self.mapping)
        self.writes = list(self.mapping.values())

    def forward(self, batch: dict) -> dict:
        for source, destination in self.mapping.items():
            batch[destination] = batch[source]
            cu_key, maximum_key, time_key = _grid_keys(source)
            if cu_key in batch:
                _write_grid(
                    batch,
                    destination,
                    batch[cu_key],
                    batch[maximum_key],
                    batch[time_key],
                )
        return batch


class ObsStack(Stage):
    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
        n_obs_steps: int = 2,
    ):
        super().__init__()
        self.in_keys = [str(key) for key in in_keys]
        self.out_keys = [str(key) for key in out_keys]
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError("ObsStack: one out_key per in_key.")
        self.n = int(n_obs_steps)
        self.reads = list(self.in_keys) + ["cu_seqlens"]
        self.writes = list(self.out_keys)

    def forward(self, batch: dict) -> dict:
        ref = batch[self.in_keys[0]]
        total, device = ref.shape[0], ref.device
        cu = batch["cu_seqlens"].to(device=device, dtype=torch.long)
        position = _within_episode_time_pos(cu, total, device)
        index = torch.arange(total, device=device)
        gathers = [
            index - torch.minimum(torch.full_like(position, offset), position)
            for offset in range(self.n - 1, -1, -1)
        ]
        for source, destination in zip(self.in_keys, self.out_keys):
            value = batch[source]
            batch[destination] = torch.cat(
                [value.index_select(0, gather) for gather in gathers], dim=-1
            )
            cu_key, maximum_key, time_key = _grid_keys(source)
            if destination != source and cu_key in batch:
                _write_grid(
                    batch,
                    destination,
                    batch[cu_key],
                    batch[maximum_key],
                    batch[time_key],
                )
        return batch


def _cosine_alphas_cumprod(steps_count: int, offset: float = 0.008):
    steps = torch.arange(steps_count + 1, dtype=torch.float64)
    value = (
        torch.cos(((steps / steps_count) + offset) / (1 + offset) * math.pi / 2) ** 2
    )
    cumulative = (value / value[0]).clamp(1e-8, 1.0)
    return cumulative[1:].float()


class _SingleStreamBlockAdaLN(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        ffn_mult: int = 4,
        moe_experts: int = 0,
        moe_top_k: int = 4,
        moe_d_expert: Optional[int] = None,
        moe_aux_weight: float = 0.01,
    ):
        super().__init__()
        del moe_top_k, moe_d_expert, moe_aux_weight
        if moe_experts:
            raise ValueError("Frozen Fold compatibility does not support MoE")
        self.h, self.d = int(n_heads), int(d)
        if self.d % self.h:
            raise ValueError("d_model must be divisible by n_heads")
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d),
            nn.GELU(),
            nn.Linear(ffn_mult * d, d),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, value, condition):
        total, length, _ = value.shape
        head_dim = self.d // self.h
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(condition).chunk(6, -1)
        query, key, content = self.qkv(self.norm1(value) * (1 + scale1) + shift1).chunk(
            3, -1
        )
        query, key, content = [
            tensor.reshape(total, length, self.h, head_dim).transpose(1, 2)
            for tensor in (query, key, content)
        ]
        attended = F.scaled_dot_product_attention(query, key, content)
        attended = attended.transpose(1, 2).reshape(total, length, self.d)
        value = value + gate1 * self.out(attended)
        value = value + gate2 * self.ffn(self.norm2(value) * (1 + scale2) + shift2)
        return value


class SingleStreamDenoiserV2(nn.Module):
    def __init__(
        self,
        d_a_in: int,
        d_s_in: Optional[int],
        action_dim: int,
        chunk_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        n_positions: Optional[int] = None,
        moe_experts: int = 0,
        moe_top_k: int = 4,
        moe_d_expert: Optional[int] = None,
        moe_aux_weight: float = 0.01,
    ):
        super().__init__()
        chunk, action = int(chunk_len), int(action_dim)
        length = int(n_positions) if n_positions else chunk
        width = int(d_model)
        self.C, self.D, self.L = chunk, action, length
        self.in_x = nn.Linear(action, width)
        self.cond_a = nn.Linear(int(d_a_in), width)
        self.cond_s = nn.Linear(int(d_s_in), width) if d_s_in else None
        self.temb = nn.Sequential(
            SinusoidalPosEmb(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.pos = nn.Parameter(torch.zeros(length, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.vout = nn.Linear(width, action)
        self.blocks = nn.ModuleList(
            _SingleStreamBlockAdaLN(
                width,
                n_heads,
                ffn_mult,
                moe_experts=moe_experts,
                moe_top_k=moe_top_k,
                moe_d_expert=moe_d_expert,
                moe_aux_weight=moe_aux_weight,
            )
            for _ in range(int(n_layers))
        )
        self.norm_f = nn.LayerNorm(width)
        self.fmod = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.fmod[1].weight)
        nn.init.zeros_(self.fmod[1].bias)

    def _temb(self, timestep, total, length):
        if timestep.dim() == 1:
            return self.temb(timestep)[:, None, :].expand(total, length, -1)
        return self.temb(timestep.reshape(-1)).reshape(total, length, -1)

    def forward(self, noisy_action, timestep, a_top, specific, emb=None):
        del emb
        total, length, _ = noisy_action.shape
        condition = self.cond_a(a_top)[:, None, :] + self._temb(timestep, total, length)
        if self.cond_s is not None and specific is not None:
            condition = condition + self.cond_s(specific)[:, None, :]
        value = self.in_x(noisy_action) + self.pos[None, :length]
        for block in self.blocks:
            value = block(value, condition)
        shift, scale = self.fmod(condition).chunk(2, -1)
        velocity = self.vout(self.norm_f(value) * (1 + scale) + shift)
        return velocity, None, None


class DiffusionHead(Stage):
    reads = ["a_top", "s", "embodiment"]
    writes = [
        "pred_action",
        "loss/ddpm",
        "log/ddpm",
        "log/vA_frac",
        "loss/moe_lb",
        "log/*",
    ]

    def __init__(
        self,
        d_a: int,
        d_s: int,
        action_dim: int,
        chunk_len: int,
        embodiments: Optional[List[str]] = None,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        d_model_a: int = 256,
        d_model_s: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
        mask_mode: str = "sym",
        denoiser: str = "dual",
        denoiser_arch: str = "adaln",
        moe_experts: int = 0,
        moe_top_k: int = 4,
        moe_d_expert: Optional[int] = None,
        moe_aux_weight: float = 0.01,
        action_dims: Optional[dict] = None,
        latent_dim: Optional[int] = None,
        enc_hidden=256,
        enc_layers=3,
        enc_residual=False,
        enc_per_stream: bool = False,
        emit_loss: bool = True,
        loss_space: str = "eps",
    ):
        super().__init__()
        del (
            d_model_s,
            mask_mode,
            denoiser_arch,
            enc_hidden,
            enc_layers,
            enc_residual,
            enc_per_stream,
        )
        if denoiser != "single":
            raise ValueError("Frozen Fold compatibility requires denoiser='single'")
        if moe_experts:
            raise ValueError("Frozen Fold compatibility does not support MoE")
        if action_dims is not None or latent_dim is not None:
            raise ValueError(
                "Frozen Fold compatibility does not support heterogeneous actions"
            )
        if loss_space != "eps":
            raise ValueError("Frozen Fold compatibility requires eps loss space")
        self.emit_loss = bool(emit_loss)
        self.loss_space = str(loss_space)
        self.C, self.D = int(chunk_len), int(action_dim)
        self.N, self.S = int(num_train_timesteps), int(num_inference_steps)
        self.register_buffer("abar", _cosine_alphas_cumprod(self.N))
        self.register_buffer(
            "inf_levels",
            torch.linspace(self.N - 1, 0, self.S).round().long(),
        )
        self.denoiser_kind = str(denoiser)
        names = [str(value) for value in embodiments] if embodiments else ["shared"]
        self.Dmap = {name: self.D for name in names}
        self.rh = False
        self.net = SingleStreamDenoiserV2(
            d_a_in=d_a,
            d_s_in=d_s,
            action_dim=action_dim,
            chunk_len=chunk_len,
            d_model=d_model_a,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            n_positions=int(chunk_len),
            moe_experts=moe_experts,
            moe_top_k=moe_top_k,
            moe_d_expert=moe_d_expert,
            moe_aux_weight=moe_aux_weight,
        )

    def _D(self, embodiment) -> int:
        return self.Dmap.get(str(embodiment), self.D)

    def forward(self, batch: dict) -> dict:
        a_top = batch["a_top"]
        specific = batch.get("s")
        embodiment = str(batch["embodiment"])
        if "target" in batch:
            target = batch["target"]
            total = target.shape[0]
            timestep = torch.randint(0, self.N, (total,), device=target.device)
            alpha = self.abar[timestep][:, None, None]
            noise = torch.randn_like(target)
            noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
            prediction, effect_a, effect_s = self.net(
                noisy,
                timestep.float(),
                a_top,
                specific,
                embodiment,
            )
            batch["aux/eps_pred"] = prediction
            batch["aux/eps_target"] = noise
            batch["aux/ddpm_t"] = timestep
            if self.emit_loss:
                loss = F.mse_loss(prediction, noise)
                batch["loss/ddpm"] = loss
                batch["log/ddpm"] = float(loss)
            if effect_a is not None and effect_s is not None:
                with torch.no_grad():
                    norm_a = effect_a.norm(dim=-1).mean()
                    norm_s = effect_s.norm(dim=-1).mean()
                    batch["log/vA_frac"] = float(norm_a / (norm_a + norm_s + 1e-8))
        if not self.training:
            with torch.no_grad():
                total = a_top.shape[0]
                streaming = "rollout_t" in batch
                selected_a = a_top[-1:] if streaming else a_top
                selected_s = (
                    (specific[-1:] if specific is not None else None)
                    if streaming
                    else specific
                )
                selected_total = selected_a.shape[0]
                sample = torch.randn(
                    selected_total,
                    self.C,
                    self._D(embodiment),
                    device=a_top.device,
                    dtype=a_top.dtype,
                )
                for index in range(self.S):
                    level = int(self.inf_levels[index])
                    timestep = torch.full(
                        (selected_total,),
                        float(level),
                        device=sample.device,
                        dtype=sample.dtype,
                    )
                    prediction, _, _ = self.net(
                        sample,
                        timestep,
                        selected_a,
                        selected_s,
                        embodiment,
                    )
                    alpha = self.abar[level]
                    estimate = (sample - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                    estimate = estimate.clamp(-1.0, 1.0)
                    if index + 1 < self.S:
                        next_alpha = self.abar[int(self.inf_levels[index + 1])]
                        sample = (
                            next_alpha.sqrt() * estimate
                            + (1 - next_alpha).sqrt() * prediction
                        )
                    else:
                        sample = estimate
                sample = sample.clamp(-1.0, 1.0)
                if streaming:
                    output = torch.zeros(
                        total,
                        self.C,
                        self._D(embodiment),
                        device=a_top.device,
                        dtype=a_top.dtype,
                    )
                    output[-1] = sample[0]
                    sample = output
                batch["pred_action"] = sample
        return batch


class MaskedActionLoss(Stage):
    train_only = True
    reads = ["aux/eps_pred", "aux/eps_target", "embodiment"]

    def __init__(
        self,
        exclude_dims: Optional[dict] = None,
        name: str = "ddpm",
        embodiments: Optional[List[str]] = None,
        weights: Optional[dict] = None,
    ):
        super().__init__()
        self.name = str(name)
        self.weights = {
            str(key): float(value) for key, value in (weights or {}).items()
        }
        self.writes = [f"loss/{self.name}", f"log/{self.name}"]
        self.exclude = {
            str(key): [int(index) for index in value]
            for key, value in (exclude_dims or {}).items()
        }
        names = [str(value) for value in (embodiments or [])]
        if names and self.exclude:
            common = (
                set.intersection(*[set(self.exclude.get(name, [])) for name in names])
                if all(name in self.exclude for name in names)
                else set()
            )
            if common:
                raise ValueError(
                    f"MaskedActionLoss: dims {sorted(common)} are excluded for "
                    f"EVERY embodiment {names}"
                )

    def forward(self, batch: dict) -> dict:
        prediction = batch.get("aux/eps_pred")
        if prediction is None:
            return batch
        target = batch["aux/eps_target"]
        embodiment = str(batch["embodiment"])
        drop = self.exclude.get(embodiment)
        if drop:
            action_dim = prediction.shape[-1]
            invalid = [index for index in drop if index >= action_dim]
            if invalid:
                raise IndexError(
                    f"MaskedActionLoss: exclude_dims {invalid} out of range for "
                    f"embodiment {embodiment!r} with action_dim {action_dim}."
                )
            keep = torch.ones(action_dim, dtype=torch.bool, device=prediction.device)
            keep[torch.tensor(drop, device=prediction.device)] = False
            prediction, target = prediction[..., keep], target[..., keep]
        loss = F.mse_loss(prediction, target)
        weight = self.weights.get(embodiment, 1.0)
        batch[f"loss/{self.name}"] = loss * weight
        batch[f"log/{self.name}"] = float(loss)
        batch[f"log/{self.name}_weight"] = float(weight)
        batch[f"log/{self.name}_dims_scored"] = float(prediction.shape[-1])
        return batch
