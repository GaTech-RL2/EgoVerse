"""
BPP — Behavior-Prompting diffusion-transformer policy as an EgoMimic algo.

Thin adapter (Option A in docs/2026-09-02_bpp_algo_option.md) around the vendored
``external/behavior_prompting`` package: the hydra model config instantiates
``DiffusionTransformerPolicy`` (+ ``PairPromptObsEncoder`` stack) directly and
this class only translates between the EgoMimic Algo contract and BPP's
``{"obs": {..., "prompt": ...}, "action": ...}`` batch format.

Normalization ownership: samples are already normalized by
``MultiDataset.__getitem__``; the BPP policy's internal ``Normalizer`` is
forced to identity for every key so its internal normalize/unnormalize calls
are no-ops, and ``forward_eval`` unnormalizes through ``self.norm_stats``.

Prompting is optional and decided by the wrapped policy: if
``policy.supports_prompting()`` is False (plain ``TransformerObsEncoder``,
see ``model/bpp_dit.yaml``) the adapter never builds, encodes, or attaches a
prompt, so the model is the unprompted diffusion transformer with the same
receding-observation path.

Two prompt sampling modes (``prompt.mode``):

- ``batch_roll``: each sample is prompted with its batch neighbor's single
  frame + action chunk (single-task smoke tests; no dataset changes).
- ``episode_pair``: the dataset (``EpisodePromptMultiDataset``) attaches a
  whole-episode prompt from the same (task, operator) group; the adapter only
  renames keys, resizes frames, and applies prompt dropout. See
  docs/2026-09-02_bpp_episode_prompting.md.
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
from behavior_prompting.train_network.model.common.normalizer import (
    SingleFieldLinearNormalizer,
)
from behavior_prompting.train_network.utils.prompt_util import PromptActionChunker
from omegaconf import OmegaConf
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class BPP(Algo):
    """Adapter exposing behavior_prompting's DiffusionTransformerPolicy as an
    EgoMimic algo. Single-embodiment only."""

    def __init__(
        self,
        norm_stats,
        camera_transforms,
        # ---------------------------
        # Image augmentations (own all image geometry here; the policy's
        # internal train/eval_image_transforms must stay null so frames are
        # never doubly augmented — it still applies CLIP mean/std internally
        # via use_vision_norm, so do NOT add a Normalize here).
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # hydra-instantiated behavior_prompting DiffusionTransformerPolicy
        # ---------------------------
        policy: nn.Module = None,
        # BPP-convention shape_meta dict; must agree with the policy's own
        # shape_meta (the model yaml interpolates the same node into both).
        shape_meta: dict = None,
        # {embodiment_name: action key}, e.g. {"eva_bimanual": "actions_cartesian"}
        ac_keys: dict = None,
        # prompt sampling options: mode ("batch_roll"), p_drop_prompt.
        # Only valid when the policy supports prompting; must be omitted for
        # the unprompted variant.
        prompt: dict = None,
        # Activation checkpointing of the timm ViT blocks (every module in
        # the policy that exposes ``set_grad_checkpointing``). Trades the
        # per-image ViT activations, which dominate memory with long
        # prompts, for a second forward pass in backward. Non-reentrant, so
        # it is safe under DDP with the shared prompt/receding encoder
        # being called several times per step. No effect on the math.
        grad_checkpointing: bool = False,
        **kwargs,
    ):
        if policy is None or shape_meta is None or not ac_keys:
            raise ValueError("BPP requires `policy`, `shape_meta`, and `ac_keys`.")

        self.norm_stats = norm_stats
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs

        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.shape_meta = shape_meta
        self.action_horizon = int(shape_meta["action"]["horizon"])

        # The wrapped policy decides whether a prompt exists at all.
        self.use_prompt = bool(policy.supports_prompting())
        prompt = dict(prompt or {})
        if self.use_prompt:
            self.prompt_mode = prompt.get("mode", "batch_roll")
            if self.prompt_mode not in ("batch_roll", "episode_pair"):
                raise NotImplementedError(
                    f"Unsupported prompt sampling mode: {self.prompt_mode!r} "
                    "(expected 'batch_roll' or 'episode_pair')"
                )
            self.p_drop_prompt = float(prompt.get("p_drop_prompt", 0.0))
            self.chunk_n_actions = int(shape_meta["prompt_chunk_n_actions"])
            self.max_prompt_len = math.ceil(
                int(shape_meta["max_sequence_length"]) / self.chunk_n_actions
            )
            if self.prompt_mode == "batch_roll":
                if self.action_horizon % self.chunk_n_actions != 0:
                    raise ValueError(
                        "batch_roll prompting requires action horizon "
                        f"({self.action_horizon}) divisible by prompt_chunk_n_actions "
                        f"({self.chunk_n_actions})"
                    )
                self.prompt_chunker = PromptActionChunker(shape_meta)
            else:
                self.prompt_chunker = None
        else:
            if prompt:
                raise ValueError(
                    "`prompt` config given but the policy's obs encoder does not "
                    "support prompting; drop the `prompt:` block or use a "
                    "PairPromptObsEncoder policy (model/bpp_prompt_dit.yaml)."
                )
            self.prompt_mode = None
            self.p_drop_prompt = 0.0
            self.chunk_n_actions = None
            self.max_prompt_len = None
            self.prompt_chunker = None

        # BPP has no multi-head/shared/OT machinery; expose the attributes the
        # HPT evaluator reads so eval_hpt works unchanged.
        self.shared_ac_key = None
        self.auxiliary_ac_keys = {}
        self.rkl_samples = 0

        self.domains = sorted(ac_keys.keys())
        if len(self.domains) != 1:
            raise ValueError(f"BPP is single-embodiment; got domains {self.domains}.")

        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ac_keys keyed by both embodiment name (config side) and id (eval side)
        self.ac_keys = dict(ac_keys)
        self.camera_keys = {}
        self.proprio_keys = {}
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.ac_keys[embodiment_id] = ac_keys[embodiment]
            self.camera_keys[embodiment_id] = [
                k
                for k in norm_stats.keys_of_type("camera_keys", embodiment_id)
                if norm_stats.is_key_with_embodiment(k, embodiment_id)
            ]
            self.proprio_keys[embodiment_id] = [
                k
                for k in norm_stats.keys_of_type("proprio_keys", embodiment_id)
                if norm_stats.is_key_with_embodiment(k, embodiment_id)
            ]

        # batch key (dotted zarr-style) -> shape_meta obs key. Cameras map to
        # their last dotted segment ("observations.images.front_img_1" ->
        # "front_img_1"), proprio to "state_" + last segment — same convention
        # HPT uses for its stem names.
        embodiment_id = get_embodiment_id(self.domains[0])
        self.obs_key_map = {}
        self._rgb_meta_keys = set()
        for key in self.camera_keys[embodiment_id]:
            name = key.rsplit(".", 1)[-1]
            if name in shape_meta["obs"]:
                self.obs_key_map[key] = name
                self._rgb_meta_keys.add(name)
        for key in self.proprio_keys[embodiment_id]:
            name = "state_" + key.rsplit(".", 1)[-1]
            if name in shape_meta["obs"]:
                self.obs_key_map[key] = name

        missing = [
            name
            for name, attr in shape_meta["obs"].items()
            if not attr.get("ignore_by_policy", False)
            and name not in self.obs_key_map.values()
        ]
        if missing:
            raise ValueError(
                f"shape_meta obs keys {missing} have no matching batch key. "
                f"Available batch keys: cameras={self.camera_keys[embodiment_id]}, "
                f"proprio={self.proprio_keys[embodiment_id]}"
            )

        self.nets = nn.ModuleDict({"policy": policy})

        self.grad_checkpointing = bool(grad_checkpointing)
        if self.grad_checkpointing:
            import timm.layers

            timm.layers.set_reentrant_ckpt(False)
            checkpointed = [
                type(m).__name__
                for m in policy.modules()
                if hasattr(m, "set_grad_checkpointing")
            ]
            for m in policy.modules():
                if hasattr(m, "set_grad_checkpointing"):
                    m.set_grad_checkpointing(True)
            if not checkpointed:
                raise ValueError(
                    "grad_checkpointing=True but no module in the policy exposes "
                    "set_grad_checkpointing (expected a timm ViT backbone)."
                )

        # Force the policy's internal Normalizer (and its prompt normalizer)
        # to identity for every key it will ever see, so all real
        # normalization stays on the EgoMimic side (self.norm_stats).
        normalizer = self.nets["policy"].normalizer
        prompt_normalizer = normalizer.get_prompt_normalizer()
        for key in list(shape_meta["obs"].keys()) + ["action"]:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
            prompt_normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        # create_identity leaves requires_grad=True (unlike Normalizer.fit);
        # freeze so the optimizer can't drift the identity scale/offset and
        # DDP (find_unused_parameters=False) doesn't see grad-less params.
        for p in normalizer.parameters():
            p.requires_grad_(False)

        self.nets = self.nets.float().to(self.device)
        self.training_step = 0

    # =====================================================================
    # Algo contract
    # =====================================================================

    @override
    def process_batch_for_training(self, batch):
        """
        Zarr-key -> key-name translation plus device/dtype handling, mirroring
        HPT.process_batch_for_training. Samples arrive already normalized by
        MultiDataset.__getitem__.
        """
        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                if key == "prompt":
                    # Nested whole-episode prompt (episode_pair); keys inside
                    # are batch keys and are renamed in _build_obs_dict. An
                    # unprompted policy drops it here so the prompt tensors
                    # never reach the device.
                    if self.use_prompt:
                        processed_batch[embodiment_id][key] = value
                    continue
                key_name = (
                    self.norm_stats.zarr_key_to_keyname(key, embodiment_id) or key
                )
                processed_batch[embodiment_id][key_name] = value

            ac_key = self.ac_keys[embodiment_id]
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError(
                    f"Expected (B, T, D) actions for {ac_key}, got shape "
                    f"{tuple(processed_batch[embodiment_id][ac_key].shape)}"
                )

            B, S, _ = processed_batch[embodiment_id][ac_key].shape
            device = processed_batch[embodiment_id][ac_key].device
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=device
            )
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )
            processed_batch[embodiment_id] = self._to_device(
                processed_batch[embodiment_id]
            )

        return processed_batch

    def _to_device(self, value):
        """Move tensors (recursing into dicts) to self.device; floats -> fp32."""
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.float()
        return value

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step += 1
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            obs_dict = self._build_obs_dict(_batch, embodiment_id, training=True)
            loss = self.nets["policy"].compute_loss(
                {"obs": obs_dict, "action": _batch[ac_key]}
            )
            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss
        return predictions

    @override
    def forward_eval(self, batch):
        """
        Two passes, mirroring HPT.forward_eval: the diffusion val loss (same
        call as training) and a DDIM-sampled action prediction, unnormalized
        via self.norm_stats.
        """
        unnorm_preds = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            obs_dict = self._build_obs_dict(_batch, embodiment_id, training=False)

            val_loss = self.nets["policy"].compute_loss(
                {"obs": obs_dict, "action": _batch[ac_key]}
            )
            unnorm_preds[f"{embodiment_name}_loss"] = val_loss

            # Whole-episode prompting: split the val loss by whether the
            # sample's operator was seen in training (held-out episode of a
            # training operator) or held out entirely (unseen operator).
            seen = _batch.get("operator_seen")
            if torch.is_tensor(seen) and seen.ndim == 1 and seen.shape[0] > 0:
                seen = seen.bool()
                for name, rows in (("seen", seen), ("unseen", ~seen)):
                    if not rows.any():
                        continue
                    sub_obs = self._select_rows(obs_dict, rows)
                    unnorm_preds[f"{embodiment_name}_loss_{name}_operator"] = self.nets[
                        "policy"
                    ].compute_loss({"obs": sub_obs, "action": _batch[ac_key][rows]})

            result = self.nets["policy"].predict_action(obs_dict)
            ref = _batch[ac_key]
            B, T, D = ref.shape
            # .clone() lifts the prediction out of predict_action's
            # inference_mode so downstream metric/viz code can use it freely.
            pred = result["action"][:, :T, :D].clone()

            unnorm_actions = self.norm_stats.unnormalize({ac_key: pred}, embodiment_id)
            unnorm_preds[f"{embodiment_name}_{ac_key}"] = unnorm_actions[ac_key]

        return unnorm_preds

    @override
    def compute_losses(self, predictions, batch):
        loss_dict = OrderedDict()
        total_loss = torch.tensor(0.0, device=self.device)
        for embodiment_id in batch:
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            total_loss = total_loss + bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss
        loss_dict["action_loss"] = total_loss / len(self.domains)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        return log

    # =====================================================================
    # Optimizer parameter groups (original BPP recipe: pretrained backbone at
    # lr * pretrained_lr_scale with no weight decay, transformer decay groups)
    # =====================================================================

    def optimizer_param_groups(self, lr: float, weight_decay: float):
        policy = self.nets["policy"]
        groups = []
        groups.extend(policy.model.get_optim_groups(weight_decay=weight_decay))
        groups.extend(
            policy.obs_encoder.get_optim_groups(lr=lr, weight_decay=weight_decay)
        )
        # The original builds some groups from parameter generators; make
        # them lists so they survive being inspected and filtered below.
        for g in groups:
            g["params"] = list(g["params"])
        covered = {id(p) for g in groups for p in g["params"]}
        missing = [
            n
            for n, p in self.nets.named_parameters()
            if p.requires_grad and id(p) not in covered
        ]
        if missing:
            raise RuntimeError(
                "BPP optimizer groups do not cover trainable params: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        # drop frozen params (identity normalizers) so the optimizer never
        # sees grad-less tensors
        for g in groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        return [g for g in groups if g["params"]]

    # =====================================================================
    # Eval helpers
    # =====================================================================

    @staticmethod
    def _select_rows(value, rows):
        """Index the batch dimension of every tensor in a (nested) obs dict."""
        if isinstance(value, dict):
            return {k: BPP._select_rows(v, rows) for k, v in value.items()}
        if (
            torch.is_tensor(value)
            and value.ndim >= 1
            and value.shape[0] == rows.shape[0]
        ):
            return value[rows]
        return value

    # =====================================================================
    # Prompting API passthrough (deployment: prompt once per episode, then
    # predict_action per step at unprompted cost)
    # =====================================================================

    def supports_prompting(self):
        return self.use_prompt

    def prompt(self, prompt_dict):
        if not self.use_prompt:
            raise RuntimeError(
                "This BPP policy was built without a prompt encoder "
                "(model/bpp_dit.yaml); prompting is unavailable."
            )
        self.nets["policy"].prompt(prompt_dict)

    def reset(self, action_exec_horizon=None):
        self.nets["policy"].reset(action_exec_horizon=action_exec_horizon)

    # =====================================================================
    # Batch assembly
    # =====================================================================

    def _build_obs_dict(self, _batch, embodiment_id, training: bool):
        """
        Assemble BPP's expected obs_dict from a processed EgoMimic batch:

            {
                <current obs keys>: (B, To, ...),
                "prompt": {
                    "obs":      {<prompt obs keys>: (B, P, ...)},
                    "action":   (B, P, chunk_n_actions, action_dim),
                    "metadata": {"mask": (B, P)},
                },
            }

        Prompt sampling is batch-roll: each sample is prompted with its batch
        neighbor's (obs, action-chunk) pair. Prompt frames always go through
        the eval transform so the demonstration stays in-distribution.

        When the policy has no prompt encoder (``self.use_prompt`` False) the
        ``"prompt"`` entry is omitted entirely and no prompt tensors are built.
        """
        ac_key = self.ac_keys[embodiment_id]
        obs = {}
        prompt_obs_src = {}
        for batch_key, meta_key in self.obs_key_map.items():
            value = _batch[batch_key]
            attr = self.shape_meta["obs"][meta_key]
            prompt_type = (
                attr.get("prompt_type", "ignore") if self.use_prompt else "ignore"
            )
            if meta_key in self._rgb_meta_keys:
                if value.dim() != 4:
                    raise ValueError(
                        f"Expected (B, C, H, W) for {batch_key}, got "
                        f"{tuple(value.shape)}"
                    )
                cur = (
                    self.train_image_augs(value)
                    if training and self.train_image_augs is not None
                    else self.eval_image_augs(value)
                )
                obs[meta_key] = cur.unsqueeze(1)
                if prompt_type == "observation":
                    prompt_obs_src[meta_key] = (
                        self.eval_image_augs(value) if training else cur
                    )
            else:
                if value.dim() != 2:
                    raise ValueError(
                        f"Expected (B, D) for {batch_key}, got {tuple(value.shape)}"
                    )
                obs[meta_key] = value.unsqueeze(1)
                if prompt_type == "proprioception":
                    prompt_obs_src[meta_key] = value

        if self.use_prompt and self.prompt_mode == "batch_roll":
            obs["prompt"] = self._batch_roll_prompt(
                prompt_obs_src, _batch[ac_key], training
            )
        elif self.use_prompt:
            if "prompt" not in _batch:
                raise ValueError(
                    "prompt.mode=episode_pair but the batch carries no `prompt`; "
                    "use data built on EpisodePromptMultiDataset "
                    "(data/bpp_folding_clothes.yaml)."
                )
            obs["prompt"] = self._episode_prompt(_batch["prompt"], training)
        return obs

    def _episode_prompt(self, prompt, training: bool):
        """Adapt a collated whole-episode prompt (batch keys, resized frames,
        normalized state/actions, ``metadata.mask``) to the policy's prompt
        dict: rename obs keys to shape_meta names, run the eval image
        transform, check chunk geometry, apply content dropout."""
        obs = {}
        for batch_key, value in prompt["obs"].items():
            meta_key = self.obs_key_map.get(batch_key)
            if meta_key is None:
                continue
            attr = self.shape_meta["obs"][meta_key]
            prompt_type = attr.get("prompt_type", "ignore")
            if meta_key in self._rgb_meta_keys and prompt_type == "observation":
                B, P = value.shape[:2]
                frames = self.eval_image_augs(value.reshape(B * P, *value.shape[2:]))
                obs[meta_key] = frames.reshape(B, P, *frames.shape[1:])
            elif (
                meta_key not in self._rgb_meta_keys and prompt_type == "proprioception"
            ):
                obs[meta_key] = value
        action = prompt["action"]
        B, P, chunk_n, _ = action.shape
        if chunk_n != self.chunk_n_actions:
            raise ValueError(
                f"prompt chunk size {chunk_n} != shape_meta.prompt_chunk_n_actions "
                f"{self.chunk_n_actions}; the data config must interpolate the "
                "model's prompt_chunk_n_actions."
            )
        if P > self.max_prompt_len:
            raise ValueError(
                f"prompt has {P} chunks > max {self.max_prompt_len} "
                "(shape_meta.max_sequence_length / prompt_chunk_n_actions)."
            )
        mask = prompt["metadata"]["mask"].to(torch.bool)
        chunked = {"obs": obs, "action": action, "metadata": {"mask": mask}}
        if training and self.p_drop_prompt > 0.0:
            chunked = self._drop_prompt_content(chunked, B, action.device)
        return chunked

    def _drop_prompt_content(self, chunked, B, device):
        # Content dropout: zero the prompt for dropped samples instead of
        # masking it out entirely (an all-True key_padding_mask row would
        # NaN the cross-attention softmax).
        keep = (torch.rand(B, device=device) >= self.p_drop_prompt).float()
        chunked["action"] = chunked["action"] * keep.view(B, 1, 1, 1)
        chunked["obs"] = {
            k: v * keep.view(B, *([1] * (v.dim() - 1)))
            for k, v in chunked["obs"].items()
        }
        return chunked

    def _batch_roll_prompt(self, prompt_obs_src, actions, training: bool):
        """Roll the batch by one so sample i is prompted with sample i+1's
        trajectory. The rolled sample's single current frame is repeated
        across the P prompt steps (batch-roll's known approximation), while
        its full action chunk is split into P chunks of chunk_n_actions."""
        B, T, _ = actions.shape
        P = T // self.chunk_n_actions

        prompt_raw = {"obs": {}, "action": torch.roll(actions, 1, dims=0)}
        for key, value in prompt_obs_src.items():
            rolled = torch.roll(value, 1, dims=0)
            prompt_raw["obs"][key] = rolled.unsqueeze(1).expand(B, P, *rolled.shape[1:])

        chunked = self.prompt_chunker.chunk_prompt(prompt_raw, obs_predownsampled=True)
        chunked["metadata"] = {
            "mask": torch.zeros(B, P, dtype=torch.bool, device=actions.device)
        }

        if training and self.p_drop_prompt > 0.0:
            chunked = self._drop_prompt_content(chunked, B, actions.device)

        return chunked
