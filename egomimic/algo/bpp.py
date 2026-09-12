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

Own-episode history (``prompt.history.enabled``, episode_pair only, needs a
``HistoryPairPromptObsEncoder`` policy): the batch also carries ``history``,
the last ``max_chunks`` completed chunks of the sample's own episode in the
prompt's chunk format. The adapter adapts it like the prompt and nests it at
``obs["prompt"]["metadata"]["history"]`` (the vendored normalizer rejects
unknown top-level obs keys but passes metadata through). See
docs/plan/2026-09-08_bpp_rollout_history.md.
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

        # ---- own-episode history ----
        history = dict(prompt.get("history") or {}) if self.use_prompt else {}
        self.use_history = bool(history.get("enabled", False))
        encoder = getattr(policy, "obs_encoder", None)
        encoder_cap = getattr(encoder, "history_max_chunks", None)
        if self.use_history:
            if self.prompt_mode != "episode_pair":
                raise ValueError(
                    "prompt.history.enabled requires prompt.mode=episode_pair"
                )
            if encoder_cap is None:
                raise ValueError(
                    "prompt.history.enabled requires the policy's obs_encoder to be "
                    "a HistoryPairPromptObsEncoder "
                    "(model/bpp_prompt_dit_episode_history.yaml)."
                )
            self.history_max_chunks = int(encoder_cap)
            cfg_cap = history.get("max_chunks")
            if cfg_cap is not None and int(cfg_cap) != self.history_max_chunks:
                raise ValueError(
                    f"prompt.history.max_chunks={cfg_cap} != the encoder's "
                    f"history_max_chunks={self.history_max_chunks}"
                )
            self.p_drop_history = float(history.get("p_drop_history", 0.2))
            self.p_drop_history_state = float(history.get("p_drop_history_state", 0.0))
            self.p_drop_history_action = float(
                history.get("p_drop_history_action", 0.0)
            )
            self.history_action_noise_std = float(history.get("action_noise_std", 0.0))
            self.history_only = bool(history.get("history_only", False))
            eval_chunks = history.get("eval_history_chunks")
            self.eval_history_chunks = None if eval_chunks is None else int(eval_chunks)
            if self.history_only and not getattr(
                encoder, "attention_sink_enabled", False
            ):
                raise ValueError(
                    "prompt.history.history_only=true requires "
                    "policy.obs_encoder.use_attention_sink=true: with every demo "
                    "token masked, a row with no history would have an all-masked "
                    "memory and NaN the cross-attention softmax."
                )
        else:
            if encoder_cap is not None:
                raise ValueError(
                    "The policy's obs_encoder is a HistoryPairPromptObsEncoder but "
                    "prompt.history.enabled is false; enable it or use "
                    "PairPromptObsEncoder."
                )
            self.history_max_chunks = None
            self.p_drop_history = 0.0
            self.p_drop_history_state = 0.0
            self.p_drop_history_action = 0.0
            self.history_action_noise_std = 0.0
            self.history_only = False
            self.eval_history_chunks = None

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
                if key == "history":
                    # Own-episode history, same treatment: only a history
                    # policy moves it to the device.
                    if self.use_history:
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

    def episode_prompt_to_policy(self, prompt):
        """Deployment: a collated whole-episode prompt with batch keys (e.g.
        from ``build_episode_prompt``) -> the policy's prompt dict, ready for
        ``prompt()``. A ``history`` entry in its metadata is adapted too."""
        prompt = self._to_device(prompt)
        out = self._episode_prompt(prompt, training=False)
        history = (prompt.get("metadata") or {}).get("history")
        if history is not None and self.use_history:
            out["metadata"]["history"] = self.history_chunk_to_policy(history)
        return out

    def history_chunk_to_policy(self, chunk):
        """Deployment: one or more history chunks with batch keys (e.g. from
        ``build_history_chunk`` + prompt normalization; ``metadata.mask`` is
        optional) -> the policy's history dict. No eval slicing."""
        if not self.use_history:
            raise RuntimeError(
                "This BPP policy was built without history (prompt.history.enabled)."
            )
        return self._adapt_chunks(
            self._to_device(chunk),
            training=False,
            max_len=self.history_max_chunks,
            name="history",
        )

    def push_history_chunk(self, chunk) -> int:
        """Deployment: append one completed chunk (batch keys, B = 1) to the
        encoder's sliding history cache. Returns the cache length."""
        return self.nets["policy"].obs_encoder.push_history_chunk(
            self.history_chunk_to_policy(chunk)
        )

    def clear_history(self):
        if self.use_history:
            self.nets["policy"].obs_encoder.clear_history()

    @property
    def history_len(self) -> int:
        if not self.use_history:
            return 0
        return int(self.nets["policy"].obs_encoder.history_len)

    def predict_action_deployed(self, _batch, embodiment_id):
        """Deployment inference after ``prompt()``: receding obs only (the
        encoder serves the demo prompt and history from its caches), DDIM
        sample, unnormalize like ``forward_eval``. Returns ``(B, T, D)``."""
        obs_dict = self._build_obs_dict(
            _batch, embodiment_id, training=False, deployed=True
        )
        result = self.nets["policy"].predict_action(obs_dict)
        ac_key = self.ac_keys[embodiment_id]
        pred = result["action"].clone()
        return self.norm_stats.unnormalize({ac_key: pred}, embodiment_id)[ac_key]

    # =====================================================================
    # Batch assembly
    # =====================================================================

    def _build_obs_dict(
        self, _batch, embodiment_id, training: bool, deployed: bool = False
    ):
        """
        Assemble BPP's expected obs_dict from a processed EgoMimic batch:

            {
                <current obs keys>: (B, To, ...),
                "prompt": {
                    "obs":      {<prompt obs keys>: (B, P, ...)},
                    "action":   (B, P, chunk_n_actions, action_dim),
                    "metadata": {
                        "mask": (B, P),
                        # own-episode history (prompt.history.enabled):
                        "history": {
                            "obs":      {<prompt obs keys>: (B, H, ...)},
                            "action":   (B, H, chunk_n_actions, action_dim),
                            "metadata": {"mask": (B, H)},
                        },
                    },
                },
            }

        Prompt sampling is batch-roll (each sample is prompted with its batch
        neighbor's (obs, action-chunk) pair) or episode_pair (the batch carries
        a collated whole-episode prompt). Prompt and history frames always go
        through the eval transform so the demonstration stays in-distribution.

        When the policy has no prompt encoder (``self.use_prompt`` False) the
        ``"prompt"`` entry is omitted entirely and no prompt tensors are built.
        ``deployed=True`` returns the receding obs only: after ``prompt()`` the
        encoder serves the demo prompt and history from its caches and
        refuses a ``prompt`` entry.
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

        if deployed:
            return obs

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
            if self.use_history:
                if "history" not in _batch:
                    raise ValueError(
                        "prompt.history.enabled but the batch carries no `history`; "
                        "use data with prompt.history.max_chunks > 0 "
                        "(data/bpp_folding_clothes_history.yaml)."
                    )
                obs["prompt"]["metadata"]["history"] = self._episode_history(
                    _batch["history"], training
                )
                if self.history_only:
                    obs["prompt"] = self._mask_demo_prompt(obs["prompt"])
        return obs

    def _adapt_chunks(self, payload, training: bool, *, max_len: int, name: str):
        """Adapt a collated chunk payload (batch keys, resized frames,
        normalized state/actions, ``metadata.mask``; the whole-episode prompt
        or the own-episode history) to the policy's prompt-dict format: rename
        obs keys to shape_meta names, run the eval image transform, check the
        chunk geometry. ``P`` may be 0 (empty history)."""
        obs = {}
        for batch_key, value in payload["obs"].items():
            meta_key = self.obs_key_map.get(batch_key)
            if meta_key is None:
                continue
            attr = self.shape_meta["obs"][meta_key]
            prompt_type = attr.get("prompt_type", "ignore")
            if meta_key in self._rgb_meta_keys and prompt_type == "observation":
                B, P = value.shape[:2]
                if P == 0:
                    obs[meta_key] = value.new_zeros((B, 0) + tuple(attr["shape"]))
                else:
                    frames = self.eval_image_augs(
                        value.reshape(B * P, *value.shape[2:])
                    )
                    obs[meta_key] = frames.reshape(B, P, *frames.shape[1:])
            elif (
                meta_key not in self._rgb_meta_keys and prompt_type == "proprioception"
            ):
                obs[meta_key] = value
        action = payload["action"]
        B, P, chunk_n, _ = action.shape
        if chunk_n != self.chunk_n_actions:
            raise ValueError(
                f"{name} chunk size {chunk_n} != shape_meta.prompt_chunk_n_actions "
                f"{self.chunk_n_actions}; the data config must interpolate the "
                "model's prompt_chunk_n_actions."
            )
        if P > max_len:
            raise ValueError(f"{name} has {P} chunks > max {max_len}.")
        mask = (payload.get("metadata") or {}).get("mask")
        if mask is None:
            mask = torch.zeros(B, P, dtype=torch.bool, device=action.device)
        return {"obs": obs, "action": action, "metadata": {"mask": mask.to(torch.bool)}}

    def _episode_prompt(self, prompt, training: bool):
        """Whole-episode prompt -> policy prompt dict (+ content dropout)."""
        chunked = self._adapt_chunks(
            prompt, training, max_len=self.max_prompt_len, name="prompt"
        )
        if training and self.p_drop_prompt > 0.0:
            action = chunked["action"]
            chunked = self._drop_prompt_content(chunked, action.shape[0], action.device)
        return chunked

    # ---- own-episode history ----

    def _episode_history(self, history, training: bool):
        """Own-episode history -> policy history dict. Training: optional
        action noise, per-modality dropout (state / action content zeroed,
        mask unchanged) and whole-history dropout (content zeroed AND every
        chunk masked; safe because the demo tokens stay in the memory).
        Eval: optional slice to the newest ``eval_history_chunks`` chunks."""
        chunked = self._adapt_chunks(
            history, training, max_len=self.history_max_chunks, name="history"
        )
        action = chunked["action"]
        B, device = action.shape[0], action.device
        if training:
            if self.history_action_noise_std > 0.0:
                chunked["action"] = (
                    action + self.history_action_noise_std * torch.randn_like(action)
                )
            chunked = self._drop_history_modalities(chunked, B, device)
            if self.p_drop_history > 0.0:
                chunked = self._drop_history(chunked, B, device)
        elif self.eval_history_chunks is not None:
            chunked = self._slice_history(chunked, self.eval_history_chunks)
        return chunked

    def _drop_history_modalities(self, chunked, B, device):
        def _keep(p):
            return (torch.rand(B, device=device) >= p).float()

        state_keys = [k for k in chunked["obs"] if k not in self._rgb_meta_keys]
        if self.p_drop_history_state > 0.0 and state_keys:
            keep = _keep(self.p_drop_history_state)
            for k in state_keys:
                v = chunked["obs"][k]
                chunked["obs"][k] = v * keep.view(B, *([1] * (v.dim() - 1)))
        if self.p_drop_history_action > 0.0:
            keep = _keep(self.p_drop_history_action)
            chunked["action"] = chunked["action"] * keep.view(B, 1, 1, 1)
        return chunked

    def _drop_history(self, chunked, B, device):
        keep = torch.rand(B, device=device) >= self.p_drop_history
        keepf = keep.float()
        chunked["action"] = chunked["action"] * keepf.view(B, 1, 1, 1)
        chunked["obs"] = {
            k: v * keepf.view(B, *([1] * (v.dim() - 1)))
            for k, v in chunked["obs"].items()
        }
        mask = chunked["metadata"]["mask"] | (~keep)[:, None]
        chunked["metadata"] = {**chunked["metadata"], "mask": mask}
        return chunked

    @staticmethod
    def _slice_history(chunked, k: int):
        """Keep only the newest ``k`` valid chunks per row (valid chunks are
        left-aligned), re-padding to the new batch maximum."""
        mask = chunked["metadata"]["mask"]
        B, H = mask.shape
        if H == 0:
            return chunked
        n_valid = (~mask).sum(dim=1)
        keep_n = n_valid.clamp(max=max(int(k), 0))
        start = n_valid - keep_n
        H_new = int(keep_n.max())
        j = torch.arange(H_new, device=mask.device)
        idx = (start[:, None] + j[None, :]).clamp(max=H - 1)  # (B, H_new)
        new_mask = j[None, :] >= keep_n[:, None]
        rows = torch.arange(B, device=mask.device)[:, None]

        def _take(v):
            out = v[rows, idx]
            return out * (~new_mask).view(B, H_new, *([1] * (out.dim() - 2))).to(
                out.dtype
            )

        return {
            "obs": {key: _take(v) for key, v in chunked["obs"].items()},
            "action": _take(chunked["action"]),
            "metadata": {**chunked["metadata"], "mask": new_mask},
        }

    @staticmethod
    def _mask_demo_prompt(prompt):
        """history_only mode: replace the demo prompt by a single zero chunk
        that is fully masked. P = 1 keeps every shape and the prompt-cache
        path valid while skipping the ViT work for the real demo frames; the
        policy then attends to the sinks and its own history only. Never in
        place: the mask may be the batch's own tensor."""
        B = prompt["action"].shape[0]
        metadata = dict(prompt["metadata"])
        metadata["mask"] = torch.ones(
            B, 1, dtype=torch.bool, device=prompt["action"].device
        )
        return {
            "obs": {k: torch.zeros_like(v[:, :1]) for k, v in prompt["obs"].items()},
            "action": torch.zeros_like(prompt["action"][:, :1]),
            "metadata": metadata,
        }

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
