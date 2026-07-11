import logging
import os
from collections import OrderedDict
from typing import Literal

import numpy as np
import openpi
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import safetensors
import torch
import torch.nn as nn
from openpi.shared.image_tools import resize_with_pad_torch
from overrides import override
from transformers import AutoTokenizer

from egomimic.algo.algo import Algo
from egomimic.models.pi0_subtask import PI0SubtaskPytorch
from egomimic.models.preprocess_pi_obs import (
    _concat_proprio,
    _empty_lang_placeholders,
    _ensure_bchw,
    _fill_missing_images,
    _SimpleObservation,
    _to_minus1_1,
)
from egomimic.rldb.annotation_processing import (
    AnnotationProcessor,
    DefaultAnnotationProcessor,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.action_utils import (
    PI05_CARTESIAN_ACTION_ENCODING_LEGACY,
    PI05_CARTESIAN_ACTION_ENCODING_NORM_ROT_6D,
    PI05_CARTESIAN_ACTION_ENCODING_RAW_ROT_6D,
    ConverterRegistry,
)

logger = logging.getLogger(__name__)
# Ensure logger propagates to root logger and has appropriate level
# Child loggers inherit from parent, but we explicitly set level to ensure INFO messages appear
logger.setLevel(logging.INFO)
logger.propagate = True  # Explicitly enable propagation (default, but ensures it works)


class PI(Algo):
    """ """

    def __init__(
        self,
        norm_stats,
        camera_transforms,
        domains,
        # ---------------------------
        # Image augmentations
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # Model params
        # ---------------------------
        config,
        # ---------------------------
        ac_keys,
        action_converters,
        # ---------------------------
        # Prompt / tokenization (moved from data config). The PI algo owns
        # the tokenizer because the prompt template is model-specific
        # (paligemma + pi0.5 anchor). See ``process_batch_for_training`` for
        # how these knobs assemble the prompt and produce ``tokenized_*``.
        # ---------------------------
        tokenizer_model_name: str = "google/paligemma-3b-mix-224",
        tokenizer_max_length: int = 128,
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key: str | None = None,
        # Modular annotation processing: a hydra-instantiated
        # AnnotationProcessor that turns the raw batch annotation lists into
        # per-role sampled strings (see egomimic/rldb/annotation_processing.py).
        # When None, a default processor is built from annotation_key +
        # sampling_mode, so existing configs behave identically.
        annotation_processor: AnnotationProcessor | None = None,
        default_prompt: str = "",
        proprio_in_prompt: bool = False,
        embodiment_label: bool = False,
        state_num_bins: int = 256,
        control_mode: dict[str, str] | None = None,
        proprio_keys_for_prompt: list[str] | None = None,
        action_encoding: str = PI05_CARTESIAN_ACTION_ENCODING_LEGACY,
        # ---------------------------
        # Hierarchical subtask prediction (sort regime). When enabled, the
        # high-level (sort) annotation conditions the model and the low-level
        # (pick-and-place) annotation is predicted as language tokens. See
        # ``egomimic/models/pi0_subtask.py``.
        # ---------------------------
        subtask_prediction: bool = False,
        subtask_key: str | None = None,
        subtask_anchor: str = "Subtask: ",
        subtask_loss_weight: float = 1.0,
        max_subtask_len: int = 48,
        # When False, the action expert cannot attend the HIGH-level language
        # instruction (+anchor): it still sees all obs and the predicted
        # subtask, but the instruction is masked from the suffix rows. The
        # subtask LM head still conditions on the instruction either way.
        action_expert_sees_lang: bool = True,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.norm_stats = norm_stats

        # ---- Prompt assembly + tokenization (was in collate_fn) ----
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.tokenizer_max_length = tokenizer_max_length
        self.sampling_mode = sampling_mode
        self.annotation_key = annotation_key
        self.annotation_processor = (
            annotation_processor
            if annotation_processor is not None
            else DefaultAnnotationProcessor(key=annotation_key, strategy=sampling_mode)
        )
        self.default_prompt = default_prompt
        self.subtask_prediction = subtask_prediction
        self.subtask_key = subtask_key
        self.subtask_anchor = subtask_anchor
        self.subtask_loss_weight = subtask_loss_weight
        self.max_subtask_len = max_subtask_len
        self.action_expert_sees_lang = action_expert_sees_lang
        self.proprio_in_prompt = proprio_in_prompt
        self.embodiment_label = embodiment_label
        self.state_num_bins = state_num_bins
        self.control_mode = control_mode
        # Default to the canonical concat key produced by each embodiment's
        # transform_list (ConcatKeys with delete_old_keys removes per-arm zarr keys).
        self.proprio_keys_for_prompt = (
            list(proprio_keys_for_prompt)
            if proprio_keys_for_prompt is not None
            else ["observations.state.ee_pose"]
        )
        self._state_bin_edges = np.linspace(-1.0, 1.0, state_num_bins + 1)[:-1]

        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        if "image_resolution" in kwargs:
            self.image_resolution = kwargs["image_resolution"]
        self.pi_cam_keys = kwargs.get(
            "pi_cam_keys", ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        )
        self.config = config
        self.action_encoding = action_encoding

        self.ac_keys = ac_keys

        self.domains = domains

        self.device = None

        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}

        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
            self.lang_keys[embodiment_id] = []
            for key in norm_stats.keys_of_type("action_keys", embodiment_id):
                if (
                    norm_stats.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys[embodiment_id] = key
            for key in norm_stats.keys_of_type("camera_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in norm_stats.keys_of_type("proprio_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.lang_keys[embodiment_id].append(key)

        self.num_steps = getattr(self.config, "num_sampling_steps", 10)
        self.is_6dof = kwargs.get("is_6dof", True)

        self.action_converters = action_converters

        self.action_registry = ConverterRegistry()

        arcfg = self.action_converters
        default_ac_key = getattr(arcfg, "ac_key", "actions_cartesian")

        for emb_name, conv_obj in arcfg.rules.items():
            emb_id = get_embodiment_id(emb_name)
            self.action_registry.register(emb_id, self.ac_keys[emb_id], conv_obj)

        fb_obj = arcfg.fallback
        self.action_registry.register("*", default_ac_key, fb_obj)
        self.action_registry.register("*", "*", fb_obj)

        # Create the model
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=self.config.pytorch_training_precision,
            action_dim=self.config.model.action_dim,
            action_horizon=self.config.model.action_horizon,
            max_token_len=self.config.model.max_token_len,
            paligemma_variant=getattr(
                self.config.model, "paligemma_variant", "gemma_2b"
            ),
            action_expert_variant=getattr(
                self.config.model, "action_expert_variant", "gemma_300m"
            ),
            pi05=getattr(config.model, "pi05", False),
        )

        if self.subtask_prediction:
            # Hierarchical variant: adds the subtask LM head + autoregressive
            # subtask decoding at inference. The auxiliary head reuses
            # PaliGemma's tied lm_head, so pretrained weights load unchanged.
            self.model = PI0SubtaskPytorch(model_cfg)
            self.model.max_subtask_len = self.max_subtask_len
            self.model.subtask_eos_id = self.tokenizer.eos_token_id
            self.model.action_expert_sees_lang = self.action_expert_sees_lang
        else:
            self.model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

        if self.config.pytorch_weight_path is not None:
            model_path = os.path.join(
                self.config.pytorch_weight_path, "model.safetensors"
            )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Pretrained weight file not found: {model_path}"
                )
            target = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )
            safetensors.torch.load_model(target, model_path)
            logger.info(
                "Loaded pretrained weights from %s (%d parameters)",
                model_path,
                sum(p.numel() for p in target.parameters()),
            )
        else:
            logger.warning("No pytorch_weight_path specified — training from scratch")
        self.nets = nn.ModuleDict()
        self.nets["policy"] = self.model

    def _control_mode_for(self, emb_name: str | None) -> str:
        if self.control_mode and emb_name is not None:
            for key, val in self.control_mode.items():
                if key.lower() in emb_name:
                    return val
        raise ValueError(
            f"control_mode has no entry matching embodiment {emb_name!r} "
            f"(declared keys: {sorted(self.control_mode or {})}). Control modes "
            "are declared per-embodiment in the model yaml; there is no fallback."
        )

    def _discretize_state_for_sample(self, _batch, sample_idx: int) -> str | None:
        """Pick the latest proprio timestep for sample i, clip to [-1, 1],
        digitize into ``state_num_bins`` bins, and return as a space-joined
        string. Returns None if no proprio key is present.
        """
        parts = []
        for k in self.proprio_keys_for_prompt:
            if k not in _batch:
                continue
            v = _batch[k]
            if isinstance(v, torch.Tensor):
                v = v[sample_idx].detach().cpu().numpy()
            else:
                v = np.asarray(v)[sample_idx]
            v = np.asarray(v, dtype=np.float32)
            while v.ndim > 1:
                v = v[-1]
            parts.append(v.reshape(-1))
        if not parts:
            return None
        state = np.concatenate(parts, axis=-1)
        state = np.clip(state, -1.0, 1.0)
        bins = np.digitize(state, bins=self._state_bin_edges) - 1
        return " ".join(map(str, bins.tolist()))

    def _sample_from_annotation(self, key: str | None, _batch, batch_size: int):
        """Sample one annotation string per item from the raw list field
        ``key`` (``list[list[str]]``). Returns ``None`` per item when that
        item's list is empty / the key is absent, so callers can apply their
        own fallback (e.g. high→low for sort-less episodes)."""
        if key is None or key not in _batch:
            return [None] * batch_size
        out = []
        for sample in _batch[key]:
            if not sample:
                out.append(None)
            elif self.sampling_mode == "random":
                out.append(sample[random.randint(0, len(sample) - 1)])
            else:  # "first"
                out.append(sample[0])
        return out

    def _prompt_blocks(self, prompt: str, emb_name: str, _batch, i: int) -> str:
        """Render the comma-joined pi0.5 prompt blocks (``Task``, optional
        ``Embodiment`` / ``Control mode`` / ``State``) for sample ``i`` without
        the trailing anchor."""
        blocks = [f"Task: {prompt}"]
        if self.embodiment_label:
            blocks.append(f"Embodiment: {emb_name}")
        if self.control_mode:
            blocks.append(f"Control mode: {self._control_mode_for(emb_name)}")
        if self.proprio_in_prompt:
            state_str = self._discretize_state_for_sample(_batch, i)
            if state_str is not None:
                blocks.append(f"State: {state_str}")
        return ", ".join(blocks)

    def _build_prompts(
        self, _batch, embodiment_name: str, batch_size: int
    ) -> list[str]:
        """Sample one prompt per item via the annotation processor and
        splice in any of the active blocks. Returns ``batch_size`` strings.

        Sampling lives in ``self.annotation_processor`` (role ``task``);
        ``None`` per item (no annotation / missing key) falls back to
        ``default_prompt``. Embodiment is known per-batch (one DataLoader
        per embodiment), so we don't re-derive it per sample.
        """
        sampled = self.annotation_processor(_batch, batch_size)["task"]
        prompts = [self.default_prompt if p is None else p for p in sampled]

        any_block_active = (
            self.proprio_in_prompt or self.embodiment_label or bool(self.control_mode)
        )
        if not any_block_active:
            return prompts

        emb_name = embodiment_name.lower().replace("_", " ")
        spliced = []
        for i, prompt in enumerate(prompts):
            spliced.append(
                self._prompt_blocks(prompt, emb_name, _batch, i) + ";\nAction: "
            )
        return spliced

    def _build_subtask_prompts(
        self, _batch, embodiment_name: str, batch_size: int
    ) -> tuple[list[str], list[str | None], list[int]]:
        """Assemble the (high-level prefix, low-level subtask-target) pair per
        item for hierarchical training.

        The high-level prompt comes from ``annotation_key`` (the sort goal); the
        low-level subtask target comes from ``subtask_key`` (pick-and-place).
        For sort-less episodes the high list is empty, so the high prompt falls
        back to the sampled low instruction — i.e. the single pick-and-place
        annotation is used as *both* the high-level and low-level instruction.
        The rendered prefix ends with ``subtask_anchor`` (where decoding begins).

        Also returns ``instr_char_lens``: per item, the char length of the
        leading ``Task: <instruction>`` block inside the rendered prefix (the
        instruction is always the FIRST block; Embodiment/Control-mode/State
        follow). Used to build ``token_gate_mask`` when
        ``action_expert_sees_lang`` is off — only these chars (+ the anchor)
        are hidden from the action expert; the State proprio block stays
        visible.
        """
        high = self._sample_from_annotation(self.annotation_key, _batch, batch_size)
        low = self._sample_from_annotation(self.subtask_key, _batch, batch_size)

        emb_name = embodiment_name.lower().replace("_", " ")
        prefixes: list[str] = []
        targets: list[str | None] = []
        instr_char_lens: list[int] = []
        for i in range(batch_size):
            low_i = low[i]
            # high falls back to low (sort-less episode), then to default_prompt.
            high_i = high[i] if high[i] is not None else low_i
            if high_i is None:
                high_i = self.default_prompt
            # The subtask anchor is appended at tokenization time (as protected
            # tokens), so it is never lost to truncation — see
            # ``_encode_prefix_with_anchor``.
            prefixes.append(self._prompt_blocks(high_i, emb_name, _batch, i) + ";\n")
            # ``Task: <instruction>`` is the first block rendered by
            # ``_prompt_blocks``, so its char span is exactly [0, this length).
            instr_char_lens.append(len(f"Task: {high_i}"))
            # The subtask target is always the low-level pick-and-place phrasing.
            # When the episode has no sort goal, ``high_i`` was set to this same
            # instruction above, so the model conditions on and predicts the
            # same text ("use as both"). ``None`` ⇒ no subtask to predict for
            # this frame (zero LM loss; nothing decoded at inference).
            targets.append(low_i)
        return prefixes, targets, instr_char_lens

    def _tokenize_prompts(self, prompts: list[str]) -> dict:
        enc = self.tokenizer(
            prompts,
            padding="max_length"
            if self.tokenizer_max_length is not None
            else "longest",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        attention_mask = enc["attention_mask"].bool()
        token_loss_mask = attention_mask.clone()
        token_loss_mask[:, -1] = False
        return {
            "tokenized_prompt": enc["input_ids"].requires_grad_(False),
            "tokenized_mask": attention_mask.requires_grad_(False),
            "token_loss_mask": token_loss_mask.requires_grad_(False),
            "token_ar_mask": attention_mask.clone().requires_grad_(False),
        }

    def _encode_prefix_with_anchor(
        self, prefix_text: str, instr_chars: int | None = None
    ) -> tuple[list[int], list[bool] | None]:
        """Tokenize ``prefix_text`` and append the subtask anchor, guaranteeing
        the anchor survives truncation to ``tokenizer_max_length``.

        The anchor (``subtask_anchor``, e.g. ``"Subtask: "``) is where decoding
        begins, so it must never be cut. We reserve room for it and truncate the
        high-level block tail instead. Returns ``(ids, gate)`` with length ≤ L.

        ``gate`` (only built when ``instr_chars`` is given; else ``None``) marks
        the tokens the action expert must not attend under
        ``action_expert_sees_lang=False``: tokens whose char span overlaps the
        leading ``Task: <instruction>`` block ``[0, instr_chars)`` plus the
        anchor tokens. BOS (empty span) and the Embodiment/Control-mode/State
        blocks stay ``False`` (visible) — proprio is obs, not instruction.
        """
        L = self.tokenizer_max_length
        anchor_ids = self.tokenizer(self.subtask_anchor, add_special_tokens=False)[
            "input_ids"
        ]
        if instr_chars is None:
            blocks = self.tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
            blocks = blocks[: max(0, L - len(anchor_ids))]
            return (blocks + anchor_ids)[:L], None

        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "action_expert_sees_lang=False needs a fast tokenizer "
                "(return_offsets_mapping) to locate the Task-instruction tokens; "
                f"{type(self.tokenizer).__name__} is not fast."
            )
        enc = self.tokenizer(
            prefix_text, add_special_tokens=True, return_offsets_mapping=True
        )
        blocks = enc["input_ids"]
        # A token is instruction iff its char span overlaps [0, instr_chars).
        # BOS/specials carry an empty (0, 0) span -> no overlap -> visible.
        gate = [s < instr_chars and e > s for (s, e) in enc["offset_mapping"]]
        keep = max(0, L - len(anchor_ids))
        blocks, gate = blocks[:keep], gate[:keep]
        ids = (blocks + anchor_ids)[:L]
        gate = (gate + [True] * len(anchor_ids))[:L]  # anchor is hidden too
        return ids, gate

    def _tokenize_prompts_with_subtask(
        self,
        prefixes: list[str],
        subtasks: list[str | None],
        instr_char_lens: list[int] | None = None,
    ) -> dict:
        """Tokenize the hierarchical sequence ``[prefix][anchor][subtask]<eos>``
        per item.

        The prefix (high-level instruction + anchor) is bidirectional with no
        loss; the subtask (low-level instruction + EOS) is the autoregressive
        target. Returns the same field names as :meth:`_tokenize_prompts` but
        with ``token_ar_mask`` = 0 over the prefix / 1 over the subtask, and
        ``token_loss_mask`` True only on subtask tokens. Sequences are
        right-padded to ``tokenizer_max_length``; the prefix (incl. anchor) is
        never truncated away (subtask is dropped first if space is tight).

        ``token_gate_mask`` marks the ``Task: <instruction>`` (+anchor) tokens
        to hide from the action expert when ``action_expert_sees_lang`` is off;
        it is all-False (nothing hidden) when ``instr_char_lens`` is ``None``
        (toggle on — skips the extra offsets tokenization pass entirely).
        """
        L = self.tokenizer_max_length
        B = len(prefixes)
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id

        ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.bool)
        ar = torch.zeros((B, L), dtype=torch.bool)
        loss = torch.zeros((B, L), dtype=torch.bool)
        gate = torch.zeros((B, L), dtype=torch.bool)

        for i in range(B):
            pre, pre_gate = self._encode_prefix_with_anchor(
                prefixes[i],
                None if instr_char_lens is None else instr_char_lens[i],
            )
            sub = []
            if subtasks[i]:
                sub = self.tokenizer(subtasks[i], add_special_tokens=False)["input_ids"]
                if eos_id is not None:
                    sub = sub + [eos_id]
            sub_full = len(sub)
            sub = sub[: max(0, L - len(pre))]
            if len(sub) < sub_full:
                # EOS is the LAST subtask token, so any truncation drops it
                # first: the sample trains continuation-without-stop (greedy
                # decode then runs to max_subtask_len for such prompts). If the
                # whole subtask is gone the frame silently contributes zero CE.
                # Surface it — silent label corruption is worse than log spam.
                self._subtask_trunc_count = (
                    getattr(self, "_subtask_trunc_count", 0) + 1
                )
                if self._subtask_trunc_count == 1 or (
                    self._subtask_trunc_count % 500 == 0
                ):
                    logger.warning(
                        "Subtask target truncated (%s): prefix=%d + subtask=%d "
                        "> max_length=%d — EOS dropped%s. Occurrence #%d. "
                        "Consider a shorter prompt (proprio_in_prompt?) or a "
                        "larger tokenizer_max_length.",
                        subtasks[i][:60],
                        len(pre),
                        sub_full,
                        L,
                        "; ENTIRE subtask dropped (zero CE)" if not sub else "",
                        self._subtask_trunc_count,
                    )
            n_pre, n = len(pre), len(pre) + len(sub)
            ids[i, :n] = torch.tensor(pre + sub, dtype=torch.long)
            attn[i, :n] = True
            ar[i, n_pre:n] = True  # subtask tokens are causal
            loss[i, n_pre:n] = True  # subtask tokens are CE targets
            if pre_gate is not None:
                gate[i, :n_pre] = torch.tensor(pre_gate, dtype=torch.bool)
        return {
            "tokenized_prompt": ids.requires_grad_(False),
            "tokenized_mask": attn.requires_grad_(False),
            "token_loss_mask": loss.requires_grad_(False),
            "token_ar_mask": ar.requires_grad_(False),
            "token_gate_mask": gate.requires_grad_(False),
        }

    def _tokenize_highonly(
        self, prefixes: list[str], instr_char_lens: list[int] | None = None
    ) -> dict:
        """Tokenize just the high-level prefix (+ anchor) for autoregressive
        subtask decoding at inference. All tokens bidirectional (``ar`` = 0),
        no loss. Uses the same anchor-preserving encoder as the full sequence so
        the decode seed always ends in the anchor. Written under ``*_hl`` keys
        so it coexists with the full sequence used for the training/val loss.
        ``token_gate_mask_hl`` mirrors :meth:`_tokenize_prompts_with_subtask`'s
        ``token_gate_mask`` (all-False when ``instr_char_lens`` is ``None``)."""
        L = self.tokenizer_max_length
        B = len(prefixes)
        pad_id = self.tokenizer.pad_token_id or 0
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.bool)
        gate = torch.zeros((B, L), dtype=torch.bool)
        for i in range(B):
            pre, pre_gate = self._encode_prefix_with_anchor(
                prefixes[i],
                None if instr_char_lens is None else instr_char_lens[i],
            )
            ids[i, : len(pre)] = torch.tensor(pre, dtype=torch.long)
            attn[i, : len(pre)] = True
            if pre_gate is not None:
                gate[i, : len(pre)] = torch.tensor(pre_gate, dtype=torch.bool)
        return {
            "tokenized_prompt_hl": ids.requires_grad_(False),
            "tokenized_mask_hl": attn.requires_grad_(False),
            "token_ar_mask_hl": torch.zeros_like(attn).requires_grad_(False),
            "token_gate_mask_hl": gate.requires_grad_(False),
        }

    def _decode_subtask_text(self, subtask_ids, subtask_mask) -> list[str]:
        """Detokenize per-sample decoded subtask ids (``[B, T]`` + bool mask)
        into strings for logging."""
        texts = []
        for i in range(subtask_ids.shape[0]):
            keep = subtask_mask[i].to(torch.bool)
            toks = subtask_ids[i][keep].tolist()
            texts.append(self.tokenizer.decode(toks, skip_special_tokens=True))
        return texts

    def _action_stats(self, embodiment_id: int, ac_key: str) -> dict:
        try:
            return self.norm_stats.norm_stats[embodiment_id][ac_key]
        except KeyError as exc:
            raise KeyError(
                f"Missing norm stats for action key {ac_key!r} "
                f"and embodiment id {embodiment_id}"
            ) from exc

    def _unnormalize_action(
        self, action: torch.Tensor, embodiment_id: int, ac_key: str
    ):
        return self.norm_stats.unnormalize(
            {ac_key: action.clone(), "embodiment": embodiment_id},
            embodiment_id,
        )[ac_key].to(action.device)

    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batchs that works with pi0.
        """
        processed_batch = {}

        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.norm_stats.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            # Carry forward episode_hash from the data dict (other keys are
            # built inline below via _tokenize_prompts).
            if "episode_hash" in _batch:
                processed_batch[embodiment_id]["episode_hash"] = _batch["episode_hash"]

            ac_key = self.ac_keys[embodiment_id]
            if ac_key not in processed_batch[embodiment_id]:
                raise KeyError(
                    f"Missing action key '{ac_key}' for embodiment {embodiment_id}. "
                    f"Incoming keys were: {list(_batch.keys())}"
                )
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")

            B, S, _ = processed_batch[embodiment_id][ac_key].shape

            # Build prompts + tokenize. Reads raw `annotations` (list[list[str]])
            # left in `_batch` by `annotation_collate`, plus per-sample proprio
            # tensors from `_batch` for the optional State block.
            if self.subtask_prediction:
                # Hierarchical: condition on the high-level prefix and predict
                # the low-level subtask. The "full" tokenization (prefix +
                # subtask) drives the training/val loss; the "high-only" one
                # (`*_hl` keys) seeds autoregressive subtask decoding at
                # inference (rollout has no subtask target, so it carries an
                # empty subtask ⇒ zero LM loss).
                prefixes, targets, instr_lens = self._build_subtask_prompts(
                    _batch, embodiment_name, B
                )
                # Instruction token spans are only located (extra offsets
                # tokenization pass) when the action expert must not see them.
                if self.action_expert_sees_lang:
                    instr_lens = None
                processed_batch[embodiment_id]["sampled_prompt"] = prefixes
                processed_batch[embodiment_id].update(
                    self._tokenize_prompts_with_subtask(prefixes, targets, instr_lens)
                )
                processed_batch[embodiment_id].update(
                    self._tokenize_highonly(prefixes, instr_lens)
                )
            else:
                prompts = self._build_prompts(_batch, embodiment_name, B)
                processed_batch[embodiment_id]["sampled_prompt"] = prompts
                processed_batch[embodiment_id].update(self._tokenize_prompts(prompts))
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=self.device
            )
            # Samples are already normalized by MultiDataset.__getitem__.
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )

            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        if not processed_batch:
            raise ValueError(
                f"No valid embodiments found in batch. Batch contained: {list(batch.keys())}, "
                f"but ac_keys only has: {list(self.ac_keys.keys())}"
            )

        return processed_batch

    @override
    def forward_training(self, batch):
        """
        One iteration of training. Sequentially, forward pass loss, Compute forward pass and compute losses.  Return predictions dictionary.  HPT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        # self.nets["policy"].train()
        predictions = OrderedDict()
        for embodiment_id, _batch in batch.items():
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            camera_keys = self.camera_keys.get(embodiment_id, self.pi_cam_keys)
            embodiment_name = get_embodiment(embodiment_id).lower()
            processed_obs, action = self._robomimic_to_pi_data(
                _batch,
                camera_keys,
                proprio_keys,
                lang_keys,
                ac_key,
                embodiment_name,
            )

            losses = self.nets["policy"].forward(processed_obs, action)

            if self.subtask_prediction:
                # PI0SubtaskPytorch returns (action_mse[B,H,D], subtask_ce, subtask_token_acc).
                action_mse, subtask_ce, subtask_acc = losses
                loss = action_mse.mean()
                predictions[f"{embodiment_name}_subtask_loss"] = subtask_ce
                predictions[f"{embodiment_name}_subtask_acc"] = subtask_acc
            else:
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(
                        losses, device=action.device, dtype=torch.float32
                    )
                loss = losses.mean()

            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss

        return predictions

    @override
    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Unnormalize data here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            unnorm_preds (dict): {
                <embodiment_name>_<ac_key>: torch.Tensor (B, Seq, D),
                <embodiment_name>_loss: torch.Tensor (1),  # flow-matching val loss
            }
        """
        unnorm_preds = {}
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_key = self.ac_keys[embodiment_id]
                camera_keys = self.camera_keys.get(embodiment_id, self.pi_cam_keys)
                embodiment_name = get_embodiment(embodiment_id).lower()
                processed_obs, action = self._robomimic_to_pi_data(
                    _batch,
                    camera_keys,
                    proprio_keys,
                    lang_keys,
                    ac_key,
                    embodiment_name,
                )

                # Flow-matching val loss — same call as forward_training.
                losses = self.nets["policy"].forward(processed_obs, action)
                if self.subtask_prediction:
                    action_mse, subtask_ce, subtask_acc = losses
                    unnorm_preds[f"{embodiment_name}_loss"] = action_mse.mean()
                    unnorm_preds[f"{embodiment_name}_subtask_loss"] = subtask_ce
                    unnorm_preds[f"{embodiment_name}_subtask_acc"] = subtask_acc
                else:
                    if isinstance(losses, (list, tuple)):
                        losses = torch.stack(losses)
                    elif not isinstance(losses, torch.Tensor):
                        losses = torch.tensor(
                            losses, device=action.device, dtype=torch.float32
                        )
                    unnorm_preds[f"{embodiment_name}_loss"] = losses.mean()

                if self.subtask_prediction:
                    # Hierarchical inference: decode the subtask from the
                    # high-only prefix, then condition action sampling on the
                    # realized [high + subtask] prefix.
                    sample_obs, _ = self._robomimic_to_pi_data(
                        _batch,
                        camera_keys,
                        proprio_keys,
                        lang_keys,
                        ac_key,
                        embodiment_name,
                        lang_prefix="hl",
                    )
                    pred_actions, subtask_ids, subtask_mask = self.nets[
                        "policy"
                    ].sample_actions(
                        device=self.device,
                        observation=sample_obs,
                        noise=None,
                        num_steps=self.num_steps,
                        return_subtask=True,
                    )
                    unnorm_preds[f"{embodiment_name}_subtask_pred"] = (
                        self._decode_subtask_text(subtask_ids, subtask_mask)
                    )
                else:
                    pred_actions = self.nets["policy"].sample_actions(
                        device=self.device,
                        observation=processed_obs,
                        noise=None,
                        num_steps=self.num_steps,
                    )

                pred_actions = pred_actions.clone()

                predictions = OrderedDict()
                ref = _batch[ac_key]
                B, T, D = ref.shape

                converter = self.action_registry.get(embodiment_id, ac_key)
                if self.action_encoding == PI05_CARTESIAN_ACTION_ENCODING_RAW_ROT_6D:
                    pred_actions_orig = converter.from32_raw_rotation(
                        pred_actions,
                        stats=self._action_stats(embodiment_id, ac_key),
                        norm_mode=self.norm_stats.norm_mode,
                        unnormalize_non_rotation=True,
                    )
                    unnorm_actions = {ac_key: pred_actions_orig[:, :T, :D]}
                elif self.action_encoding == PI05_CARTESIAN_ACTION_ENCODING_NORM_ROT_6D:
                    # Extract the normalized xyz+6D(+gripper) action, then
                    # unnormalize via the standard pipeline (stats were computed
                    # over the 6D representation) to get raw 6D actions.
                    pred_6d = converter.from32_norm_6d(pred_actions)
                    predictions[ac_key] = pred_6d[:, :T, :D]
                    unnorm_actions = self.norm_stats.unnormalize(
                        predictions, embodiment_id
                    )
                elif self.action_encoding == PI05_CARTESIAN_ACTION_ENCODING_LEGACY:
                    pred_actions_orig = converter.from32(pred_actions)
                    pred = pred_actions_orig[:, :T, :D]
                    predictions[ac_key] = pred
                    unnorm_actions = self.norm_stats.unnormalize(
                        predictions, embodiment_id
                    )
                else:
                    raise ValueError(
                        f"Unsupported PI0.5 action_encoding: {self.action_encoding!r}"
                    )
                for key in unnorm_actions:
                    unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)
        """
        loss_dict = OrderedDict()
        total_action_loss = None
        total_subtask_loss = None
        total_subtask_acc = None
        n_subtask = 0

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            if total_action_loss is None:
                total_action_loss = torch.tensor(0.0, device=bc_loss.device)
            total_action_loss += bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss  # for logging

            subtask_key = f"{embodiment_name}_subtask_loss"
            if subtask_key in predictions:
                sub_loss = predictions[subtask_key]
                if total_subtask_loss is None:
                    total_subtask_loss = torch.tensor(0.0, device=sub_loss.device)
                total_subtask_loss += sub_loss
                n_subtask += 1
                loss_dict[subtask_key] = sub_loss  # for logging

            acc_key = f"{embodiment_name}_subtask_acc"
            if acc_key in predictions:
                sub_acc = predictions[acc_key]
                if total_subtask_acc is None:
                    total_subtask_acc = torch.tensor(0.0, device=sub_acc.device)
                total_subtask_acc += sub_acc
                loss_dict[acc_key] = sub_acc  # per-embodiment token accuracy (logging)

        # in the case we put all embodiments in one batch, get rid of this norm.
        action_loss = total_action_loss / len(self.domains)
        if total_subtask_loss is not None:
            # Hierarchical training: action flow-matching + weighted subtask CE.
            subtask_loss = total_subtask_loss / len(self.domains)
            loss_dict["subtask_loss"] = subtask_loss
            loss_dict["action_loss"] = (
                action_loss + self.subtask_loss_weight * subtask_loss
            )
        else:
            loss_dict["action_loss"] = action_loss

        # Mean teacher-forced subtask token accuracy (logged, not backpropped).
        if total_subtask_acc is not None:
            loss_dict["subtask_token_acc"] = total_subtask_acc / max(n_subtask, 1)

        return loss_dict

    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def _robomimic_to_pi_data(
        self,
        batch,
        cam_keys,
        proprio_keys,
        lang_keys,
        ac_key,
        embodiment,
        lang_prefix="full",
    ):
        """Wrap a processed batch into a ``_SimpleObservation`` + 32D actions.

        ``lang_prefix`` selects which tokenization to feed the model:
          - ``"full"``  → the ``[prefix][subtask]`` sequence (training/val loss).
          - ``"hl"``    → the high-only prefix (``*_hl`` keys) used to seed
            autoregressive subtask decoding at inference; the loss mask is
            zeroed so it never contributes a language loss.
        Only meaningful when ``subtask_prediction`` is on; otherwise ``"full"``
        reads the single tokenization built by :meth:`_tokenize_prompts`.
        """
        if ac_key not in batch:
            raise KeyError(f"Missing action key '{ac_key}' in batch")

        device = self.device
        action = batch[ac_key].to(device)
        image_resolution = getattr(self, "image_resolution", (224, 224))
        required_cam_keys = getattr(self, "pi_cam_keys", cam_keys)

        present_flags = {
            k: (
                k in batch and isinstance(batch[k], torch.Tensor) and batch[k].ndim == 4
            )
            for k in required_cam_keys
        }

        emb_id = get_embodiment_id(embodiment)  # embodiment is a name string
        converter = self.action_registry.get(emb_id, ac_key)
        if self.action_encoding == PI05_CARTESIAN_ACTION_ENCODING_RAW_ROT_6D:
            raw_action = self._unnormalize_action(action, emb_id, ac_key)
            action32 = converter.to32_raw_rotation(
                raw_action,
                normalized_actions=action,
                stats=self._action_stats(emb_id, ac_key),
                norm_mode=self.norm_stats.norm_mode,
            )
        elif self.action_encoding == PI05_CARTESIAN_ACTION_ENCODING_NORM_ROT_6D:
            # Action is already a normalized xyz+6D(+gripper) chunk (the
            # ypr->6D conversion happened in the CartesianYPRToRot6D data
            # transform). Just pack it into the 32D vector.
            action32 = converter.to32_norm_6d(action)
        elif self.action_encoding == PI05_CARTESIAN_ACTION_ENCODING_LEGACY:
            action32 = converter.to32(action)
        else:
            raise ValueError(
                f"Unsupported PI0.5 action_encoding: {self.action_encoding!r}"
            )

        # OpenPI expects a fixed camera tuple. Human datasets only provide
        # `base_0_rgb`, so duplicate that view into the missing wrist slots and
        # mark those synthesized views as masked out below.
        raw_images = _fill_missing_images(batch, required_cam_keys, device)

        # ---- Images (dict[str, Tensor]) ----
        images = {}
        for k in required_cam_keys:
            img = _ensure_bchw(raw_images[k])
            img = _to_minus1_1(img)
            if img.shape[2:] != tuple(image_resolution):
                img = resize_with_pad_torch(img, *image_resolution)
            if img.ndim != 4:
                raise ValueError(
                    f"Expected 4D BCHW image for key '{k}', got shape {tuple(img.shape)}"
                )
            images[k] = img

        if not images:
            raise ValueError("No camera tensors found for the provided cam_keys.")

        # ---- Proprio -> state [B, D] ----
        state = _concat_proprio(batch, proprio_keys, device)
        if state.numel() == 0:
            B = next(iter(images.values())).shape[0]
            state = torch.zeros(B, 0, device=device)
        else:
            B = state.shape[0]

        # ---- Masks for duplicated images + empty language fields ----
        image_masks = {
            k: (
                torch.ones(B, dtype=torch.bool, device=device)
                if present_flags[k]
                else torch.zeros(B, dtype=torch.bool, device=device)
            )
            for k in images.keys()
        }

        if lang_prefix == "hl":
            prompt_key, mask_key, ar_key, gate_key = (
                "tokenized_prompt_hl",
                "tokenized_mask_hl",
                "token_ar_mask_hl",
                "token_gate_mask_hl",
            )
        else:
            prompt_key, mask_key, ar_key, gate_key = (
                "tokenized_prompt",
                "tokenized_mask",
                "token_ar_mask",
                "token_gate_mask",
            )
        has_lang = prompt_key in batch and batch[prompt_key].numel() > 0
        if has_lang:
            tokenized_prompt = batch[prompt_key].to(device)
            tokenized_prompt_mask = batch[mask_key].to(device)
            token_ar_mask = batch[ar_key].to(device)
            # The high-only prefix carries no subtask target, so it has no loss
            # mask; zero it to match the prompt length.
            if lang_prefix == "hl" or "token_loss_mask" not in batch:
                token_loss_mask = torch.zeros_like(tokenized_prompt_mask)
            else:
                token_loss_mask = batch["token_loss_mask"].to(device)
            token_gate_mask = (
                batch[gate_key].to(device)
                if gate_key in batch
                else torch.zeros_like(tokenized_prompt_mask)
            )
        else:
            tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
                _empty_lang_placeholders(B, device)
            )
            token_gate_mask = torch.zeros_like(tokenized_prompt_mask)

        # ---- Wrap into simple observation (helpers) ----
        observation = _SimpleObservation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
            # Instruction-token gate for action_expert_sees_lang=False. Rides on
            # the raw observation (openpi's preprocess drops unknown fields, so
            # the model reads it via getattr before preprocessing; token fields
            # pass through preprocessing unchanged, so alignment holds).
            token_gate_mask=token_gate_mask,
        )

        # Do NOT call _preprocessing here; the PI model does it internally.
        return observation, action32

    def _clone_batch(self, batch):
        """Recursively clones all tensors inside a nested dictionary."""
        if isinstance(batch, dict):
            return {key: self._clone_batch(val) for key, val in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.clone()
        else:
            return batch  # Return as is for non-tensor types

    def _extract_xyz(self, x):
        """
        Extract xyz (3D position) and rotation from 6DoF or 6DoF+gripper actions.

        Supports:
        - 6: 6DoF (single arm)
        - 7: 6DoF + gripper (single arm)
        - 12: 2 arms × 6DoF
        - 14: 2 arms × (6DoF + gripper)

        Returns:
            xyz: Tensor with only xyz per arm (shape: ..., 3) or (..., 6) for dual-arm.
            rot: Tensor with only rotation per arm (shape: ..., 3) or (..., 6) for dual-arm.
        """
        if x.shape[-1] == 6:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 7:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 12:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 6:9]
            rot_left = x[..., 9:12]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        elif x.shape[-1] == 14:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 7:10]
            rot_left = x[..., 10:13]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        else:
            raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
