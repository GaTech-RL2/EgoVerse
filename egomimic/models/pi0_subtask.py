"""PI0.5 with hierarchical subtask prediction.

``PI0SubtaskPytorch`` extends openpi's action-only :class:`PI0Pytorch` with an
auxiliary language-modeling head over the *subtask* portion of the prompt, so a
single forward pass jointly trains:

    prefix  = [images] + HIGH-level instruction + "Subtask:" anchor
                                                  (bidirectional, no loss)
    subtask = LOW-level instruction + EOS         (causal, cross-entropy loss)
    suffix  = action tokens                       (flow-matching MSE)

The high-level (sort) instruction conditions the model; the low-level
(pick-and-place) instruction is the prediction target. Episodes without a sort
track reuse their single pick-and-place instruction as both — handled upstream
in :class:`egomimic.algo.pi.PI`, which assembles the token sequence and the
``token_ar_mask`` / ``token_loss_mask`` this module consumes.

At inference the subtask is decoded autoregressively and the action expert is
conditioned on the realized ``[HIGH + subtask]`` prefix (full hierarchical
inference).

This subclass is intentionally self-contained in egomimic so the pinned
``external/openpi`` submodule stays untouched. It mirrors the parent's idioms
(gradient-checkpoint wrappers, bf16 casts, eager attention for KV-cache decode).
"""

import math

import torch
import torch.nn.functional as F
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

# Default cap on autoregressively-decoded subtask length (tokens). Overridable
# per-run via ``PI`` setting ``model.max_subtask_len`` on the instance.
DEFAULT_MAX_SUBTASK_LEN = 48


class PI0SubtaskPytorch(PI0Pytorch):
    """PI0.5 action model + auxiliary subtask language-modeling head."""

    def __init__(self, config):
        super().__init__(config)
        # The parent compiled ``self.sample_actions`` in __init__; our override
        # does dynamic-length autoregressive decoding (compile-hostile), so drop
        # the cached compiled bound method and fall back to this class's method.
        self.__dict__.pop("sample_actions", None)
        # Populated by PI after construction (tokenizer-owned values).
        self.subtask_eos_id: int | None = None
        self.max_subtask_len: int = DEFAULT_MAX_SUBTASK_LEN
        # When False, the action expert (suffix) is masked off the
        # ``Task: <instruction>`` tokens (+"Subtask:" anchor) via the
        # tokenization-time ``token_gate_mask``; it still attends every obs
        # token — images AND the Embodiment/Control-mode/State proprio blocks —
        # and the predicted subtask. Set by PI from the model config. See
        # forward / sample_actions where the (suffix, instruction) mask block
        # is zeroed. The subtask LM head conditions on the instruction either way.
        self.action_expert_sees_lang: bool = True

    # ------------------------------------------------------------------ #
    # embedding helpers (shared by training forward and AR-decode)
    # ------------------------------------------------------------------ #
    @property
    def _lm_head(self):
        return self.paligemma_with_expert.paligemma.lm_head

    def _embed_images(self, images, img_masks):
        """Embed each camera with SigLIP → ``(image_embs[B,I,w], image_pad[B,I])``."""
        embs = []
        pad_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1)

    def _embed_lang(self, lang_tokens):
        """Embed language tokens with PaliGemma's (sqrt-scaled) embedding."""

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        return self._apply_checkpoint(lang_embed_func, lang_tokens)

    def _assemble_prefix(
        self, image_embs, image_pad, lang_tokens, lang_masks, lang_ar_mask=None
    ):
        """Concat image + language embeddings into a prefix.

        ``lang_ar_mask`` (``token_ar_mask`` restricted to the language tokens)
        is 0 over the bidirectional high-level prompt and 1 over the causal
        subtask tokens. Images are always bidirectional (0). When ``None`` the
        whole language span is bidirectional (used for the high-only prefix
        before any subtask has been decoded).
        """
        lang_embs = self._embed_lang(lang_tokens)
        embs = torch.cat([image_embs, lang_embs], dim=1)
        pad_masks = torch.cat([image_pad, lang_masks], dim=1)

        bsize, num_img = image_pad.shape
        device = image_pad.device
        img_att = torch.zeros(bsize, num_img, dtype=torch.bool, device=device)
        if lang_ar_mask is None:
            lang_att = torch.zeros(
                bsize, lang_tokens.shape[1], dtype=torch.bool, device=device
            )
        else:
            lang_att = lang_ar_mask.to(device=device, dtype=torch.bool)
        att_masks = torch.cat([img_att, lang_att], dim=1)
        return embs, pad_masks, att_masks

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, lang_ar_mask=None
    ):
        """Override: honor per-token ``lang_ar_mask`` (causal subtask tokens).

        The parent forces every language token bidirectional; here the subtask
        block is made causal so the LM head can be trained autoregressively
        while the action suffix still attends to the full prefix+subtask.
        """
        image_embs, image_pad = self._embed_images(images, img_masks)
        return self._assemble_prefix(
            image_embs, image_pad, lang_tokens, lang_masks, lang_ar_mask
        )

    # ------------------------------------------------------------------ #
    # optional gating: hide the Task instruction from the action expert
    # ------------------------------------------------------------------ #
    def _prefix_hide_cols(self, token_gate_mask, lang_pad_masks, num_img):
        """Boolean ``[B, prefix_len]`` columns the action expert must not
        attend: the ``Task: <instruction>`` (+``Subtask:`` anchor) tokens marked
        by ``token_gate_mask`` (built at tokenization from the instruction's
        char span — see ``PI._encode_prefix_with_anchor``), ANDed with the real
        (non-pad) positions. Image columns and the Embodiment/Control-mode/
        State proprio blocks are never hidden — those are obs, not instruction.
        """
        bsize = token_gate_mask.shape[0]
        img = torch.zeros(
            bsize, num_img, dtype=torch.bool, device=token_gate_mask.device
        )
        return torch.cat(
            [img, token_gate_mask.bool() & lang_pad_masks.bool()], dim=1
        )

    def _hide_cols_from_suffix(self, att_2d_masks, hide_cols):
        """Zero the ``(suffix row, hidden col)`` block of the full 2D mask so
        the action expert cannot attend the instruction, while still attending
        images, State, and the subtask. Prefix rows are untouched, so the
        subtask LM head's conditioning on the instruction is unchanged.
        """
        bsize, seqlen, _ = att_2d_masks.shape
        prefix_len = hide_cols.shape[1]
        device = att_2d_masks.device
        cols = torch.zeros(bsize, seqlen, dtype=torch.bool, device=device)
        cols[:, :prefix_len] = hide_cols.to(device)
        rows = torch.zeros(bsize, seqlen, dtype=torch.bool, device=device)
        rows[:, prefix_len:] = True
        return att_2d_masks & ~(rows[:, :, None] & cols[:, None, :])

    def _require_gate(self, observation):
        """Fetch ``token_gate_mask`` off the RAW observation (openpi's
        preprocess drops unknown fields; token tensors pass through unchanged,
        so alignment holds). Hard-fails when gating is on but the mask is
        missing — silently hiding nothing (or everything) would be worse."""
        gate = getattr(observation, "token_gate_mask", None)
        if gate is None:
            raise RuntimeError(
                "action_expert_sees_lang=False but the observation carries no "
                "token_gate_mask — the batch was not tokenized by the subtask "
                "path of PI.process_batch_for_training."
            )
        return gate

    def _preprocess_with_masks(self, observation, *, train):
        """Like the parent's ``_preprocess_observation`` but also returns the
        subtask ``token_ar_mask`` / ``token_loss_mask`` (preserved through
        :func:`preprocess_observation_pytorch`)."""
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(obs.images.values()),
            list(obs.image_masks.values()),
            obs.tokenized_prompt,
            obs.tokenized_prompt_mask,
            obs.state,
            obs.token_ar_mask,
            obs.token_loss_mask,
        )

    # ------------------------------------------------------------------ #
    # training forward: action flow-matching + subtask CE
    # ------------------------------------------------------------------ #
    def forward(self, observation, actions, noise=None, time=None):
        """Returns ``(action_mse[B,H,D], subtask_ce[scalar], subtask_token_acc[scalar])``.

        ``action_mse`` matches the parent's per-element MSE (so the PI algo's
        ``.mean()`` is unchanged); ``subtask_ce`` is the masked next-token loss
        over the subtask tokens, and ``subtask_token_acc`` the teacher-forced
        next-token accuracy over the same positions.
        """
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            token_ar_mask,
            token_loss_mask,
        ) = self._preprocess_with_masks(
            observation, train=True
        )  # mirror PI0Pytorch.forward

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, lang_ar_mask=token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if not self.action_expert_sees_lang:
            # Hide the Task instruction (+anchor) from the action suffix only —
            # via the tokenization-time gate, NOT an ar-mask heuristic, so the
            # Embodiment/Control-mode/State proprio blocks stay visible. Prefix
            # rows (incl. subtask->instruction) and position_ids (built from
            # pad_masks below) are untouched.
            hide_cols = self._prefix_hide_cols(
                self._require_gate(observation),
                lang_masks,
                num_img=prefix_pad_masks.shape[1] - lang_tokens.shape[1],
            )
            att_2d_masks = self._hide_cols_from_suffix(att_2d_masks, hide_cols)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        # ---- action flow-matching loss (same as parent) ----
        action_out = suffix_out[:, -self.config.action_horizon :].to(
            dtype=torch.float32
        )
        v_t = self.action_out_proj(action_out)
        action_mse = F.mse_loss(u_t, v_t, reduction="none")

        # ---- subtask language-modeling loss + token accuracy ----
        subtask_ce, subtask_acc = self._subtask_ce_loss(
            prefix_out, lang_tokens, token_loss_mask
        )
        return action_mse, subtask_ce, subtask_acc

    def _subtask_ce_loss(self, prefix_out, lang_tokens, token_loss_mask):
        """Masked, shifted next-token cross-entropy + token accuracy over the
        subtask tokens. Returns ``(ce[scalar], token_acc[scalar])``.

        ``prefix_out`` are the (post-final-norm) hidden states for the
        ``[images, language]`` prefix; the language slice is its last
        ``lang_len`` positions. With next-token shift, position ``t-1`` predicts
        token ``t``; ``token_loss_mask`` marks the subtask target positions.
        ``token_acc`` is the teacher-forced fraction of those positions whose
        argmax prediction matches the target (0 when there are no targets).
        """
        lang_len = lang_tokens.shape[1]
        lang_hidden = prefix_out[:, -lang_len:, :]

        # Next-token shift: hidden[t] predicts token[t+1]. Gather ONLY the
        # subtask target positions BEFORE the (full-vocab) LM head, so the
        # projection / FP32 logits / cross-entropy / argmax run over the handful
        # of subtask tokens (~N) instead of every prefix position (B*lang_len).
        # Equivalent to the masked full-prefix version (non-target positions
        # contributed 0 to the CE and were excluded from the acc denominator).
        pred_hidden = lang_hidden[:, :-1, :]
        flat_hidden = pred_hidden.reshape(-1, pred_hidden.shape[-1])
        flat_labels = lang_tokens[:, 1:].long().reshape(-1)
        flat_mask = token_loss_mask[:, 1:].to(torch.bool).reshape(-1)
        sel_hidden = flat_hidden[flat_mask]
        sel_labels = flat_labels[flat_mask]

        # No subtask targets in this microbatch (e.g. a frame outside any
        # annotation span). Still run the LM head on one dummy token, zeroed
        # below, so its params are in EVERY rank's autograd graph every step --
        # conditionally skipping it desyncs DDP's gradient all-reduce and hangs
        # NCCL. (The old full-prefix version always ran the head, so was safe.)
        empty = sel_hidden.shape[0] == 0
        if empty:
            sel_hidden = flat_hidden[:1]
            sel_labels = flat_labels[:1]

        logits = self._lm_head(sel_hidden.to(self._lm_head.weight.dtype)).float()
        ce = F.cross_entropy(logits, sel_labels)
        token_acc = (logits.argmax(dim=-1) == sel_labels).float().mean()
        if empty:
            ce = ce * 0.0
            token_acc = token_acc * 0.0
        return ce, token_acc

    # ------------------------------------------------------------------ #
    # hierarchical inference: decode subtask, then sample actions
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_actions(
        self, device, observation, noise=None, num_steps=10, return_subtask=False
    ):
        """Decode the subtask autoregressively from the high-level prompt, then
        flow-match actions conditioned on ``[HIGH + subtask]``.

        The incoming ``observation`` carries only the high-level prompt (ending
        at the subtask anchor) with an all-zero ``token_ar_mask`` — PI builds it
        that way at eval time. Returns the action tensor, or
        ``(actions, subtask_token_ids, subtask_mask)`` when ``return_subtask``.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, token_ar_mask, _ = (
            self._preprocess_with_masks(observation, train=False)
        )

        # Embed images once; only the (growing) language tokens are re-embedded
        # during decoding.
        image_embs, image_pad = self._embed_images(images, img_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        cur_tokens = lang_tokens.clone().long()
        cur_mask = lang_masks.clone().to(torch.bool)
        cur_ar = (
            token_ar_mask.clone().to(torch.bool)
            if token_ar_mask is not None
            else torch.zeros_like(cur_mask)
        )
        finished = torch.zeros(bsize, dtype=torch.bool, device=device)
        ar = torch.arange(bsize, device=device)

        for _ in range(self.max_subtask_len):
            next_tok = self._decode_next_token(
                image_embs, image_pad, cur_tokens, cur_mask, cur_ar
            )
            # Insert each sample's new token right after its last real token,
            # keeping every sequence right-padded and contiguous from index 0.
            insert_idx = cur_mask.long().sum(dim=1)  # == current real length
            zeros_col = torch.zeros(bsize, 1, dtype=cur_tokens.dtype, device=device)
            cur_tokens = torch.cat([cur_tokens, zeros_col], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.zeros(bsize, 1, dtype=torch.bool, device=device)],
                dim=1,
            )
            cur_ar = torch.cat(
                [cur_ar, torch.zeros(bsize, 1, dtype=torch.bool, device=device)], dim=1
            )
            active = ~finished
            cur_tokens[ar, insert_idx] = torch.where(active, next_tok, zeros_col[:, 0])
            cur_mask[ar, insert_idx] = active
            cur_ar[ar, insert_idx] = active  # subtask tokens are causal
            if self.subtask_eos_id is not None:
                finished = finished | (active & (next_tok == self.subtask_eos_id))
            if bool(finished.all()):
                break

        # ---- condition action expert on [images + HIGH + decoded subtask] ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._assemble_prefix(
            image_embs, image_pad, cur_tokens, cur_mask, lang_ar_mask=cur_ar
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Optionally hide the Task instruction from the action expert. The KV
        # cache above still holds every prefix token (subtask decode + KV
        # positions unchanged); only the denoise attention skips the instruction
        # columns, so obs (images + State block) + decoded subtask stay visible.
        attn_prefix_pad_masks = prefix_pad_masks
        if not self.action_expert_sees_lang:
            gate = self._require_gate(observation).bool()
            # cur_tokens grew by the decoded-subtask columns; the gate did not —
            # pad it with False (decoded subtask must stay visible).
            grown = cur_tokens.shape[1] - gate.shape[1]
            gate_full = torch.cat(
                [
                    gate.to(device=device),
                    torch.zeros(bsize, grown, dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            hide_cols = self._prefix_hide_cols(
                gate_full, cur_mask, num_img=image_embs.shape[1]
            )
            attn_prefix_pad_masks = prefix_pad_masks & ~hide_cols

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                attn_prefix_pad_masks=attn_prefix_pad_masks,
            )
            x_t = x_t + dt * v_t
            time += dt

        if return_subtask:
            # The decoded subtask tokens are exactly the real, causally-flagged
            # ones (ar == 1): the high-only prompt arrived right-padded with
            # ar == 0, so tokens were inserted at each sample's real length —
            # inside the original padding region, not the appended columns.
            # A boolean gather over (real & causal) recovers them in order.
            subtask_mask = cur_mask & cur_ar
            return x_t, cur_tokens, subtask_mask
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        attn_prefix_pad_masks=None,
    ):
        """Parent ``denoise_step`` with the suffix->prefix ATTENTION mask
        decoupled from the position/length mask.

        ``attn_prefix_pad_masks`` (defaults to ``prefix_pad_masks``) governs
        which prefix tokens the action expert attends; ``prefix_pad_masks`` still
        sets the KV position offset, so hiding the language instruction never
        shifts the suffix's ``position_ids``.
        """
        if attn_prefix_pad_masks is None:
            attn_prefix_pad_masks = prefix_pad_masks
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )
        suffix_len = suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        prefix_pad_2d_masks = attn_prefix_pad_masks[:, None, :].expand(
            bsize, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        # positions come from the TRUE prefix length, not the (possibly gated)
        # attention mask, so masking the instruction leaves KV offsets intact.
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001, E501
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(
            dtype=torch.float32
        )
        return self.action_out_proj(suffix_out)

    def _decode_next_token(self, image_embs, image_pad, cur_tokens, cur_mask, cur_ar):
        """One greedy decoding step: forward the current prefix and read the
        next-token logits at each sample's last real position."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._assemble_prefix(
            image_embs, image_pad, cur_tokens, cur_mask, lang_ar_mask=cur_ar
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_4d = self._prepare_attention_masks_4d(att_2d)

        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        lang_out = prefix_out[:, -cur_tokens.shape[1] :, :]
        last_idx = cur_mask.long().sum(dim=1) - 1  # last real language position
        bsize = lang_out.shape[0]
        gathered = lang_out[torch.arange(bsize, device=lang_out.device), last_idx]
        logits = self._lm_head(gathered.to(self._lm_head.weight.dtype))
        return torch.argmax(logits, dim=-1)
