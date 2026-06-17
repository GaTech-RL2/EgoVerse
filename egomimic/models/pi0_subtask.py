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
from torch import Tensor

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
        """Returns ``(action_mse[B,H,D], subtask_ce[scalar])``.

        ``action_mse`` matches the parent's per-element MSE (so the PI algo's
        ``.mean()`` is unchanged); ``subtask_ce`` is the masked next-token loss
        over the subtask tokens.
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

        # ---- subtask language-modeling loss ----
        subtask_ce = self._subtask_ce_loss(prefix_out, lang_tokens, token_loss_mask)
        return action_mse, subtask_ce

    def _subtask_ce_loss(self, prefix_out, lang_tokens, token_loss_mask) -> Tensor:
        """Masked, shifted next-token cross-entropy over the subtask tokens.

        ``prefix_out`` are the (post-final-norm) hidden states for the
        ``[images, language]`` prefix; the language slice is its last
        ``lang_len`` positions. With next-token shift, position ``t-1`` predicts
        token ``t``; ``token_loss_mask`` marks the subtask target positions.
        """
        lang_len = lang_tokens.shape[1]
        lang_hidden = prefix_out[:, -lang_len:, :]
        logits = self._lm_head(lang_hidden.to(self._lm_head.weight.dtype)).float()

        shift_logits = logits[:, :-1, :]
        shift_labels = lang_tokens[:, 1:].long()
        shift_mask = token_loss_mask[:, 1:].to(torch.bool)

        vocab = shift_logits.shape[-1]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            reduction="none",
        )
        ce = ce * shift_mask.reshape(-1).to(ce.dtype)
        denom = shift_mask.sum().clamp(min=1).to(ce.dtype)
        return ce.sum() / denom

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

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, expanded_time
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
