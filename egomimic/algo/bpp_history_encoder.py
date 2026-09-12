"""
History-aware prompt encoder for the BPP algo (docs/plan/2026-09-08_bpp_rollout_history.md).

``HistoryPairPromptObsEncoder`` extends the vendored ``PairPromptObsEncoder``
(``external/behavior_prompting``, never modified) with a second memory
stream: the policy's *own* history, the last ``history_max_chunks`` completed
chunks of the current episode, in exactly the demo prompt's chunk format
(frame at the chunk start + proprio state + the chunk's actions). History
chunks are tokenized by the same tokenizer / attention pool as the demo
prompt (one token per chunk), given their own age-indexed sinusoidal position
and a learned segment embedding (demo vs. self), and concatenated after the
demo prompt tokens as the memory of the existing prompt-with-obs
cross-attention decoder. Nothing downstream changes: the decoder output has
the same token count, so the diffusion head is untouched.

Payload: the history travels inside the prompt metadata, because the vendored
normalizer indexes its parameter table for every *top-level* obs key but
passes ``prompt["metadata"]`` through untouched::

    obs_dict["prompt"]["metadata"]["history"] = {
        "obs": {<prompt obs keys>: (B, H, ...)},
        "action": (B, H, chunk_n_actions, action_dim),
        "metadata": {"mask": (B, H) bool, True = padded},
    }

``H`` may be 0 (episode start). Valid chunks are left-aligned, oldest first,
so the newest chunk is the last unmasked one and gets age 0.

Deployment: ``prompt()`` caches the demo tokens (and any history carried in
the metadata); ``push_history_chunk()`` encodes one new chunk and appends
its raw token to a sliding cache of the last ``history_max_chunks`` chunks.
This incremental cache is exact because ``prompt_encoder_enabled`` is False
(asserted): every chunk token is computed independently and the age
positions are re-applied at every ``forward``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from behavior_prompting.train_network.model.prompt.prompt_obs_encoder import (
    PairPromptObsEncoder,
)
from behavior_prompting.train_network.utils.model_util import init_weights


def sinusoidal_age_embedding(age: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed transformer-style sinusoidal embedding of integer ages:
    ``(...,)`` -> ``(..., dim)``. Computed on the fly, so any age works."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=age.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = age.to(torch.float32).unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class HistoryPairPromptObsEncoder(PairPromptObsEncoder):
    """``PairPromptObsEncoder`` whose prompt-decoder memory is
    ``[sinks?, demo prompt tokens, own-history tokens]``."""

    def __init__(self, *args, history_max_chunks: int = 15, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.prompt_with_obs_decoder_enabled:
            raise ValueError(
                "HistoryPairPromptObsEncoder needs prompt_with_obs_decoder_enabled=True"
            )
        if self.prompt_encoder_enabled:
            raise ValueError(
                "HistoryPairPromptObsEncoder needs prompt_encoder_enabled=False: the "
                "incremental history cache assumes chunk tokens are independent."
            )
        tok = self.obs_encoder
        if getattr(tok, "num_output_modalities_prompt", None) != 1:
            raise ValueError(
                "HistoryPairPromptObsEncoder needs one token per prompt chunk "
                "(merge_prompt_tokens: obs_and_action, or obs with ignore_prompt_action)."
            )
        if getattr(tok, "only_use_last_step_of_prompt", False):
            raise ValueError(
                "only_use_last_step_of_prompt is not supported with history"
            )
        if getattr(tok, "cut_first_n_steps_of_prompt", 0):
            raise ValueError(
                "cut_first_n_steps_of_prompt is not supported with history"
            )
        self.history_max_chunks = int(history_max_chunks)
        if self.history_max_chunks < 1:
            raise ValueError(
                f"history_max_chunks must be >= 1, got {history_max_chunks}"
            )

        self.n_emb = int(self.prompt_pos_emb.shape[-1])
        # Age-indexed positions: a fixed sinusoid of the chunk age (0 = newest)
        # through a learned linear map, so no length is baked into the weights.
        self.history_pos_proj = nn.Linear(self.n_emb, self.n_emb)
        self.history_pos_proj.apply(init_weights)
        # Row 0 is added to demo tokens, row 1 to history tokens. Zero-init so
        # the demo path is unchanged at init and the separation is learned.
        self.segment_emb = nn.Parameter(torch.zeros(2, 1, 1, self.n_emb))

        self.history_raw_cache: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # token helpers
    # ------------------------------------------------------------------

    def _prepare_demo_tokens(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Base ``prepare_prompt_tokens`` (token dropout + learned forward
        position) plus the demo segment embedding."""
        prompt_tokens = self.drop_tokens(prompt_tokens)
        assert prompt_tokens.shape[1] <= self.prompt_pos_emb.shape[1], (
            f"prompt length ({prompt_tokens.shape[1]}) is greater than prompt "
            f"position embedding length ({self.prompt_pos_emb.shape[1]})"
        )
        prompt_tokens = (
            prompt_tokens + self.prompt_pos_emb[:, : prompt_tokens.shape[1], :]
        )
        return prompt_tokens + self.segment_emb[0]

    def _encode_history_raw(
        self,
        history: Optional[dict],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a history payload with the prompt tokenizer (prompt-only
        branch): ``(B, H, D)`` raw chunk tokens (no position / segment yet)
        and a ``(B, H)`` bool padding mask. Empty tensors when ``H == 0``."""
        if history is None or history["action"].shape[1] == 0:
            B = batch_size if history is None else history["action"].shape[0]
            return (
                torch.zeros(B, 0, self.n_emb, device=device, dtype=self.dtype),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )
        tokens, mask = self.obs_encoder({"prompt": history})
        tokens = self.drop_tokens(tokens)
        B, H, D = tokens.shape
        if H > self.history_max_chunks:
            raise ValueError(
                f"history has {H} chunks > history_max_chunks={self.history_max_chunks}"
            )
        if mask is None:
            mask = torch.zeros(B, H, dtype=torch.bool, device=tokens.device)
        return tokens, mask.to(torch.bool)

    def _embed_history(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add age-indexed positions (newest valid chunk = age 0; valid
        chunks are left-aligned) and the history segment embedding."""
        B, H, D = raw.shape
        if H == 0:
            return raw
        n_valid = (~mask).sum(dim=1)  # (B,)
        age = (
            n_valid[:, None] - 1 - torch.arange(H, device=raw.device)[None, :]
        ).clamp(min=0)  # (B, H); padded chunks get an arbitrary (masked) age
        pos = self.history_pos_proj(sinusoidal_age_embedding(age, D).to(raw.dtype))
        return raw + pos + self.segment_emb[1]

    def _history_from_cache(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.history_raw_cache:
            return (
                torch.zeros(1, 0, self.n_emb, device=device, dtype=self.dtype),
                torch.zeros(1, 0, dtype=torch.bool, device=device),
            )
        raw = torch.cat(self.history_raw_cache[-self.history_max_chunks :], dim=1)
        mask = torch.zeros(1, raw.shape[1], dtype=torch.bool, device=raw.device)
        return raw.to(device), mask

    def _merge_memory(
        self,
        demo_tokens: torch.Tensor,
        demo_mask: Optional[torch.Tensor],
        hist_raw: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``[demo, history]`` memory and its key-padding mask."""
        B, P, D = demo_tokens.shape
        # Under autocast the demo tokens may be bf16 while the (possibly
        # empty) history placeholders are fp32: match dtypes before cat.
        hist_tokens = self._embed_history(hist_raw, hist_mask).to(demo_tokens.dtype)
        if hist_tokens.shape[0] != B:
            if hist_tokens.shape[0] == 1:
                hist_tokens = hist_tokens.expand(B, -1, -1)
                hist_mask = hist_mask.expand(B, -1)
            elif B == 1:
                demo_tokens = demo_tokens.expand(hist_tokens.shape[0], -1, -1)
                if demo_mask is not None:
                    demo_mask = demo_mask.expand(hist_tokens.shape[0], -1)
                B = hist_tokens.shape[0]
            else:
                raise ValueError(
                    f"demo prompt batch {B} and history batch {hist_tokens.shape[0]} differ"
                )
        H = hist_tokens.shape[1]
        memory = torch.cat([demo_tokens, hist_tokens], dim=1)
        if H == 0:
            # Keep the history parameters in the graph so every trainable
            # tensor receives a grad (DDP, find_unused_parameters=False).
            zero = (
                self.history_pos_proj.weight.sum()
                + self.history_pos_proj.bias.sum()
                + self.segment_emb[1].sum()
            )
            memory = memory + 0.0 * zero
            return memory, demo_mask
        if demo_mask is None:
            if not bool(hist_mask.any()):
                return memory, None
            demo_mask = torch.zeros(B, P, dtype=torch.bool, device=memory.device)
        mask = torch.cat([demo_mask.to(torch.bool), hist_mask.to(torch.bool)], dim=1)
        return memory, mask

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        need_weights: bool = False,
        average_attn_weights: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        if self.is_prompted:
            assert "prompt" not in obs_dict, (
                "prompt should not be provided in the obs_dict after `prompt` is "
                "called. Call `reset` to reset the policy to remove the prompt."
            )
        obs_dict = dict(obs_dict)
        prompt_present = "prompt" in obs_dict
        receding_present = any(key in obs_dict for key in self.current_obs_keys)
        if receding_present:
            assert all(key in obs_dict for key in self.current_obs_keys)
        else:
            assert not any(key in obs_dict for key in self.current_obs_keys)

        # Never mutate the caller's dicts: the policy's normalizer reuses the
        # same metadata object, and ``prompt()`` reads the history from it.
        history = None
        if prompt_present:
            prompt = dict(obs_dict["prompt"])
            metadata = dict(prompt.get("metadata") or {})
            history = metadata.pop("history", None)
            prompt["metadata"] = metadata
            obs_dict["prompt"] = prompt

        if prompt_present and receding_present:
            # training / offline eval
            prompt_receding_tokens, receding_tokens, prompt_tokens, prompt_mask = (
                self.obs_encoder(obs_dict)
            )
            demo_tokens = self._prepare_demo_tokens(prompt_tokens)
            hist_raw, hist_mask = self._encode_history_raw(
                history, demo_tokens.shape[0], demo_tokens.device
            )
        elif prompt_present:
            # ``prompt()``: encode and return the demo tokens for the cache
            prompt_tokens, prompt_mask = self.obs_encoder(obs_dict)
            return self._prepare_demo_tokens(prompt_tokens), prompt_mask
        else:
            # ``predict_action`` after ``prompt()``: cached demo + history
            prompt_receding_tokens, receding_tokens = self.obs_encoder(obs_dict)
            assert self.is_prompted
            demo_tokens = self.prompt_tokens_cache
            prompt_mask = self.prompt_mask_cache
            hist_raw, hist_mask = self._history_from_cache(demo_tokens.device)

        memory, memory_mask = self._merge_memory(
            demo_tokens, prompt_mask, hist_raw, hist_mask
        )

        prompt_B = memory.shape[0]
        receding_B = receding_tokens.shape[0]
        if self.attention_sink_enabled:
            sink = self.attention_sink_tokens.expand(
                prompt_B, self.num_attention_sink_tokens, -1
            )
            memory = torch.cat([sink, memory], dim=1)
            if memory_mask is not None:
                sink_mask = torch.zeros(
                    (memory_mask.shape[0], self.num_attention_sink_tokens),
                    device=memory_mask.device,
                    dtype=memory_mask.dtype,
                )
                memory_mask = torch.cat([sink_mask, memory_mask], dim=1)

        if prompt_B == 1 and prompt_B != receding_B:
            memory = memory.expand(receding_B, -1, -1)
            if memory_mask is not None:
                memory_mask = memory_mask.expand(receding_B, -1)

        ret = self.prompt_with_obs_decoder(
            tgt=prompt_receding_tokens + self.prompt_current_obs_pos_emb,
            memory=memory,
            memory_key_padding_mask=memory_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        cross_attn_weights = None
        if need_weights:
            encoded_prompt_tokens, cross_attn_weights = ret
            if self.attention_sink_enabled:
                cross_attn_weights = cross_attn_weights[
                    ..., self.num_attention_sink_tokens :
                ]
        else:
            encoded_prompt_tokens = ret

        if self.ignore_prompt:
            receding_and_prompt_tokens = receding_tokens + 0 * encoded_prompt_tokens
        else:
            receding_and_prompt_tokens = torch.cat(
                [receding_tokens, encoded_prompt_tokens], dim=1
            )

        if self.flatten_output:
            receding_and_prompt_tokens = receding_and_prompt_tokens.view(
                receding_and_prompt_tokens.shape[0], -1
            )

        metadata = {}
        if not self.flatten_output:
            metadata["token_mask"] = None
        if need_weights:
            metadata["prompt_with_obs_transfomer_attention_weights"] = (
                cross_attn_weights
            )
        return receding_and_prompt_tokens, metadata

    # ------------------------------------------------------------------
    # deployment API
    # ------------------------------------------------------------------

    def prompt(self, prompt_dict: dict) -> None:
        # Read the history before the base call: our ``forward`` copies the
        # metadata, but read first so no ordering assumption is needed.
        history = (prompt_dict.get("metadata") or {}).get("history")
        super().prompt(prompt_dict)
        self.clear_history()
        if history is not None and history["action"].shape[1] > 0:
            device = self.prompt_tokens_cache.device
            raw, mask = self._encode_history_raw(history, 1, device)
            if raw.shape[0] != 1:
                raise ValueError(
                    "a deployment prompt's history must have batch size 1, got "
                    f"{raw.shape[0]}"
                )
            n_valid = int((~mask[0]).sum())
            for i in range(n_valid):
                self.history_raw_cache.append(raw[:, i : i + 1].clone())
            del self.history_raw_cache[: -self.history_max_chunks]

    @torch.inference_mode()
    def push_history_chunk(self, chunk: dict) -> int:
        """Encode one completed chunk (policy-side payload, B = 1, H = 1) and
        append it to the sliding history cache. Returns the cache length."""
        if not self.is_prompted:
            raise RuntimeError("call prompt() before push_history_chunk()")
        device = self.prompt_tokens_cache.device
        raw, mask = self._encode_history_raw(chunk, 1, device)
        if raw.shape[0] != 1 or raw.shape[1] != 1 or bool(mask.any()):
            raise ValueError(
                "push_history_chunk expects one valid chunk with batch size 1, got "
                f"tokens {tuple(raw.shape)}, mask {mask.tolist()}"
            )
        self.history_raw_cache.append(raw.clone())
        del self.history_raw_cache[: -self.history_max_chunks]
        return len(self.history_raw_cache)

    def clear_history(self) -> None:
        self.history_raw_cache = []

    @property
    def history_len(self) -> int:
        return len(self.history_raw_cache)

    def reset(self) -> None:
        super().reset()
        self.clear_history()

    # ------------------------------------------------------------------
    # optimizer / introspection
    # ------------------------------------------------------------------

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        groups = super().get_optim_groups(lr=lr, weight_decay=weight_decay)
        groups.append(
            {
                "params": [self.history_pos_proj.weight],
                "weight_decay": weight_decay,
                "lr": lr,
            }
        )
        groups.append(
            {
                "params": [self.history_pos_proj.bias, self.segment_emb],
                "weight_decay": 0.0,
                "lr": lr,
            }
        )
        return groups

    def get_prompt_cross_attn_dim_names(self, prompt_len: int, history_len: int = 0):
        prompt_current_obs_names, prompt_obs_names = (
            super().get_prompt_cross_attn_dim_names(prompt_len)
        )
        history_names = [
            f"history age {history_len - 1 - i}" for i in range(history_len)
        ]
        return prompt_current_obs_names, prompt_obs_names + history_names
