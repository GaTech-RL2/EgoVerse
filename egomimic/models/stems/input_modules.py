"""Composable input modules for HNetPolicy.

Each module produces a per-token contribution of shape ``(B, T, d_model)`` (or
``(T_total, d_model)`` in packed mode, or ``(B, 1, d_model)`` in step mode)
that ``HNetPolicy`` sums together and feeds to the H-Net trunk.

The default `ActionInToken` reproduces the legacy autoregressive behaviour
(BOS + shifted action embedding); `ObsToken` provides obs-as-input where
the model reads encoded obs directly with no action conditioning. Sum both
to get obs+AR.

Each module is wholly responsible for any obs encoder it needs — they do NOT
share weights with `HNetPolicy.cond_encoder` (which is the AdaLN cond path).
This keeps AdaLN-on and AdaLN-off variants directly comparable without param
double-counting.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from egomimic.models.stems.cond_encoders import CondEncoderModule


class InputModule(nn.Module):
    """Base interface.

    Subclasses implement the three forward paths used by HNetPolicy.

    All three return tensors at d_model. Modules are summed in HNetPolicy.
    """

    def forward_padded(
        self,
        *,
        actions: Optional[torch.Tensor],
        obs: dict,
        B: int,
        T: int,
        device,
        dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_packed(
        self,
        *,
        actions_packed: Optional[torch.Tensor],
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        T_total: int,
        device,
        dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def step(
        self,
        *,
        prev_action_norm: Optional[torch.Tensor],
        obs_norm: dict,
        t: int,
        B: int,
        device,
        dtype,
    ) -> torch.Tensor:
        raise NotImplementedError


class ActionInToken(InputModule):
    """Legacy AR input: ``Linear(action_dim -> d_model)`` + learned BOS at t=0,
    teacher-forced with right-shift in training, fed previous prediction at
    inference.

    State-dict layout matches pre-refactor HNetPolicy attributes so legacy
    checkpoints load cleanly via the remap hook in :class:`HNetPolicy`:
      - ``action_in.weight`` / ``action_in.bias``
      - ``bos``
    """

    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.action_in = nn.Linear(action_dim, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)

    def forward_padded(self, *, actions, obs, B, T, device, dtype):
        x = self.action_in(actions)
        x = torch.cat([self.bos.expand(B, -1, -1).to(x.dtype), x[:, :-1]], dim=1)
        return x

    def forward_packed(
        self, *, actions_packed, obs_packed, cu_seqlens, T_total, device, dtype
    ):
        a_emb = self.action_in(actions_packed)  # (T_total, D)
        x_shifted = torch.cat(
            [torch.zeros(1, self.d_model, device=device, dtype=a_emb.dtype), a_emb[:-1]],
            dim=0,
        )
        bos = self.bos.squeeze(0).squeeze(0).to(a_emb.dtype)  # (D,)
        starts = cu_seqlens[:-1]
        x_shifted = x_shifted.clone()
        x_shifted[starts] = bos
        return x_shifted

    def step(self, *, prev_action_norm, obs_norm, t, B, device, dtype):
        if t == 0 or prev_action_norm is None:
            return self.bos.expand(B, 1, self.d_model).to(dtype)
        return self.action_in(prev_action_norm).to(dtype)


class ObsToken(InputModule):
    """Obs-as-input: encode obs through a dedicated `CondEncoderModule` at
    width ``d_model`` and feed the per-frame embedding directly as input.

    No BOS, no shift — input at position t IS the encoder output at frame t.
    Causal attention masking still prevents leakage (the model decoding a_t
    only attends to obs_{0..t}, which is what we want).

    Args:
        d_model: trunk hidden width.
        obs_encoder: an already-constructed `CondEncoderModule` whose
            ``d_cond == d_model`` and whose ``output_key`` is the fused token.
            Instantiated by Hydra alongside (and independently of) the policy's
            cond_encoder so this module owns its own MLPs/img encoders.
    """

    def __init__(self, d_model: int, obs_encoder: CondEncoderModule):
        super().__init__()
        if obs_encoder.d_cond != d_model:
            raise ValueError(
                f"ObsToken expects obs_encoder.d_cond == d_model "
                f"(got {obs_encoder.d_cond} vs {d_model}). Set d_cond={d_model} "
                f"in the obs_encoder config."
            )
        self.d_model = d_model
        self.obs_encoder = obs_encoder
        self._output_key = obs_encoder.output_key

    def _encode(self, obs: dict, T_action: int) -> torch.Tensor:
        out = self.obs_encoder.encode(obs, T_action)
        if self._output_key not in out:
            raise KeyError(
                f"ObsToken: obs_encoder did not produce '{self._output_key}' "
                f"(got keys {list(out)}). Check that obs_specs/img_encoders are "
                f"populated."
            )
        return out[self._output_key]

    def forward_padded(self, *, actions, obs, B, T, device, dtype):
        c = self._encode(obs, T)  # (B, T, d_model)
        return c.to(dtype)

    def forward_packed(
        self, *, actions_packed, obs_packed, cu_seqlens, T_total, device, dtype
    ):
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        c_padded = self._encode(obs_for_encode, T_action=T_total)  # (1, T_total, D)
        return c_padded.squeeze(0).to(dtype)

    def step(self, *, prev_action_norm, obs_norm, t, B, device, dtype):
        c_seq = self._encode(obs_norm, T_action=1)  # (B, 1, d_model)
        return c_seq.to(dtype)
