"""Chunked dual-stream outer stage (v2): feeds the inner chain a StreamBundle.

Unlike :class:`DualStreamOuterStage` (v1: concat A+S -> single masked trunk) and
:class:`MultiStreamOuterStage` (NoChunk: A,S via ctx.extras['ms']), this outer
stage builds a :class:`StreamBundle` and returns it as the flowing ``x`` so a
``DualStreamChunkerStage`` (and, recursively, an agnostic-only apex ComputeStage)
can carry both streams explicitly through the chunked H-Net.

Flow (inherited ``HNetOuterStage.forward``):
  encode(batch) -> StreamBundle(A, S)         # A shared-agnostic, S per-emb
  h = inner_stage(bundle)                      # DualStreamChunkerStage -> apex -> dechunk
  decode(h) -> PartitionedGMMActionHead(h.A, h.S)

Closed-loop AR (``step``): recompute-over-prefix, identical strategy to the v1
DualStreamOuterStage but the inner chain now returns a StreamBundle.

This is a NEW class; it does not modify DualStreamOuterStage / MultiStreamOuterStage,
so existing 2trunk / NoChunk configs are byte-identical (BC preserved).
"""

from typing import Optional

import torch

from egomimic.algo.hnet.dual_stream_outer import DualStreamOuterStage
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.stream_bundle import StreamBundle


class DualStreamChunkedOuterStage(DualStreamOuterStage):
    """Dual-stream outer stage whose ``encode`` emits a :class:`StreamBundle`.

    Reuses the parent's obs encoders (``agnostic_input`` for A, ``input_modules``
    for S), ``_within_episode_time_pos``, ``init_step_state`` and
    ``_append_obs_frame``. Overrides ``encode`` / ``decode`` / ``step`` to carry a
    StreamBundle instead of the v1 concat tensor.
    """

    # ------------------------------------------------------------------
    # encode: build A (agnostic) + S (specific) -> StreamBundle.
    # ------------------------------------------------------------------
    def encode(self, batch: dict, ctx: HNetContext) -> StreamBundle:  # type: ignore[override]
        if not ctx.packed:
            raise NotImplementedError(
                "DualStreamChunkedOuterStage supports the PACKED path only."
            )
        actions_packed = batch["actions"]  # (T_total, A)
        obs_packed = batch["__obs"]
        T_total = actions_packed.shape[0]
        device = actions_packed.device
        dtype = actions_packed.dtype
        cu_seqlens = ctx.cu_seqlens.to(device=device, dtype=torch.long)

        # SPECIFIC stream S: sum the per-embodiment input_modules.
        S = None
        for mod in self.input_modules:
            contrib = mod.forward_packed(
                actions_packed=actions_packed, obs_packed=obs_packed,
                cu_seqlens=cu_seqlens, T_total=T_total, device=device, dtype=dtype,
                embodiment_id=ctx.embodiment_id,
            )
            S = contrib if S is None else S + contrib
        if S is None:
            raise RuntimeError("input_modules produced no specific tokens")

        # AGNOSTIC stream A: the shared ObsToken (encoder ignores embodiment_id).
        A = self.agnostic_input.forward_packed(
            actions_packed=actions_packed, obs_packed=obs_packed,
            cu_seqlens=cu_seqlens, T_total=T_total, device=device, dtype=dtype,
            embodiment_id=ctx.embodiment_id,
        )

        time_pos = self._within_episode_time_pos(cu_seqlens, T_total, device)
        ctx.extras["dual_mode"] = True  # cheap assert/logging flag (not source of truth)

        # cond_dict (mirror parent; unused by the agnostic apex but read downstream).
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(
            obs_for_encode, T_action=T_total, embodiment_id=ctx.embodiment_id
        )
        ctx.cond_dict = {k: v.squeeze(0) for k, v in cond_padded.items()}

        return StreamBundle(
            A=A, S=S, T_total=int(T_total), cu_seqlens=cu_seqlens, time_pos=time_pos
        )

    # ------------------------------------------------------------------
    # decode: feed (A_top, S) from the returned StreamBundle into the head.
    # ------------------------------------------------------------------
    def decode(self, h, batch: dict, ctx: HNetContext) -> None:  # type: ignore[override]
        if isinstance(h, StreamBundle):
            A_top, S = h.A, h.S
        else:  # BC fallback (compute-only chain stashes specific_tokens)
            A_top = h
            S = ctx.extras.get("specific_tokens")
            if S is None:
                raise RuntimeError(
                    "DualStreamChunkedOuterStage.decode: inner chain returned a "
                    "bare tensor without ctx.extras['specific_tokens']."
                )
        head = (
            self.action_out[ctx.embodiment_id]
            if isinstance(self.action_out, torch.nn.ModuleDict)
            else self.action_out
        )
        object.__setattr__(self, "_cur_head", head)
        head._emb_id = ctx.embodiment_id
        # emb-partitioned head (shared-agnostic + per-emb specific/gate) needs emb_id;
        # the legacy per-emb-ModuleDict head takes (A_top, S) only.
        if getattr(head, "is_emb_partitioned", False):
            batch["pred_action"] = head(A_top, S, ctx.embodiment_id)
        else:
            batch["pred_action"] = head(A_top, S)
        ctx.extras["gmm_head"] = head

    # ------------------------------------------------------------------
    # Closed-loop AR — recompute-over-prefix; inner chain returns a StreamBundle.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self, state: dict, obs_norm: dict, t: int, embodiment_id: Optional[str] = None
    ) -> torch.Tensor:  # type: ignore[override]
        self._append_obs_frame(state["obs_prefix"], obs_norm)
        prefix = state["obs_prefix"]
        T_total = len(prefix)
        device = state["device"]
        dtype = state["dtype"]

        obs_packed = {}
        for k in prefix[0].keys():
            vs = [f[k] for f in prefix]
            obs_packed[k] = torch.stack(vs, dim=0).to(device) if torch.is_tensor(vs[0]) else vs[-1]

        actions_packed = torch.zeros((T_total, self.action_dim), device=device, dtype=dtype)
        cu_seqlens = torch.tensor([0, T_total], device=device, dtype=torch.long)
        batch = {"actions": actions_packed, "__obs": obs_packed}
        ctx = HNetContext(
            cond_dict={}, aux=[], inference_params=state["params"], extras={},
            cu_seqlens=cu_seqlens, max_seqlen=int(T_total), embodiment_id=embodiment_id,
        )

        bundle = self.encode(batch, ctx)
        h = self.inner_stage(bundle, ctx)  # StreamBundle (chunker -> apex -> dechunk)
        if not isinstance(h, StreamBundle):
            raise RuntimeError(
                "DualStreamChunkedOuterStage.step: inner chain did not return a StreamBundle."
            )
        a_top_last = h.A[T_total - 1].reshape(1, 1, -1)
        s_last = h.S[T_total - 1].reshape(1, 1, -1)
        head = (
            self.action_out[embodiment_id]
            if isinstance(self.action_out, torch.nn.ModuleDict)
            else self.action_out
        )
        if getattr(head, "is_emb_partitioned", False):
            return head(a_top_last, s_last, embodiment_id)
        return head(a_top_last, s_last)
