"""Dual-stream outer stage for the cross-embodiment GMM H-Net (v1).

Extends :class:`HNetOuterStage` to feed the inner trunk a TWO-stream token
sequence:

  * the existing ``input_modules`` (an ``ObsToken`` over a per-embodiment
    ``MultiEmbodimentCondEncoder``) produce the per-embodiment **specific**
    stream ``S``;
  * a NEW ``self.agnostic_input`` (an ``ObsToken`` over a SINGLE shared
    ``CondEncoderModule``, weight-shared across embodiments) produces the
    **agnostic** stream ``A``.

``encode`` concatenates them along the packed token axis (``x = cat([A, S], 0)``,
length ``2*T_total``) and stashes the layout in ``ctx.extras['dual']`` so
``DualStreamComputeStage`` can build its channel-aware mask. ``decode`` pulls the
post-trunk agnostic tokens ``A_top`` (the value the inner stage returned) and the
specific tokens ``S`` (stashed by the stage in ``ctx.extras['specific_tokens']``)
and feeds BOTH into the per-embodiment partitioned GMM head.

Training path: encode -> inner_stage -> decode via the inherited ``forward``.

Closed-loop AR / sim path (``init_step_state`` + ``step``, used by
``PackedAlgoBase.inference_step``): RECOMPUTE-OVER-PREFIX, no KV-cache. At each
env step we append the current frame's obs to a running prefix, REBUILD the full
A+S two-stream packed sequence over that prefix (exactly like training
``encode``), re-run the channel-aware masked trunk
(``DualStreamComputeStage.forward``), and feed the LAST timestep's
``(A_top, S)`` into the per-embodiment partitioned GMM head. This sidesteps an
incremental KV-cache for ``forward_masked`` (which has none) and keeps TF and AR
byte-identical through the trunk: same (L,L) channel mask, same RoPE, same
mask_mode / depth_split_k. Correct + simple for eval (sim is pymunk-bound; the
O(L^2)-per-step trunk recompute is negligible).
"""

from typing import Optional

import torch

from egomimic.algo.hnet.algo import HNetOuterStage
from egomimic.models.hnet.context import HNetContext
from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.stems.input_modules import ObsToken


class DualStreamOuterStage(HNetOuterStage):
    """HNetOuterStage with an added weight-shared agnostic obs token stream.

    Args (additional to HNetOuterStage):
        agnostic_obs_encoder: a single shared ``CondEncoderModule`` (weight-
            shared across embodiments) used to build the agnostic stream ``A``.
            Wrapped in an ``ObsToken`` internally. Required.
    """

    def __init__(
        self, *args, agnostic_obs_encoder: Optional[CondEncoderModule] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if agnostic_obs_encoder is None:
            raise ValueError(
                "DualStreamOuterStage requires agnostic_obs_encoder "
                "(a shared CondEncoderModule)."
            )
        # ObsToken validates agnostic_obs_encoder.d_cond == d_model.
        self.agnostic_input = ObsToken(
            d_model=self.d_model, obs_encoder=agnostic_obs_encoder
        )

    # ------------------------------------------------------------------
    # encode: build S (specific) + A (agnostic), concat, stash layout.
    # ------------------------------------------------------------------

    @staticmethod
    def _within_episode_time_pos(
        cu_seqlens: torch.Tensor, T_total: int, device
    ) -> torch.Tensor:
        """Per-token within-episode time index (0..len-1 per episode)."""
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        t = torch.arange(T_total, device=device, dtype=torch.long)
        # Subtract each token's episode start. Build a per-token start offset.
        starts = torch.zeros(T_total, dtype=torch.long, device=device)
        for b in range(cu.shape[0] - 1):
            starts[cu[b] : cu[b + 1]] = cu[b]
        return t - starts

    def encode(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        if not ctx.packed:
            raise NotImplementedError(
                "DualStreamOuterStage v1 supports the PACKED path only."
            )
        actions_packed = batch["actions"]  # (T_total, A)
        obs_packed = batch["__obs"]
        T_total = actions_packed.shape[0]
        device = actions_packed.device
        dtype = actions_packed.dtype
        cu_seqlens = ctx.cu_seqlens.to(device=device, dtype=torch.long)

        # SPECIFIC stream: sum the existing input_modules' packed contributions
        # (mirrors HNetOuterStage._encode_packed). Routed per-embodiment by
        # ctx.embodiment_id inside the MultiEmbodimentCondEncoder.
        S = None
        for mod in self.input_modules:
            contrib = mod.forward_packed(
                actions_packed=actions_packed,
                obs_packed=obs_packed,
                cu_seqlens=cu_seqlens,
                T_total=T_total,
                device=device,
                dtype=dtype,
                embodiment_id=ctx.embodiment_id,
            )
            S = contrib if S is None else S + contrib
        if S is None:
            raise RuntimeError("input_modules produced no specific tokens")

        # AGNOSTIC stream: the shared ObsToken (its encoder ignores
        # embodiment_id, so weights are shared across embodiments).
        A = self.agnostic_input.forward_packed(
            actions_packed=actions_packed,
            obs_packed=obs_packed,
            cu_seqlens=cu_seqlens,
            T_total=T_total,
            device=device,
            dtype=dtype,
            embodiment_id=ctx.embodiment_id,
        )

        # Concat along the packed token axis: A first, then S.
        x = torch.cat([A, S], dim=0)  # (2*T_total, d_model)

        # Within-episode time positions (shared by A_t and S_t).
        time_pos = self._within_episode_time_pos(cu_seqlens, T_total, device)

        ctx.extras["dual"] = {
            "T_total": int(T_total),
            "cu_seqlens": cu_seqlens,
            "time_pos": time_pos,
            "n_tokens_per_stream": int(T_total),
        }

        # Keep cond_dict valid (cond is unused since main_network cond:false,
        # but downstream _get_cond reads ctx.cond_dict). Mirror the parent's
        # packed cond build so any config that DOES use cond stays correct.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(
            obs_for_encode, T_action=T_total, embodiment_id=ctx.embodiment_id
        )
        ctx.cond_dict = {k: v.squeeze(0) for k, v in cond_padded.items()}
        return x

    # ------------------------------------------------------------------
    # decode: feed (A_top, S) into the per-embodiment partitioned GMM head.
    # ------------------------------------------------------------------

    def decode(self, h: torch.Tensor, batch: dict, ctx: HNetContext) -> None:
        A_top = h  # (T_total, d_model) — value returned by the inner stage.
        S = ctx.extras.get("specific_tokens")
        if S is None:
            raise RuntimeError(
                "DualStreamOuterStage.decode requires ctx.extras['specific_tokens'] "
                "(set by DualStreamComputeStage.forward)."
            )
        head = (
            self.action_out[ctx.embodiment_id]
            if isinstance(self.action_out, torch.nn.ModuleDict)
            else self.action_out
        )
        # NOT a registered submodule (mirrors HNetOuterStage.decode): avoids
        # duplicate _cur_head.* keys in state_dict.
        object.__setattr__(self, "_cur_head", head)
        head._emb_id = ctx.embodiment_id  # label for the mode-weighting probe
        # emb-partitioned head (shared-agnostic + per-emb specific/gate) needs emb_id;
        # legacy per-emb-ModuleDict head takes (A_top, S) only.
        if getattr(head, "is_emb_partitioned", False):
            batch["pred_action"] = head(A_top, S, ctx.embodiment_id)
        else:
            batch["pred_action"] = head(A_top, S)
        ctx.extras["gmm_head"] = head

    # ------------------------------------------------------------------
    # Closed-loop AR / sim path — recompute-over-prefix (NO KV-cache).
    #
    # ``PackedAlgoBase.inference_step`` calls ``init_step_state`` once at t==0,
    # then ``step(state, obs_norm, t, embodiment_id=...)`` every obs-step and
    # expects ``raw`` of shape ``(B, 1, chunk_len*per_step)`` so its
    # ``head.decode(raw)`` returns ``(B, 1, C, D)``.
    #
    # We override BOTH (the inherited HNetOuterStage versions assume the inner
    # stage supports an incremental KV-cache via ``inner_stage.step``;
    # DualStreamComputeStage does not — it recomputes). Instead we keep a running
    # buffer of per-frame obs in ``state`` and, each step, REBUILD the packed
    # A+S sequence over the prefix and re-run the masked trunk ``forward``.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR state for the recompute-over-prefix rollout.

        No KV-cache is needed; the state just carries a growing per-key buffer of
        normalized obs frames (the prefix) plus bookkeeping. ``batch_size`` must
        be 1 (the sim drives one env at a time).
        """
        if int(batch_size) != 1:
            raise NotImplementedError(
                "DualStreamOuterStage AR (recompute-over-prefix) supports "
                f"batch_size==1 only (got {batch_size})."
            )
        dtype = dtype or next(self.parameters()).dtype
        # Trivial stage cache (recompute => nothing cached). Kept so the contract
        # (inner_stage.allocate_inference_cache returns something) holds.
        params = self.inner_stage.allocate_inference_cache(
            batch_size=int(batch_size),
            max_seqlen=int(T_max),
            device=device,
            dtype=dtype,
        )
        return {
            "params": params,
            "obs_prefix": [],  # list of per-frame obs dicts (each value (…,))
            "batch_size": int(batch_size),
            "device": device,
            "dtype": dtype,
            "T_max": int(T_max),
        }

    @staticmethod
    def _append_obs_frame(prefix: list, obs_norm: dict) -> None:
        """Append one env frame's obs to the running prefix.

        ``obs_norm`` values arrive with a leading B==1 dim (env-to-zarr emits
        ``(1, …)``); we squeeze it so each stored frame is ``(…,)`` and a later
        ``torch.stack`` over the buffer yields the packed ``(T_total, …)`` tensor
        the dual-stream ``encode`` consumes. Non-tensor / non-B1 entries are
        stored as-is.
        """
        frame = {}
        for k, v in obs_norm.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
                frame[k] = v[0]
            else:
                frame[k] = v
        prefix.append(frame)

    @torch.no_grad()
    def step(
        self, state: dict, obs_norm: dict, t: int, embodiment_id: Optional[str] = None
    ) -> torch.Tensor:
        """One closed-loop AR step via recompute-over-prefix.

        Appends ``obs_norm`` to the prefix, rebuilds the packed A+S sequence over
        the whole prefix, re-runs the masked trunk, and decodes the LAST
        timestep through the per-embodiment partitioned GMM head.

        Returns ``raw`` of shape ``(B=1, 1, chunk_len*per_step)`` — the flat GMM
        params for the latest frame — so the caller's ``head.decode(raw)`` yields
        ``(B, 1, C, D)``.
        """
        self._append_obs_frame(state["obs_prefix"], obs_norm)
        prefix = state["obs_prefix"]
        T_total = len(prefix)
        device = state["device"]
        dtype = state["dtype"]

        # Stack the per-frame obs into packed (T_total, …) tensors. Pass through
        # any non-per-frame entries from the latest frame unchanged.
        obs_packed = {}
        for k in prefix[0].keys():
            vs = [f[k] for f in prefix]
            if torch.is_tensor(vs[0]):
                obs_packed[k] = torch.stack(vs, dim=0).to(device)
            else:
                obs_packed[k] = vs[-1]

        # Dummy packed actions: ObsToken ignores them, but encode reads
        # batch["actions"] for T_total / device / dtype. Shape (T_total, A).
        actions_packed = torch.zeros(
            (T_total, self.action_dim), device=device, dtype=dtype
        )
        cu_seqlens = torch.tensor([0, T_total], device=device, dtype=torch.long)

        batch = {"actions": actions_packed, "__obs": obs_packed}
        ctx = HNetContext(
            cond_dict={},
            aux=[],
            inference_params=state["params"],
            extras={},
            cu_seqlens=cu_seqlens,
            max_seqlen=int(T_total),
            embodiment_id=embodiment_id,
        )

        # Build the 2T two-stream sequence (sets ctx.extras['dual']) and run the
        # masked trunk forward over the whole prefix. forward() stashes S in
        # ctx.extras['specific_tokens'] and returns A_top (T_total, d_model).
        x = self.encode(batch, ctx)
        A_top = self.inner_stage(x, ctx)  # (T_total, d_model)
        S = ctx.extras.get("specific_tokens")
        if S is None:
            raise RuntimeError(
                "DualStreamOuterStage.step: inner_stage.forward did not stash "
                "ctx.extras['specific_tokens'] (expected from "
                "DualStreamComputeStage.forward)."
            )

        # Take the LAST timestep of each stream and add a (B=1, T=1) lead so the
        # head emits raw shaped (1, 1, C*per_step) -> decode -> (1, 1, C, D).
        a_top_last = A_top[T_total - 1].reshape(1, 1, -1)  # (1, 1, d_model)
        s_last = S[T_total - 1].reshape(1, 1, -1)  # (1, 1, d_model)

        head = (
            self.action_out[embodiment_id]
            if isinstance(self.action_out, torch.nn.ModuleDict)
            else self.action_out
        )
        if getattr(head, "is_emb_partitioned", False):
            raw = head(a_top_last, s_last, embodiment_id)
        else:
            raw = head(a_top_last, s_last)  # (1, 1, chunk_len*per_step)
        return raw


class MultiStreamOuterStage(DualStreamOuterStage):
    """Per-stream-dim variant: the agnostic stream A (width d_A) and specific
    stream S (width d_S) are kept SEPARATE (different widths, so no concat) and
    handed to MultiStreamComputeStage via ctx.extras['ms']. Only ``encode`` is
    overridden; ``decode`` / ``step`` / ``init_step_state`` are inherited (they
    call self.encode + self.inner_stage + the per-stream partitioned GMM head).
    """

    def encode(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        if not ctx.packed:
            raise NotImplementedError(
                "MultiStreamOuterStage supports the PACKED path only."
            )
        actions_packed = batch["actions"]
        obs_packed = batch["__obs"]
        T_total = actions_packed.shape[0]
        device = actions_packed.device
        dtype = actions_packed.dtype
        cu_seqlens = ctx.cu_seqlens.to(device=device, dtype=torch.long)

        # SPECIFIC stream S (width d_S): sum the per-embodiment input_modules.
        S = None
        for mod in self.input_modules:
            contrib = mod.forward_packed(
                actions_packed=actions_packed,
                obs_packed=obs_packed,
                cu_seqlens=cu_seqlens,
                T_total=T_total,
                device=device,
                dtype=dtype,
                embodiment_id=ctx.embodiment_id,
            )
            S = contrib if S is None else S + contrib
        if S is None:
            raise RuntimeError("input_modules produced no specific tokens")

        # AGNOSTIC stream A (width d_A): the shared ObsToken.
        A = self.agnostic_input.forward_packed(
            actions_packed=actions_packed,
            obs_packed=obs_packed,
            cu_seqlens=cu_seqlens,
            T_total=T_total,
            device=device,
            dtype=dtype,
            embodiment_id=ctx.embodiment_id,
        )

        time_pos = self._within_episode_time_pos(cu_seqlens, T_total, device)
        # Keep A, S SEPARATE (different widths) -> list for MultiStreamComputeStage.
        ctx.extras["ms"] = {
            "streams": [A, S],
            "T_total": int(T_total),
            "cu_seqlens": cu_seqlens,
            "time_pos": time_pos,
        }

        # cond_dict (mirror parent; unused by the trunk but read downstream).
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(
            obs_for_encode, T_action=T_total, embodiment_id=ctx.embodiment_id
        )
        ctx.cond_dict = {k: v.squeeze(0) for k, v in cond_padded.items()}
        # Return A as the flowing x (the stage ignores x, reads ctx.extras['ms']);
        # its width == outer d_model so HNetOuterStage's dim-check passes.
        return A
