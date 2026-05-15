"""
H-Net policy for EgoVerse — stage-based architecture.

The policy treats the action sequence as the input modality (autoregressive,
causal). Observations are encoded by a ``CondEncoderModule`` into a
``cond_dict`` carried by an ``HNetContext`` that is threaded through the
stage tree. Each stage reads whichever cond key it wants (or ignores cond
entirely).

Loss = action MSE (next-action prediction) +
       sum_over_chunkers( weight * ratio_loss(boundary_predictions) ).

The per-chunker ratio-loss weights live inside the chunker stages themselves;
this algo just calls ``ratio_loss_from_aux(ctx.aux)`` after forward.
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.context import HNetContext
from egomimic.models.hnet_nets.hnet import HNet as HNetCore
from egomimic.models.hnet_nets.hnet import chunk_stats_from_aux, ratio_loss_from_aux
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class HNetPolicy(nn.Module):
    """
    action-tokenizer → stage-based H-Net → action-detokenizer.

    Owns action_in / action_out projections, BOS token, positional embedding,
    the ``CondEncoderModule``, and the ``HNetCore`` (stage tree).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model

        self.action_in = nn.Linear(action_dim, d_model)
        self.action_out = nn.Linear(d_model, action_dim)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.cond_encoder = cond_encoder
        self.hnet = hnet

        # Sanity-check that the stage tree's outer hidden dim matches d_model.
        if self.hnet.input_hidden_dim != d_model:
            raise ValueError(
                f"hnet.input_hidden_dim ({self.hnet.input_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )
        if self.hnet.output_hidden_dim != d_model:
            raise ValueError(
                f"hnet.output_hidden_dim ({self.hnet.output_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )

    def _build_ctx(self, obs: dict) -> HNetContext:
        cond_dict = self.cond_encoder.encode(obs, self.action_horizon)
        return HNetContext(cond_dict=cond_dict, aux=[], inference_params=None)

    def forward(self, actions: torch.Tensor, obs: dict):
        """
        actions: (B, T, action_dim) ground-truth actions for teacher-forcing.
        obs:     dict of (B, ...) obs tensors.

        Returns: (pred_actions (B, T, action_dim), aux list).
        """
        B, T, _ = actions.shape
        x = self.action_in(actions)
        x = torch.cat([self.bos.expand(B, -1, -1), x[:, :-1]], dim=1)
        x = x + self.pos_emb[:, :T]

        ctx = self._build_ctx(obs)
        h = self.hnet(x, ctx)
        return self.action_out(h), ctx.aux

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward.

        Mirrors :meth:`forward` for variable-length episodes packed into a
        single flat stream (FlashAttention-style varlen). The BOS shift and
        ``pos_emb`` indexing happen per sub-sequence — for each subseq
        ``[s, e)``, position s gets BOS and positions s+1..e-1 get
        ``action_in(actions[s..e-2])``; ``pos_emb`` is indexed by ``t - s``
        within each subseq so every episode starts at position 0.

        Args:
            actions_packed: (T_total, action_dim) packed ground-truth actions.
            obs_packed:     dict of (T_total, ...) per-frame obs tensors.
            cu_seqlens:     (B+1,) long, cumulative subseq lengths (starts 0).
            max_seqlen:     int, longest subseq length.

        Returns: (pred_packed (T_total, action_dim), aux).
        """
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        if not torch.is_tensor(cu_seqlens):
            cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.long)
        else:
            cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        if max_seqlen > self.action_horizon:
            raise ValueError(
                f"max_seqlen={max_seqlen} exceeds pos_emb length "
                f"action_horizon={self.action_horizon}; increase action_horizon "
                f"or chunk episodes to <= action_horizon frames."
            )

        # 1. Tokenize, then global shift-right by 1, then overwrite each
        #    subseq's first slot with BOS. Under autocast (e.g. bf16) the
        #    activations are downcast but ``self.bos`` stays in fp32; we
        #    match the activation dtype to keep the index-put happy.
        a_emb = self.action_in(actions_packed)  # (T_total, D)
        x_shifted = torch.cat(
            [
                torch.zeros(1, self.d_model, device=device, dtype=a_emb.dtype),
                a_emb[:-1],
            ],
            dim=0,
        )  # (T_total, D)
        bos = self.bos.squeeze(0).squeeze(0).to(a_emb.dtype)  # (D,)
        starts = cu_seqlens[:-1]  # (B,)
        x_shifted = x_shifted.clone()
        x_shifted[starts] = bos

        # 2. Per-sub-sequence pos_emb: position t in subseq [s, e) gets index t-s.
        #    Build seq_idx (which subseq each token belongs to), then subtract
        #    that subseq's start to get the local position index.
        pos_t = torch.arange(T_total, device=device)
        # seq_idx[t] = number of subseq starts strictly less-than-or-equal to t
        # Same as: (cu_seqlens[1:] <= t).sum() ... but easier:
        seq_idx = (pos_t[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)  # (T_total,)
        local_pos = pos_t - cu_seqlens[seq_idx]  # (T_total,)
        pos = self.pos_emb.squeeze(0)[local_pos].to(x_shifted.dtype)  # (T_total, D)
        x_packed = x_shifted + pos

        # 3. Packed cond. ``cond_encoder.encode`` expects (B, T, ...); for a
        #    packed stream the simplest path is to feed (1, T_total, ...) and
        #    squeeze the leading dim back out. The encoder's per-frame branch
        #    (state x.dim()==3 / images x.dim()==5) fires correctly because
        #    obs_packed already carries the per-frame dim.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(obs_for_encode, T_action=T_total)
        cond_packed = {k: v.squeeze(0) for k, v in cond_padded.items()}

        # 4. Build packed ctx and run.
        ctx = HNetContext(
            cond_dict=cond_packed,
            aux=[],
            inference_params=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
        )
        h = self.hnet(x_packed, ctx)  # (T_total, D)
        return self.action_out(h), ctx.aux

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout from BOS for ``T`` steps (default
        ``action_horizon``). ``T`` may be < ``action_horizon`` when rolling
        out an individual episode whose length is known to be shorter — the
        pos_emb is sized at action_horizon so any T <= action_horizon
        works."""
        if T is None:
            T = self.action_horizon
        if T > self.action_horizon:
            raise ValueError(
                f"generate T={T} exceeds pos_emb length action_horizon="
                f"{self.action_horizon}"
            )
        cond_dict = self.cond_encoder.encode(obs, T)
        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        dtype = next(self.parameters()).dtype

        inference_params = self.hnet.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=T,
            device=device,
            dtype=dtype,
        )

        # Per-step cond_dict slice (B, d_cond) — AdaLN broadcasts over the
        # single-token sequence dim inside the encoder.
        def slice_cond(t: int) -> dict:
            return {k: v[:, t] if v.dim() == 3 else v for k, v in cond_dict.items()}

        cur = self.bos.expand(batch_size, -1, -1) + self.pos_emb[:, 0:1]
        for t in range(T):
            ctx = HNetContext(
                cond_dict=slice_cond(t),
                aux=[],
                inference_params=inference_params,
            )
            h = self.hnet.step(cur, ctx)
            a_t = self.action_out(h)
            actions[:, t : t + 1] = a_t
            if t < T - 1:
                cur = self.action_in(a_t) + self.pos_emb[:, t + 1 : t + 2]
        return actions


class FlatFusedPolicy(nn.Module):
    """Flat transformer with interleaved [c_t, a_t] tokens.

    Drop-in replacement for ``HNetPolicy`` that bypasses the H-Net stage
    hierarchy (no chunker, no ratio loss). Each timestep contributes TWO
    tokens to the input sequence: a cond token and a (shifted) action token.
    Causal masking means the model predicting ``a_t`` (at sequence position
    2t+1) has seen ``c_0, BOS, c_1, a_0, c_2, a_1, ..., c_t, a_{t-1}``.

    Input layout (length 2T):
      x[:, 0]  = cond_in(c_0)         x[:, 1]  = BOS
      x[:, 2]  = cond_in(c_1)         x[:, 3]  = action_in(a_0)
      ...
      x[:, 2t]   = cond_in(c_t)       x[:, 2t+1] = action_in(a_{t-1})

    Output extraction:
      pred[:, t] = action_out(out[:, 2t+1])

    AR rollout walks the sequence one token at a time (2T model steps for T
    actions). Each "outer" step emits one action prediction and adds two
    tokens (the new cond + the new predicted action) to the cache.

    Same ``forward(actions, obs)`` / ``forward_packed(...)`` / ``generate(...)``
    contracts as ``HNetPolicy`` so the same ``HNet`` algo wrapper consumes
    it. Aux is always ``[]`` (no chunker contributions).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        arch_layout: str = "T8",
        num_heads: int = 4,
        d_intermediate: int = 512,
    ):
        super().__init__()
        from egomimic.models.hnet_nets.isotropic_builder import build_isotropic

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.d_cond = d_cond
        self.cond_encoder = cond_encoder

        self.action_in = nn.Linear(action_dim, d_model)
        self.action_out = nn.Linear(d_model, action_dim)
        self.cond_in = nn.Linear(d_cond, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        # pos_emb covers the 2T-token interleaved sequence.
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Single Isotropic stack, causal, no per-block cond (the fusion happens
        # at the input layer).
        self.backbone = build_isotropic(
            {
                "arch_layout": arch_layout,
                "d_model": d_model,
                "d_intermediate": d_intermediate,
                "num_heads": num_heads,
                "cond": False,
            },
            d_cond=0,
            causal=True,
        )

    def _encode_cond(self, obs: dict, T: int) -> torch.Tensor:
        cond_dict = self.cond_encoder.encode(obs, T)
        c = cond_dict.get("fused_cond")
        if c is None:
            raise KeyError(
                "FlatFusedPolicy requires 'fused_cond' in cond_encoder output."
            )
        return c  # (B, T, d_cond)

    def forward(self, actions: torch.Tensor, obs: dict):
        """Padded teacher-forced forward.

        actions: (B, T, action_dim)
        obs:     dict of per-frame (B, T, ...) obs tensors.
        Returns: (pred (B, T, action_dim), aux=[]).
        """
        B, T, _ = actions.shape
        c = self._encode_cond(obs, T)  # (B, T, d_cond)
        c_tok = self.cond_in(c)  # (B, T, d_model)
        a_tok = self.action_in(actions)  # (B, T, d_model)
        # Shift: BOS at position 0, a_0..a_{T-2} after.
        a_shifted = torch.cat([self.bos.expand(B, -1, -1), a_tok[:, :-1]], dim=1)

        # Interleave: x[:, 0::2] = c_tok, x[:, 1::2] = a_shifted.
        x = torch.empty(
            B, 2 * T, self.d_model, device=actions.device, dtype=a_tok.dtype
        )
        x[:, 0::2] = c_tok
        x[:, 1::2] = a_shifted
        x = x + self.pos_emb[:, : 2 * T].to(x.dtype)

        x = self.backbone(x)  # (B, 2T, d_model)
        pred = self.action_out(x[:, 1::2])  # (B, T, action_dim)
        return pred, []

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward.

        Builds a fused-token packed stream where each sub-sequence ``[s, e)``
        becomes a length-``2*(e-s)`` interleaved chunk. Returns predictions
        in packed format ``(T_total, action_dim)`` matching the input
        ``actions_packed`` layout.
        """
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        if max_seqlen > self.action_horizon:
            raise ValueError(
                f"max_seqlen={max_seqlen} exceeds action_horizon={self.action_horizon}"
            )

        # Encode cond. Feed obs as (1, T_total, ...) via cond_encoder.encode.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_seq = self._encode_cond(obs_for_encode, T_total).squeeze(
            0
        )  # (T_total, d_cond)
        c_tok = self.cond_in(cond_seq)  # (T_total, d_model)
        a_tok = self.action_in(actions_packed)  # (T_total, d_model)

        # Build BOS-shifted actions per sub-sequence.
        a_shifted = torch.empty_like(a_tok)
        bos = self.bos.squeeze(0).squeeze(0).to(a_tok.dtype)  # (d_model,)
        a_shifted[cu_seqlens[:-1]] = bos
        # For positions that aren't sub-seq starts, copy a_tok[t-1].
        non_start = torch.ones(T_total, dtype=torch.bool, device=device)
        non_start[cu_seqlens[:-1]] = False
        # Indices of non-start positions:
        idx_non_start = torch.nonzero(non_start, as_tuple=False).squeeze(-1)
        a_shifted[idx_non_start] = a_tok[idx_non_start - 1]

        # Build per-sub-seq position indices: 0, 1, ..., (e-s)-1 within each
        # sub-seq. Then the 2-token interleave doubles to 2*(e-s).
        pos = torch.arange(T_total, device=device)
        seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
        local_pos = pos - cu_seqlens[seq_idx]  # (T_total,)

        # The interleaved stream has 2T_total tokens. cond_t at 2*local_pos
        # within each sub-seq; action_t at 2*local_pos+1.
        # Compute new cu_seqlens for the doubled stream.
        sub_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        new_lens = 2 * sub_lens
        new_cu = torch.zeros(len(cu_seqlens), dtype=torch.long, device=device)
        new_cu[1:] = torch.cumsum(new_lens, dim=0)
        new_T_total = int(new_cu[-1].item())

        # Build the packed interleaved stream and apply pos_emb based on
        # 2*local_pos / 2*local_pos+1.
        x = torch.empty(new_T_total, self.d_model, device=device, dtype=a_tok.dtype)
        # For each t, write c_tok[t] at new_cu[seq_idx[t]] + 2*local_pos[t]
        # and a_shifted[t] at the next position.
        target_c = new_cu[seq_idx] + 2 * local_pos
        target_a = target_c + 1
        x[target_c] = c_tok
        x[target_a] = a_shifted
        # pos_emb indexing: doubled local positions within each sub-seq.
        # pos_emb is (1, 2*action_horizon, d_model). Apply pos_emb[2*local_pos]
        # to cond positions and pos_emb[2*local_pos+1] to action positions.
        pos_c = (2 * local_pos).clamp(max=2 * self.action_horizon - 1)
        pos_a = (2 * local_pos + 1).clamp(max=2 * self.action_horizon - 1)
        x[target_c] = x[target_c] + self.pos_emb[0, pos_c].to(x.dtype)
        x[target_a] = x[target_a] + self.pos_emb[0, pos_a].to(x.dtype)

        # Run the backbone on the doubled packed stream.
        out = self.backbone(
            x,
            cu_seqlens=new_cu,
            max_seqlen=2 * int(max_seqlen),
        )

        # Predictions: out at action positions (target_a). action_out projects
        # to (T_total, action_dim). Order matches actions_packed.
        pred = self.action_out(out[target_a])
        return pred, []

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """AR rollout for T action steps over a 2T-token interleaved stream.

        Each AR outer step does TWO inner step() calls: one for the cond
        token, one for the (predicted) action token. The output of the
        action-step is the next action prediction.
        """
        if T is None:
            T = self.action_horizon
        if T > self.action_horizon:
            raise ValueError(f"generate T={T} exceeds action_horizon")

        cond_seq = self._encode_cond(obs, T)  # (B, T, d_cond)
        c_tok = self.cond_in(cond_seq)  # (B, T, d_model)
        dtype = c_tok.dtype

        # Allocate the backbone's inference cache sized for the doubled stream.
        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T,
            device=device,
            dtype=dtype,
        )

        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        a_prev = self.bos.expand(batch_size, -1, -1).to(dtype)  # (B, 1, d_model)
        for t in range(T):
            # Cond step.
            x_c = c_tok[:, t : t + 1] + self.pos_emb[:, 2 * t : 2 * t + 1].to(dtype)
            _ = self.backbone.step(x_c, params)
            # Action step.
            x_a = a_prev + self.pos_emb[:, 2 * t + 1 : 2 * t + 2].to(dtype)
            h = self.backbone.step(x_a, params)
            a_t = self.action_out(h)  # (B, 1, action_dim)
            actions[:, t : t + 1] = a_t
            # Prepare next-step's a_prev (a_t becomes a_{t-1} for the next outer step).
            a_prev = self.action_in(a_t)

        return actions


class HNet(Algo):
    """
    H-Net policy Algo. Single-domain action-sequence model with per-frame
    obs conditioning -- each action token sees the obs at its own timestep.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        init_weights_range: Optional[float] = None,
        lr_multipliers: Optional[list] = None,
        use_parameter_groups: bool = False,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Training recipe knobs (all OFF by default — opt-in):

        - ``init_weights_range``: if set (e.g. ``0.02``), call
          ``hnet.init_weights(init_weights_range)`` after policy construction
          so ``out_proj`` / ``fc2`` weights get ``1/sqrt(n_residuals)``
          scaling.
        - ``lr_multipliers``: list of per-stage LR scales (outer→inner). If
          set, call ``hnet.apply_lr_multiplier(...)`` which stamps every
          parameter's ``_optim`` dict with ``lr_multiplier``.
        - ``use_parameter_groups``: if True, expose
          ``self.parameter_groups()`` so ``pl_model.configure_optimizers``
          builds AdamW param groups (bias / norm weights get
          ``weight_decay=0``; per-group ``lr = base_lr * lr_multiplier``).
        - ``weight_decay``: the base WD used when building parameter groups
          (only consulted when ``use_parameter_groups=True``). Outside of
          that, the optimizer config in the model YAML drives WD.

        Leaving all of these at their defaults reproduces "standard
        training": PyTorch default init, single LR for all params, single
        WD for all params from the optimizer config.
        """
        super().__init__()
        # ``norm_stats`` is a ``MultiDataset`` instance that owns the
        # per-embodiment per-feature normalization stats AND the key-topology
        # helpers (``keys_of_type``, ``is_key_with_embodiment``,
        # ``zarr_key_to_keyname``). pl_model._instantiate_model passes it in
        # automatically; the previous ``data_schematic`` parameter was legacy
        # and is gone.
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.d_cond = d_cond
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Cache training-recipe knobs for configure_optimizers.
        self.use_parameter_groups = bool(use_parameter_groups)
        self.lr_multipliers = list(lr_multipliers) if lr_multipliers else None
        self.weight_decay = float(weight_decay)

        policy = HNetPolicy(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_model=d_model,
            cond_encoder=cond_encoder,
            hnet=hnet,
        )
        # Apply opt-in training recipe BEFORE moving to device so the init
        # writes hit cpu params (matches upstream's pattern of init pre-move).
        if init_weights_range is not None:
            hnet.init_weights(initializer_range=float(init_weights_range))
        if self.lr_multipliers is not None:
            hnet.apply_lr_multiplier(self.lr_multipliers)
        # Stash the inner HNetCore so parameter_groups can reach it without
        # going through nets["policy"].hnet.
        self._hnet_core = hnet

        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via norm_stats (which owns the
        # MultiDataset key topology — same surface HPT uses).
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)

    # ---- Algo API --------------------------------------------------------

    # Keys emitted by ``pack_collate`` that aren't zarr-mapped data tensors
    # and must be passed through ``process_batch_for_training`` unchanged.
    _PACKED_META_KEYS = (
        "cu_seqlens",
        "max_seq_len",
        "seq_lens",
        "batch_size",
        "embodiment",
        "episode_idx",
        "chunk_offset",
    )

    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            processed[emb_id] = {}
            # Detect packed batches by the presence of cu_seqlens. Packed and
            # padded batches have a different key topology; treat them
            # separately so we don't try to keyname-resolve the meta keys.
            is_packed = "cu_seqlens" in _batch

            for key, value in _batch.items():
                if is_packed and key in self._PACKED_META_KEYS:
                    processed[emb_id][key] = value
                    continue
                key_name = self.norm_stats.zarr_key_to_keyname(key, emb_id)
                # Pre-existing typo: tested ``key is not None`` instead of
                # ``key_name``, which caused unrelated batch keys (e.g.
                # ``metadata.robot_name`` from _read_span) to be stored under
                # the None key. Skip silently when keyname can't be resolved.
                if key_name is not None:
                    processed[emb_id][key_name] = value

            ac_key = self.resolved_ac_keys[emb_id]
            if is_packed:
                # Packed actions are (T_total, action_dim). No pad_mask needed
                # since variable-length is expressed via cu_seqlens.
                processed[emb_id]["pad_mask"] = None
                processed[emb_id]["_packed"] = True
            else:
                B, S, _ = processed[emb_id][ac_key].shape
                processed[emb_id]["pad_mask"] = torch.ones(
                    B, S, 1, device=processed[emb_id][ac_key].device
                )
                processed[emb_id]["_packed"] = False
            # Per-feature normalization via MultiDataset stats: each tensor
            # gets ``(x - mean) / std`` (or quantile equivalent) broadcast
            # against (action_dim,) / (proprio_dim,) stats. Works for both
            # padded ``(B, T, D)`` and packed ``(T_total, D)`` shapes.
            processed[emb_id] = self.norm_stats.normalize(processed[emb_id], emb_id)
            processed[emb_id]["embodiment"] = torch.tensor(
                [emb_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed[emb_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed[emb_id][key] = value
        return processed

    def _build_obs(self, _batch, emb_id):
        obs = {}
        for key in (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        ):
            if key in _batch:
                obs[key] = _batch[key]
        return obs

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)

            if _batch.get("_packed", False):
                cu_seqlens = _batch["cu_seqlens"]
                max_seqlen = int(_batch["max_seq_len"])
                pred, aux = policy.forward_packed(actions, obs, cu_seqlens, max_seqlen)
            else:
                pred, aux = policy(actions, obs)

            mse = nn.functional.mse_loss(pred, actions)
            rloss = ratio_loss_from_aux(aux, device=mse.device)
            predictions[f"{emb_id}_pred"] = pred
            predictions[f"{emb_id}_action_loss"] = mse
            predictions[f"{emb_id}_ratio_loss"] = rloss

            # Per-chunker stats for logging. avg_chunk_len = T / max(#boundaries, 1)
            # so it == 1/F when F>0 and falls back to T when F==0 (no compression).
            stats = chunk_stats_from_aux(aux)
            for k, v in stats.items():
                predictions[f"{emb_id}_{k}"] = torch.tensor(v, device=mse.device)
        return predictions

    @override
    def forward_eval(self, batch):
        """Per-frame teacher-forced eval.

        For each batch entry we run a SINGLE forward pass with the GT
        action stream (same as training) and compare per-frame predictions
        against GT in raw / unnormalized action space. AR rollout used to
        live here but is pointless when the obs sequence is fixed: the
        predicted action doesn't change what the model sees next, so AR
        just compounds exposure-bias error without testing anything
        useful. To test closed-loop behavior, run a separate
        envrollout evaluator that steps a simulator with predicted
        actions.

        Returns a dict with keys ``emb{id}_{ac_key}`` carrying
        ``(B, T_max, action_dim)`` unnormalized predictions (zero-padded
        past each episode's length) and ``emb{id}_seq_lens`` ``(B,)`` so
        downstream metric code can mask the padded positions.
        """
        unnorm = {}
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            if _batch.get("_packed", False):
                preds_padded, seq_lens = self._teacher_forced_packed(_batch, emb_id)
                preds = OrderedDict()
                preds[ac_key] = preds_padded
                unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
                for key, val in unnorm_actions.items():
                    unnorm[f"emb{emb_id}_{key}"] = val
                unnorm[f"emb{emb_id}_seq_lens"] = seq_lens
                continue
            # Padded mode (legacy): no packed dataset wired but kept for
            # completeness.
            obs = self._build_obs(_batch, emb_id)
            actions = _batch[ac_key]
            pred, _ = policy(actions, obs)
            preds = OrderedDict()
            preds[ac_key] = pred
            unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
            for key, val in unnorm_actions.items():
                unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

    @torch.no_grad()
    def _teacher_forced_packed(self, _batch: dict, emb_id: int):
        """Single-pass teacher-forced eval for a packed validation batch.

        Runs ``policy.forward_packed`` (the same path used in training)
        and unpacks the resulting ``(T_total, action_dim)`` predictions
        into ``(B, T_max, action_dim)`` zero-padded per-episode for
        downstream metric / viz code.
        """
        policy = self.nets["policy"]
        ac_key = self.resolved_ac_keys[emb_id]
        actions = _batch[ac_key]
        obs = self._build_obs(_batch, emb_id)
        cu = _batch["cu_seqlens"]
        max_seqlen = int(_batch["max_seq_len"])
        seq_lens = _batch["seq_lens"].clone()
        pred_packed, _ = policy.forward_packed(actions, obs, cu, max_seqlen)

        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        action_dim = policy.action_dim
        preds_padded = torch.zeros(
            B,
            T_max,
            action_dim,
            device=pred_packed.device,
            dtype=pred_packed.dtype,
        )
        for b in range(B):
            s = int(cu[b].item())
            e = int(cu[b + 1].item())
            preds_padded[b, : e - s] = pred_packed[s:e]
        return preds_padded, seq_lens

    @torch.no_grad()
    def _ar_rollout_packed(self, _batch: dict, emb_id: int):
        """Per-episode AR rollout for a packed validation batch.

        For each sub-sequence ``[s, e)`` in ``cu_seqlens``:
          1. Slice that episode's obs into ``(1, T_ep, ...)``.
          2. Call ``policy.generate(obs_ep, batch_size=1, T=T_ep)`` to AR
             rollout exactly ``T_ep`` steps from BOS.
          3. Stash the prediction.

        Returns:
            preds_padded: ``(B, T_max, action_dim)`` (zero-padded past each
                episode's length).
            seq_lens:     ``(B,)`` long, the per-episode rollout lengths
                (matches ``_batch['seq_lens']`` and used for masking the
                padding in downstream MSE).
        """
        policy = self.nets["policy"]
        cu = _batch["cu_seqlens"]
        seq_lens = _batch["seq_lens"].clone()
        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        action_dim = policy.action_dim
        device = self.device

        # Gather the obs keys we need for the cond encoder.
        obs_keys = (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        )
        obs_keys = [k for k in obs_keys if k in _batch]

        preds_padded = torch.zeros(B, T_max, action_dim, device=device)

        for b in range(B):
            s = int(cu[b].item())
            e = int(cu[b + 1].item())
            T_ep = e - s
            # Slice each obs key to the episode's range and add a leading
            # batch dim. The packed tensor is (T_total, ...) so slicing along
            # dim 0 gives (T_ep, ...) → unsqueeze → (1, T_ep, ...).
            obs_ep = {k: _batch[k][s:e].unsqueeze(0) for k in obs_keys}
            a_ep = policy.generate(
                obs_ep,
                batch_size=1,
                device=device,
                T=T_ep,
            )  # (1, T_ep, action_dim)
            preds_padded[b, :T_ep] = a_ep.squeeze(0)

        return preds_padded, seq_lens

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            a = predictions[f"{emb_id}_action_loss"]
            r = predictions[f"{emb_id}_ratio_loss"]
            loss_dict[f"emb{emb_id}_action_loss"] = a
            loss_dict[f"emb{emb_id}_ratio_loss"] = r
            # Ratio-loss weights are baked into each chunker stage, so r
            # is already a properly-weighted sum.
            total = total + a + r

            # Pass non-loss stats through to logging (boundary_rate /
            # avg_chunk_len, per-chunker and aggregate). They are 0-dim
            # tensors so log_info.item() still works.
            for key, value in predictions.items():
                prefix = f"{emb_id}_"
                if not key.startswith(prefix):
                    continue
                tail = key[len(prefix) :]
                if tail in ("pred", "action_loss", "ratio_loss"):
                    continue
                loss_dict[f"emb{emb_id}_{tail}"] = value
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log

    # ----- Optional training recipe hook for pl_model.configure_optimizers ----- #

    def parameter_groups(self, base_lr: float):
        """Return AdamW-ready ``list[dict]`` if ``use_parameter_groups``,
        else ``None`` (caller falls back to ``self.nets.parameters()``).

        Groups are built by the inner HNet stage tree via
        ``HNetCore.parameter_groups(weight_decay=self.weight_decay)``, then
        each group's ``lr`` is set to ``base_lr * lr_multiplier``. Params
        that aren't part of the HNet stage tree (e.g. ``action_in``,
        ``action_out``, ``cond_encoder``, ``bos``, ``pos_emb``) are added in
        a single extra group with ``lr_multiplier=1.0`` so optimizer
        instantiation still sees every learnable parameter exactly once.
        """
        if not self.use_parameter_groups:
            return None

        # Groups for params inside the HNet stage tree.
        groups = self._hnet_core.parameter_groups(weight_decay=self.weight_decay)
        for g in groups:
            g["lr"] = float(base_lr) * float(g.get("lr_multiplier", 1.0))

        # Extra group for everything *not* inside the HNet stage tree.
        hnet_param_ids = {id(p) for g in groups for p in g["params"]}
        extra_params, extra_bias_norm = [], []
        for name, p in self.nets.named_parameters():
            if id(p) in hnet_param_ids or not p.requires_grad:
                continue
            # Bias / norm-weight detection (same rule as parameter_groups).
            if name.endswith(".bias") or ".norm" in name or "rmsnorm" in name.lower():
                extra_bias_norm.append(p)
            else:
                extra_params.append(p)
        if extra_params:
            groups.append(
                {
                    "params": extra_params,
                    "lr": float(base_lr),
                    "lr_multiplier": 1.0,
                    "weight_decay": self.weight_decay,
                }
            )
        if extra_bias_norm:
            groups.append(
                {
                    "params": extra_bias_norm,
                    "lr": float(base_lr),
                    "lr_multiplier": 1.0,
                    "weight_decay": 0.0,
                }
            )
        return groups


class HNetFused(HNet):
    """Flat-transformer (no chunker) variant: interleaved [c_t, a_t] tokens.

    Same Algo contract as :class:`HNet`. Replaces the H-Net stage hierarchy
    with a single :class:`FlatFusedPolicy` (one Isotropic stack over a 2T
    interleaved input). ``aux`` is always ``[]`` so ratio_loss is 0 and
    chunk_stats is empty; logging surfaces only action_loss.

    Reuses HNet's ``process_batch_for_training`` / ``forward_training`` /
    ``forward_eval`` / ``compute_losses`` / ``log_info`` /
    ``_ar_rollout_packed`` unchanged because the policy interface matches
    HNetPolicy verbatim.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        arch_layout: str = "T8",
        num_heads: int = 4,
        d_intermediate: int = 512,
        **kwargs,
    ):
        # Skip HNet.__init__ — it requires a HNetCore. We re-implement the
        # tiny init here for the flat path.
        Algo.__init__(self)
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.d_cond = d_cond
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_parameter_groups = False
        self.lr_multipliers = None
        self.weight_decay = 0.0
        self._hnet_core = None
        self.action_dim = action_dim

        policy = FlatFusedPolicy(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_model=d_model,
            d_cond=d_cond,
            cond_encoder=cond_encoder,
            arch_layout=arch_layout,
            num_heads=num_heads,
            d_intermediate=d_intermediate,
        )
        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Replicate HNet's key-resolution loop (norm_stats-based).
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)
