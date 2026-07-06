"""
Register-token masked-transformer chunk interface (design: vault
"Expressive Chunk Downsampler-Upsampler", CURRENT PROPOSAL 2026-06-10/11).

Replaces first-token-down + duplicate-up + EMA with ONE masked-attention
pattern. m learned REGISTER tokens are inserted at each chunk's END; a
structured mask makes them act as the down/inner/up interface:

  down:  register r of chunk c  -> attends member tokens of chunk c
  inner: register r of chunk c  -> attends registers of chunks c' <= c
  up:    member t of chunk c    -> attends PREVIOUS chunk's registers
                                   + own-chunk earlier-or-equal members

Causality is exact: registers sit AFTER their chunk's members so they may
see the whole chunk; a member sees only earlier chunks' (complete)
registers + its own causal prefix -> no future leak, KV-cacheable for AR.
This is the shift-by-one-chunk semantics realized as a mask.

This module builds the augmented stream + the boolean attention mask; the
actual attention is a standard transformer block run with that mask. Member
outputs are recovered by ``out[member_idx]``.

Packed convention: hidden (T, D), cu_seqlens (B+1,), boundary_mask (T,) with
chunk starts forced True.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet.blocks import MultiHeadAttention, RMSNorm


def _fourier_phase(phase: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(n_freqs, device=phase.device)) * math.pi
    ang = phase[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


@dataclass
class RegisterStream:
    aug_hidden: torch.Tensor  # (L, D) interleaved members+registers
    attn_mask: torch.Tensor  # (L, L) bool, True = ALLOWED
    member_idx: torch.Tensor  # (T,) aug positions of original members
    reg_idx: torch.Tensor  # (m*N,) aug positions of registers
    seg_id_aug: torch.Tensor  # (L,) chunk id per aug position
    is_register: torch.Tensor  # (L,) bool
    aug_cu_seqlens: torch.Tensor  # (B+1,) subseq boundaries in aug space
    # (m*N,) subseq id per register token, in reg_idx (chunk-major) order. Used
    # by the DOWN-only encoder to build register-rate cu_seqlens. Optional so
    # the FUSED forward (which doesn't need it) stays byte-identical.
    seq_aug_per_register: Optional[torch.Tensor] = None


class RegisterInterface(nn.Module):
    """Builds the augmented register stream + mask, and adds member phase.

    Args:
      d_model: token dim.
      m: registers per chunk.
      n_phase_freqs: fourier features for member phase.
      component_end_norm: when True, expose an RMSNorm to be applied at the
        encoder's OUTPUT (the paper's "network normalization" at the
        component/encoder end). DEFAULT False -> no extra param, byte-identical
        to the pre-knob arch. Built here (not on the trunk) so it lives next to
        the register tokens it normalizes; the trunk applies it on the chunk-rate
        register outputs in the DOWN-only path.
    """

    def __init__(
        self,
        d_model: int,
        m: int = 4,
        n_phase_freqs: int = 4,
        component_end_norm: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.m = int(m)
        self.n_phase_freqs = n_phase_freqs
        # learned register initial content + a small projection of the
        # member-phase fourier features added onto member tokens.
        self.register_emb = nn.Parameter(torch.randn(m, d_model) * 0.02)
        self.phase_proj = nn.Linear(2 * n_phase_freqs, d_model, **fk)
        # Gated encoder-end normalization (off by default => no new params).
        self.component_end_norm = bool(component_end_norm)
        self.end_norm = RMSNorm(d_model) if self.component_end_norm else None

    @torch.no_grad()
    def _layout(self, boundary_mask, cu_seqlens, causal_phase: bool = False):
        """Compute the interleaving layout (index-only, no grad).

        ``causal_phase``: when True, the member phase is the prefix-causal
        ``offset/(offset+1)`` (a function of the running offset alone,
        reproducible at AR emit-time) instead of the teacher-forced full-chunk
        ``offset/chunk_len``. Mirrors ``scan_interface._phase_from_offset_len``
        so the register AR ``step`` can compare against a TF reference under the
        SAME phase definition (GATE-1, exact match). Default False keeps the
        training forward byte-identical to GATE-0."""
        device = boundary_mask.device
        T = boundary_mask.shape[0]
        # ROBUST LAYOUT (cotrain tiling-gap fix): force a boundary at every
        # subseq start so (a) ``seg`` never goes negative — the first token of
        # the packed stream is always a chunk start, so ``cumsum(0)-1`` >= 0 —
        # and (b) no chunk straddles a subseq boundary (every chunk is wholly
        # contained in one subseq, which ``chunk_seq``/``seq_aug`` rely on). The
        # real packed routers (cossim/scan/PSER) already set
        # ``boundary_prob[cu_seqlens[:-1]] = 1.0``, so for a well-formed mask
        # this OR is a NO-OP and the layout is byte-identical; it only repairs
        # a mask with a leading/interior tiling gap (negative chunk ids or an
        # uncovered aug slot feeding ``torch.bincount``). The clone keeps the
        # caller's tensor intact.
        boundary_mask = boundary_mask.clone()
        if T > 0:
            boundary_mask[cu_seqlens[:-1]] = True
        seg = boundary_mask.cumsum(0) - 1  # (T,) chunk id, global, >= 0
        N = int(seg[-1].item()) + 1
        m = self.m
        # subseq id per token.
        tok_seq = (torch.arange(T, device=device)[:, None] >= cu_seqlens[None, 1:]).sum(
            -1
        )
        # First token position of each chunk == the boundary positions, in chunk
        # order. With every subseq start forced True above, ``boundary_mask`` has
        # exactly N True entries (one per chunk) so ``nonzero`` yields the N chunk
        # starts sorted ascending == ``first_pos`` directly. This is DETERMINISTIC
        # — it replaces the previous ``scatter_`` with duplicate ``seg`` indices,
        # whose "last writer wins" semantics are UNDEFINED on CUDA (the duplicate
        # writes raced, giving wrong/negative ``offset`` -> negative ``member_idx``
        # -> uncovered ``seq_aug`` slots -> the bincount crash). Indices here are
        # unique, so CPU and GPU agree exactly.
        first_pos = boundary_mask.nonzero(
            as_tuple=False
        ).flatten()  # (N,) min pos per chunk
        # subseq id per chunk = subseq of its first token (deterministic gather).
        chunk_seq = tok_seq[first_pos]  # (N,)
        # member offset within chunk.
        offset = torch.arange(T, device=device) - first_pos[seg]
        chunk_len = torch.bincount(seg, minlength=N)
        # ---- build aug order: per subseq, chunks in order, each = members then m regs
        # aug length
        L = T + m * N
        # position of each member and register block in aug space:
        # walk chunks; member block size = chunk_len[c], then m regs.
        block_sizes = chunk_len + m  # (N,)
        chunk_aug_start = F.pad(block_sizes.cumsum(0), (1, 0))[:N]  # (N,)
        member_idx = chunk_aug_start[seg] + offset  # (T,)
        reg_local = torch.arange(m, device=device)
        reg_idx = (
            chunk_aug_start[:, None] + chunk_len[:, None] + reg_local[None, :]
        ).reshape(-1)  # (m*N,)
        # seg/register/phase in aug space. ``zeros`` (not ``empty``) so an
        # uncovered slot — impossible now that ``offset``/``member_idx`` are
        # correct, but cheap insurance — is a valid non-negative bincount input
        # rather than uninitialized garbage. Identical output when coverage is
        # complete (every slot is overwritten below).
        seg_aug = torch.zeros(L, dtype=torch.long, device=device)
        seg_aug[member_idx] = seg
        seg_aug[reg_idx] = torch.repeat_interleave(torch.arange(N, device=device), m)
        is_reg = torch.zeros(L, dtype=torch.bool, device=device)
        is_reg[reg_idx] = True
        if causal_phase:
            o = offset.float()
            phase = o / (o + 1.0)
        else:
            phase = offset.float() / chunk_len[seg].clamp(min=1).float()
        # aug subseq id + cu. ``zeros`` (not ``empty``) for the same insurance.
        seq_aug = torch.zeros(L, dtype=torch.long, device=device)
        seq_aug[member_idx] = tok_seq
        seq_aug[reg_idx] = torch.repeat_interleave(chunk_seq, m)
        B = cu_seqlens.shape[0] - 1
        counts = torch.bincount(seq_aug, minlength=B)
        aug_cu = F.pad(counts.cumsum(0), (1, 0))
        return dict(
            L=L,
            N=N,
            seg_aug=seg_aug,
            is_reg=is_reg,
            member_idx=member_idx,
            reg_idx=reg_idx,
            phase=phase,
            seq_aug=seq_aug,
            aug_cu=aug_cu,
            offset=offset,
            seg=seg,
            chunk_len=chunk_len,
            chunk_aug_start=chunk_aug_start,
            tok_seq=tok_seq,
        )

    @torch.no_grad()
    def _build_mask(self, lay):
        """(L, L) bool, True = query i ALLOWED to attend key j."""
        seg = lay["seg_aug"]
        is_reg = lay["is_reg"]
        seq = lay["seq_aug"]
        L = lay["L"]
        device = seg.device
        same_subseq = seq[:, None] == seq[None, :]
        seg_i = seg[:, None]
        seg_j = seg[None, :]
        regi = is_reg[:, None]
        regj = is_reg[None, :]
        memi = ~regi
        memj = ~regj
        # aug position for member causal-prefix test
        augpos = torch.arange(L, device=device)
        pos_i = augpos[:, None]
        pos_j = augpos[None, :]

        # 1 down: register i attends member j of same chunk
        down = regi & memj & (seg_i == seg_j)
        # 2 inner: register i attends register j of chunk <= c
        inner = regi & regj & (seg_j <= seg_i)
        # 3a up-members: member i attends member j of same chunk with pos<=i
        up_mem = memi & memj & (seg_i == seg_j) & (pos_j <= pos_i)
        # 3b up-registers: member i attends registers of PREVIOUS chunk
        up_reg = memi & regj & (seg_j == seg_i - 1)
        allow = (down | inner | up_mem | up_reg) & same_subseq
        # every token attends itself (numerical safety for softmax rows)
        allow = allow | torch.eye(L, dtype=torch.bool, device=device)
        return allow

    def forward(
        self, hidden_states, boundary_mask, cu_seqlens, causal_phase: bool = False
    ) -> RegisterStream:
        lay = self._layout(boundary_mask, cu_seqlens, causal_phase=causal_phase)
        L = lay["L"]
        aug = hidden_states.new_zeros(L, self.d_model)
        # members: token + phase projection
        ph = self.phase_proj(_fourier_phase(lay["phase"], self.n_phase_freqs))
        aug[lay["member_idx"]] = hidden_states + ph
        # registers: learned init (m,D) tiled per chunk
        reg_init = self.register_emb.repeat(lay["N"], 1)  # (m*N, D)
        aug[lay["reg_idx"]] = reg_init.to(aug.dtype)
        mask = self._build_mask(lay)
        return RegisterStream(
            aug_hidden=aug,
            attn_mask=mask,
            member_idx=lay["member_idx"],
            reg_idx=lay["reg_idx"],
            seg_id_aug=lay["seg_aug"],
            is_register=lay["is_reg"],
            aug_cu_seqlens=lay["aug_cu"],
            # subseq id per register, in reg_idx (chunk-major) order — for the
            # DOWN-only encoder's register-rate cu_seqlens.
            seq_aug_per_register=lay["seq_aug"][lay["reg_idx"]],
        )

    # ------------------------------------------------------------------ #
    # AR helpers — build the per-token member embedding the SAME way the TF
    # forward does, but from a running offset (prefix-causal phase). Used by
    # the register AR step so member tokens entering the trunk match the TF
    # ``aug[member_idx]`` byte-for-byte under causal_phase=True.
    # ------------------------------------------------------------------ #

    def member_embed_from_offset(
        self, x_t: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """x_t: (B, D) raw member token. offset: (B,) running position within
        its (still-open) chunk. Returns x_t + phase_proj(fourier(phi)) with the
        prefix-causal phase phi = offset/(offset+1) — identical to the TF
        ``forward(causal_phase=True)`` member content."""
        o = offset.float()
        phi = o / (o + 1.0)
        ph = self.phase_proj(_fourier_phase(phi, self.n_phase_freqs))
        return x_t + ph

    def register_init(self, batch_size: int, device, dtype) -> torch.Tensor:
        """The learned m-register initial content tiled for ``batch_size``
        chunks: (B, m, D). Matches the TF ``register_emb.repeat`` content."""
        return (
            self.register_emb[None]
            .to(device=device, dtype=dtype)
            .expand(batch_size, self.m, self.d_model)
        )


# --------------------------------------------------------------------------- #
# RegisterTrunk — the fused masked transformer over the augmented stream.
#
# One stack of ``depth`` prenorm masked-attention + MLP blocks realizes
# down (registers <- own members) + inner (registers <- earlier registers,
# the chunk-rate trunk) + up (members <- prev-chunk registers + own causal
# prefix) through the single (L, L) RegisterStream mask. The TF forward runs
# the whole augmented stream; the AR path (``step`` machinery in ChunkerStage)
# uses per-layer KV caches split into a PER-CHUNK REGISTER cache (finalized at
# each boundary) + a CURRENT-CHUNK MEMBER cache, exploiting that a member never
# attends its own-chunk registers nor any future token (GATE-0 causality).
# --------------------------------------------------------------------------- #


@dataclass
class RegisterTrunkState:
    """Per-layer KV caches for the register AR step.

    Two caches per layer:
      * ``reg_kv`` — K/V of all FINALIZED registers so far (prev chunks). A
        register is finalized JOINTLY with its m siblings at the chunk boundary
        (same-chunk registers are mutually visible, so they finalize together).
        Layout (B, H, cap_reg, Dh) with a per-row fill counter ``reg_fill``.
      * ``mem_kv`` — K/V of the CURRENT (open) chunk's members. Reset to empty
        when a boundary opens a new chunk. Layout (B, H, cap_mem, Dh) with
        ``mem_fill``.
    Plus chunk bookkeeping: ``offset`` (position within the open chunk),
    ``has_seen`` (episode started), and ``mem_x`` — the raw post-phase member
    inputs of the open chunk (needed to recompute the chunk through the trunk
    at finalization, where registers attend members' per-layer reps)."""

    reg_k: list  # depth x (B, H, cap_reg, Dh)
    reg_v: list  # depth x (B, H, cap_reg, Dh)
    reg_fill: torch.Tensor  # (B,) long — # finalized register slots
    mem_k: list  # depth x (B, H, cap_mem, Dh)
    mem_v: list  # depth x (B, H, cap_mem, Dh)
    mem_fill: torch.Tensor  # (B,) long — # members in the open chunk
    mem_x: torch.Tensor  # (B, cap_mem, D) post-phase member inputs (open chunk)
    offset: torch.Tensor  # (B,) long — running offset within the open chunk
    has_seen: torch.Tensor  # (B,) bool


class RegisterTrunk(nn.Module):
    """``depth`` prenorm (masked-attn -> MLP) blocks consuming a RegisterStream.

    ``forward(stream)`` runs the full augmented stream with the (L, L) mask and
    returns the trunk output at every aug position (members recovered by the
    caller via ``stream.member_idx``). The per-block structure mirrors
    ``transformer_router``'s self-contained stack (RMSNorm prenorm, repo
    ``MultiHeadAttention.forward_masked`` for attention, GELU MLP, residual).
    Attention is bidirectional-within-the-mask (causal=False); the MASK alone
    enforces causality, exactly as GATE-0 verified."""

    def __init__(
        self,
        d_model: int,
        depth: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        # rotary off by default — the register trunk gets order from the
        # member-phase features + the structural mask, keeping the AR phase
        # exactly reproducible. (Exposed as a knob for ablations.)
        self.rotary_emb_dim = int(rotary_emb_dim)
        self.attn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(self.depth)])
        self.attns = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model,
                    n_heads,
                    causal=False,
                    dropout=0.0,
                    rotary_emb_dim=self.rotary_emb_dim,
                    rotary_emb_base=rotary_emb_base,
                )
                for _ in range(self.depth)
            ]
        )
        self.mlp_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(self.depth)])
        ff_dim = int(ff_mult * d_model)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_dim, **fk),
                    nn.GELU(),
                    nn.Linear(ff_dim, d_model, **fk),
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, stream: "RegisterStream") -> torch.Tensor:
        """Run the augmented stream through the masked trunk. Returns (L, D)."""
        x = stream.aug_hidden
        mask = stream.attn_mask
        for i in range(self.depth):
            x = x + self.attns[i].forward_masked(self.attn_norms[i](x), mask)
            x = x + self.mlps[i](self.mlp_norms[i](x))
        return x

    def n_residual_adds(self) -> int:
        """# residual-stream adds this trunk contributes: 2 per block (attn +
        MLP) over ``depth`` blocks (mirrors the Isotropic ``height`` counting in
        ``_init_isotropic_linears``)."""
        return 2 * self.depth

    def _init_weights(self, initializer_range: float, n_residuals: int) -> None:
        """FIX #7: residual-aware init. The trunk's attn ``out_proj`` and each
        MLP's second Linear (``fc2`` — the residual-stream writer) are scaled by
        ``initializer_range / sqrt(n_residuals)`` so the residual norm stays
        bounded with depth; all other Linears use ``initializer_range``. Mirrors
        ``_init_isotropic_linears`` but matches the RegisterTrunk's own module
        naming (the MLP is an ``nn.Sequential`` so its second Linear is at index
        2, not named ``fc2``). ``n_residuals`` is the CUMULATIVE count (parent
        residuals + this trunk's ``n_residual_adds()``).

        GATED at the call site (``ChunkerStage.reg_trunk_residual_init``): OFF by
        default so existing register configs are byte-unchanged. Param count is
        unaffected — only init values change.
        """
        scaled_std = initializer_range / max(n_residuals, 1) ** 0.5
        for i in range(self.depth):
            # Attention out-projection writes into the residual stream.
            op = getattr(self.attns[i], "out_proj", None)
            if op is not None and not getattr(op.weight, "_no_reinit", False):
                nn.init.normal_(op.weight, mean=0.0, std=scaled_std)
                if op.bias is not None:
                    nn.init.zeros_(op.bias)
            # Other attention Linears (q/k/v/in projections) use base range.
            for name, mod in self.attns[i].named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                if "out_proj" in name:
                    continue  # already handled above (scaled)
                if getattr(mod.weight, "_no_reinit", False):
                    continue
                nn.init.normal_(mod.weight, mean=0.0, std=initializer_range)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            # MLP: Sequential(Linear_in, GELU, Linear_out). Linear_out (idx 2,
            # the "fc2") writes the residual -> scaled; Linear_in -> base range.
            mlp = self.mlps[i]
            fc1, fc2 = mlp[0], mlp[2]
            if not getattr(fc1.weight, "_no_reinit", False):
                nn.init.normal_(fc1.weight, mean=0.0, std=initializer_range)
                if fc1.bias is not None:
                    nn.init.zeros_(fc1.bias)
            if not getattr(fc2.weight, "_no_reinit", False):
                nn.init.normal_(fc2.weight, mean=0.0, std=scaled_std)
                if fc2.bias is not None:
                    nn.init.zeros_(fc2.bias)

    # ------------------------------------------------------------------ #
    # DOWN-ONLY encoder use.
    #
    # The same masked trunk, but the caller consumes the REGISTER outputs
    # (chunk-rate tokens, m per chunk) instead of the member outputs. This is
    # the H-Net "transformer chunker": the register encoder produces the
    # contextualized chunk-rate stream (down + register<-register inner) which a
    # SEPARATE inner_stage + up_interface + router then consume — exactly the
    # role ``ChunkScanEncoder`` plays for the scan/spline path (it returns a
    # ``chunked`` (N, D) tensor), here at register rate (N*m, D).
    #
    # The member ("up") outputs are simply DISCARDED in this mode: the bespoke
    # dechunker (NEXT build step) reconstructs members from the inner-processed
    # register tokens via its own attention, so the trunk's own up-rule is not
    # used. Computing it is cheap and keeps the TF path a single attention call;
    # gating it out would only complicate the mask for no param savings.
    # ------------------------------------------------------------------ #

    def encode_down(
        self,
        interface: "RegisterInterface",
        hidden_states: torch.Tensor,  # (T, D)
        boundary_mask: torch.Tensor,  # (T,) bool
        cu_seqlens: torch.Tensor,  # (B+1,)
        causal_phase: bool = False,
    ):
        """DOWN-only forward: returns ``(reg_tokens, next_cu, next_max, stream)``.

          reg_tokens (N*m, D): trunk outputs at the m register positions of each
            chunk, in chunk-major / register-minor order (chunk 0 regs 0..m-1,
            chunk 1 regs 0..m-1, ...). This is the chunk-rate stream the inner
            stage + up_interface consume.
          next_cu (B+1,): register-rate cu_seqlens = m * (chunks per subseq).
          next_max (int): max registers per subseq = m * max chunks per subseq.
          stream: the RegisterStream (member_idx / reg_idx / aug_cu / masks) so
            the up_interface / dechunker can map chunk-rate tokens back to tokens.

        Same masked attention as ``forward``; only the GATHER differs (reg_idx
        not member_idx) plus the optional encoder-end RMSNorm."""
        stream = interface(
            hidden_states, boundary_mask, cu_seqlens, causal_phase=causal_phase
        )
        trunk_out = self.forward(stream)  # (L, D)
        reg_tokens = trunk_out[stream.reg_idx]  # (N*m, D), chunk-major
        if interface.end_norm is not None:
            reg_tokens = interface.end_norm(reg_tokens)
        # register-rate cu_seqlens: m registers per chunk. ``reg_idx`` is built
        # chunk-major (chunk c contributes slots [c*m .. c*m+m)); ``seq_aug`` at
        # those slots is the subseq id per register, already in reg_idx order. So
        # per-subseq register counts are bincount(seq_aug[reg_idx]). This is the
        # register-rate analogue of ChunkScanEncoder.next_cu (which counts chunks
        # per subseq) and equals m * (chunks per subseq) because every chunk
        # contributes exactly m registers and no chunk straddles a subseq (the
        # layout forces a boundary at each subseq start).
        reg_seq = stream.seq_aug_per_register  # (N*m,) subseq id per register
        B = cu_seqlens.shape[0] - 1
        counts = torch.bincount(reg_seq, minlength=B)  # registers per subseq
        next_cu = F.pad(counts.cumsum(0), (1, 0))
        next_max = int(counts.max().item()) if B > 0 else 0
        return reg_tokens, next_cu, next_max, stream

    # ------------------------------------------------------------------ #
    # AR inference (chunk-incremental KV cache).
    #
    # Exactness rests on the GATE-0 causality of the mask: a member attends
    # only own-chunk earlier-or-equal members + PREVIOUS chunks' (finalized)
    # registers — never its own-chunk registers, never future tokens. So a
    # member is fully computable token-by-token (output one-per-step) from the
    # finalized-register cache + the open-chunk member cache. Registers of a
    # chunk are mutually visible, so they finalize JOINTLY at the boundary,
    # attending [finalized prev registers] + [their own chunk's member KV] +
    # [the m same-chunk registers]. After finalization their KV is appended to
    # the register cache to serve the NEXT chunk.
    # ------------------------------------------------------------------ #

    def allocate(self, batch_size, cap_reg, cap_mem, d_model, device, dtype):
        H, Dh = self.n_heads, self.head_dim
        mk = lambda cap: [  # noqa: E731
            torch.zeros(batch_size, H, cap, Dh, device=device, dtype=dtype)
            for _ in range(self.depth)
        ]
        return RegisterTrunkState(
            reg_k=mk(cap_reg),
            reg_v=mk(cap_reg),
            reg_fill=torch.zeros(batch_size, device=device, dtype=torch.long),
            mem_k=mk(cap_mem),
            mem_v=mk(cap_mem),
            mem_fill=torch.zeros(batch_size, device=device, dtype=torch.long),
            mem_x=torch.zeros(batch_size, cap_mem, d_model, device=device, dtype=dtype),
            offset=torch.zeros(batch_size, device=device, dtype=torch.long),
            has_seen=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    def _qkv(self, attn: MultiHeadAttention, h):
        """Project (B, n, D) -> q,k,v each (B, H, n, Dh). No RoPE (trunk default)."""
        B, n, _ = h.shape
        qkv = attn.qkv(h).reshape(B, n, 3, attn.num_heads, attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def step_member(self, x_t, state: RegisterTrunkState, m: int):
        """Emit ONE member token. ``x_t``: (B, D) post-phase member input for the
        CURRENT open chunk. ``m``: registers per chunk. Appends the member's
        per-layer KV to the open-chunk member cache and returns the member's
        trunk output (B, D).

        At each layer the member attends [the immediately-PREVIOUS chunk's m
        finalized registers] ++ [open-chunk member KV INCLUDING itself].
        Mirrors the TF up-rule (members <- prev-chunk regs + own causal prefix);
        the prev chunk is the LAST m finalized register slots. Chunk/reset
        bookkeeping is owned by the caller (ChunkerStage._step_register)."""
        B = x_t.shape[0]
        # Stash the raw post-phase member input at the current member slot so the
        # boundary finalization can recompute this chunk through the trunk for the
        # register side.
        slot = state.mem_fill  # (B,)
        bidx = torch.arange(B, device=x_t.device)
        state.mem_x[bidx, slot] = x_t.to(state.mem_x.dtype)
        # immediately-previous chunk's register window [reg_fill - m, reg_fill).
        reg_lo = (state.reg_fill - m).clamp(min=0)
        reg_hi = state.reg_fill
        mem_hi = slot + 1  # include self

        x = x_t.unsqueeze(1)  # (B, 1, D)
        for i in range(self.depth):
            h = self.attn_norms[i](x)  # (B, 1, D)
            q, k, v = self._qkv(self.attns[i], h)  # (B, H, 1, Dh)
            # scatter this member's k/v into the open-chunk member cache.
            state.mem_k[i][bidx, :, slot] = k.squeeze(2).to(state.mem_k[i].dtype)
            state.mem_v[i][bidx, :, slot] = v.squeeze(2).to(state.mem_v[i].dtype)
            attn_out = self._attend(
                q,
                state,
                i,
                reg_lo=reg_lo,
                reg_hi=reg_hi,
                mem_lo=None,
                mem_hi=mem_hi,
            )  # (B, 1, D)
            x = x + self.attns[i].out_proj(attn_out)
            x = x + self.mlps[i](self.mlp_norms[i](x))
        # advance the open-chunk member counters.
        state.mem_fill = state.mem_fill + 1
        return x.squeeze(1)

    def finalize_chunk(
        self,
        state: RegisterTrunkState,
        finished_mask,
        reg_interface: "RegisterInterface",
    ):
        """Finalize the just-completed chunk's m registers for the rows in
        ``finished_mask`` and append their per-layer KV to the register cache.

        Processes the m registers JOINTLY through the trunk. At layer i the
        registers attend [finalized prev regs] + [this chunk's member KV at
        layer i] + [the m same-chunk registers (mutually visible)]. After all
        layers their KV is appended to ``reg_k/reg_v`` and ``reg_fill += m``.
        Then resets the open-chunk member cache for the rows that finalized
        (a fresh chunk is about to open).

        Runs on ALL B rows (cheap) and writes back only ``finished_mask`` rows
        so the per-batch slice/scatter stays uniform."""
        B = state.reg_fill.shape[0]
        m = reg_interface.m
        device = state.reg_fill.device
        dtype = state.mem_x.dtype
        # register initial content (B, m, D) — same as TF register_emb tile.
        x = reg_interface.register_init(B, device, dtype)  # (B, m, D)
        bidx = torch.arange(B, device=device)
        for i in range(self.depth):
            h = self.attn_norms[i](x)  # (B, m, D)
            q, k, v = self._qkv(self.attns[i], h)  # (B, H, m, Dh)
            # registers attend: ALL prior finalized regs (inner: seg_j<=seg_i,
            # the j<seg_i part) + this chunk's members (down) + the m same-chunk
            # registers (the seg_j==seg_i part of inner, mutually visible).
            attn_out = self._attend(
                q,
                state,
                i,
                reg_lo=None,
                reg_hi=state.reg_fill,
                mem_lo=None,
                mem_hi=state.mem_fill,
                extra_k=k,
                extra_v=v,  # full m x m visibility
            )  # (B, m, D)
            x = x + self.attns[i].out_proj(attn_out)
            x = x + self.mlps[i](self.mlp_norms[i](x))
            # cache the FINALIZED register k/v at this layer (append at reg_fill).
            # write all rows into a scratch then scatter only finished rows.
            for j in range(m):
                pos = state.reg_fill + j  # (B,)
                fk = state.reg_k[i]
                fv = state.reg_v[i]
                # only write finished rows; others keep their (unused) slots.
                kj = k[:, :, j].to(fk.dtype)  # (B, H, Dh)
                vj = v[:, :, j].to(fv.dtype)
                wmask = finished_mask.view(B, 1, 1)
                fk[bidx, :, pos] = torch.where(wmask, kj, fk[bidx, :, pos])
                fv[bidx, :, pos] = torch.where(wmask, vj, fv[bidx, :, pos])
        # bump register fill + reset member cache for finalized rows.
        state.reg_fill = torch.where(finished_mask, state.reg_fill + m, state.reg_fill)
        state.mem_fill = torch.where(
            finished_mask, torch.zeros_like(state.mem_fill), state.mem_fill
        )
        # DECHUNK-AR: return the post-trunk register representations (B, m, D) of
        # the just-finalized chunk (previously discarded). The inner stage +
        # msublatent dechunker consume these; caller reads only finished rows.
        return x

    def _attend(
        self,
        q,
        state,
        layer,
        reg_lo,
        reg_hi,
        mem_lo,
        mem_hi,
        extra_k=None,
        extra_v=None,
    ):
        """SDPA of ``q`` (B, H, n, Dh) over the concatenation of
        [register-cache slots in [reg_lo, reg_hi)] ++ [member-cache slots in
        [mem_lo, mem_hi)] ++ optional ``extra`` KV (the m same-chunk registers,
        fully visible, used only in finalize). All ranges are per-row (B,) long
        tensors (or None => 0 lo). Per-row masks select each row's valid window;
        softmax is permutation-invariant over keys so the concat order is
        irrelevant. Returns (B, n, D)."""
        B, H, n, Dh = q.shape
        device = q.device
        keys, vals, masks = [], [], []

        def _window(kc, vc, lo, hi):
            cap = kc.shape[2]
            ar = torch.arange(cap, device=device)[None, :]  # (1, cap)
            lo_t = (
                torch.zeros(B, 1, dtype=torch.long, device=device)
                if lo is None
                else lo[:, None]
            )
            valid = (ar >= lo_t) & (ar < hi[:, None])  # (B, cap)
            keys.append(kc)
            vals.append(vc)
            masks.append(valid)

        if reg_hi is not None:
            _window(state.reg_k[layer], state.reg_v[layer], reg_lo, reg_hi)
        if mem_hi is not None:
            _window(state.mem_k[layer], state.mem_v[layer], mem_lo, mem_hi)
        if extra_k is not None:
            keys.append(extra_k)
            vals.append(extra_v)
            masks.append(
                torch.ones(B, extra_k.shape[2], dtype=torch.bool, device=device)
            )

        K = torch.cat(keys, dim=2)  # (B, H, S, Dh)
        V = torch.cat(vals, dim=2)
        M = torch.cat(masks, dim=1)  # (B, S)
        attn_mask = M[:, None, None, :]  # broadcast over (H, n)
        out = F.scaled_dot_product_attention(
            q, K, V, attn_mask=attn_mask, dropout_p=0.0
        )
        return out.transpose(1, 2).reshape(B, n, H * Dh)
