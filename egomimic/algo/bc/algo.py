"""Windowed BC policy/algo for EgoVerse -- faithful robomimic BC-RNN(-GMM) recipe.

Canonical class names are ``WindowedBC`` (algo) and ``WindowedBCPolicy`` (the
nn.Module). ``BCRNN`` / ``BCRNNPolicy`` are kept as module-level aliases at the
bottom of this file for backward compatibility (old configs / ported yamls
and saved checkpoints). The classes live in ``egomimic.algo.bc.algo`` and are
re-exported by the ``egomimic.algo.bc`` package (one folder per model, peer of
``hpt`` / ``act`` / ``pi``), so the canonical import path stays
``egomimic.algo.bc.WindowedBC``; the ``egomimic.algo.bc_rnn`` re-export shim was
deleted at the DESIGN.md step-13 final flip.

The RNN conditions on OBSERVATION HISTORY only: an LSTM is unrolled over the
per-frame obs embeddings; the action is the OUTPUT decoded from each LSTM
hidden state by a GMM head. NO past-action input (the difference from the
H-Net / fused autoregressive policies). Pipeline per step t:
    obs_t --ObsEncoder--> e_t --LSTM(history)--> h_t --GMMHead--> a_t

EXACT-REPLICA alignment to robomimic BC_RNN_GMM (robomimic/algo/bc.py +
robomimic/models/policy_nets.py::RNNGMMActorNetwork):

* TRAINING ON LENGTH-`rnn_horizon` (=10) WINDOWS, hidden zero-init per window.
  robomimic's SequenceDataset serves length-`seq_length` (=rnn.horizon) demo
  subsequences; ``RNN_MIMO_MLP.forward`` is called with ``rnn_init_state=None``
  (so a fresh ZERO hidden per window). We replicate this by cutting all
  consecutive length-`H` windows out of each padded episode (respecting the
  validity mask), stacking them as the batch, and unrolling each with a fresh
  zero hidden. This is DIFFERENT from a full-episode unroll: each window's
  hidden starts at zero, so the LSTM only ever sees up to `H` steps of history
  at train time (matching robomimic).

* ROLLOUT HIDDEN RE-INIT EVERY `rnn_horizon` STEPS. robomimic
  ``BC_RNN.get_action`` (bc.py):
        if self._rnn_hidden_state is None or \\
           self._rnn_counter % self._rnn_horizon == 0:
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(...)
        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(...)
  i.e. the hidden is RE-ZEROED every `rnn_horizon` env steps (NOT carried
  indefinitely). We replicate this exactly in ``WindowedBCPolicy.step`` via a step
  counter that re-inits (h, c) whenever ``counter % rnn_horizon == 0``.

Reuses PackedAlgoBase batch processing.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.algo.hnet import PackedAlgoBase
from egomimic.models.cores.hnet_core import HNetCore
from egomimic.models.cores.lstm_core import LSTMCore
from egomimic.models.cores.transformer_core import TransformerCore
from egomimic.models.heads.gmm_head import GMMActionHead
from egomimic.models.heads.query_decoder import QueryActionDecoder
from egomimic.models.stems.obs_encoder import ObsEncoder
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _pack_to_padded(x, cu_seqlens, B, T_max):
    """Packed (T_total, ...) -> padded (B, T_max, ...) (zero-padded)."""
    out = x.new_zeros((B, T_max) + tuple(x.shape[1:]))
    for b in range(B):
        s = int(cu_seqlens[b].item())
        e = int(cu_seqlens[b + 1].item())
        out[b, : e - s] = x[s:e]
    return out


def _cut_windows(obs_padded, actions_padded, mask, horizon, max_windows=None,
                 pad_mode="zero_masked", obs_stride=1, chunk_len=1,
                 window_anchor="uniform", pad_actions=None):
    """Sample length-`horizon` windows from padded episodes.

    robomimic SequenceDataset serves length-`seq_length` (=rnn.horizon)
    subsequences: each training example is a (demo, start-index) pair sampled
    from the set of all valid start indices, and a fixed ``batch_size`` of them
    forms a batch. We replicate that by enumerating all valid (episode b, start
    s) pairs with s in [0, seq_len_b - 1] and, if there are more than
    ``max_windows``, randomly sampling ``max_windows`` of them WITHOUT
    replacement (a uniform sample over start indices, like robomimic's
    uniform-over-demos-and-starts sampler). Each window [s, s+horizon) is cut
    with a FRESH ZERO hidden at train time. Windows running past the episode end
    are right-zero-padded and their tail steps masked (mask=0) -- robomimic uses
    ``pad_seq_length`` + ``pad_mask`` for the same boundary handling, so the NLL
    ignores padded steps.

    ``max_windows`` bounds memory/throughput (one frame can appear in up to H
    windows; without a cap the effective batch is ~H x larger -> OOM). Defaults
    to no cap when None.

    OBS STRIDING + ACTION CHUNKING (obs_stride > 1 or chunk_len > 1):
      obs-step k of a window reads frame ``s + obs_stride*k`` (so the H obs are
      the SUBSAMPLED frames s, s+σ, s+2σ, ..., s+(H-1)σ with σ=obs_stride), and
      its action TARGET is the ``chunk_len`` GT actions at frames
      ``[s+σk, s+σk+chunk_len)``. Returns ``act_w`` of shape
      ``(Nw, H, chunk_len, D)`` (an extra chunk axis) for the chunked GMM head.

      With ``obs_stride == 1 and chunk_len == 1`` (the default) the EXACT
      pre-existing code path runs and the outputs are byte-identical (act_w is
      ``(Nw, H, D)`` with NO chunk axis, same windows, same masks).

    Returns:
        obs_w:     dict key -> (Nw, horizon, ...)
        act_w:     (Nw, horizon, D)              if chunk_len == 1
                   (Nw, horizon, chunk_len, D)   if chunk_len > 1
        mask_w:    (Nw, horizon)   1=valid step, 0=pad
    """
    B, T_max = mask.shape
    H = int(horizon)
    sigma = int(obs_stride)
    C = int(chunk_len)
    seq_lens = mask.sum(dim=1).to(torch.long)  # (B,)

    # WINDOW ANCHORING (single knob; default 'uniform' is byte-identical --
    # ported from the proven EgoVerse2 bc_rnn.py c3000v2 implementation):
    #   'uniform' (default, pre-existing): enumerate all valid (episode, start)
    #     pairs with start s in [0, L-1] (a uniform sample over start frames,
    #     robomimic-style); tails handled by padding below. s indexes a real
    #     frame and is UNCHANGED by striding/chunking.
    #   'start' (full-history): exactly ONE window per episode, anchored at frame
    #     0 -> pairs == [(b, 0) for each episode b]. The window then spans frames
    #     0, sigma, 2*sigma, ... (length-H over the SUBSAMPLED obs), repeat-padded
    #     at the tail. This makes the training context semantics match a
    #     NEVER-RESET rollout (the buffer is never re-zeroed mid-episode), which
    #     is the point of the full-history variant. No max_windows subsample is
    #     applied (there are at most B windows, one per episode in the batch).
    if window_anchor not in ("uniform", "start"):
        raise ValueError(
            f"window_anchor must be uniform|start, got {window_anchor!r}"
        )
    if window_anchor == "start":
        # one (b, 0) per episode; no random subsample (<= B windows total).
        pairs = [(b, 0) for b in range(B)]
    else:
        # enumerate all valid (episode, start) pairs. starts are uniform over all
        # in-episode start frames s in [0, L-1] (tails handled by padding below);
        # this is UNCHANGED by striding/chunking -- s still indexes a real frame.
        pairs = []
        for b in range(B):
            L = int(seq_lens[b].item())
            for s in range(L):
                pairs.append((b, s))
        if max_windows is not None and len(pairs) > int(max_windows):
            # uniform sample WITHOUT replacement over valid starts (robomimic-style)
            idx = torch.randperm(len(pairs), device=actions_padded.device)[: int(max_windows)]
            pairs = [pairs[int(i)] for i in idx]

    # pad_mode (robomimic SequenceDataset boundary handling):
    #   "zero_masked" (default, pre-existing): window tails past the episode end
    #     are ZERO-padded and MASKED out of the NLL (mask=0 on pad steps).
    #   "repeat_unmasked" (robomimic pad_same=True, dataset.py:519 +
    #     get_pad_mask=False train_utils.py:146 + plain mean bc.py:623): window
    #     tails are padded by REPEATING the last real frame (obs AND action),
    #     and EVERY step (incl. pad) is counted in the NLL (mask all-ones).
    #   "repeat_pusher_unmasked" (hold-position pad): identical to
    #     repeat_unmasked for OBS (repeat last real frame) and the mask
    #     (all-ones NLL), but the padded ACTION target is ``pad_actions[b]`` --
    #     the episode's LAST PUSHER POSITION expressed in ACTION-normalized
    #     space (computed by WindowedBC._pusher_pad_actions). Rationale: the
    #     dataset's actions are CURSOR positions that sit ~5-10px off the
    #     pusher; repeating the last cursor action teaches a persistent offset
    #     push after the solve frame, while the pusher position is the true
    #     "hold position" command.
    repeat = pad_mode in ("repeat_unmasked", "repeat_pusher_unmasked")
    pusher_pad = pad_mode == "repeat_pusher_unmasked"
    if pusher_pad and pad_actions is None:
        raise ValueError(
            "pad_mode='repeat_pusher_unmasked' requires pad_actions (B, D) "
            "(the per-episode last-pusher-position action pad)."
        )

    strided = (sigma != 1) or (C != 1)
    if strided:
        return _cut_windows_strided(
            obs_padded, actions_padded, seq_lens, pairs, H, sigma, C, repeat,
            pusher_pad=pusher_pad, pad_actions=pad_actions,
        )

    # ===== pre-existing (byte-identical) path: obs_stride==1, chunk_len==1 =====
    obs_chunks = {k: [] for k in obs_padded}
    act_chunks = []
    mask_chunks = []
    for b, s in pairs:
        L = int(seq_lens[b].item())
        if repeat:
            # clamp the real read to the last valid frame index (L-1); the tail
            # [n:H] is filled by repeating that last real frame below.
            e = min(s + H, L)
        else:
            # pre-existing default path (byte-identical): read up to T_max; the
            # tail beyond episode end is zero and gets masked out.
            e = min(s + H, T_max)
        n = e - s  # real steps available in this window
        # action window
        aw = actions_padded.new_zeros((H,) + actions_padded.shape[2:])
        aw[:n] = actions_padded[b, s:e]
        if repeat and n < H:
            if pusher_pad:
                # hold-position pad: last pusher position in ACTION-norm space
                aw[n:] = pad_actions[b]
            else:
                aw[n:] = actions_padded[b, e - 1]  # repeat last real action
        act_chunks.append(aw)
        # validity mask
        mw = mask.new_zeros((H,))
        if repeat:
            mw[:] = 1.0  # every step counted (robomimic plain mean, no pad mask)
        else:
            valid_n = max(0, min(s + H, L) - s)  # steps before episode end
            mw[:valid_n] = 1.0
        mask_chunks.append(mw)
        for k, v in obs_padded.items():
            ow = v.new_zeros((H,) + v.shape[2:])
            ow[:n] = v[b, s:e]
            if repeat and n < H:
                ow[n:] = v[b, e - 1]  # repeat last real obs frame
            obs_chunks[k].append(ow)

    if not act_chunks:  # degenerate empty batch
        obs_w = {k: v[:, :H] for k, v in obs_padded.items()}
        return obs_w, actions_padded[:, :H], mask[:, :H]

    obs_w = {k: torch.stack(obs_chunks[k], dim=0) for k in obs_chunks}
    act_w = torch.stack(act_chunks, dim=0)
    mask_w = torch.stack(mask_chunks, dim=0)
    return obs_w, act_w, mask_w


def _cut_windows_strided(obs_padded, actions_padded, seq_lens, pairs, H, sigma,
                         C, repeat, pusher_pad=False, pad_actions=None):
    """Strided-obs + action-chunking window cut (obs_stride>1 or chunk_len>1).

    For window start s and each obs-step k in [0, H):
      * obs frame ``f_k = s + sigma*k``. If ``f_k < L`` read the real frame;
        else (tail) under repeat_unmasked repeat the LAST real obs (frame L-1),
        under zero_masked zero it and mask the step out.
      * action TARGET = the ``C`` frames ``[f_k, f_k+C)``. For each target frame
        ``g``: real action if ``g < L``; else (tail) repeat action[L-1]
        (repeat_unmasked), the episode's last-pusher-position pad
        ``pad_actions[b]`` (repeat_pusher_unmasked / ``pusher_pad=True``), or
        zero (zero_masked).

    Mask (per obs-step, NOT per chunk position):
      * repeat_unmasked: all-ones (every step incl. repeat-padded tails counted,
        exactly as paper-exact for the chunk8 run).
      * zero_masked: step k counted iff its obs frame ``f_k < L``.

    Returns obs_w (Nw,H,...), act_w (Nw,H,C,D), mask_w (Nw,H).
    """
    obs_chunks = {k: [] for k in obs_padded}
    act_chunks = []
    mask_chunks = []
    A_tail = actions_padded.shape[2:]  # (D,)
    for b, s in pairs:
        L = int(seq_lens[b].item())
        last = L - 1  # last real frame index
        # obs frame index per step, clamped to [0, L-1] for the repeat path.
        f = torch.arange(H, device=actions_padded.device) * sigma + s  # (H,)
        in_ep = f < L  # (H,) bool: obs frame is a real (in-episode) frame
        f_clamped = torch.clamp(f, max=last)  # (H,)

        # --- obs window: (H, ...) ---
        for k, v in obs_padded.items():
            if repeat:
                ow = v[b, f_clamped]  # repeat-pad tail via clamp to last real
            else:
                ow = v.new_zeros((H,) + tuple(v.shape[2:]))
                ow[in_ep] = v[b, f_clamped[in_ep]]
            obs_chunks[k].append(ow)

        # --- action target chunk: (H, C, D) ---
        # target frame g[k, j] = f_k + j for j in [0, C). clamp to last real
        # frame for repeat-pad; zero for the masked path on out-of-episode g.
        j = torch.arange(C, device=actions_padded.device)  # (C,)
        g = f[:, None] + j[None, :]  # (H, C)
        g_valid = g < L  # (H, C)
        g_clamped = torch.clamp(g, max=last)  # (H, C)
        aw = actions_padded[b][g_clamped]  # (H, C, D) gathered (repeat-clamped)
        if not repeat:
            # zero out target frames past episode end (zero_masked semantics)
            aw = aw * g_valid[..., None].to(aw.dtype)
        elif pusher_pad:
            # hold-position pad: target frames past episode end get the
            # episode's LAST PUSHER POSITION in ACTION-normalized space
            # (instead of the clamp-gathered last cursor action).
            aw = aw.clone()
            aw[~g_valid] = pad_actions[b].to(aw.dtype)
        act_chunks.append(aw)

        # --- per-obs-step mask: (H,) ---
        mw = actions_padded.new_zeros((H,))
        if repeat:
            mw[:] = 1.0  # every step counted (repeat-padded tails included)
        else:
            mw[in_ep] = 1.0  # count step iff its obs frame is in-episode
        mask_chunks.append(mw)

    if not act_chunks:  # degenerate empty batch
        obs_w = {k: v[:, :H] for k, v in obs_padded.items()}
        zero_act = actions_padded.new_zeros(
            (actions_padded.shape[0], H, C) + tuple(A_tail)
        )
        return obs_w, zero_act, seq_lens.new_zeros((actions_padded.shape[0], H))

    obs_w = {k: torch.stack(obs_chunks[k], dim=0) for k in obs_chunks}
    act_w = torch.stack(act_chunks, dim=0)  # (Nw, H, C, D)
    mask_w = torch.stack(mask_chunks, dim=0)  # (Nw, H)
    return obs_w, act_w, mask_w


class WindowedBCPolicy(nn.Module):
    """ObsEncoder -> LSTM (obs history) -> GMM head. One action per step.

    forward(obs) is the per-window teacher-forced unroll (zero hidden init);
    init_step_state/step run the closed-loop single-step rollout, re-zeroing the
    LSTM hidden every ``rnn_horizon`` steps (robomimic get_action semantics).
    """

    def __init__(self, obs_encoder, core_net=None, gmm_head=None, action_dim=None,
                 action_horizon=None, rnn_horizon=10, actor_mlp_dims=(1024, 1024),
                 core="lstm", obs_stride=1, chunk_len=1, chunk_head="linear",
                 query_decoder=None, lstm=None):
        super().__init__()
        # CONFIG-FACING name is ``core_net``; ``lstm`` is the DEPRECATED ALIAS
        # (kept so old configs / EgoVerse-pact-2 ported yamls keep working). Pass
        # exactly one of the two.
        if core_net is not None and lstm is not None:
            raise ValueError(
                "pass core_net OR lstm (deprecated alias), not both"
            )
        core_net = core_net if core_net is not None else lstm
        if core_net is None:
            raise ValueError("pass core_net (or the deprecated lstm alias)")
        # ``core_net`` is the recurrent/attention/H-Net CORE (LSTMCore by default,
        # TransformerCore when core=="transformer", HNetCore when core=="hnet").
        # attribute kept named .lstm for state_dict stability across cores;
        # config-facing name is core_net (lstm = deprecated alias)
        # -> default core=="lstm" is byte-for-byte the old policy. All cores
        # honor the same interface: ``.input_dim``, ``.hidden_dim``,
        # ``forward(emb)->(out,hidden)``, ``init_hidden(...)``, ``step(emb_t,h)``.
        if core not in ("lstm", "transformer", "hnet"):
            raise ValueError(f"core must be lstm|transformer|hnet, got {core!r}")
        self.core = str(core)
        lstm = core_net
        if lstm.input_dim != obs_encoder.embed_dim:
            raise ValueError("core.input_dim != obs_encoder.embed_dim")
        if gmm_head.action_dim != action_dim:
            raise ValueError("gmm_head.action_dim != action_dim")
        self.obs_encoder = obs_encoder
        self.lstm = lstm
        # robomimic per-step actor MLP (mlp_layer_dims=actor_layer_dims=[1024,
        # 1024]): RNN_MIMO_MLP inserts MLP(rnn_output_dim -> [1024,1024]) between
        # the LSTM output and the GMM decoder. We replicate it here. Empty list
        # => no MLP (LSTM output feeds the head directly).
        dims = list(actor_mlp_dims or [])
        if dims:
            layers = []
            prev = lstm.hidden_dim
            for w in dims:
                layers += [nn.Linear(prev, w), nn.ReLU()]
                prev = int(w)
            self.actor_mlp = nn.Sequential(*layers)
            head_in = dims[-1]
        else:
            self.actor_mlp = nn.Identity()
            head_in = lstm.hidden_dim
        if gmm_head.proj.in_features != head_in:
            raise ValueError(
                f"gmm_head d_model ({gmm_head.proj.in_features}) must equal the "
                f"actor MLP output dim ({head_in}); set gmm_head.d_model to "
                f"{head_in} (= actor_mlp_dims[-1] or lstm.hidden_dim)."
            )
        self.gmm_head = gmm_head
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.rnn_horizon = int(rnn_horizon)
        # obs striding + action chunking. Defaults 1/1 == today's behavior.
        #   obs_stride: the policy observes/encodes only every obs_stride env
        #     steps; the LSTM/transformer history is over the SUBSAMPLED obs.
        #   chunk_len: each obs-step emits chunk_len actions (gmm_head.chunk_len).
        self.obs_stride = int(obs_stride)
        self.chunk_len = int(chunk_len)
        if self.obs_stride < 1:
            raise ValueError(f"obs_stride must be >= 1, got {obs_stride}")
        if self.chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
        if int(getattr(gmm_head, "chunk_len", 1)) != self.chunk_len:
            raise ValueError(
                f"policy.chunk_len ({self.chunk_len}) must equal "
                f"gmm_head.chunk_len ({getattr(gmm_head, 'chunk_len', 1)}); set "
                "both in the config (model.robomimic_model.chunk_len and "
                "model.robomimic_model.gmm_head.chunk_len)."
            )
        # CHUNK READOUT SELECTION (single knob; default byte-identical):
        #   "linear"  -> the chunk_len GMMs are read out by the gmm_head's wide
        #     Linear(d_model -> chunk_len*per_step) of EACH obs-step feature h_k
        #     (the pre-existing path; with chunk_head="linear" NOTHING about the
        #     module graph or the forward changes, so the live chunk8 build is
        #     byte-for-byte identical -- proven by torch.equal).
        #   "queries" -> an ACT/HPT-style QueryActionDecoder refines chunk_len
        #     learnable query embeddings (self-attention among queries + causal
        #     cross-attention over h_0..h_k) and a SHARED Linear(d_model ->
        #     per_step) maps each refined query to its GMM params. The flat
        #     output layout (.., chunk_len*per_step) is IDENTICAL to the linear
        #     head's, so gmm_head.nll / decode / _make_dist are reused unchanged.
        if chunk_head not in ("linear", "queries"):
            raise ValueError(
                f"chunk_head must be linear|queries, got {chunk_head!r}"
            )
        self.chunk_head = str(chunk_head)
        self.query_decoder = None
        if self.chunk_head == "queries":
            if query_decoder is None:
                raise ValueError(
                    "chunk_head='queries' requires a query_decoder module "
                    "(instantiated by WindowedBC under the model 'query_decoder:' "
                    "slot); none was passed."
                )
            # the decoder consumes the core's per-step features (d_model) and
            # emits the SAME flat per-obs-step GMM-param layout the gmm_head
            # would (chunk_len * per_step). Guard the dims so a yaml mismatch
            # fails at construction, not at the first forward.
            if int(query_decoder.chunk_len) != self.chunk_len:
                raise ValueError(
                    f"query_decoder.chunk_len ({query_decoder.chunk_len}) must "
                    f"equal policy.chunk_len ({self.chunk_len})."
                )
            if int(query_decoder.per_step) != int(gmm_head.per_step):
                raise ValueError(
                    f"query_decoder.per_step ({query_decoder.per_step}) must "
                    f"equal gmm_head.per_step ({gmm_head.per_step}) = "
                    "num_modes*(2*action_dim+1)."
                )
            if int(query_decoder.d_model) != int(head_in):
                raise ValueError(
                    f"query_decoder.d_model ({query_decoder.d_model}) must equal "
                    f"the core feature width feeding the readout ({head_in})."
                )
            self.query_decoder = query_decoder

    def _readout(self, feats):
        """Map per-obs-step core features ``(.., d_model)`` -> flat GMM params.

        ``chunk_head=="linear"`` (default): the pre-existing path -- the wide
        gmm_head projection of each obs-step feature. BYTE-IDENTICAL.

        ``chunk_head=="queries"``: route through the QueryActionDecoder, which
        returns the SAME flat ``(.., chunk_len*per_step)`` layout the gmm_head
        would, so all downstream nll/decode code is unchanged.
        """
        if self.chunk_head == "queries":
            return self.query_decoder(feats)
        return self.gmm_head(feats)

    def forward(self, obs):
        """Teacher-forced unroll over (B, T, ...) obs -> GMM params per step.

        Hidden is left as ``None`` (zero init) — for the windowed training path
        each (B,) entry is a length-`rnn_horizon` window, so this matches
        robomimic's per-window zero-init unroll.
        """
        emb = self.obs_encoder(obs)
        out, _ = self.lstm(emb)
        return self._readout(self.actor_mlp(out))

    def init_step_state(self, batch_size, T_max, device, dtype=None):
        dtype = dtype or next(self.parameters()).dtype
        # "counter" counts OBS-STEPS (incremented only on an obs step); it drives
        # the rnn_horizon buffer re-init exactly as before. "queue" holds the
        # actions emitted by the last obs-step that have not yet been consumed by
        # the env (chunk_len rollout). Both 1/1 defaults reduce to the prior
        # single-step semantics: every env step is an obs step, queue len 1.
        return {"hidden": None, "dtype": dtype, "T_max": int(T_max),
                "counter": 0, "queue": [], "last_action": None}

    @torch.no_grad()
    def step(self, state, obs_norm, t):
        """Return ONE action for env step ``t`` (called every env step).

        obs_stride == 1, chunk_len == 1 (default): every env step is an obs step
        and the head emits one action -> BYTE-IDENTICAL to the prior code path
        (the queue trivially holds a single action consumed the same step).

        obs_stride > 1 / chunk_len > 1: the policy observes/encodes ONLY on env
        steps where ``t % obs_stride == 0`` (obs steps). On an obs step it
        re-inits the buffer every ``rnn_horizon`` OBS-steps (== every
        ``obs_stride*rnn_horizon`` env frames), core-steps the SUBSAMPLED obs,
        the head emits ``chunk_len`` actions, and those REPLACE the queue. On a
        non-obs env step it pops the next queued action WITHOUT any model call.
        Episode start (t==0) is obs step 0 (counter==0 -> buffer re-init).
        """
        is_obs_step = (t % self.obs_stride == 0)
        if is_obs_step:
            emb = self.obs_encoder(obs_norm)
            emb_t = emb[:, 0]
            B = emb_t.shape[0]
            # robomimic BC_RNN.get_action: re-init hidden every rnn_horizon
            # OBS-steps -> if hidden is None or counter % rnn_horizon == 0: reset.
            counter = state.get("counter", 0)
            if state.get("hidden", None) is None or (counter % self.rnn_horizon == 0):
                state["hidden"] = self.lstm.init_hidden(
                    B, device=emb_t.device, dtype=emb_t.dtype
                )
            state["counter"] = counter + 1
            if self.chunk_head == "queries":
                # the query decoder needs the FULL causal context h_0..h_k for
                # the current buffer (not just the last feature h_t). Grow the
                # core's rolling obs buffer by ONE frame manually (the same cat
                # the core's ``step`` would do internally) and encode it ONCE via
                # the core's forward to get the per-step features h_0..h_k
                # (B, S, hidden) -- the SAME features training row k cross-attends
                # over. This avoids the redundant double-encode of calling
                # ``self.lstm.step`` (which would _encode the buffer just to
                # return h_t = out[:, -1], which the queries path discards) and
                # THEN re-encoding the same buffer for the full context. One
                # encode now serves both: ctx_feats[:, -1] IS h_t, and the full
                # ctx_feats is the causal context. (No core change; the buffer
                # layout {"obs": (B, S, input_dim)} matches TransformerCore.step.)
                prev = state["hidden"].get("obs", None)
                emb_step = emb_t.unsqueeze(1)  # (B, 1, input_dim)
                buf = emb_step if prev is None else torch.cat([prev, emb_step], dim=1)
                state["hidden"] = {"obs": buf}  # grown buffer (B, S, input_dim)
                ctx_feats, _ = self.lstm(buf)  # (B, S, hidden) -- single encode
                ctx_feats = self.actor_mlp(ctx_feats)
                raw = self.query_decoder.forward_step(ctx_feats)  # (B, C*P)
            else:
                h_t, state["hidden"] = self.lstm.step(emb_t, state["hidden"])
                raw = self.gmm_head(self.actor_mlp(h_t))
            chunk = self.gmm_head.decode(raw)  # (B,D) if C==1 else (B,C,D)
            if self.chunk_len > 1:
                # (B, C, D) -> list of C tensors (B, D), one per chunk position.
                state["queue"] = [chunk[:, j] for j in range(self.chunk_len)]
            else:
                state["queue"] = [chunk]  # (B, D)
        # pop the next action for THIS env step.
        if state["queue"]:
            a = state["queue"].pop(0)
        else:
            # queue drained between obs steps (only possible if chunk_len <
            # obs_stride, never for the chunk8 run where chunk_len==obs_stride);
            # safe fallback: repeat the last emitted action so the env always
            # gets a valid frame action.
            a = state["last_action"]
        state["last_action"] = a
        return a.unsqueeze(1)  # (B, 1, D)


class WindowedBC(PackedAlgoBase):
    def __init__(
        self,
        action_dim,
        action_horizon,
        obs_encoder,
        gmm_head,
        norm_stats,
        core_net=None,
        rnn_horizon=10,
        max_windows_per_batch=256,
        actor_mlp_dims=(1024, 1024),
        pad_mode="zero_masked",
        core="lstm",
        obs_stride=1,
        chunk_len=1,
        chunk_head="linear",
        query_decoder=None,
        lstm=None,
        domains=None,
        ac_keys=None,
        device=None,
        window_anchor="uniform",
        pad_pusher_obs_key="state_agent_obj",
        pad_pusher_slice=(0, 2),
        **kwargs,
    ):
        Algo.__init__(self)
        # WindowedBC calls Algo.__init__ (not PackedAlgoBase.__init__), so
        # PackedAlgoBase-base attributes that the INHERITED
        # ``PackedAlgoBase.process_batch_for_training`` reads are never set.
        # WindowedBC has no outer-stage and
        # no train-only obs augmentation, so make ``train_obs_transforms`` an
        # empty list: the ``if self.train_obs_transforms and ...`` guard then
        # short-circuits on the falsy first operand and never touches the
        # (nonexistent) ``self.outer_stage``. Without this, both the train and
        # the closed-loop sim-eval validation paths raise AttributeError before
        # any rollout — which is exactly what blocked BC-RNN sim eval (DESIGN
        # step 10 headline). Pre-existing latent bug from the H-Net restructure.
        self.train_obs_transforms: list = []
        # CONFIG-FACING name is ``core_net``; ``lstm`` is the DEPRECATED ALIAS
        # (kept so old configs / EgoVerse-pact-2 ported yamls keep working). Pass
        # exactly one of the two. Downstream code below keeps the local name
        # ``lstm`` (and the attribute stays ``self.lstm``) for state_dict
        # stability across cores.
        if core_net is not None and lstm is not None:
            raise ValueError(
                "pass core_net OR lstm (deprecated alias), not both"
            )
        lstm = core_net if core_net is not None else lstm
        if lstm is None:
            raise ValueError("pass core_net (or the deprecated lstm alias)")
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.rnn_horizon = int(rnn_horizon)
        # obs striding + action chunking knobs (defaults 1/1 == byte-identical).
        #   obs_stride: build TRAINING windows over the SUBSAMPLED obs (frame
        #     s+obs_stride*k for obs-step k) and observe only every obs_stride
        #     env steps at rollout.
        #   chunk_len: each obs-step's TARGET is the chunk_len GT actions at
        #     [s+obs_stride*k, s+obs_stride*k+chunk_len); the head emits chunk_len
        #     actions per obs-step and the rollout queues them.
        self.obs_stride = int(obs_stride)
        self.chunk_len = int(chunk_len)
        if self.obs_stride < 1:
            raise ValueError(f"obs_stride must be >= 1, got {obs_stride}")
        if self.chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
        if int(getattr(gmm_head, "chunk_len", 1)) != self.chunk_len:
            raise ValueError(
                f"WindowedBC.chunk_len ({self.chunk_len}) must equal "
                f"gmm_head.chunk_len ({getattr(gmm_head, 'chunk_len', 1)}); set "
                "both in the config (chunk_len and gmm_head.chunk_len)."
            )
        # pad_mode: "zero_masked" (default, pre-existing), "repeat_unmasked"
        # (robomimic pad_same=True + unmasked plain-mean NLL), or
        # "repeat_pusher_unmasked" (like repeat_unmasked but the padded ACTION
        # target is the episode's last PUSHER POSITION re-expressed with the
        # ACTION norm stats -- the "hold position" command -- instead of the
        # last recorded cursor action). See _cut_windows.
        if pad_mode not in (
            "zero_masked", "repeat_unmasked", "repeat_pusher_unmasked"
        ):
            raise ValueError(
                "pad_mode must be zero_masked|repeat_unmasked|"
                f"repeat_pusher_unmasked, got {pad_mode!r}"
            )
        self.pad_mode = str(pad_mode)
        # window_anchor: "uniform" (default, pre-existing) or "start"
        # (full-history: ONE window per episode anchored at frame 0). Ported
        # from the EgoVerse2 c3000v2 implementation. See _cut_windows.
        if window_anchor not in ("uniform", "start"):
            raise ValueError(
                f"window_anchor must be uniform|start, got {window_anchor!r}"
            )
        self.window_anchor = str(window_anchor)
        # repeat_pusher_unmasked plumbing: which NORMALIZED obs key holds the
        # pusher/agent position and which slice of it is the xy that maps onto
        # the 2-dim action space. Defaults match the pushshapes keymap
        # (state_agent_obj[:, 0:2] = agent xy; actions = cursor xy).
        self.pad_pusher_obs_key = str(pad_pusher_obs_key)
        self.pad_pusher_slice = tuple(int(v) for v in pad_pusher_slice)
        if len(self.pad_pusher_slice) != 2:
            raise ValueError(
                f"pad_pusher_slice must be (start, end), got {pad_pusher_slice!r}"
            )
        # core: "lstm" (default, byte-identical to the pre-existing build),
        # "transformer" (causal self-attention over the rnn_horizon window), or
        # "hnet" (the real dynamic-chunking H-Net over the rnn_horizon window).
        # The actual core OBJECT is instantiated by Hydra under the ``core_net:``
        # config slot (``lstm:`` is the deprecated alias) -- LSTMCore /
        # TransformerCore / HNetCore; this string is a guard that the config's
        # _target_ matches the declared intent, and is threaded to the policy.
        # All cores share the LSTMCore interface so the rest of the algo
        # (forward_training/forward_eval/inference_step) is core-agnostic.
        if core not in ("lstm", "transformer", "hnet"):
            raise ValueError(f"core must be lstm|transformer|hnet, got {core!r}")
        self.core = str(core)
        _is_tx = isinstance(lstm, TransformerCore)
        _is_lstm = isinstance(lstm, LSTMCore)
        _is_hnet = isinstance(lstm, HNetCore)
        if self.core == "transformer" and not _is_tx:
            raise ValueError(
                "core='transformer' but the instantiated core under the model "
                f"'core_net:' slot is {type(lstm).__name__}, not TransformerCore. "
                "Point core_net._target_ at TransformerCore (see the *_tx config)."
            )
        if self.core == "lstm" and not _is_lstm:
            raise ValueError(
                "core='lstm' but the instantiated core under the model 'core_net:' "
                f"slot is {type(lstm).__name__}, not LSTMCore."
            )
        if self.core == "hnet" and not _is_hnet:
            raise ValueError(
                "core='hnet' but the instantiated core under the model 'core_net:' "
                f"slot is {type(lstm).__name__}, not HNetCore. Point "
                "core_net._target_ at HNetCore (see the *_hnet config)."
            )
        # Config-resilience guard: the transformer's positional table (and the
        # H-Net core's window guard) hold ``max_window`` slots and every training
        # window is length ``rnn_horizon`` (see _cut_windows). If a yaml-only
        # edit bumps rnn_horizon but forgets core_net.max_window, the model would
        # BUILD fine and only crash at the first training forward (window length
        # rnn_horizon > max_window). Fail at CONSTRUCTION instead — before
        # norm-stats compute + dataloader spin-up + a wasted SLURM launch. (LSTM
        # is recurrent: no such constraint.)
        if (_is_tx or _is_hnet) and lstm.max_window < self.rnn_horizon:
            raise ValueError(
                f"{type(lstm).__name__}.max_window ({lstm.max_window}) must be "
                f">= rnn_horizon ({self.rnn_horizon}); set core_net.max_window = "
                f"rnn_horizon in the config (they default to 10/10)."
            )
        # chunk_head: "linear" (default, byte-identical) | "queries" (ACT/HPT
        # action-query readout). The actual decoder OBJECT is instantiated by
        # Hydra under the model ``query_decoder:`` slot (a QueryActionDecoder);
        # this string guards that the config + algo agree and is threaded to the
        # policy. With "linear" the query_decoder slot is ignored (and may be
        # absent) so the live chunk8 build is byte-identical.
        if chunk_head not in ("linear", "queries"):
            raise ValueError(
                f"chunk_head must be linear|queries, got {chunk_head!r}"
            )
        self.chunk_head = str(chunk_head)
        if self.chunk_head == "queries":
            # NOTE: chunk_len == 1 with chunk_head='queries' is INTENTIONALLY
            # permitted (a single-query ACT-style readout is a legitimate
            # degenerate config: one learnable query cross-attending the causal
            # context, one shared GMM projection). The absence of a chunk_len>1
            # guard here is deliberate, not an oversight.
            if query_decoder is None:
                raise ValueError(
                    "chunk_head='queries' requires the model 'query_decoder:' "
                    "slot to instantiate a QueryActionDecoder; none was passed."
                )
            if not isinstance(query_decoder, QueryActionDecoder):
                raise ValueError(
                    "chunk_head='queries' but the object under the model "
                    f"'query_decoder:' slot is {type(query_decoder).__name__}, "
                    "not QueryActionDecoder. Point query_decoder._target_ at "
                    "egomimic.models.heads.query_decoder.QueryActionDecoder."
                )
            # the query decoder cross-attends over up to rnn_horizon context
            # features; mirror the core's max_window guard so a yaml-only
            # rnn_horizon bump fails at construction, not the first forward.
            if int(query_decoder.max_window) < self.rnn_horizon:
                raise ValueError(
                    f"query_decoder.max_window ({query_decoder.max_window}) must "
                    f">= rnn_horizon ({self.rnn_horizon}); set "
                    "query_decoder.max_window = rnn_horizon in the config."
                )
            # queries only make sense for the transformer core (it produces the
            # per-step causal features h_0..h_k the decoder cross-attends over,
            # and the rollout re-encodes its obs buffer to recover them).
            if not _is_tx:
                raise ValueError(
                    "chunk_head='queries' is only supported with "
                    "core='transformer' (the query decoder cross-attends over "
                    f"the transformer's per-step features); core is {self.core!r}."
                )
        self.max_windows_per_batch = (
            None if max_windows_per_batch is None else int(max_windows_per_batch)
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_parameter_groups = False
        self.lr_multipliers = None
        self.weight_decay = 0.0
        self._hnet_core = None
        policy = WindowedBCPolicy(
            obs_encoder=obs_encoder,
            core_net=lstm,
            gmm_head=gmm_head,
            action_dim=action_dim,
            action_horizon=action_horizon,
            rnn_horizon=rnn_horizon,
            actor_mlp_dims=actor_mlp_dims,
            core=core,
            obs_stride=obs_stride,
            chunk_len=chunk_len,
            chunk_head=chunk_head,
            query_decoder=query_decoder,
        )
        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via norm_stats. Shared base helper
        # (collapse c5): see Algo._resolve_embodiment_keys.
        self._resolve_embodiment_keys(norm_stats)

    def _unpack_obs_actions(self, _batch, emb_id):
        ac_key = self.resolved_ac_keys[emb_id]
        obs = self._build_obs(_batch, emb_id)
        if not _batch.get("_packed", False):
            actions = _batch[ac_key]
            B, T = actions.shape[0], actions.shape[1]
            mask = actions.new_ones((B, T))
            seq_lens = actions.new_full((B,), T, dtype=torch.long)
            return obs, actions, mask, seq_lens
        cu = _batch["cu_seqlens"]
        seq_lens = _batch["seq_lens"].clone().to(torch.long)
        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        obs_padded = {k: _pack_to_padded(v, cu, B, T_max) for k, v in obs.items()}
        actions_padded = _pack_to_padded(_batch[ac_key], cu, B, T_max)
        ar = torch.arange(T_max, device=seq_lens.device)
        mask = (ar[None, :] < seq_lens[:, None]).to(actions_padded.dtype)
        return obs_padded, actions_padded, mask, seq_lens

    def _stats_for_key(self, key, emb_id):
        """Norm stats dict for ``key`` (accepts keyname OR zarr key)."""
        stats_map = self.norm_stats.norm_stats.get(emb_id, {})
        if key in stats_map:
            return stats_map[key]
        zk_to_kn = {
            v: k for k, v in self.norm_stats.zarr_keys.get(emb_id, {}).items()
        }
        kn = zk_to_kn.get(key)
        if kn is not None and kn in stats_map:
            return stats_map[kn]
        raise KeyError(
            f"no norm stats for key {key!r} (emb {emb_id}); available: "
            f"{sorted(stats_map)}"
        )

    def _pusher_pad_actions(self, obs_padded, seq_lens, emb_id):
        """Per-episode hold-position action pad: (B, action_dim).

        The batch reaching forward_training is ALREADY normalized
        (PackedAlgoBase.process_batch_for_training normalizes obs with the
        PROPRIO stats and actions with the ACTION stats BEFORE any window
        cutting). The correct normalization-consistent pad is therefore:
        take the LAST REAL frame's pusher obs (proprio-normalized), unnormalize
        it with the PROPRIO stats back to raw pixels, slice the xy, and
        re-normalize with the ACTION stats. Under minmax both live in [-1,1]
        but with different min/max ranges (cursor vs pusher travel), so the two
        normalized spaces are NOT interchangeable.
        """
        key = self.pad_pusher_obs_key
        if key not in obs_padded:
            raise KeyError(
                f"pad_mode='repeat_pusher_unmasked': obs key {key!r} not in "
                f"batch obs {sorted(obs_padded)}; set pad_pusher_obs_key."
            )
        B = int(seq_lens.shape[0])
        idx = (seq_lens.to(torch.long) - 1).clamp(min=0)  # (B,) last real frame
        last_obs = obs_padded[key][torch.arange(B, device=idx.device), idx]
        p_stats = self._stats_for_key(key, emb_id)
        a_stats = self._stats_for_key(self.resolved_ac_keys[emb_id], emb_id)
        raw = self.norm_stats._apply_unnorm_one(last_obs, p_stats)  # (B, P) px
        lo, hi = self.pad_pusher_slice
        raw_xy = raw[:, lo:hi]  # (B, action_dim) raw pusher position
        pad = self.norm_stats._apply_norm_one(raw_xy, a_stats)  # action-norm
        return pad.to(obs_padded[key].dtype)

    def forward_training(self, batch):
        """Length-`rnn_horizon` windowed teacher-forced NLL (robomimic recipe).

        For each embodiment: unpack packed episodes -> padded, cut all length-H
        windows (fresh zero hidden per window), unroll the LSTM per window, GMM
        NLL vs the demo action at every valid (unpadded) step.
        """
        predictions = OrderedDict()
        policy = self.nets["policy"]
        H = self.rnn_horizon
        for emb_id, _batch in batch.items():
            obs_padded, actions_padded, mask, seq_lens = self._unpack_obs_actions(
                _batch, emb_id
            )
            pad_actions = None
            if self.pad_mode == "repeat_pusher_unmasked":
                pad_actions = self._pusher_pad_actions(
                    obs_padded, seq_lens, emb_id
                )
            obs_w, act_w, mask_w = _cut_windows(
                obs_padded, actions_padded, mask, H,
                max_windows=self.max_windows_per_batch,
                pad_mode=self.pad_mode,
                obs_stride=self.obs_stride,
                chunk_len=self.chunk_len,
                window_anchor=self.window_anchor,
                pad_actions=pad_actions,
            )
            raw = policy(obs_w)  # (Nw, H, M*(2D+1)); zero hidden per window
            aloss = policy.gmm_head.nll(raw, act_w, mask=mask_w)
            predictions[f"{emb_id}_pred"] = raw
            predictions[f"{emb_id}_action_loss"] = aloss
            predictions[f"{emb_id}_ratio_loss"] = torch.tensor(
                0.0, device=aloss.device
            )
        return predictions

    @torch.no_grad()
    def forward_eval(self, batch):
        """Teacher-forced val overlay: dense per-step prediction over the whole
        episode (GT-vs-pred across all frames). Under eval mode the GMM head uses
        low_noise_eval (1e-4 std) so the sampled action ~= the chosen mode's mean.

        Core-dependent unroll (handled inside the core's ``forward``):
          * LSTM: a true full-episode recurrent unroll (any T).
          * TransformerCore: the positional table only holds ``max_window``
            slots, so the full episode is encoded in NON-OVERLAPPING
            length-``max_window`` windows (each fresh, start_pos=0) and concat'd.
            This makes the overlay match ROLLOUT semantics (the rollout re-inits
            the buffer every ``rnn_horizon`` steps), not a full-length attention
            the model never sees in training. Consequence: the LSTM overlay
            (unbounded history) and the TF overlay (max_window-windowed history)
            are NOT directly comparable step-for-step — compare each to its own
            rollout instead. NOTE: neither path affects training, which is
            strictly windowed for both cores.
        """
        unnorm = {}
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            obs_padded, _, _, seq_lens = self._unpack_obs_actions(_batch, emb_id)
            raw = policy(obs_padded)
            pred = policy.gmm_head.decode(raw)  # (B,T,D) or (B,T,chunk_len,D)
            if policy.chunk_len > 1:
                # chunked head: the overlay is a per-frame GT-vs-pred diagnostic,
                # so take chunk position 0 (the action AT the obs frame). The
                # full chunk is only used at rollout (queued). seq_lens unchanged.
                pred = pred[..., 0, :]
            preds = OrderedDict()
            preds[ac_key] = pred
            unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
            for key, val in unnorm_actions.items():
                unnorm[f"emb{emb_id}_{key}"] = val
            unnorm[f"emb{emb_id}_seq_lens"] = seq_lens
        return unnorm

    @torch.no_grad()
    def inference_step(self, obs_zarr, t, emb_id, T_max=None):
        """One closed-loop env-tick. Returns the absolute (action_dim,) action.

        ``T_max`` is accepted to match the eval contract
        (``PackedSimEval`` calls ``inference_step(obs_zarr, t, emb_id,
        T_max=self.max_steps)`` — eval_sim.py:251). It is the *sim rollout
        horizon* (max env steps), which is a different quantity from the
        policy's internal action-queue length: the queue is sized by the
        policy's ``action_horizon`` (how many actions one obs-step emits), not
        by how long the episode runs. So we intentionally keep sizing
        ``init_step_state`` from ``policy.action_horizon`` and only accept the
        evaluator's ``T_max`` so the call signature matches (DESIGN step 10 /
        PORT_NOTES item 2: "accept/ignore T_max"). DFoT's ``inference_step``
        already takes ``T_max`` the same way; this closes the BC-RNN-only gap.
        """
        policy = self.nets["policy"]
        if t == 0:
            device = next(policy.parameters()).device
            queue_len = int(getattr(policy, "action_horizon", 1024))
            self._sim_state = policy.init_step_state(
                batch_size=1, T_max=queue_len, device=device
            )
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = (
            self.ac_keys[embodiment_name]
            if embodiment_name in self.ac_keys
            else self.ac_keys[emb_id]
        )
        obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
        action_norm = policy.step(self._sim_state, obs_norm, t)
        action_unnorm = self.norm_stats.unnormalize(
            {ac_key: action_norm.squeeze(0).squeeze(0)}, emb_id,
        )[ac_key]
        return action_unnorm.detach().cpu().numpy().reshape(-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Backward-compat aliases (DESIGN.md step 5, amended). The classes were renamed
# BCRNN -> WindowedBC and BCRNNPolicy -> WindowedBCPolicy. These module-level
# aliases keep the old names importable so existing configs (`_target_:
# egomimic.algo.bc.BCRNN`), saved
# checkpoints, and downstream code resolve unchanged. They are the SAME class
# objects (``is`` identity), so isinstance / pickling / Hydra instantiate behave
# identically regardless of which name is used.
BCRNN = WindowedBC
BCRNNPolicy = WindowedBCPolicy

__all__ = [
    "WindowedBC",
    "WindowedBCPolicy",
    "BCRNN",
    "BCRNNPolicy",
]
