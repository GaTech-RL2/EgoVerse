"""Windowed K-curriculum AR training (fused H-Net) — VECTORIZED over windows.

A (obs_source=demo) and B (obs_source=sim) are the SAME training; the only
difference is where each step's obs comes from. Both are now **parallel over
windows**: instead of looping windows one-at-a-time, all W windows of a batch
roll out TOGETHER — at each of the K steps we do ONE batched (W,...) model
forward (and, for B, step all W envs). Only the K rollout steps stay sequential
(the AR chain).

Per episode: slice FULL-K windows at stride = K//2 (episodes shorter than K are
skipped). Per window, the per-step loss is MSE vs the demo's GT action; the loss
is MEANed over the K steps (per-window mean), then over windows. Action feedback
is detached -> per-step gradient (each step trains its own prediction; matches
the non-differentiable sim in B). K grows 1->2->4->8->16->32 via the callback.

SPEED: A's win is the batched GPU forward (W windows ~ free on GPU). B additionally
needs the W envs stepped across CPU cores -> TODO-VEC-MP (multiprocessing pool);
v1 steps them in a loop (correct; the MP pool is the remaining ~Nx speedup).
B(sim) only: the train dataset must use get_keymap_eval (so goal_pose is present).
"""
import os
from collections import OrderedDict

import numpy as np
import torch

from egomimic.algo.hnet import HNetFused
from egomimic.eval.eval_sim import _state_to_init, _format_pushshapes_obs
# DAgger expert: the SAME scripted policy used by data collection + eval_sim's
# _dagger_record. Returns a (2,) world-coord absolute-cursor target (clipped to
# [0, world_size]) -- exactly the frame ac_key unnormalizes to.
from Tsimulation.collect.scripted_collect import scripted_action as _scripted_expert


def _dagger_sanity(algo, chunk_idx, chunk_pred, demo_chunk, expert_chunk, emb_id, ac_key):
    """Print DAgger relabel sanity checks (guarded by DAGGER_SANITY env var so it
    only fires in the smoke). Reports, in UNNORMALIZED world px:
      - first chunk (chunk_idx==0, ~on-manifold demo anchor): ||expert - demo||
        should be SMALL (expert ~ demo near the demo).
      - later chunks (drift): ||expert - demo|| should be clearly LARGER.
    expert_chunk/demo_chunk are normalized (W, cs, A); unnormalize to compare."""
    import os
    if os.environ.get("DAGGER_SANITY", "0") not in ("1", "true", "True"):
        return
    with torch.no_grad():
        e_w = algo.norm_stats.unnormalize({ac_key: expert_chunk}, emb_id)[ac_key]
        d_w = algo.norm_stats.unnormalize({ac_key: demo_chunk}, emb_id)[ac_key]
        diff = (e_w - d_w).norm(dim=-1)                 # (W, cs) per-step world-px L2
        mean_px = float(diff.mean())
        # per-j profile: j=0 of chunk 0 is the TRUE on-manifold point (env was just
        # set_state'd to the demo anchor; the policy hasn't drifted yet). Later j
        # within the chunk drift as the model action diverges from the demo. j=0
        # gap small + growing-with-j == frame correct + relabel doing its job.
        per_j = diff.mean(dim=0)                        # (cs,) mean over windows
        j0 = float(per_j[0]); jmid = float(per_j[per_j.numel() // 2]); jlast = float(per_j[-1])
        # also the model-pred vs expert (the actual training residual), world px
        p_w = algo.norm_stats.unnormalize({ac_key: chunk_pred.detach()}, emb_id)[ac_key]
        pe_px = float((p_w - e_w).norm(dim=-1).mean())
        tag = "FIRST(on-manifold)" if chunk_idx == 0 else f"chunk{chunk_idx}(drift)"
        print(
            f"[DAGGER_SANITY] {tag} emb={emb_id} "
            f"mean||expert-demo||={mean_px:.2f}px (j0={j0:.2f} jmid={jmid:.2f} jlast={jlast:.2f}) "
            f"mean||pred-expert||={pe_px:.2f}px",
            flush=True,
        )


class HNetFusedClosedLoop(HNetFused):
    """Fused H-Net trained closed-loop with a growing-K curriculum, vectorized over windows."""

    def __init__(self, *args, k_schedule=None, image_size: int = 96,
                 obs_source: str = "demo", image_key: str = "front_img_1",
                 max_windows: int = 0, max_window_steps: int = 0,
                 n_env_workers: int = 16,
                 chunk_rollout: bool = False, chunk_size: int = 32,
                 dagger: bool = False, n_obs_history: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        # n_obs_history (N): condition each chunk on the LAST N observation
        # frames (sliding window, oldest->newest, frame N-1 == current frame),
        # encoded as N cond tokens so the backbone attends over them, with the
        # K-action chunk read at a trailing BOS. OBS-ONLY -- NO past actions are
        # fed (avoids the documented action-copying collapse; BC-RNN-style). N=1
        # (default) is the current single-frame / memoryless behavior and is
        # BYTE-IDENTICAL to before this knob existed. Episode-start frames are
        # padded by repeating frame 0 (clamp the per-window frame indices at >=
        # the anchor's episode start).
        self.n_obs_history = max(1, int(n_obs_history))
        # chunk_rollout: chunk-B recipe. Each macro-step observes ONE frame and
        # predicts a chunk_size-action chunk in ONE shot via the chunk head
        # (chunk_k=chunk_size), executes it open-loop to advance the sim, then
        # re-observes for the next chunk. self.current_k is reinterpreted as the
        # NUMBER OF CHUNKS (not frames). Off by default -> the single-step A/B
        # path below is byte-identical.
        self.chunk_rollout = bool(chunk_rollout)
        self.chunk_size = int(chunk_size)
        # dagger: DAgger-style expert relabeling (chunk-B + sim only). When on, the
        # per-chunk supervision target is NOT the demo's GT chunk but the SCRIPTED
        # EXPERT's action queried at each (drifted) state the policy actually
        # visits during the chunk's execution. This trains recovery instead of
        # demo-mimicry on off-manifold states. Off by default -> the chunk-B path
        # is byte-identical (gt_chunk = actions[step_idx]).
        self.dagger = bool(dagger)
        self.k_schedule = list(
            k_schedule or [(0, 1), (50, 2), (100, 4), (150, 8), (250, 16), (400, 32)]
        )
        self.current_k = 1
        self._cl_image_size = int(image_size)
        self._image_key = str(image_key)
        assert obs_source in ("demo", "sim")
        self.obs_source = obs_source
        # max_windows: cap windows per batch (0 = no cap). Bounds memory/compute
        # when small K yields very many windows; a random subset is sampled.
        self.max_windows = int(max_windows)
        # max_window_steps: cap on W*K^2 (the dominant memory term — each of the K steps
        # re-encodes the length-(k+1) window). 0=off. At high K this auto-shrinks W so the
        # batched rollout stays within GPU memory while low-K phases keep all W windows
        # (a bigger effective batch -> a smoother loss).
        self.max_window_steps = int(max_window_steps)
        # n_env_workers: CPU processes for B's vectorized env pool (sim only).
        self.n_env_workers = int(n_env_workers)
        self._vec = None  # lazy VecPushShapes (sim only)

    def _get_vec(self, W):
        """Persistent multiprocessing env pool sized to hold >= W envs."""
        n_envs = max(self.max_windows or W, W)
        if self._vec is None or self._vec.n_envs < n_envs:
            if self._vec is not None:
                self._vec.close()
            from egomimic.algo.vec_pushshapes import VecPushShapes
            self._vec = VecPushShapes(
                n_envs=n_envs, n_workers=self.n_env_workers, image_size=self._cl_image_size
            )
        return self._vec

    def set_curriculum_epoch(self, epoch: int):
        k = 1
        for start, kk in self.k_schedule:
            if epoch >= start:
                k = kk
        self.current_k = int(k)

    # ---------- batched growing-window AR forward (over W windows) ----------
    def _predict_action_grad_batched(self, window_obs, prev_actions):
        """window_obs: list (len L=k+1) of per-frame obs dicts, each value (W,...).
        prev_actions: list (len k) of (W, A) DETACHED preds.
        Returns (W, A) prediction for a_k with grad on the current step only."""
        policy = self.nets["policy"]
        L = len(window_obs)
        W = window_obs[0][self._image_key].shape[0]
        Adim = self.action_dim
        obs_win = {key: torch.stack([o[key] for o in window_obs], dim=1)  # (W, L, ...)
                   for key in window_obs[0]}
        if prev_actions:
            acts = torch.cat(
                [torch.stack(prev_actions, dim=1),                 # (W, k, A)
                 torch.zeros(W, 1, Adim, device=self.device)], dim=1)  # (W, L, A)
        else:
            acts = torch.zeros(W, 1, Adim, device=self.device)
        raw, _ = policy(acts, obs_win)                              # (W, L, A[, n_bins])
        from egomimic.models.hnet_nets.action_heads import action_head_decode
        return action_head_decode(policy, raw[:, L - 1])            # (W, A)

    # ---------- chunk-B: grad chunk prediction from the obs window ----------
    def _predict_chunk_grad(self, obs_window):
        """obs_window: dict of normalized obs tensors for the chunk anchor.
        Each value is either single-frame ``(W, ...)`` (n_obs_history=1) or a
        history window ``(W, N, ...)`` (n_obs_history=N>1; oldest->newest, frame
        N-1 == current). Returns the chunk_size-action chunk ``(W, chunk_size,
        A[/params])`` WITH grad.

        Reuses the chunk head via ``policy.chunk_forward_history``: cond tokens
        ``[c_0..c_{N-1}, BOS]`` -> the head's one-shot K-action plan read at the
        trailing BOS (OBS-ONLY -- no past-action tokens). For N=1 this is the
        single-frame ``[c_0, BOS]`` forward, BYTE-IDENTICAL to the prior
        ``policy(dummy, obs_single)`` T=1 path."""
        # RAW (NOT decoded): continuous -> (W, K, A); GMM -> (W, K, M*(2D+1)).
        # The chunk loss (MSE or GMM NLL) consumes the raw params directly.
        return self.nets["policy"].chunk_forward_history(obs_window)  # (W, K, A[/params])

    def _chunk_loss_per_window(self, chunk_pred, gt_chunk):
        """Per-window chunk loss ``(W,)`` from RAW head output + GT chunk.

        ``chunk_pred``: RAW head output for the chunk -- ``(W, cs, A)`` for a
        continuous/discrete head, ``(W, cs, M*(2D+1))`` for GMM. ``gt_chunk``:
        the (normalized) supervision target ``(W, cs, A)`` (demo chunk or, under
        DAgger, the scripted-expert chunk).

        GMM: split the raw params, evaluate the diagonal-Gaussian-mixture
        log-prob of the GT action at each chunk step, and return the per-window
        mean NLL ``(-logp).mean(dim=1)`` -- matching the MSE branch's per-window
        reduction so the caller's ``torch.stack(per_chunk, dim=1).mean()`` is
        unchanged. continuous/discrete: the EXACT original inline MSE (mean over
        cs * A), byte-identical to before this head-aware split was added.
        """
        policy = self.nets["policy"]
        if getattr(policy, "_act_gmm", False):
            from egomimic.models.hnet_nets.action_heads import (
                _gmm_log_prob,
                _gmm_split,
            )
            if chunk_pred.shape[:-1] != gt_chunk.shape[:-1]:
                raise ValueError(
                    "GMM chunk loss leading-shape mismatch: chunk_pred "
                    f"{tuple(chunk_pred.shape)} vs gt_chunk {tuple(gt_chunk.shape)} "
                    "(expected matching (W, cs) leading dims)"
                )
            means, log_stds, logits = _gmm_split(policy, chunk_pred)  # (W,cs,M,D)x2,(W,cs,M)
            log_prob = _gmm_log_prob(policy, means, log_stds, logits, gt_chunk)  # (W, cs)
            return (-log_prob).mean(dim=1)                                       # (W,)
        # continuous / discrete: original inline MSE (byte-identical).
        return ((chunk_pred - gt_chunk) ** 2).mean(dim=(1, 2))   # (W,) per-window mean over cs*A

    def _format_batch_sim(self, obs_env_list, emb_id, device):
        """Format+normalize W raw sim obs -> dict of (W, ...) normalized tensors."""
        per = [self._format_pushshapes_norm(o, emb_id, device) for o in obs_env_list]
        return {key: torch.cat([p[key] for p in per], dim=0) for key in per[0]}

    def _format_pushshapes_norm(self, obs_env, emb_id, device):
        raw = _format_pushshapes_obs(obs_env, device)              # (1, ...)
        return self.norm_stats.normalize(raw, emb_id)

    # ---------- vectorized closed-loop windowed K-rollout ----------
    def forward_training(self, batch):
        if self.chunk_rollout:
            return self._forward_training_chunk(batch)
        # AR (non-chunk) path: the per-step loss is MSE on the DECODED action
        # (a_pred from action_head_decode). For a GMM head that would silently
        # MSE-on-the-committed-mode -- the WRONG objective (it must be the GMM
        # NLL on RAW params, like the chunk path). Refuse rather than train the
        # wrong loss; use the chunk-B path (chunk_rollout=true) for GMM, or wire
        # NLL here as a follow-up.
        if getattr(self.nets["policy"], "_act_gmm", False):
            raise NotImplementedError(
                "GMM action head is not supported on the AR (non-chunk) "
                "closed-loop path: the per-step loss here is MSE on the decoded "
                "action, which would mis-train the mixture. Use chunk_rollout=true "
                "(chunk-B), which applies the GMM NLL on the raw params."
            )
        K = self.current_k
        device = self.device
        stride = max(1, K // 2)
        use_sim = self.obs_source == "sim"
        predictions = OrderedDict()

        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            cu = _batch["cu_seqlens"].long()
            actions = _batch[ac_key]                  # (T_total, A) normalized GT
            state_seq = _batch["state_agent_obj"]     # (T_total, Dstate)
            goal_seq = _batch.get("goal_pose")
            imgs = None if use_sim else _batch.get(self._image_key)
            B_ep = int(cu.numel() - 1)

            # 1. enumerate FULL-K window anchors (global frame indices).
            anchors, ep_ids = [], []
            for b in range(B_ep):
                s, e = int(cu[b]), int(cu[b + 1])
                T = e - s
                if T < K:
                    continue                          # skip episodes shorter than K
                for w0 in range(0, T - K + 1, stride):
                    anchors.append(s + w0)
                    ep_ids.append(b)
            if not anchors:
                predictions[f"{emb_id}_action_loss"] = torch.zeros((), device=device, requires_grad=True)
                predictions[f"{emb_id}_ratio_loss"] = torch.zeros((), device=device)
                predictions[f"{emb_id}_current_k"] = torch.tensor(float(K), device=device)
                continue
            # cap windows to bound memory (~W*K^2). max_window_steps is that budget
            # (0=off); eff cap = min(max_windows, budget // K^2) so low K keeps more
            # windows (smoother loss) while high K shrinks W to avoid CUDA OOM.
            eff_max = self.max_windows
            if self.max_window_steps:
                budget_cap = max(8, self.max_window_steps // max(1, K * K))
                eff_max = budget_cap if not eff_max else min(eff_max, budget_cap)
            if eff_max and len(anchors) > eff_max:
                sel = torch.randperm(len(anchors))[:eff_max].tolist()
                anchors = [anchors[i] for i in sel]
                ep_ids = [ep_ids[i] for i in sel]
            anchors_t = torch.tensor(anchors, device=device, dtype=torch.long)  # (W,)
            W = anchors_t.numel()

            # 2. B only: set_state W envs to their anchors (raw world-coords) — PARALLEL.
            if use_sim:
                vec = self._get_vec(W)
                states = []
                for i in range(W):
                    a, b = anchors[i], ep_ids[i]
                    raw_state = (
                        self.norm_stats.unnormalize(
                            {"state_agent_obj": state_seq[a].unsqueeze(0)}, emb_id
                        )["state_agent_obj"].reshape(-1).detach().cpu().numpy()
                    )
                    ap, op = _state_to_init(raw_state)
                    gp = (
                        tuple(float(x) for x in goal_seq[b].detach().cpu().numpy().reshape(-1)[:3])
                        if goal_seq is not None else None
                    )
                    states.append((ap, op, gp))
                obs_cur = vec.set_states(states)        # W obs dicts (workers in parallel)

            # 3. batched K-step rollout over all W windows.
            obs_hist, prev_acts, per_step = [], [], []
            for k in range(K):
                if use_sim:
                    obs_k = self._format_batch_sim(obs_cur, emb_id, device)        # (W,...)
                else:
                    fidx = anchors_t + k                                           # (W,)
                    obs_k = {                                                     # already normalized
                        self._image_key: imgs[fidx].to(device),
                        "state_agent_obj": state_seq[fidx].to(device),
                    }
                obs_hist.append(obs_k)
                a_pred = self._predict_action_grad_batched(obs_hist, prev_acts)    # (W, A)
                gt = actions[anchors_t + k]                                        # (W, A)
                per_step.append(((a_pred - gt) ** 2).mean(dim=-1))                 # (W,)
                prev_acts.append(a_pred.detach())
                if use_sim:                                                       # advance all W envs (PARALLEL)
                    aw = self.norm_stats.unnormalize(
                        {ac_key: a_pred.detach()}, emb_id
                    )[ac_key]                                                      # (W, A)
                    actions_xy = [
                        np.array([float(aw[i, 0]), float(aw[i, 1])], np.float64)
                        for i in range(W)
                    ]
                    obs_cur = vec.step(actions_xy)

            # per-window MEAN over K, then mean over windows.
            aloss = torch.stack(per_step, dim=1).mean()
            predictions[f"{emb_id}_action_loss"] = aloss
            predictions[f"{emb_id}_ratio_loss"] = torch.zeros((), device=device)
            predictions[f"{emb_id}_current_k"] = torch.tensor(float(K), device=device)
            predictions[f"{emb_id}_n_windows"] = torch.tensor(float(W), device=device)
        return predictions

    # ---------- chunk-B: closed-loop CHUNKED rollout ----------
    def _forward_training_chunk(self, batch):
        """chunk-B recipe. self.current_k = NUMBER OF CHUNKS this epoch. Each
        macro-step (chunk) observes ONE frame, predicts a chunk_size-action
        chunk from THAT obs with grad (reusing the chunk head -- one forward,
        current obs only), supervises it per-step against the demo's GT chunk,
        then (sim) executes the predicted chunk DETACHED to advance the sim to
        the next chunk's obs. Loss = mean over chunks of (per-window mean over
        the chunk_size steps), then mean over windows."""
        C = self.current_k                       # number of chunks this epoch
        cs = self.chunk_size                     # frames per chunk
        window_length = C * cs                   # total frames spanned
        device = self.device
        stride = max(1, window_length // 2)
        use_sim = self.obs_source == "sim"
        predictions = OrderedDict()

        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            cu = _batch["cu_seqlens"].long()
            actions = _batch[ac_key]                  # (T_total, A) normalized GT
            state_seq = _batch["state_agent_obj"]     # (T_total, Dstate)
            goal_seq = _batch.get("goal_pose")
            imgs = None if use_sim else _batch.get(self._image_key)
            B_ep = int(cu.numel() - 1)

            # 1. enumerate full window anchors; skip episodes shorter than window.
            anchors, ep_ids = [], []
            for b in range(B_ep):
                s, e = int(cu[b]), int(cu[b + 1])
                T = e - s
                if T < window_length:
                    continue
                for w0 in range(0, T - window_length + 1, stride):
                    anchors.append(s + w0)
                    ep_ids.append(b)
            if not anchors:
                predictions[f"{emb_id}_action_loss"] = torch.zeros((), device=device, requires_grad=True)
                predictions[f"{emb_id}_ratio_loss"] = torch.zeros((), device=device)
                predictions[f"{emb_id}_current_k"] = torch.tensor(float(C), device=device)
                continue
            # cap windows to bound memory (dominant term ~ W*C). max_window_steps
            # is that budget (0=off); eff cap = min(max_windows, budget // C) so
            # low chunk-count phases keep more windows.
            eff_max = self.max_windows
            if self.max_window_steps:
                budget_cap = max(8, self.max_window_steps // max(1, C))
                eff_max = budget_cap if not eff_max else min(eff_max, budget_cap)
            if eff_max and len(anchors) > eff_max:
                sel = torch.randperm(len(anchors))[:eff_max].tolist()
                anchors = [anchors[i] for i in sel]
                ep_ids = [ep_ids[i] for i in sel]
            anchors_t = torch.tensor(anchors, device=device, dtype=torch.long)  # (W,)
            W = anchors_t.numel()

            # 2. B only: set_state W envs to their anchors (raw world-coords).
            if use_sim:
                vec = self._get_vec(W)
                states = []
                for i in range(W):
                    a, b = anchors[i], ep_ids[i]
                    raw_state = (
                        self.norm_stats.unnormalize(
                            {"state_agent_obj": state_seq[a].unsqueeze(0)}, emb_id
                        )["state_agent_obj"].reshape(-1).detach().cpu().numpy()
                    )
                    ap, op = _state_to_init(raw_state)
                    gp = (
                        tuple(float(x) for x in goal_seq[b].detach().cpu().numpy().reshape(-1)[:3])
                        if goal_seq is not None else None
                    )
                    states.append((ap, op, gp))
                obs_cur = vec.set_states(states)        # W obs dicts

            # 3. C-chunk rollout over all W windows.
            N = self.n_obs_history                  # obs-history window (N>=1)
            # sim: rolling buffer of the last N chunk-anchor obs dicts (each
            # (W,...) normalized). Chunk anchors are cs env-steps apart, so this
            # is the history of the frames the policy actually re-observes. At
            # episode start (<N seen) we repeat the earliest available frame.
            sim_hist = [] if use_sim else None
            per_chunk = []
            for c in range(C):
                if use_sim:
                    obs_c = self._format_batch_sim(obs_cur, emb_id, device)        # (W,...) current frame
                    if N > 1:
                        sim_hist.append(obs_c)
                        if len(sim_hist) > N:
                            sim_hist = sim_hist[-N:]
                        # pad-by-repeat (oldest) so the window is always length N.
                        win = ([sim_hist[0]] * (N - len(sim_hist))) + sim_hist
                        # HIST_TRACE: gated diagnostic (byte-identical when unset).
                        # Prints the REAL accumulated history depth (len(sim_hist),
                        # which grows 1..C as chunks roll) vs the padded window len N.
                        # Evidence that history windows assemble with growing length.
                        if os.environ.get("HIST_TRACE", "0") in ("1", "true", "True"):
                            _stk = torch.stack([f["state_agent_obj"] for f in win], dim=1)
                            print(
                                f"[HIST_TRACE] sim chunk c={c}/{C-1} N={N} "
                                f"real_hist_depth={len(sim_hist)} win_len={len(win)} "
                                f"stacked_shape={tuple(_stk.shape)}",
                                flush=True,
                            )
                        obs_in = {
                            k: torch.stack([f[k] for f in win], dim=1)             # (W, N, ...)
                            for k in obs_c
                        }
                    else:
                        obs_in = obs_c
                else:
                    if N > 1:
                        # history frame indices: anchor + (c-h)*cs for h=N-1..0,
                        # clamped to >= anchor (repeat frame 0 at episode start).
                        offs = (c - torch.arange(N - 1, -1, -1, device=device)).clamp(min=0) * cs  # (N,)
                        fidx = anchors_t.unsqueeze(1) + offs.unsqueeze(0)           # (W, N) frame indices
                        obs_in = {                                                  # already normalized
                            self._image_key: imgs[fidx].to(device),               # (W, N, C, H, W)
                            "state_agent_obj": state_seq[fidx].to(device),        # (W, N, Ds)
                        }
                    else:
                        fidx = anchors_t + c * cs                                   # (W,) anchor of this chunk
                        obs_in = {                                                  # already normalized
                            self._image_key: imgs[fidx].to(device),
                            "state_agent_obj": state_seq[fidx].to(device),
                        }
                chunk_pred = self._predict_chunk_grad(obs_in)                       # (W, cs, A) grad
                # demo GT chunk: actions[anchor + c*cs : anchor + (c+1)*cs] per window.
                step_idx = (anchors_t + c * cs).unsqueeze(1) + torch.arange(cs, device=device).unsqueeze(0)  # (W, cs)
                demo_chunk = actions[step_idx]                                      # (W, cs, A) normalized demo
                do_dagger = self.dagger and use_sim
                if use_sim:                                                         # advance sim cs steps (detached)
                    # The sim is driven by the DECODED continuous action. For a
                    # continuous head chunk_pred already IS that action (decode is
                    # identity) -> keep the exact original detach (byte-identical).
                    # For GMM, chunk_pred is RAW params (W, cs, M*(2D+1)); decode
                    # the full chunk to a mode-committed action (W, cs, A) BEFORE
                    # unnormalizing -- the loss above still uses the raw params.
                    if getattr(self.nets["policy"], "_act_gmm", False):
                        from egomimic.models.hnet_nets.action_heads import (
                            action_head_decode_chunk,
                        )
                        chunk_act = action_head_decode_chunk(
                            self.nets["policy"], chunk_pred.detach()
                        )                                                            # (W, cs, A) decoded
                    else:
                        chunk_act = chunk_pred.detach()
                    aw = self.norm_stats.unnormalize(
                        {ac_key: chunk_act}, emb_id
                    )[ac_key]                                                        # (W, cs, A) world-coord model action
                    expert_cols = [] if do_dagger else None
                    for j in range(cs):
                        if do_dagger:
                            # BEFORE stepping: query the scripted expert at each env's
                            # CURRENT (drifted) world state, normalize -> expert target
                            # for this step. obs_cur[i] holds agent_pos/object_pose/
                            # goal_pose in WORLD coords (the SAME fields _dagger_record
                            # / _rollout_split read). World frame == the cursor world
                            # coords ac_key unnormalizes to (env.step clips action to
                            # [0,WORLD_SIZE] as the absolute pusher target).
                            e_world = np.empty((W, self.action_dim), np.float64)    # (W, A)
                            for i in range(W):
                                oe = obs_cur[i]
                                agent = np.asarray(oe["agent_pos"], np.float64).reshape(-1)[:2]
                                obj = np.asarray(oe["object_pose"], np.float64).reshape(-1)
                                goal = np.asarray(oe["goal_pose"], np.float64).reshape(-1)
                                e_world[i] = _scripted_expert(
                                    agent_xy=agent, object_xy=obj[:2], goal_xy=goal[:2],
                                    world_size=512.0,
                                )
                            # normalize the expert world action. normalize() keys
                            # off the ZARR key (not the algo key_name), so feed/read
                            # under the action's zarr_key. assert it actually applied
                            # (zarr_key present + stats exist) -- a silent no-op here
                            # would mean comparing a world-coord target against a
                            # normalized pred (sanity #1 would also blow up).
                            ac_zkey = self.norm_stats.keyname_to_zarr_key(ac_key, emb_id) or ac_key
                            e_in = {ac_zkey: torch.from_numpy(e_world).float().to(device)}
                            e_norm = self.norm_stats.normalize(e_in, emb_id)[ac_zkey]   # (W, A) normalized
                            if c == 0 and j == 0:
                                assert not torch.equal(
                                    e_norm, e_in[ac_zkey]
                                ), f"[DAGGER] normalize() was a no-op for action zarr_key={ac_zkey!r}"
                            expert_cols.append(e_norm)
                        # step with the MODEL's action (UNCHANGED -- policy drives rollout).
                        actions_xy = [
                            np.array([float(aw[i, j, 0]), float(aw[i, j, 1])], np.float64)
                            for i in range(W)
                        ]
                        obs_cur = vec.step(actions_xy)
                    if do_dagger:
                        # (W, cs, A) DETACHED expert target. grad flows ONLY via chunk_pred.
                        gt_chunk = torch.stack(expert_cols, dim=1).detach()
                        # Sanity compares the model action (world px) to the expert,
                        # so pass the DECODED chunk_act (== chunk_pred for continuous,
                        # mode-committed action for GMM) -- raw GMM params can't be
                        # unnormalized inside _dagger_sanity.
                        _dagger_sanity(self, c, chunk_act, demo_chunk, gt_chunk, emb_id, ac_key)
                    else:
                        gt_chunk = demo_chunk
                else:
                    gt_chunk = demo_chunk
                # HEAD-AWARE per-window loss (W,). GMM -> NLL of the GT chunk
                # under the per-step mixture; continuous/discrete -> the exact
                # original inline MSE (byte-identical). DAgger is unchanged:
                # gt_chunk is still the supervision target, only the loss fn
                # differs. chunk_pred is RAW head output (params for GMM).
                per_chunk.append(self._chunk_loss_per_window(chunk_pred, gt_chunk))

            # mean over chunks of per-window means, then mean over windows.
            aloss = torch.stack(per_chunk, dim=1).mean()
            predictions[f"{emb_id}_action_loss"] = aloss
            predictions[f"{emb_id}_ratio_loss"] = torch.zeros((), device=device)
            predictions[f"{emb_id}_current_k"] = torch.tensor(float(C), device=device)
            predictions[f"{emb_id}_n_windows"] = torch.tensor(float(W), device=device)
            # Optional per-step stdout trace of the chunk loss (GMM NLL or MSE),
            # gated by CHUNK_LOSS_TRACE so it only fires in smokes -- wandb buffers
            # on_epoch metrics, so this is the reliable way to SEE the loss fall.
            if os.environ.get("CHUNK_LOSS_TRACE", "0") in ("1", "true", "True"):
                head = "gmm" if getattr(self.nets["policy"], "_act_gmm", False) else "cont"
                print(
                    f"[CHUNK_LOSS_TRACE] emb={emb_id} head={head} C={C} cs={cs} "
                    f"W={W} action_loss={float(aloss):.5f}",
                    flush=True,
                )
        return predictions


# ---------- curriculum + env-pool-teardown callback ----------
def _find_algo(pl_module):
    return getattr(pl_module, "model", pl_module)


try:
    import lightning.pytorch as pl

    class KCurriculumCallback(pl.Callback):
        """Bumps K at each epoch (the curriculum) AND closes B's env pool on exit
        (the workers don't auto-die while Lightning teardown is waiting -> the
        job hangs after training; closing the pool fixes it)."""

        def on_train_epoch_start(self, trainer, pl_module):
            algo = _find_algo(pl_module)
            if hasattr(algo, "set_curriculum_epoch"):
                algo.set_curriculum_epoch(trainer.current_epoch)

        def _close_pool(self, pl_module):
            algo = _find_algo(pl_module)
            vec = getattr(algo, "_vec", None)
            if vec is not None:
                try:
                    vec.close()
                except Exception:
                    pass
                algo._vec = None

        def on_fit_end(self, trainer, pl_module):
            self._close_pool(pl_module)

        def teardown(self, trainer, pl_module, stage=None):
            self._close_pool(pl_module)
except Exception:
    KCurriculumCallback = None
