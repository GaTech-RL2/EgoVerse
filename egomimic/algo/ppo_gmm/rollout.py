"""On-policy rollout collection for exact-log-prob PPO on the BC-RNN GMM policy.

One RL transition = one obs-step decision: encode the current strided obs (frozen
encoder), re-run the transformer over the rolling window (reset every rnn_horizon
obs-steps, EXACTLY like BCRNNPolicy.step), sample a chunk_len action chunk from
the GMM (softplus-std -> real exploration variance), execute all chunk_len actions
in the env, record the EXACT joint log-prob + value + shaped reward. Because
obs_stride == chunk_len (==8 here), each decision's chunk exactly fills the next
obs_stride env frames -> clean one-decision-per-chunk transitions (no queue
straddling, same as the FPO chunk-openloop transition model).

For exact log-prob recompute at update time we store, per episode, the sequence
of frozen-encoder strided-obs embeddings; a decision at obs-step k uses the window
emb[(k//rnn)*rnn : k+1]. The TRAINABLE core+head are re-run over that window at
update time (grouped by window length) -> exact PPO ratio.

Fixed goal pinned each episode via the shared _reset_fixed_goal (overrides ONLY
the goal pose -- the documented non-idempotent-object-pose gotcha).
"""
from __future__ import annotations
import numpy as np
import torch

from egomimic.eval.eval_sim import _OBS_FORMATTERS as _ENV_TO_ZARR
from egomimic.algo.fpo.rollout import _reset_fixed_goal, _obj_goal_dist


def _place_object_near_goal(env, goal, radius, rng, ori_max=0.4):
    """Reverse-curriculum init: move the object to goal + a random POSITION offset
    (radius, clipped to arena) AND a random ORIENTATION error (+/- ori_max rad).
    Annealing BOTH radius (short->long approaches) and ori_max (small->large
    rotations, 0.4->pi) teaches the precise in-place rotation the policy needs for
    the mis-rotated 'stubborn' seeds (eval has uniform +/-pi orientation, so a
    tiny ori jitter under-trained rotation -> the diagnosed ~0.80 ceiling)."""
    import math
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    r = float(rng.uniform(0.0, radius))
    ox = float(np.clip(goal[0] + r * math.cos(ang), 60.0, 452.0))
    oy = float(np.clip(goal[1] + r * math.sin(ang), 60.0, 452.0))
    oth = float(goal[2] + rng.uniform(-ori_max, ori_max))
    env.set_state(object_pose=(ox, oy, oth))
    return env._get_obs(), {"coverage": float(env._coverage())}


def collect_rollouts_gmm(pol, env, seeds, goal, max_steps=400, shape_w=0.5,
                         ref_pol=None, init_radius_max=None, cur_rng=None,
                         init_ori_max=0.4):
    """Returns dict with per-episode embedding sequences + per-transition records:
      ep_embs: list over episodes of tensor (K_ep, E) on CPU,
      k [M], ep_id [M], a_norm [M,C,D], logp_old [M], rewards [M], values [M],
      dones [M], last_values {ep_id: bootstrap}, final_covs [list]."""
    emb = pol.emb_name
    conv = _ENV_TO_ZARR[emb]
    rnn, chunk = pol.rnn_horizon, pol.chunk

    ep_embs = []
    ks, ep_ids, a_norms, logps, rewards, values, dones = [], [], [], [], [], [], []
    logp_refs = []
    last_values, final_covs = {}, []

    for ep, seed in enumerate(seeds):
        obs, info = _reset_fixed_goal(env, seed, goal, emb)
        if init_radius_max is not None:
            # reverse-curriculum: re-place the object within init_radius_max of the
            # goal. VARIED-goal (goal=None): use the env's natural per-episode goal.
            gtarget = goal if goal is not None else tuple(float(x) for x in env._goal_pose)
            obs, info = _place_object_near_goal(env, gtarget, init_radius_max,
                                                cur_rng if cur_rng is not None else np.random,
                                                ori_max=init_ori_max)
        prev_cov = float(info["coverage"])
        prev_d = _obj_goal_dist(obs, env)
        strided = []                                   # this episode's (E,) frame embeddings
        t, k, terminated = 0, 0, False
        while t < max_steps:
            obs_zarr = conv(obs, pol.device)
            obs_norm = pol.normalize_obs(obs_zarr)
            emb_t = pol.encode_frame(obs_norm)          # (1, E)
            strided.append(emb_t.squeeze(0).detach().cpu())
            block = (k // rnn) * rnn
            window = torch.stack(strided[block:k + 1], dim=0).unsqueeze(0).to(pol.device)  # (1,L,E)
            a, logp, value = pol.act(window)            # (1,C,D),(1,),(1,)
            a_chunk = a.squeeze(0)                       # (C, D)
            chunk_world = pol.unnormalize_chunk(a_chunk).detach().cpu().numpy()            # (C, D)
            # FROZEN-BC log-prob of the SAME sampled chunk (same frozen-encoder
            # window) -> STORED for the loss-side BC-distillation anchor in the
            # update (coef*(logp_new - logp_ref)^2). NOT folded into the reward:
            # doing so polluted the returns/value (per-position log-density is
            # +-1-2 for a peaked GMM, so the penalty dwarfed the ~0.01-0.1 coverage
            # reward and the value diverged). The reward stays pure task coverage.
            logp_ref_t = float(logp.item())              # default (no ref): anchor inert
            if ref_pol is not None:
                logp_ref_t = float(ref_pol.logp_given(window, a).item())
            cov = prev_cov
            for j in range(chunk):
                obs, reward, terminated, truncated, info = env.step(chunk_world[j])
                cov = float(info["coverage"])
                if terminated or truncated:
                    break
            d = _obj_goal_dist(obs, env)
            shaped = (cov - prev_cov) + shape_w * (prev_d - d)
            prev_d = d
            ks.append(k); ep_ids.append(ep)
            a_norms.append(a_chunk.detach().cpu())
            logps.append(float(logp.item())); logp_refs.append(logp_ref_t)
            rewards.append(shaped); values.append(float(value.item()))
            dones.append(1.0 if terminated else 0.0)
            prev_cov = cov; t += chunk; k += 1
            if terminated:
                break
        # bootstrap value for the last (non-terminal) state
        if terminated:
            last_values[ep] = 0.0
        else:
            obs_zarr = conv(obs, pol.device)
            emb_t = pol.encode_frame(pol.normalize_obs(obs_zarr))
            seq = strided + [emb_t.squeeze(0).detach().cpu()]
            block = (k // rnn) * rnn
            window = torch.stack(seq[block:k + 1], dim=0).unsqueeze(0).to(pol.device)
            last_values[ep] = float(pol.value_of(window).item())
        ep_embs.append(torch.stack(strided, dim=0) if strided
                       else torch.zeros(0, pol.core.input_dim))
        final_covs.append(prev_cov)

    return {
        "ep_embs": ep_embs,
        "k": torch.tensor(ks, dtype=torch.int64),
        "ep_id": torch.tensor(ep_ids, dtype=torch.int64),
        "a_norm": torch.stack(a_norms),                       # (M, C, D)
        "logp_old": torch.tensor(logps, dtype=torch.float32),
        "logp_ref": torch.tensor(logp_refs, dtype=torch.float32),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.float32),
        "last_values": last_values,
        "final_covs": final_covs,
    }


def build_windows(roll, idx, rnn, device):
    """Group minibatch indices by window length L=(k%rnn)+1 so each group is a
    same-length batch -> one core.forward per group. Returns list of
    (sub_idx (list of global i), window_batch (G, L, E))."""
    ks, eps, ep_embs = roll["k"], roll["ep_id"], roll["ep_embs"]
    by_len = {}
    for i in idx:
        k = int(ks[i]); ep = int(eps[i])
        block = (k // rnn) * rnn
        by_len.setdefault(k - block + 1, []).append((i, ep, block, k))
    groups = []
    for items in by_len.values():
        wins = [ep_embs[ep][block:k + 1] for (_i, ep, block, k) in items]
        sub = [i for (i, _ep, _b, _k) in items]
        groups.append((sub, torch.stack(wins, dim=0).to(device)))   # (G, L, E)
    return groups
