"""On-policy rollout collection for FPO, reusing the existing PushShapes sim.

One RL transition = one action-chunk decision (chunk_openloop, k=chunk):
observe -> sample H-step chunk -> execute all H in env -> record. Reward is the
per-chunk coverage improvement (delta), which telescopes to (final - init)
coverage, so maximizing return == maximizing final coverage. Fixed goal pinned
to the training target each episode (matches eval_act8_peek fixed_goal).
"""
from __future__ import annotations
import numpy as np
import torch

from egomimic.eval.eval_sim import _OBS_FORMATTERS as _ENV_TO_ZARR


def _reset_fixed_goal(env, seed, goal, emb_name):
    obs, info = env.reset(seed=int(seed))
    if goal is not None:
        # Match EV2 _apply_fixed_goal EXACTLY: override ONLY the goal, leaving
        # agent_pos/object_pose=None so the reset's random T+pusher are kept.
        # Passing object_pose back through set_state re-applies it through the
        # T-shape CoG offset (angle-first), which is NOT idempotent and silently
        # SHIFTS the object -> corrupted init -> ~0 coverage. (Root cause of the
        # earlier all-zero rollouts.)
        env.set_state(goal_pose=np.asarray(goal, dtype=np.float32))
        obs = env._get_obs()
        info = {"coverage": float(env._coverage())}
    return obs, info


def _obj_goal_dist(obs, env):
    """Normalized object->goal distance (xy), for dense reward shaping."""
    o = np.asarray(obs["object_pose"]).reshape(-1)
    g = np.asarray(env._goal_pose).reshape(-1)
    return float(np.hypot(o[0] - g[0], o[1] - g[1])) / 512.0


def collect_rollouts(fpo, env, seeds, goal, max_steps=500, shape_w=0.5, explore_std=0.0):
    """Returns dict with stacked tensors over all transitions across episodes:
      data_list (list of per-transition HPT data dicts, on CPU),
      a_norm [M,H,D], rewards [M], values [M], dones [M], ep_id [M],
      last_values {ep_id: bootstrap}, and per-episode final coverage list.
    """
    emb = fpo.emb_name
    conv = _ENV_TO_ZARR[emb]
    chunk = fpo.chunk
    obs_list, a_norms, rewards, values, dones, ep_ids = [], [], [], [], [], []
    last_values, final_covs = {}, []

    for ep, seed in enumerate(seeds):
        obs, info = _reset_fixed_goal(env, seed, goal, emb)
        prev_cov = float(info["coverage"])
        prev_d = _obj_goal_dist(obs, env)
        t, terminated = 0, False
        while t < max_steps:
            obs_zarr = conv(obs, fpo.device)
            data = fpo.build_data(obs_zarr)
            a_norm, value, chunk_world = fpo.sample(data, explore_std=explore_std)
            cov = prev_cov
            for j in range(chunk):
                obs, reward, terminated, truncated, info = env.step(chunk_world[j])
                cov = float(info["coverage"])
                if terminated or truncated:
                    break
            d = _obj_goal_dist(obs, env)
            # dense shaped reward: coverage gain + progress toward goal (so the
            # hard-init seeds that never reach high coverage still get gradient
            # signal for pushing the object closer). shape_w=0 -> pure Δcoverage.
            shaped = (cov - prev_cov) + shape_w * (prev_d - d)
            prev_d = d
            obs_list.append({k: _to_cpu(v) for k, v in obs_zarr.items()})
            a_norms.append(a_norm.squeeze(0).cpu())
            rewards.append(shaped)
            values.append(value)
            dones.append(1.0 if terminated else 0.0)
            ep_ids.append(ep)
            prev_cov = cov
            t += chunk
            if terminated:
                break
        # bootstrap value for the last (non-terminal) state
        if terminated:
            last_values[ep] = 0.0
        else:
            obs_zarr = conv(obs, fpo.device)
            last_values[ep] = fpo.value_of(fpo.build_data(obs_zarr))
        final_covs.append(prev_cov)

    return {
        "obs_list": obs_list,
        "a_norm": torch.stack(a_norms),                       # [M,H,D]
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.float32),
        "ep_id": torch.tensor(ep_ids, dtype=torch.int64),
        "last_values": last_values,
        "final_covs": final_covs,
    }


def _to_cpu(v):
    if torch.is_tensor(v):
        return v.detach().cpu()
    return v


def collect_grpo(fpo, env, seeds, goal, group_size=6, max_steps=500,
                 shape_w=0.5, explore_std=0.1, adv_eps=1e-4):
    """GRPO-style collection: for each init seed, roll out `group_size` times
    with exploration noise; advantage = group-relative (return - mean)/std over
    the group. No critic. Returns transitions with per-transition advantages
    already computed.

    return per rollout = sum of shaped rewards (coverage gain + goal-progress),
    so hard-init seeds where exploration makes *partial* progress get positive
    relative advantage, while fully-solved seeds (all rollouts ~equal) get ~0
    advantage and are left alone.
    """
    conv = _ENV_TO_ZARR[fpo.emb_name]
    chunk = fpo.chunk
    obs_list, a_norms, trans_rid = [], [], []
    rollout_return, rollout_seed, final_covs = [], [], []
    rid = 0
    for si, seed in enumerate(seeds):
        for _g in range(group_size):
            obs, info = _reset_fixed_goal(env, seed, goal, fpo.emb_name)
            prev_cov = float(info["coverage"]); prev_d = _obj_goal_dist(obs, env)
            R = 0.0; t = 0; terminated = False
            while t < max_steps:
                oz = conv(obs, fpo.device)
                a_norm, _value, cw = fpo.sample(fpo.build_data(oz), explore_std=explore_std)
                cov = prev_cov
                for j in range(chunk):
                    obs, _r, terminated, truncated, info = env.step(cw[j])
                    cov = float(info["coverage"])
                    if terminated or truncated:
                        break
                d = _obj_goal_dist(obs, env)
                R += (cov - prev_cov) + shape_w * (prev_d - d)
                obs_list.append({k: _to_cpu(v) for k, v in oz.items()})
                a_norms.append(a_norm.squeeze(0).cpu())
                trans_rid.append(rid)
                prev_cov = cov; prev_d = d; t += chunk
                if terminated:
                    break
            rollout_return.append(R); rollout_seed.append(si); final_covs.append(prev_cov)
            rid += 1
    rr = np.asarray(rollout_return, dtype=np.float64)
    rs = np.asarray(rollout_seed)
    radv = np.zeros_like(rr)
    for si in range(len(seeds)):
        m = rs == si
        g = rr[m]
        radv[m] = (g - g.mean()) / (g.std() + adv_eps)
    adv = torch.tensor([radv[r] for r in trans_rid], dtype=torch.float32)
    return {
        "obs_list": obs_list,
        "a_norm": torch.stack(a_norms),
        "adv": adv,
        "final_covs": final_covs,        # G*N entries (with exploration noise)
        "group_returns": rr.tolist(),
    }
