"""Policy-based data generation for PushShapes (circle_3000 varied-goal task).

Generate-and-filter: roll the trained GMM BC policy on FRESH varied-goal seeds,
record every frame via the repo's ZarrDemoWriter (exact circle_3000 schema +
JPEG codec + episode_init), and COMMIT only episodes whose final coverage clears
--success-threshold. Per seed we try up to --attempts-per-seed rollouts (attempt 0
deterministic mode = the deployable BC; the rest stochastic softplus-std sampling
for best-of-N), committing the FIRST success. Failures are aborted (atomic — no
partial episodes land on disk). Fresh seed per saved episode => diverse goals.

Reuses Tsimulation.collect.zarr_writer.ZarrDemoWriter and bucket balancing so the
output directory is a drop-in circle_* dataset the existing data configs can load.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from Tsimulation.collect.balance import N_BUCKETS, BucketTracker, bucket_for, count_existing_buckets
from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv
from scripts.train_ppo_gmm import load_algo_from_ckpt
from egomimic.algo.ppo_gmm.gmm_policy import GMMPolicy
from egomimic.eval.eval_sim import _OBS_FORMATTERS


def rollout_record(env, pol, conv, rnn, chunk, seed, max_steps, stochastic):
    """One full episode from reset(seed) (natural varied goal). Returns
    (final_cov, frames, ep_init). frames: list of per-env-step dicts."""
    pol.head.low_noise_eval = (not stochastic)
    obs, info = env.reset(seed=int(seed))
    ep_init = env.get_episode_init()
    frames = []
    strided, t, k, term = [], 0, 0, False
    cov = float(info.get("coverage", 0.0))
    while t < max_steps:
        oz = conv(obs, pol.device)
        emb_t = pol.encode_frame(pol.normalize_obs(oz))
        strided.append(emb_t.squeeze(0).detach().cpu())
        blk = (k // rnn) * rnn
        window = torch.stack(strided[blk:k + 1], 0).unsqueeze(0).to(pol.device)
        a, _, _ = pol.act(window)
        cw = pol.unnormalize_chunk(a.squeeze(0)).detach().cpu().numpy()
        for j in range(chunk):
            frames.append(dict(
                image=np.asarray(obs["image"], dtype=np.uint8),
                agent_pos=np.asarray(obs["agent_pos"], dtype=np.float64),
                object_pose=np.asarray(obs["object_pose"], dtype=np.float64),
                action=np.asarray(cw[j], dtype=np.float64),
                goal_pose=np.asarray(obs["goal_pose"], dtype=np.float64),
            ))
            obs, r, term, trunc, info = env.step(cw[j])
            cov = float(info.get("coverage", 0.0))
            if term or trunc:
                break
        t += chunk; k += 1
        if term:
            break
    return cov, frames, ep_init


def write_episode(writer, frames, ep_init):
    writer.start_episode(init_state=ep_init)
    for f in frames:
        writer.add_step(image=f["image"], pusher_obs_pose=f["agent_pos"],
                        object_obs_pose=f["object_pose"], pusher_cmd_pose=f["action"],
                        action=f["action"], reward=0.0, goal_pose=f["goal_pose"])
    return writer.commit_episode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--num-episodes", type=int, default=1000)
    p.add_argument("--success-threshold", type=float, default=0.85)
    p.add_argument("--attempts-per-seed", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed-start", type=int, default=100000,
                   help="far from training seeds 0..N to avoid overlap")
    p.add_argument("--max-tries", type=int, default=12,
                   help="give up after num_episodes*max_tries seeds")
    p.add_argument("--balance", action="store_true",
                   help="balance saved episodes across the 16 obj/goal-quadrant buckets")
    p.add_argument("--per-bucket", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    device = torch.device("cuda")
    algo, _ = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.nets.eval()
    pol = GMMPolicy(algo, emb_name=args.emb, device=device)
    conv = _OBS_FORMATTERS[args.emb]
    rnn, chunk = pol.rnn_horizon, pol.chunk
    torch.manual_seed(1234)

    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0,
                        image_size=96, render_mode="rgb_array")
    world = float(env.WORLD_SIZE)
    env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0,
                "image_size": 96, "fps": args.fps, "max_episode_steps": args.max_steps,
                "collector": "gmm_bc_policy"}
    writer = ZarrDemoWriter(path=args.output, env_args=env_args, image_size=96,
                            fps=args.fps, tag=args.tag)

    tracker = None
    if args.balance:
        per_bucket = args.per_bucket or max(1, -(-args.num_episodes // N_BUCKETS))
        tracker = BucketTracker(target_per_bucket=per_bucket,
                                initial_counts=count_existing_buckets(Path(args.output), world))
        target_total = tracker.goal_total
        print(f"balance: {per_bucket}/bucket x {N_BUCKETS} = {target_total}", flush=True)
    else:
        target_total = args.num_episodes

    saved, seed_ctr, attempts = 0, 0, 0
    max_seeds = target_total * args.max_tries
    while True:
        if tracker is not None and tracker.filled:
            break
        if tracker is None and saved >= target_total:
            break
        if seed_ctr >= max_seeds:
            print(f"hit max_seeds={max_seeds}; stopping early", flush=True)
            break
        seed = args.seed_start + seed_ctr
        seed_ctr += 1
        best = None
        for att in range(args.attempts_per_seed):
            cov, frames, ep_init = rollout_record(env, pol, conv, rnn, chunk, seed,
                                                   args.max_steps, stochastic=(att > 0))
            attempts += 1
            if best is None or cov > best[0]:
                best = (cov, frames, ep_init)
            if cov >= args.success_threshold:
                break
        cov, frames, ep_init = best
        bucket = bucket_for(ep_init, world)
        if cov >= args.success_threshold and (tracker is None or tracker.has_room(bucket)):
            idx = write_episode(writer, frames, ep_init)
            if idx >= 0:
                saved += 1
                if tracker is not None:
                    tracker.increment(bucket)
                if saved % 10 == 0 or saved <= 5:
                    print(f"saved {saved}/{target_total}  seed={seed}  cov={cov:.3f}  "
                          f"bucket={bucket}  (rollouts={attempts}, accept={saved/attempts:.2%})", flush=True)
    writer.close(); env.close()
    print(f"DONE saved={saved}/{target_total} rollouts={attempts} "
          f"accept_rate={saved/max(1,attempts):.2%} -> {args.output}", flush=True)
    if tracker is not None:
        print(tracker.histogram(), flush=True)


if __name__ == "__main__":
    main()
