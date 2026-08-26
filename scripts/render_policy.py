"""Render the GMM policy doing the fixed-target push task -> per-seed mp4s."""
import argparse, numpy as np, torch
from scripts.train_ppo_gmm import load_algo_from_ckpt
from egomimic.algo.ppo_gmm.gmm_policy import GMMPolicy
from egomimic.algo.fpo.rollout import _reset_fixed_goal
from egomimic.eval.eval_sim import _OBS_FORMATTERS

GOAL = [256.0, 256.0, 0.7853981633974483]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seeds", default="0,1,2,6")
    p.add_argument("--outdir", default="/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/policy_videos")
    args = p.parse_args()
    import os, imageio
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda")
    algo, _ = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.nets.eval()
    pol = GMMPolicy(algo, emb_name=args.emb, device=device)
    if args.resume:
        torch.load(args.resume, map_location=device)
        sd = torch.load(args.resume, map_location=device)
        algo.nets["policy"].load_state_dict(sd["policy"]); print(f"[resume] {args.resume}")
    pol.head.low_noise_eval = True
    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0,
                        image_size=96, render_mode="rgb_array")
    conv = _OBS_FORMATTERS[args.emb]; rnn, chunk = pol.rnn_horizon, pol.chunk
    torch.manual_seed(1234)
    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, info = _reset_fixed_goal(env, seed, GOAL, args.emb)
        frames = [env.render()]
        strided, t, k, term = [], 0, 0, False
        while t < args.max_steps:
            oz = conv(obs, pol.device)
            emb_t = pol.encode_frame(pol.normalize_obs(oz))
            strided.append(emb_t.squeeze(0).detach().cpu())
            blk = (k // rnn) * rnn
            window = torch.stack(strided[blk:k+1], 0).unsqueeze(0).to(pol.device)
            a, _, _ = pol.act(window)
            cw = pol.unnormalize_chunk(a.squeeze(0)).detach().cpu().numpy()
            for j in range(chunk):
                obs, _r, term, trunc, info = env.step(cw[j])
                frames.append(env.render())
                if term or trunc: break
            t += chunk; k += 1
            if term: break
        cov = float(info["coverage"])
        out = f"{args.outdir}/seed{seed}_cov{cov:.2f}.mp4"
        imageio.mimwrite(out, [f for i, f in enumerate(frames) if i % 2 == 0], fps=30)
        print(f"[seed {seed}] final_cov={cov:.3f} -> {out}", flush=True)

if __name__ == "__main__":
    main()
