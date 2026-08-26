"""Reverse-curriculum FEASIBILITY diagnostic.
For the it29 PPO policy: initialize the object at goal_pose + offset (various radii)
and roll out; measure init vs final coverage. If the policy drives near-goal inits
to high coverage, a reverse curriculum (anneal init radius small->large) is viable.
"""
import argparse, math, numpy as np, torch
from scripts.train_ppo_gmm import load_algo_from_ckpt
from egomimic.algo.ppo_gmm.gmm_policy import GMMPolicy
from egomimic.eval.eval_sim import _OBS_FORMATTERS

GOAL = (256.0, 256.0, 0.7853981633974483)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)      # BC algo
    p.add_argument("--resume", default="")        # PPO policy state (it29)
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--nseeds", type=int, default=20)
    args = p.parse_args()
    device = torch.device("cuda")
    algo, _ = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.nets.eval()
    pol = GMMPolicy(algo, emb_name=args.emb, device=device)
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        algo.nets["policy"].load_state_dict(sd["policy"])
        print(f"[resume] {args.resume}", flush=True)
    pol.head.low_noise_eval = True   # deployable inference

    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)
    conv = _OBS_FORMATTERS[args.emb]
    rnn, chunk = pol.rnn_horizon, pol.chunk
    torch.manual_seed(1234)

    for radius in [0.0, 40.0, 80.0, 150.0]:
        covs0, covsF = [], []
        for seed in range(args.nseeds):
            obs, info = env.reset(seed=int(seed))
            env.set_state(goal_pose=np.asarray(GOAL, dtype=np.float32))
            # place object at goal + offset of this radius
            ang = (seed * 2.399963)  # golden-angle spread of directions
            ox = GOAL[0] + radius * math.cos(ang)
            oy = GOAL[1] + radius * math.sin(ang)
            ox = float(np.clip(ox, 60, 452)); oy = float(np.clip(oy, 60, 452))
            otheta = GOAL[2] + float(np.clip((seed - 10) * 0.05, -0.4, 0.4))
            env.set_state(object_pose=(ox, oy, otheta))
            obs = env._get_obs()
            cov0 = float(env._coverage()); covs0.append(cov0)
            # roll out the policy from this near-goal init
            strided, t, k, term = [], 0, 0, False
            while t < args.max_steps:
                oz = conv(obs, pol.device)
                emb_t = pol.encode_frame(pol.normalize_obs(oz))
                strided.append(emb_t.squeeze(0).detach().cpu())
                blk = (k // rnn) * rnn
                window = torch.stack(strided[blk:k+1], 0).unsqueeze(0).to(pol.device)
                a, _, _ = pol.act(window)
                cw = pol.unnormalize_chunk(a.squeeze(0)).detach().cpu().numpy()
                cov = cov0
                for j in range(chunk):
                    obs, _r, term, trunc, info = env.step(cw[j])
                    cov = float(info["coverage"])
                    if term or trunc: break
                t += chunk; k += 1
                if term: break
            covsF.append(cov)
        print(f"[radius={radius:5.0f}px] init_cov={np.mean(covs0):.3f}  final_cov={np.mean(covsF):.3f}  "
              f"(SR>=0.8: {np.mean([c>=0.8 for c in covsF]):.2f})", flush=True)

if __name__ == "__main__":
    main()
