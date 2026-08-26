"""Best-of-N yield test for success-filtered data generation on circle_3000.

For each VARIED-goal seed (natural env goal, no override): run 1 deterministic
rollout (low_noise_eval=True, the deployable BC) + N stochastic rollouts
(low_noise_eval=False, softplus-std sampling). Same seed reset each attempt ->
identical goal+init, only the policy's action sampling varies. Report how many
seeds reach the >=THR coverage bar one-shot-deterministic vs best-of-N. This
decides whether policy+filter data-gen is viable and how many attempts/episode
it costs, WITHOUT needing a uniformly-good policy.
"""
import argparse, numpy as np, torch
from scripts.train_ppo_gmm import load_algo_from_ckpt
from egomimic.algo.ppo_gmm.gmm_policy import GMMPolicy
from egomimic.eval.eval_sim import _OBS_FORMATTERS


def rollout_once(pol, env, seed, conv, rnn, chunk, max_steps, stochastic):
    """One full episode from reset(seed) (natural varied goal). Returns final cov."""
    pol.head.low_noise_eval = (not stochastic)
    obs, info = env.reset(seed=int(seed))   # natural per-episode goal, NO override
    strided, t, k, term = [], 0, 0, False
    cov = float(info["coverage"])
    while t < max_steps:
        oz = conv(obs, pol.device)
        emb_t = pol.encode_frame(pol.normalize_obs(oz))
        strided.append(emb_t.squeeze(0).detach().cpu())
        blk = (k // rnn) * rnn
        window = torch.stack(strided[blk:k + 1], 0).unsqueeze(0).to(pol.device)
        a, _, _ = pol.act(window)
        cw = pol.unnormalize_chunk(a.squeeze(0)).detach().cpu().numpy()
        for j in range(chunk):
            obs, _r, term, trunc, info = env.step(cw[j])
            cov = float(info["coverage"])
            if term or trunc:
                break
        t += chunk; k += 1
        if term:
            break
    return cov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--n-stoch", type=int, default=8)
    p.add_argument("--seeds", default="0-39")
    p.add_argument("--thr", type=float, default=0.9)
    args = p.parse_args()

    if "-" in args.seeds:
        a, b = args.seeds.split("-"); seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    device = torch.device("cuda")
    algo, _ = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.nets.eval()
    pol = GMMPolicy(algo, emb_name=args.emb, device=device)
    conv = _OBS_FORMATTERS[args.emb]
    rnn, chunk = pol.rnn_horizon, pol.chunk

    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0,
                        image_size=96, render_mode="rgb_array")
    torch.manual_seed(1234)

    det_covs, bestN_covs, det_hits, bestN_hits, attempts_to_hit = [], [], 0, 0, []
    for s in seeds:
        dcov = rollout_once(pol, env, s, conv, rnn, chunk, args.max_steps, stochastic=False)
        scovs = [rollout_once(pol, env, s, conv, rnn, chunk, args.max_steps, stochastic=True)
                 for _ in range(args.n_stoch)]
        best = max([dcov] + scovs)
        det_covs.append(dcov); bestN_covs.append(best)
        det_hits += int(dcov >= args.thr)
        hit = best >= args.thr
        bestN_hits += int(hit)
        if hit:
            # attempts (det counts as 1) until first success
            seq = [dcov] + scovs
            for i, c in enumerate(seq):
                if c >= args.thr:
                    attempts_to_hit.append(i + 1); break
        print(f"[seed {s:3d}] det={dcov:.2f}  stoch_max={max(scovs):.2f}  "
              f"best={best:.2f}  hit={hit}  stoch={[round(c,2) for c in scovs]}",
              flush=True)

    n = len(seeds)
    print("\n=== YIELD SUMMARY ===", flush=True)
    print(f"n_seeds={n}  n_stoch={args.n_stoch}  thr={args.thr}  max_steps={args.max_steps}", flush=True)
    print(f"deterministic one-shot:  SR>={args.thr} = {det_hits}/{n} = {det_hits/n:.2f}   mean_cov={np.mean(det_covs):.3f}", flush=True)
    print(f"best-of-(1+{args.n_stoch}):       SR>={args.thr} = {bestN_hits}/{n} = {bestN_hits/n:.2f}   mean_best={np.mean(bestN_covs):.3f}", flush=True)
    if attempts_to_hit:
        print(f"avg attempts to first success (succeeding seeds): {np.mean(attempts_to_hit):.2f}", flush=True)
    print(f"DONE", flush=True)


if __name__ == "__main__":
    main()
