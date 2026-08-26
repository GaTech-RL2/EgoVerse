"""Standard PPO (exact GMM log-prob) fine-tuning of a BC-RNN GMM policy on
PushShapes (fixed-target boundary task). EgoVerse2.

The GMM head gives EXACT action log-probs, so the PPO ratio is the textbook
exp(logp_new - logp_old) -- no Monte-Carlo CFM surrogate (FPO) and no off-policy
value bootstrap (RQL), the two routes that proved insufficient / unstable here.
Perception (obs encoder) is FROZEN; PPO tunes the temporal core + GMM head +
a fresh value head. GAE advantages, clipped surrogate, value loss, small entropy
bonus. Eval coverage is measured with the deployable low-noise BC inference path.

Run (GPU node): PYTHONPATH=<EgoVerse2> MUJOCO_GL=egl python scripts/train_ppo_gmm.py --ckpt <bc_ckpt>
"""
from __future__ import annotations
import argparse, os, numpy as np, torch

from egomimic.algo.ppo_gmm.gmm_policy import GMMPolicy
from egomimic.algo.ppo_gmm.rollout import collect_rollouts_gmm, build_windows
from egomimic.algo.fpo import fpo_core as F
from egomimic.algo.fpo.rollout import _reset_fixed_goal
from egomimic.eval.eval_sim import _OBS_FORMATTERS


def load_algo_from_ckpt(ckpt_path):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters") or {}
    cfg = OmegaConf.create(hp["config_tree"])
    norm = MultiDataset.from_state(hp["norm_stats_state"])
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm)
    sd = ck["state_dict"]; new = {}
    for k, v in sd.items():
        for p in ("nets.", "model.nets."):
            if k.startswith(p):
                new[k[len(p):]] = v; break
        else:
            new[k] = v
    missing, unexpected = algo.nets.load_state_dict(new, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return algo, cfg


def gae_per_episode(roll, gamma, lam, device):
    M = roll["rewards"].shape[0]
    adv = torch.zeros(M); ret = torch.zeros(M)
    for ep in roll["ep_id"].unique().tolist():
        idx = (roll["ep_id"] == ep).nonzero(as_tuple=True)[0]
        r = roll["rewards"][idx].unsqueeze(1); v = roll["values"][idx].unsqueeze(1)
        d = roll["dones"][idx].unsqueeze(1); lv = torch.tensor([roll["last_values"][ep]])
        a, rt = F.compute_gae(r, v, d, lv, gamma, lam)
        adv[idx] = a.squeeze(1); ret[idx] = rt.squeeze(1)
    return adv.to(device), ret.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="BC-RNN GMM checkpoint (gmmBCpaper)")
    p.add_argument("--out", default="./ppo_gmm_out")
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--goal", default="256.0,256.0,0.7853981633974483")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seeds-per-iter", type=int, default=48)   # diverse training seeds (user rule)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=5)            # PPO epochs per batch
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)           # policy (core+head)
    p.add_argument("--vlr", type=float, default=1e-3)
    p.add_argument("--eps-clip", type=float, default=0.2)      # standard PPO
    p.add_argument("--ent-coef", type=float, default=1e-3)
    p.add_argument("--target-kl", type=float, default=0.02, help="KL early-stop: skip update + break the iter's epochs once approx KL(old||new) exceeds 1.5x this (prevents on-policy trust-region blowup of the sharp BC policy)")
    p.add_argument("--bc-anchor-coef", type=float, default=0.0, help="Loss-side BC-distillation anchor: add coef * mean((logp_new - logp_ref_BC)^2) to the loss (reward/value stay clean). Pulls the policy toward the frozen BC's probabilities on the rollout actions -> prevents forgetting the high-coverage BC behavior on solved seeds while a strong advantage can still improve unsolved seeds. 0 disables.")
    p.add_argument("--curriculum", action="store_true", help="reverse-curriculum init: place the object within an annealing radius of the goal during TRAINING rollouts (eval stays full-random). Teaches short approaches first, then progressively longer ones -> targets the far-init seeds the policy fails.")
    p.add_argument("--cur-r-start", type=float, default=50.0, help="curriculum init radius (px) at iter 0")
    p.add_argument("--cur-r-end", type=float, default=260.0, help="curriculum init radius (px) at/after cur-anneal-iters (~full arena range)")
    p.add_argument("--cur-anneal-iters", type=int, default=70, help="iters over which the curriculum radius anneals start->end")
    p.add_argument("--cur-ori-start", type=float, default=0.4, help="curriculum initial-orientation-error (rad) at iter 0 (~23deg)")
    p.add_argument("--cur-ori-end", type=float, default=3.14159, help="curriculum initial-orientation-error (rad) at/after anneal (full +/-pi) -- teaches the precise rotation the mis-rotated stubborn seeds need")
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--shape-w", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--critic-warmup", type=int, default=3)
    p.add_argument("--freeze-core", action="store_true", help="tune only GMM head + value")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--eval-det-seed", type=int, default=1234)
    p.add_argument("--eval-nseeds", type=int, default=40)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    goal = [float(x) for x in args.goal.split(",")] if args.goal else None

    algo, cfg = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.device = device
    algo.nets.eval()                       # freeze BN/dropout (per-step B=1 rollouts)
    pol = GMMPolicy(algo, emb_name=args.emb, device=device, freeze_core=args.freeze_core)

    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)

    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        algo.nets["policy"].load_state_dict(sd["policy"])
        pol.value.load_state_dict(sd["value"])
        print(f"[resume] from {args.resume} (prior eval_cov={sd.get('eval_cov')})", flush=True)

    pi_params = pol.policy_parameters()
    opt_pi = torch.optim.Adam(pi_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    opt_v = torch.optim.Adam(list(pol.value.parameters()), lr=args.vlr)
    print(f"[ppo-gmm] policy_trainable={sum(q.numel() for q in pi_params)/1e6:.2f}M "
          f"freeze_core={args.freeze_core} eps_clip={args.eps_clip} ent={args.ent_coef} "
          f"lr={args.lr} seeds/iter={args.seeds_per_iter}", flush=True)

    # Frozen BC reference for the KL-to-BC anchor (RLHF-style). Loaded from the
    # SAME bootstrap ckpt; never updated -> measures the live policy's drift off BC.
    ref_pol = None
    if args.bc_anchor_coef > 0.0:
        ref_algo, _ = load_algo_from_ckpt(args.ckpt)
        ref_algo.nets = ref_algo.nets.to(device); ref_algo.nets.eval()
        ref_pol = GMMPolicy(ref_algo, emb_name=args.emb, device=device, freeze_core=True)
        for q in ref_pol.policy.parameters():
            q.requires_grad_(False)
        print(f"[ppo-gmm] BC-distillation anchor ON: bc_anchor_coef={args.bc_anchor_coef}", flush=True)

    rng = np.random.default_rng(0)
    rnn = pol.rnn_horizon

    def eval_coverage(seeds, low_noise=True, n_rep=1):
        """Coverage eval. low_noise=True -> deployable DETERMINISTIC GMM-mode (BC inference).
        low_noise=False -> the STOCHASTIC sampling policy PPO actually optimizes; n_rep
        averages several sampled rollouts/seed to tame sampling+CUDA noise. Reporting BOTH
        is mandatory: PPO can raise sampled-action reward (stoch) while the argmax mode (det)
        stays flat -> a flat det curve alone is NOT evidence of a capability ceiling."""
        prev_ln = pol.head.low_noise_eval
        pol.head.low_noise_eval = low_noise
        gen_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(args.eval_det_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.eval_det_seed)
        reps = []
        for _ in range(max(1, n_rep)):
            r = collect_rollouts_gmm(pol, env, seeds, goal, max_steps=args.max_steps, shape_w=0.0)
            reps.append(np.asarray(r["final_covs"]))
        torch.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        pol.head.low_noise_eval = prev_ln
        per_seed = np.mean(reps, axis=0)
        r = {"final_covs": list(per_seed)}
        return float(np.mean(r["final_covs"])), [round(c, 2) for c in r["final_covs"]]

    eval_seeds = list(range(args.eval_nseeds))
    bc, bl = eval_coverage(eval_seeds, low_noise=True)
    bc_stoch, _ = eval_coverage(eval_seeds, low_noise=False, n_rep=4)
    print(f"[baseline] eval det_cov={bc:.3f} stoch_cov={bc_stoch:.3f} {bl}", flush=True)
    if args.eval_only:
        return

    best = max(bc, bc_stoch)
    cur_rng = np.random.default_rng(7)
    for it in range(args.iters):
        seeds = rng.integers(0, 1_000_000, size=args.seeds_per_iter).tolist()
        init_radius_max = None
        init_ori_max = args.cur_ori_start
        if args.curriculum:
            frac = min(1.0, it / max(1, args.cur_anneal_iters))
            init_radius_max = args.cur_r_start + frac * (args.cur_r_end - args.cur_r_start)
            init_ori_max = args.cur_ori_start + frac * (args.cur_ori_end - args.cur_ori_start)
        roll = collect_rollouts_gmm(pol, env, seeds, goal, max_steps=args.max_steps,
                                    shape_w=args.shape_w, ref_pol=ref_pol,
                                    init_radius_max=init_radius_max, cur_rng=cur_rng,
                                    init_ori_max=init_ori_max)
        M = roll["a_norm"].shape[0]
        adv, ret = gae_per_episode(roll, args.gamma, args.lam, device)
        adv = F.normalize_adv(adv)
        a_norm = roll["a_norm"].to(device)
        logp_old = roll["logp_old"].to(device)
        logp_ref = roll["logp_ref"].to(device)
        warmup = it < args.critic_warmup
        st = {"pg": 0.0, "v": 0.0, "ent": 0.0, "bc": 0.0, "rho": 0.0, "clipf": 0.0, "kl": 0.0, "n": 0}
        stop_iter = False
        for _ep in range(args.epochs):
            if stop_iter:
                break
            perm = torch.randperm(M).tolist()
            for s in range(0, M, args.minibatch):
                idx = perm[s:s + args.minibatch]
                groups = build_windows(roll, idx, rnn, device)
                pg_terms, v_terms, ent_terms, bc_terms, rho_terms, clip_terms = [], [], [], [], [], []
                for (sub, window_batch) in groups:
                    lp_new, value, ent = pol.logp_value_entropy(window_batch, a_norm[sub])
                    A = adv[sub]
                    ratio = torch.exp((lp_new - logp_old[sub]).clamp(-10, 10))
                    unclipped = ratio * A
                    clipped = torch.clamp(ratio, 1 - args.eps_clip, 1 + args.eps_clip) * A
                    pg_terms.append(-torch.minimum(unclipped, clipped))      # (G,)
                    v_terms.append((value - ret[sub]) ** 2)                  # (G,)
                    ent_terms.append(ent)                                    # (G,)
                    # loss-side BC anchor: keep logp_new near the frozen BC's logp
                    # of the same rollout actions (reward/value untouched).
                    bc_terms.append((lp_new - logp_ref[sub]) ** 2)           # (G,)
                    rho_terms.append(ratio.detach())
                    clip_terms.append(((ratio - 1.0).abs() > args.eps_clip).float().detach())
                pg = torch.cat(pg_terms).mean()
                vloss = args.vcoef * torch.cat(v_terms).mean()
                ent = torch.cat(ent_terms).mean()
                bc_loss = args.bc_anchor_coef * torch.cat(bc_terms).mean()
                r_all = torch.cat(rho_terms)
                approx_kl = float(((r_all - 1.0) - r_all.clamp_min(1e-8).log()).mean())
                # KL trust-region early-stop: once the policy has drifted too far from
                # the rollout policy, SKIP this update and stop the iter's remaining
                # epochs (prevents the sharp-BC-policy collapse seen with plain PPO).
                if not warmup and approx_kl > 1.5 * args.target_kl:
                    st["kl"] += approx_kl; st["n"] = max(st["n"], 1)
                    stop_iter = True
                    break
                opt_v.zero_grad(set_to_none=True); opt_pi.zero_grad(set_to_none=True)
                if warmup:
                    vloss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.value.parameters(), args.max_grad_norm)
                    opt_v.step()
                else:
                    (pg + vloss - args.ent_coef * ent + bc_loss).backward()
                    torch.nn.utils.clip_grad_norm_(pi_params, args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(pol.value.parameters(), args.max_grad_norm)
                    opt_pi.step(); opt_v.step()
                st["pg"] += float(pg); st["v"] += float(vloss); st["ent"] += float(ent)
                st["bc"] += float(bc_loss)
                st["rho"] += float(r_all.mean())
                st["clipf"] += float(torch.cat(clip_terms).mean())
                st["kl"] += approx_kl; st["n"] += 1
        n = max(st["n"], 1)
        tag = "warmup" if warmup else "train"
        mc = float(np.mean(roll["final_covs"]))
        curtag = f" cur_r={init_radius_max:.0f} cur_ori={init_ori_max:.2f}" if init_radius_max is not None else ""
        mean_kl = st['kl'] / n
        joint_kl = mean_kl * pol.chunk   # logp is MEANed over chunk positions -> true per-decision KL ~ chunk x larger
        print(f"[it {it:03d}] [{tag}] pg={st['pg']/n:.4f} v={st['v']/n:.4f} "
              f"ent={st['ent']/n:.3f} bc={st['bc']/n:.4f} rho={st['rho']/n:.3f} clipf={st['clipf']/n:.2f} "
              f"kl={mean_kl:.4f} jkl={joint_kl:.4f} steps={st['n']}{' STOP' if stop_iter else ''}{curtag} train_cov={mc:.3f}", flush=True)
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            ec, el = eval_coverage(eval_seeds, low_noise=True)
            ec_stoch, _ = eval_coverage(eval_seeds, low_noise=False, n_rep=4)
            best_metric = max(ec, ec_stoch)
            star = " *** >0.9 ***" if best_metric > 0.9 else (" ** >0.85 **" if best_metric > 0.85 else "")
            print(f"[eval it{it:03d}] det_cov={ec:.3f} stoch_cov={ec_stoch:.3f}{star} {el}", flush=True)
            ckpt = {"policy": algo.nets["policy"].state_dict(), "value": pol.value.state_dict(),
                    "iter": it, "eval_cov": ec, "eval_cov_stoch": ec_stoch}
            torch.save(ckpt, os.path.join(args.out, f"ppo_iter{it:04d}_cov{ec:.3f}_s{ec_stoch:.3f}.pt"))
            if best_metric > best:
                best = best_metric
                torch.save(ckpt, os.path.join(args.out, "ppo_best.pt"))
    print(f"[done] best eval_cov(max det/stoch)={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
