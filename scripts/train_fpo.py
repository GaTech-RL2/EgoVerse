"""FPO (2507.21053) + FPO++ ASPO (2602.02481 §III) + DiPOD (2606.13795) RL
fine-tuning of a pretrained HPT FMPolicy on PushShapes. EgoVerse2 port.

Faithful to the two papers' continuous-control recipe:
  - FPO ratio from the conditional-flow-matching loss (per-sample, FPO++ Eq.10).
  - ASPO asymmetric trust region (FPO++ Eq.11-13): PPO-clip for A>=0, SPO
    quadratic penalty for A<0 (no zero-gradient dead zone -> no erosion).
  - DiPOD: an INITIAL self-distillation stage (tighten ELBO on the pretrained
    policy's own rollouts) before policy-gradient. No per-batch beta for
    continuous control (that's DiPOD's diffusion-LLM form).
  - GAE advantages with a learned critic (FPO/FPO++ default). eps_clip=0.01.

Run (GPU node): PYTHONPATH=<EgoVerse2> MUJOCO_GL=egl python scripts/train_fpo.py --ckpt <ckpt>
"""
from __future__ import annotations
import argparse, os, numpy as np, torch

from egomimic.algo.fpo.fpo_policy import FPOPolicy
from egomimic.algo.fpo import fpo_core as F
from egomimic.algo.fpo.rollout import collect_rollouts, _reset_fixed_goal
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
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
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


def collate_obs(obs_list, idx, device):
    keys = obs_list[0].keys()
    return {k: torch.cat([obs_list[i][k] for i in idx], dim=0).to(device) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="./fpo_out")
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--goal", default="256.0,256.0,0.7853981633974483")
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--seeds-per-iter", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--n-mc", type=int, default=16)            # FPO++ MT default
    p.add_argument("--epochs", type=int, default=5)           # FPO++ MT default
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)          # head-only finetune
    p.add_argument("--vlr", type=float, default=1e-3)
    p.add_argument("--critic-warmup", type=int, default=5)
    p.add_argument("--freeze-trunk", action="store_true")
    p.add_argument("--eps-clip", type=float, default=0.01)    # FPO++ MT default (tight)
    p.add_argument("--beta", type=float, default=0.0)         # per-batch DiPOD reg (0 = continuous-control form)
    p.add_argument("--distill-steps", type=int, default=200)  # DiPOD initial self-distillation
    p.add_argument("--distill-seeds", type=int, default=16)
    p.add_argument("--explore-std", type=float, default=0.1)
    p.add_argument("--shape-w", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--vcoef", type=float, default=0.25)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", default="", help="prior fpo_iter*.pt to resume policy+value from")
    p.add_argument("--eval-det-seed", type=int, default=1234, help="fixed torch seed for eval rollouts (reproducible eval through flow-sampling noise)")
    p.add_argument("--eval-nseeds", type=int, default=40, help="number of fixed held-out eval seeds (0..N-1) for the coverage metric")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    goal = [float(x) for x in args.goal.split(",")] if args.goal else None

    algo, cfg = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.device = device
    algo.nets.eval()  # freeze BN/dropout (per-step B=1 rollouts); gradients still flow
    fpo = FPOPolicy(algo, emb_name=args.emb, device=device)

    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)

    obs, _ = _reset_fixed_goal(env, 0, goal, args.emb)
    _ = fpo.sample(fpo.build_data(_OBS_FORMATTERS[args.emb](obs, device)))
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        algo.nets["policy"].load_state_dict(sd["policy"])
        if "value" in sd and fpo.value is not None:
            fpo.value.load_state_dict(sd["value"])
        print(f"[resume] loaded policy(+value) from {args.resume} (prior eval_cov={sd.get('eval_cov')})", flush=True)
    if args.freeze_trunk:
        for q in fpo.policy.parameters(): q.requires_grad_(False)
        for q in fpo.head.parameters(): q.requires_grad_(True)
        pi_params = [q for q in fpo.head.parameters() if q.requires_grad]
        print("[fpo] FROZEN trunk+encoder; training head only")
    else:
        pi_params = list(fpo.policy.parameters())
    opt_pi = torch.optim.Adam(pi_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    opt_v = torch.optim.Adam(list(fpo.value.parameters()), lr=args.vlr)
    print(f"[fpo] policy_trainable={sum(q.numel() for q in pi_params)/1e6:.2f}M "
          f"eps_clip={args.eps_clip} n_mc={args.n_mc} distill={args.distill_steps} beta={args.beta}", flush=True)

    rng = np.random.default_rng(0)

    def draw_mc(M):
        return (torch.rand(M, args.n_mc, device=device) * 0.999 + 0.001,
                torch.randn(M, args.n_mc, fpo.chunk, fpo.action_dim, device=device))

    def eval_coverage(seeds):
        # fixed torch seed -> reproducible flow-sampling noise, so the SAME policy
        # scores identically across iters and a real improvement shows through the
        # otherwise ~0.1 stochastic-sampling eval noise.
        gen_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(args.eval_det_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.eval_det_seed)
        r = collect_rollouts(fpo, env, seeds, goal, max_steps=args.max_steps)  # explore_std=0
        torch.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return float(np.mean(r["final_covs"])), [round(c, 2) for c in r["final_covs"]]

    eval_seeds = list(range(args.eval_nseeds))   # held-out generalization-eval seeds (NOT trained on)
    bc, bl = eval_coverage(eval_seeds)
    print(f"[baseline] eval mean_cov={bc:.3f} {bl}", flush=True)
    if args.eval_only:
        return

    # ---------- DiPOD initial self-distillation stage (tighten ELBO on pretrained-policy rollouts) ----------
    # Skip on resume: distillation is a ONE-TIME bound-tightening at the start; re-running
    # it on every preemption-restart would repeatedly perturb the (already-tight) policy.
    if args.distill_steps > 0 and not args.resume:
        dseeds = rng.integers(0, 1_000_000, size=args.distill_seeds).tolist()
        droll = collect_rollouts(fpo, env, dseeds, goal, max_steps=args.max_steps, explore_std=0.0)
        Md = droll["a_norm"].shape[0]; da = droll["a_norm"].to(device)
        print(f"[distill] {Md} ref transitions; {args.distill_steps} steps", flush=True)
        for ds in range(args.distill_steps):
            idx = rng.integers(0, Md, size=min(args.minibatch, Md)).tolist()
            data = fpo.build_data(collate_obs(droll["obs_list"], idx, device))
            tt, nn = draw_mc(len(idx))
            per, _ = fpo.cfm_per_sample_loss(data, da[idx], tt, nn)
            loss = per.mean()                       # min CFM = max ELBO (tighten bound)
            opt_pi.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pi_params, args.max_grad_norm); opt_pi.step()
            if ds % 50 == 0: print(f"  [distill {ds}] cfm={float(loss):.4f}", flush=True)
        dc, _ = eval_coverage(eval_seeds)
        print(f"[after-distill] eval mean_cov={dc:.3f}", flush=True)

    # ---------- FPO++ policy gradient ----------
    for it in range(args.iters):
        seeds = rng.integers(0, 1_000_000, size=args.seeds_per_iter).tolist()
        roll = collect_rollouts(fpo, env, seeds, goal, max_steps=args.max_steps,
                                shape_w=args.shape_w, explore_std=args.explore_std)
        M = roll["a_norm"].shape[0]
        adv, ret = gae_per_episode(roll, args.gamma, args.lam, device)
        adv = F.normalize_adv(adv)
        a_norm = roll["a_norm"].to(device)
        time, noise = draw_mc(M)
        with torch.no_grad():
            l_old = torch.empty(M, args.n_mc, device=device)
            for s in range(0, M, args.minibatch):
                idx = list(range(s, min(s + args.minibatch, M)))
                data = fpo.build_data(collate_obs(roll["obs_list"], idx, device))
                lp, _ = fpo.cfm_per_sample_loss(data, a_norm[idx], time[idx], noise[idx])
                l_old[idx] = lp
        warmup = it < args.critic_warmup
        st = {"aspo": 0.0, "v": 0.0, "rho": 0.0, "n": 0}
        for _ep in range(args.epochs):
            perm = torch.randperm(M).tolist()
            for s in range(0, M, args.minibatch):
                idx = perm[s:s + args.minibatch]
                data = fpo.build_data(collate_obs(roll["obs_list"], idx, device))
                l_new, value = fpo.cfm_per_sample_loss(data, a_norm[idx], time[idx], noise[idx])
                vloss = F.value_loss(value, ret[idx], args.vcoef)
                opt_v.zero_grad(set_to_none=True); opt_pi.zero_grad(set_to_none=True)
                if warmup:
                    vloss.backward()
                    torch.nn.utils.clip_grad_norm_(fpo.value.parameters(), args.max_grad_norm)
                    opt_v.step()
                    aspo = torch.zeros((), device=device); rho = torch.ones(1, device=device)
                else:
                    rho = F.fpo_ratio_persample(l_old[idx], l_new)        # [b,Nmc]
                    aspo = F.aspo_loss(rho, adv[idx], args.eps_clip)
                    reg = args.beta * l_new.mean() if args.beta > 0 else 0.0
                    (aspo + reg + vloss).backward()
                    torch.nn.utils.clip_grad_norm_(pi_params, args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(fpo.value.parameters(), args.max_grad_norm)
                    opt_pi.step(); opt_v.step()
                st["aspo"] += float(aspo); st["v"] += float(vloss)
                st["rho"] += float(rho.mean()); st["n"] += 1
        n = max(st["n"], 1)
        tag = "warmup" if warmup else "train"
        print(f"[it {it:03d}] [{tag}] aspo={st['aspo']/n:.4f} v={st['v']/n:.4f} rho={st['rho']/n:.3f}", flush=True)
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            ec, el = eval_coverage(eval_seeds)
            star = " *** >0.85 ***" if ec > 0.85 else ""
            print(f"[eval it{it:03d}] eval mean_cov={ec:.3f}{star} {el}", flush=True)
            torch.save({"policy": algo.nets["policy"].state_dict(), "value": fpo.value.state_dict(),
                        "iter": it, "eval_cov": ec}, os.path.join(args.out, f"fpo_iter{it:04d}_cov{ec:.3f}.pt"))
    print("[done]")


if __name__ == "__main__":
    main()
