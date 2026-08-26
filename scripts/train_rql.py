"""Offline Reversal Q-Learning on pushshapes_paper, fine-tuning the pretrained
HPT FMPolicy. Reuses FPO's sim-eval (PushShapesEnv rollout) to measure coverage.

Run (GPU node): PYTHONPATH=<EgoVerse2> MUJOCO_GL=egl python scripts/train_rql.py --ckpt <ckpt>
"""
from __future__ import annotations
import argparse, os, glob, numpy as np, torch

from egomimic.algo.rql.rql_policy import RQLPolicy
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
            if k.startswith(p): new[k[len(p):]] = v; break
        else: new[k] = v
    algo.nets.load_state_dict(new, strict=False)
    return algo, cfg


def load_offline(ds, chunk, device, max_eps=None):
    """Load pushshapes_paper into transitions (state, image, a_chunk, r, next_state,
    next_image, nonterminal). Images decoded to [3,96,96] float[0,1]. Raw (un-normalized)
    — normalization happens in RQLPolicy.build_data / via norm_stats."""
    import zarr, simplejpeg
    eps = sorted(glob.glob(ds + "/*.zarr"))
    if max_eps: eps = eps[:max_eps]
    S, I, A, Rw, NS, NI, NT = [], [], [], [], [], [], []
    for ep in eps:
        z = zarr.open_group(ep, mode="r")
        st = np.asarray(z["observations.state"])           # (T,5)
        ac = np.asarray(z["actions"])                       # (T,2)
        rw = np.asarray(z["reward"]).reshape(-1)            # (T,)
        imgs_raw = z["observations.images.front_img_1"]
        T = st.shape[0]
        def dec(i):
            b = imgs_raw[i]
            if isinstance(b, np.ndarray):
                b = b.item()           # 0-d object array -> underlying jpeg blob
            raw = bytes(b)             # force clean 1-D python bytes for simplejpeg
            arr = simplejpeg.decode_jpeg(raw)          # [H,W,3] uint8
            return np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
        for t in range(0, T - chunk, 1):
            S.append(st[t]); I.append(dec(t))
            A.append(ac[t:t + chunk])                       # (chunk,2)
            Rw.append(float(rw[t:t + chunk].sum()))         # chunk reward = sum
            nt = t + chunk
            NS.append(st[nt]); NI.append(dec(nt))
            NT.append(0.0 if nt >= T - chunk else 1.0)
    return {
        "state": torch.tensor(np.array(S), dtype=torch.float32),
        "img": torch.tensor(np.array(I), dtype=torch.float32),
        "act": torch.tensor(np.array(A), dtype=torch.float32),
        "rew": torch.tensor(np.array(Rw), dtype=torch.float32),
        "nstate": torch.tensor(np.array(NS), dtype=torch.float32),
        "nimg": torch.tensor(np.array(NI), dtype=torch.float32),
        "nt": torch.tensor(np.array(NT), dtype=torch.float32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="./rql_out")
    p.add_argument("--ds", default="/coc/flash7/paphiwetsa3/datasets/pushshapes_paper")
    p.add_argument("--emb", default="pushshapes_sim")
    p.add_argument("--goal", default="256.0,256.0,0.7853981633974483")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--kappa", type=float, default=0.7)     # expectile
    p.add_argument("--alpha", type=float, default=3.0)     # BC coeff (key knob)
    p.add_argument("--rho", type=float, default=0.5)       # ensemble pessimism
    p.add_argument("--F", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-nseeds", type=int, default=40)
    p.add_argument("--eval-det-seed", type=int, default=1234)
    p.add_argument("--max-eps", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.max_eps, args.batch = 30, 15, 8, 32
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    goal = [float(x) for x in args.goal.split(",")]

    algo, cfg = load_algo_from_ckpt(args.ckpt)
    algo.nets = algo.nets.to(device); algo.device = device; algo.nets.eval()
    rql = RQLPolicy(algo, emb_name=args.emb, device=device, F=args.F, K=args.K)
    emb_id = rql.fpo.emb_id; ac_key = rql.fpo.ac_key

    print("[rql] loading offline data...", flush=True)
    D = load_offline(args.ds, rql.chunk, device, max_eps=(args.max_eps or None))
    N = D["state"].shape[0]; print(f"[rql] {N} transitions", flush=True)

    # optimizers: actor=flow head, critic=V-ensemble (built lazily on first feats)
    opt_pi = torch.optim.Adam(rql.trainable_policy(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    opt_v = None
    from Tsimulation.pushshapes import PushShapesEnv
    env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)

    def obsdict(state, img):  # raw -> build_data input (build_data normalizes)
        return {"state_agent_obj": state.to(device), "front_img_1": img.to(device)}

    def normed_actions(act):  # raw chunk -> normalized
        return algo.norm_stats.normalize({ac_key: act.to(device)}, emb_id)[ac_key]

    def eval_cov():
        gs = torch.get_rng_state(); cs = torch.cuda.get_rng_state_all()
        torch.manual_seed(args.eval_det_seed); torch.cuda.manual_seed_all(args.eval_det_seed)
        r = collect_rollouts(rql.fpo, env, list(range(args.eval_nseeds)), goal, max_steps=500)
        torch.set_rng_state(gs); torch.cuda.set_rng_state_all(cs)
        return float(np.mean(r["final_covs"]))

    print(f"[baseline] eval mean_cov={eval_cov():.3f}", flush=True)
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = rng.integers(0, N, size=args.batch)
        obs = obsdict(D["state"][idx], D["img"][idx])
        nobs = obsdict(D["nstate"][idx], D["nimg"][idx])
        a = normed_actions(D["act"][idx])
        r = D["rew"][idx].to(device); nt = D["nt"][idx].to(device)
        # critic
        closs, qm = rql.critic_loss(obs, a, r, nobs, nt, args.gamma, args.kappa, args.rho)
        if opt_v is None:
            opt_v = torch.optim.Adam(rql.value.parameters(), lr=args.lr)
        opt_v.zero_grad(set_to_none=True); closs.backward()
        torch.nn.utils.clip_grad_norm_(rql.value.parameters(), 10.0); opt_v.step()
        # actor
        aloss, qa = rql.actor_loss(obs, a, args.alpha)
        opt_pi.zero_grad(set_to_none=True); aloss.backward()
        torch.nn.utils.clip_grad_norm_(rql.trainable_policy(), 10.0); opt_pi.step()
        rql.soft_update(args.tau)
        if step % 200 == 0:
            print(f"[step {step}] critic={float(closs):.4f} actor={float(aloss):.4f} V={qm:.3f}", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            ec = eval_cov(); star = " *** >0.85 ***" if ec > 0.85 else ""
            print(f"[eval step{step}] mean_cov={ec:.3f}{star}", flush=True)
            torch.save({"policy": algo.nets["policy"].state_dict(), "step": step, "eval_cov": ec},
                       os.path.join(args.out, f"rql_step{step:06d}_cov{ec:.3f}.pt"))
    print("[done]")


if __name__ == "__main__":
    main()
