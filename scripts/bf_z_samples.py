"""Multimodality probe: decode with K prior z-samples under teacher-forced obs.

Per episode: GT (black), z=0 (dashed), K z~N(0,I) decodes (thin colors),
posterior-z decode (green dashed) for reference. Prints per-sample MSE and a
diversity score (mean pairwise L2 between sampled trajectories).
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.pipeline import packed as packed_mod


def run_plan(algo, seed):
    runnable, _ = algo.policy.plan(list(seed.keys()))
    out = dict(seed)
    for st in runnable:
        out = st(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--eps-per-emb", type=int, default=2)
    ap.add_argument("--out-dir", default="bf_viz")
    ap.add_argument("--tag", default="zsamples")
    args = ap.parse_args()

    algo, _ = load_algo_from_ckpt(args.ckpt, None)
    algo.nets.cuda(); algo.nets.eval()
    cfg = OmegaConf.load(str(Path(args.ckpt).resolve().parent.parent / ".hydra" / "config.yaml"))
    dm = instantiate(cfg.data); dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)
    torch.manual_seed(0)

    for emb_id, eb in batch.items():
        seed0 = algo._seed(emb_id, eb)
        seed0 = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in seed0.items()}
        gt = seed0.pop("actions")
        cu = seed0["cu_seqlens"]
        n_eps = min(args.eps_per_emb, int(cu.numel()) - 1)
        B = int(cu.numel()) - 1
        zdim = 32

        def decode(z):
            s = dict(seed0)
            if z is not None:
                s["cvae_z"] = z
                s["cvae_z_tok"] = packed_mod.broadcast_per_episode(z, cu)
            with torch.no_grad():
                out = run_plan(algo, s)
            return out["pred_action"][:, 0, :].float().cpu().numpy()  # (T, D) first-action curve

        curves = {"z0": decode(torch.zeros(B, zdim, device=cu.device))}
        for k in range(args.k):
            curves[f"s{k}"] = decode(torch.randn(B, zdim, device=cu.device))
        # posterior reference (full pipeline with actions)
        with torch.no_grad():
            full = algo.policy({**seed0, "actions": gt})
        gt_np = gt.float().cpu().numpy()
        cu_np = cu.cpu().numpy()

        fig, axes = plt.subplots(1, n_eps, figsize=(7 * n_eps, 6))
        axes = np.atleast_1d(axes)
        div_all = []
        for e in range(n_eps):
            a, b = int(cu_np[e]), int(cu_np[e + 1])
            ax = axes[e]
            samp = [curves[f"s{k}"][a:b] for k in range(args.k)]
            pd = [np.linalg.norm(samp[i] - samp[j], axis=-1).mean()
                  for i in range(args.k) for j in range(i + 1, args.k)]
            div = float(np.mean(pd)); div_all.append(div)
            for k in range(args.k):
                ax.plot(samp[k][:, 0], samp[k][:, 1], lw=0.8, alpha=0.7)
            ax.plot(curves["z0"][a:b, 0], curves["z0"][a:b, 1], "b--", lw=1.6, label="z=0")
            ax.plot(gt_np[a:b, 0], gt_np[a:b, 1], "k-", lw=2.2, label="GT")
            ax.set_title(f"emb{emb_id} ep{e}  sample-diversity={div:.4f}")
            ax.legend(fontsize=8)
        out = Path(args.out_dir) / f"{args.tag}_emb{emb_id}.png"
        fig.suptitle(f"{args.tag} emb{emb_id}: {args.k} prior z-samples (TF obs)")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"wrote {out}  mean diversity={np.mean(div_all):.5f}")
    print("Z_SAMPLES_DONE")


if __name__ == "__main__":
    main()
