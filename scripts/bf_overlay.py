"""TF-overlay for BATCHFLOW ckpts: pred_action (decoded) vs GT, per emb.

Usage: python scripts/bf_overlay.py --ckpt <last.ckpt> --out-dir <dir> [--tag x]
With TargetBuilder stride == chunk_len (4), each kept token's C=4 chunk tiles
the episode exactly -> full-rate predicted curve = chunks concatenated.
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
from torch.utils.data import DataLoader

from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.pl_utils.pl_data_utils import _collate_fn_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="bf_viz")
    ap.add_argument("--tag", default="bf")
    ap.add_argument("--zero-z", action="store_true")
    args = ap.parse_args()

    algo, _ = load_algo_from_ckpt(args.ckpt)
    algo.nets.eval()
    cfg = OmegaConf.load(str(Path(args.ckpt).resolve().parent.parent / ".hydra" / "config.yaml"))

    raw_batch = {}
    for name in cfg.data.train_datasets:
        ds = instantiate(cfg.data.train_datasets[name])
        ds.set_norm_stats_from(algo.norm_stats)
        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=_collate_fn_for(ds))
        raw_batch[name] = next(iter(dl))

    Path(args.out_dir).mkdir(exist_ok=True)
    with torch.no_grad():
        proc = algo.process_batch_for_training(raw_batch)
        for emb_id, _b in proc.items():
            seeded = algo._seed(emb_id, _b)
            if args.zero_z:
                # rollout-path probe: drop actions -> plan() excludes the
                # posterior (and TargetBuilder) -> CVAEHead decodes with z=0,
                # teacher-forced obs. Isolates z-starvation from closed-loop
                # compounding. Full-rate obs (no stride decimation).
                gt_actions = seeded.pop("actions")
                runnable, _ = algo.policy.plan(list(seeded.keys()))
                out = seeded
                for stage in runnable:
                    out = stage(out)
                out["actions"] = gt_actions
                # GT chunks at FULL rate (no TargetBuilder in the rollout plan)
                from egomimic.pipeline import packed as _packed
                _C = next(st.chunk_len for st in algo.policy.stages
                          if hasattr(st, "chunk_len"))
                _cu = out["cu_seqlens"].to(gt_actions.device)
                out["target"] = _packed.chunk_targets(gt_actions, _cu, _C)
            else:
                out = algo.policy(seeded)
            pred = out["pred_action"].float().cpu().numpy()   # (T_kept, C, D)
            tgt = out["target"].float().cpu().numpy()          # (T_kept, C, D)
            cu = out["cu_seqlens"].cpu().numpy()
            # first episode only; tile chunks -> full-rate curve
            n0 = int(cu[1])
            p = pred[:n0].reshape(-1, pred.shape[-1])
            g = tgt[:n0].reshape(-1, tgt.shape[-1])
            mse = float(np.mean((p - g) ** 2))
            T = p.shape[0]
            t = np.arange(T)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            for d in range(2):
                axes[d].plot(t, g[:, d], "k-", lw=2, label="ground truth")
                axes[d].plot(t, p[:, d], lw=1.2, alpha=0.9, label="batchflow pred")
                axes[d].set_title(f"emb{emb_id} action dim {d} (normalized)")
            axes[2].plot(g[:, 0], g[:, 1], "k-", lw=2)
            axes[2].plot(p[:, 0], p[:, 1], lw=1.2, alpha=0.9)
            axes[2].set_title(f"emb{emb_id} action path (x vs y)")
            axes[2].set_aspect("equal")
            fig.legend(loc="upper center", ncol=2, fontsize=9)
            fig.suptitle(f"{args.tag} — emb{emb_id} TF overlay (mse {mse:.6f})", y=1.05)
            outp = f"{args.out_dir}/{args.tag}_overlay_emb{emb_id}.png"
            fig.savefig(outp, dpi=130, bbox_inches="tight")
            _pm = ((pred - tgt) ** 2).mean(axis=(1, 2))
            print(f"  first-chunk mse={_pm[0]:.6f}  rest mse={_pm[1:].mean():.6f}  first/rest={_pm[0] / max(_pm[1:].mean(), 1e-9):.1f}x")
            print("wrote", outp, "T=", T, "mse=", mse)
    print("BF_OVERLAY_DONE")


if __name__ == "__main__":
    main()
