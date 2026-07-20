"""Compare trained policies on the real-robot teleop validation set.

Standard offline comparator for the pickplace task: per-checkpoint, per-block MAE
on the teleop LeRobot dataset (real images + REAL measured proprio), split into
nav / manip phases (manip = GT chunk base |dx|+|dy| below --manip-thresh).

Usage (from repo root, emimic venv):
  python egomimic/scripts/eval_on_teleop.py \
      --ckpt crop100=logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt \
      --ckpt pickplace_v1=logs/aria_pickplace/pickplace_v1_2k/checkpoints/last.ckpt \
      [--dataset datasets/rby1_teleop_pickplace_val_rgb] [--frames-per-ep 40]
      [--proprio real|zero] [--out /path/out_dir]

Outputs: printed table, <out>/results.json, <out>/blocks_chart.png, and a
pred-vs-GT trajectory overlay for the first checkpoint on --overlay-episode.
"""

import argparse
import json
import os

import numpy as np
import torch

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.serving.egoverse_policy import EgoVersePolicy
from egomimic.rldb.utils import FolderRLDBDataset

H = 32
DT = {"actions.joint_base_torso_head_arm_hand": [round(0.1 * i, 1) for i in range(H)]}
BLOCKS = [("base", 0, 3), ("torso", 3, 9), ("head", 9, 11), ("l_arm", 11, 18),
          ("r_arm", 18, 25), ("l_hand", 25, 37), ("r_hand", 37, 49)]


def to_np(v):
    v = v.numpy() if torch.is_tensor(v) else np.asarray(v)
    return np.asarray(v, np.float32)


def rgb224(sample):
    img = to_np(sample["obs.aria_image"])
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = img * 255
    return img.astype(np.uint8)


def obs_of(sample, proprio_mode):
    jp = to_np(sample["obs.robot0_joint_pos_no_wheel"]).ravel()
    hl = to_np(sample["obs.hand_left_qpos"]).ravel()
    hr = to_np(sample["obs.hand_right_qpos"]).ravel()
    if proprio_mode == "zero":
        jp, hl, hr = np.zeros_like(jp), np.zeros_like(hl), np.zeros_like(hr)
    return {"front_img_1": rgb224(sample)[..., ::-1].copy(),  # serving contract: BGR
            "robot0_joint_pos": jp, "hand_left_qpos": hl, "hand_right_qpos": hr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="label=path/to/last.ckpt (repeatable)")
    ap.add_argument("--dataset", default="datasets/rby1_teleop_pickplace_val_rgb")
    ap.add_argument("--frames-per-ep", type=int, default=40)
    ap.add_argument("--proprio", choices=["real", "zero"], default="real")
    ap.add_argument("--manip-thresh", type=float, default=0.05,
                    help="GT chunk base |dx|+|dy| (m) below which a frame is manip phase")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overlay-episode", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out or os.path.join("eval_results", "teleop_" + "_".join(
        c.split("=")[0] for c in args.ckpt))
    os.makedirs(out_dir, exist_ok=True)

    ds = FolderRLDBDataset(folder_path=args.dataset, embodiment="rby1", mode="train",
                           valid_ratio=0.0, local_files_only=True, delta_timestamps=DT)
    # evenly sample the flat index space, recording episode ids as we go
    idxs = np.linspace(0, len(ds) - 1, min(len(ds), args.frames_per_ep * 8)).astype(int)
    samples, gts, eps = [], [], []
    for i in idxs:
        s = ds[int(i)]
        samples.append(s)
        gts.append(to_np(s["actions.joint_base_torso_head_arm_hand"]))
        eps.append(int(to_np(s.get("episode_index", 0)).ravel()[0]) if "episode_index" in s else 0)
    gts = np.stack(gts)  # (N, 32, 49)
    eps = np.asarray(eps)
    manip = np.abs(gts[:, :, 0:2]).sum(axis=(1, 2)) < args.manip_thresh
    print(f"dataset={args.dataset}  frames={len(samples)}  episodes={sorted(set(eps.tolist()))}")
    print(f"manip-phase frames: {manip.sum()}/{len(manip)} (thresh {args.manip_thresh} m)")
    # freeze-baseline reference for the shape metrics (constant prediction):
    # centered -> GT's own centered magnitude; velocity -> GT's velocity magnitude
    gc = np.abs(gts - gts.mean(axis=1, keepdims=True)).mean()
    gv = np.abs(np.diff(gts, axis=1)).mean()
    print(f"freeze-baseline shape reference: centered={gc:.4f}  velocity={gv:.4f} "
          "(a policy must be BELOW these to beat 'stand still' on shape)")

    results = {}
    first_preds = None
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        m = ModelWrapper.load_from_checkpoint(path, weights_only=False)
        policy = EgoVersePolicy(m)
        preds = []
        for s in samples:
            p = np.asarray(policy.infer(obs_of(s, args.proprio))["actions"], np.float32)
            preds.append(p[0] if p.ndim == 3 else p)
        preds = np.stack(preds)
        if first_preds is None:
            first_preds = preds
        err = np.abs(preds - gts)
        # shape metrics: offset-invariant (centered) and pose-invariant (velocity)
        cent_p = preds - preds.mean(axis=1, keepdims=True)
        cent_g = gts - gts.mean(axis=1, keepdims=True)
        cerr = np.abs(cent_p - cent_g)
        verr = np.abs(np.diff(preds, axis=1) - np.diff(gts, axis=1))
        r = {"overall_mae": float(err.mean()),
             "manip_mae": float(err[manip].mean()) if manip.any() else None,
             "nav_mae": float(err[~manip].mean()) if (~manip).any() else None,
             "t1_mae": float(err[:, 0].mean()),
             "short8_mae": float(err[:, :8].mean()),
             "centered_mae": float(cerr.mean()),
             "velocity_mae": float(verr.mean()),
             "blocks": {}, "blocks_manip": {}, "blocks_centered": {}, "blocks_velocity": {},
             "per_episode_mae": {}}
        for bn, a, b in BLOCKS:
            r["blocks"][bn] = float(err[:, :, a:b].mean())
            r["blocks_centered"][bn] = float(cerr[:, :, a:b].mean())
            r["blocks_velocity"][bn] = float(verr[:, :, a:b].mean())
            if manip.any():
                r["blocks_manip"][bn] = float(err[manip][:, :, a:b].mean())
        for e in sorted(set(eps.tolist())):
            r["per_episode_mae"][str(e)] = float(err[eps == e].mean())
        results[label] = r
        del m, policy
        print(f"\n=== {label} (proprio={args.proprio}) ===")
        print(f"  overall={r['overall_mae']:.4f}  nav={r['nav_mae']}  manip={r['manip_mae']}"
              f"  t1={r['t1_mae']:.4f}  short8={r['short8_mae']:.4f}"
              f"  centered={r['centered_mae']:.4f}  velocity={r['velocity_mae']:.4f}")
        print("  blocks:        " + "  ".join(f"{bn}={r['blocks'][bn]:.4f}" for bn, _, _ in BLOCKS))
        if r["blocks_manip"]:
            print("  blocks(manip): " + "  ".join(f"{bn}={r['blocks_manip'][bn]:.4f}" for bn, _, _ in BLOCKS))
        print("  blocks(cent):  " + "  ".join(f"{bn}={r['blocks_centered'][bn]:.4f}" for bn, _, _ in BLOCKS))
        print("  blocks(vel):   " + "  ".join(f"{bn}={r['blocks_velocity'][bn]:.4f}" for bn, _, _ in BLOCKS))
        print("  per-episode:   " + "  ".join(f"ep{e}={v:.4f}" for e, v in r["per_episode_mae"].items()))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    # charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    x = np.arange(len(BLOCKS))
    w = 0.8 / max(1, len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), dpi=120)
    for pi, (title, key) in enumerate([("all frames", "blocks"), ("manip phase only", "blocks_manip")]):
        ax = axes[pi]
        for li, lab in enumerate(labels):
            vals = [results[lab][key].get(bn, np.nan) for bn, _, _ in BLOCKS]
            ax.bar(x + (li - (len(labels) - 1) / 2) * w, vals, w * 0.9, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BLOCKS], fontsize=9)
        ax.set_title(f"per-block MAE — {title}"); ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        if pi == 0:
            ax.set_ylabel("MAE (rad / m)"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blocks_chart.png"), bbox_inches="tight")

    # pred-vs-GT overlay for the first ckpt on one episode (chunk t+1 stitching)
    sel = np.where(eps == args.overlay_episode)[0]
    if len(sel) > 2:
        fig, axes = plt.subplots(len(BLOCKS), 1, figsize=(12, 2.0 * len(BLOCKS)), sharex=True)
        order = sel[np.argsort(idxs[sel])]
        t = np.arange(len(order))
        for bi, (bn, a, b) in enumerate(BLOCKS):
            ax = axes[bi]
            ax.plot(t, gts[order][:, 1, a:b].mean(-1), "k-", lw=1.2, label="GT (t+1)")
            ax.plot(t, first_preds[order][:, 1, a:b].mean(-1), "r--", lw=1.1,
                    label=f"{labels[0]} (t+1)")
            ax.set_ylabel(bn, fontsize=8); ax.grid(alpha=0.3)
            if bi == 0:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel(f"sampled frames along episode {args.overlay_episode}")
        plt.suptitle(f"block-mean pred vs GT — {labels[0]}", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overlay_ep.png"), bbox_inches="tight")

    print(f"\nwrote {out_dir}/results.json, blocks_chart.png, overlay_ep.png")


if __name__ == "__main__":
    main()
