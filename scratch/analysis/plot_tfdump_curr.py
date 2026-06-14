import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

D = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/tf_dump_react_curr"
files = sorted(glob.glob(f"{D}/tf_ep*.npz"))
n = len(files)
fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
for i, f in enumerate(files):
    d = np.load(f)
    pred, gt = d["pred"], d["gt"]
    T = len(gt)
    err = np.sqrt(((pred - gt) ** 2).sum(1))
    mae = np.abs(pred - gt).mean()

    ax = axes[0, i]
    ax.plot(gt[:, 0], gt[:, 1], "-", color="green", lw=2.2, alpha=0.85, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], "-", color="red", lw=1.5, alpha=0.85, label="pred")
    ax.scatter([gt[0, 0]], [gt[0, 1]], c="lime", marker="o", s=70, edgecolor="k", zorder=5)
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], c="darkgreen", marker="X", s=90, edgecolor="k", zorder=5)
    ax.set_title(f"ep{i}  T={T}  MAE={mae:.1f}px  RMSE={err.mean():.1f}px")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(fontsize=8)
    ax.set_xlabel("cursor x"); ax.set_ylabel("cursor y")

    ax2 = axes[1, i]
    ax2.plot(err, color="purple", lw=1.2)
    ax2.axhline(err.mean(), color="gray", ls="--", lw=1, label=f"mean {err.mean():.1f}px")
    ax2.axvline(32, color="orange", ls=":", lw=1.5, label="frame 32")
    ax2.set_title(f"ep{i} per-frame L2 error")
    ax2.set_xlabel("frame"); ax2.set_ylabel("L2 px err"); ax2.legend(fontsize=8)

fig.suptitle(
    "reactiveB_curr_lr2p5e7 (reactive curriculum K1->32, LR 2.5e-7, epoch 399) -- TEACHER-FORCED: GT (green) vs pred (red)\n"
    "BARELY-TRAINED (LR 40x below base): expect poor tracking even teacher-forced",
    fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{D}/WAR_reactiveB_curr_lr2p5e7_tf_gt_vs_pred.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("SAVED", out)
for i, f in enumerate(files):
    d = np.load(f); pred, gt = d["pred"], d["gt"]
    err = np.sqrt(((pred - gt) ** 2).sum(1))
    print(f"ep{i}: MAE={np.abs(pred-gt).mean():.2f}px RMSE={err.mean():.2f}px")
