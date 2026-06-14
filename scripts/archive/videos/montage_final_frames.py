import glob, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.io as tvio

D = sys.argv[1]
files = sorted(glob.glob(f"{D}/rolloutA_*.mp4"))
if not files:
    print("NO VIDEOS in", D); sys.exit(1)
n = len(files)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = np.array(axes).reshape(-1)
for i, f in enumerate(files):
    try:
        v, _, _ = tvio.read_video(f, pts_unit="sec")  # (T,H,W,3) uint8
        last = v[-1].numpy()
        axes[i].imshow(last)
        nm = f.split("/")[-1].replace("rolloutA_", "").replace(".mp4", "")
        axes[i].set_title(f"{nm}  (T={v.shape[0]})", fontsize=9)
    except Exception as e:
        axes[i].set_title(f"ERR {e}", fontsize=7)
    axes[i].axis("off")
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("B (closed-loop GMM, ep-400) — FINAL frame of each rollout (look: is the T at the goal?)", fontsize=12)
fig.tight_layout()
out = f"{D}/final_frames_montage.png"
fig.savefig(out, dpi=90, bbox_inches="tight")
print("SAVED", out)
print("FILES:", [f.split("/")[-1] for f in files])
