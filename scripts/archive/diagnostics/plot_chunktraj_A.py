#!/usr/bin/env python3
"""Plot chunk-A executed (predicted) cursor trajectories from chunktraj npz dumps.
One combined PNG: per episode a row of 2 panels.
  (a) 2D executed path x vs y, scatter+line colored by timestep, chunk-boundary markers.
  (b) x(t), y(t) vs t, vertical dashed gridlines every K steps.
Also prints raw numeric diagnostics (within-chunk vs boundary step magnitudes).
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

D = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/chunktraj_A"
OUT = os.path.join(D, "chunkA_predicted_path.png")

files = sorted(glob.glob(os.path.join(D, "chunktraj_ep*.npz")),
               key=lambda p: int(p.split("ep")[-1].split(".")[0]))
files = files[:4]
assert files, "no chunktraj npz found"

eps = []
for f in files:
    z = np.load(f)
    a = np.asarray(z["actions"], dtype=np.float64)   # (T,2)
    K = int(z["K"])
    cov = float(z["coverage"])
    epi = int(f.split("ep")[-1].split(".")[0])
    eps.append((epi, a, K, cov))

nrow = len(eps)
fig, axes = plt.subplots(nrow, 2, figsize=(13, 3.4 * nrow), squeeze=False)

print("=== RAW NUMERIC DIAGNOSTICS (world coords, units = px in 512 world) ===")
for r, (epi, a, K, cov) in enumerate(eps):
    T = a.shape[0]
    t = np.arange(T)
    d = np.diff(a, axis=0)                 # (T-1,2) step vectors a_{t+1}-a_t
    step_mag = np.linalg.norm(d, axis=1)   # (T-1,)
    # boundary index j means the step from t=j-1 -> t=j where j is a multiple of K.
    # i.e. step index s (a_s -> a_{s+1}); the step that CROSSES a boundary is the
    # one landing on t=K,2K,... -> step index s = K-1, 2K-1, ... (0-based diff).
    bnd_steps = np.array([s for s in range(T - 1) if (s + 1) % K == 0], dtype=int)
    in_steps = np.array([s for s in range(T - 1) if (s + 1) % K != 0], dtype=int)
    mb = float(step_mag[bnd_steps].mean()) if bnd_steps.size else float("nan")
    mi = float(step_mag[in_steps].mean()) if in_steps.size else float("nan")
    maxb = float(step_mag[bnd_steps].max()) if bnd_steps.size else float("nan")
    maxi = float(step_mag[in_steps].max()) if in_steps.size else float("nan")
    print(f"ep{epi}: T={T} K={K} cov={cov:.3f} | "
          f"mean|a_t+1-a_t| within-chunk={mi:.2f} boundary={mb:.2f} | "
          f"max within={maxi:.2f} boundary={maxb:.2f} | "
          f"overall mean={step_mag.mean():.2f} std={step_mag.std():.2f} "
          f"min={step_mag.min():.2f} max={step_mag.max():.2f}")

    # ---- panel (a): 2D path colored by timestep ----
    axa = axes[r][0]
    pts = a.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=t[:-1], linewidth=1.4, alpha=0.9)
    axa.add_collection(lc)
    sc = axa.scatter(a[:, 0], a[:, 1], c=t, cmap="viridis", s=8, zorder=3)
    bnd_t = np.arange(0, T, K)
    axa.scatter(a[bnd_t, 0], a[bnd_t, 1], facecolors="none", edgecolors="red",
                s=70, linewidths=1.4, zorder=4, label="chunk boundary (t=0,K,2K,...)")
    axa.scatter([a[0, 0]], [a[0, 1]], c="lime", s=60, marker="s", zorder=5, label="start")
    axa.scatter([a[-1, 0]], [a[-1, 1]], c="black", s=60, marker="X", zorder=5, label="end")
    axa.set_xlabel("cursor x (world px)")
    axa.set_ylabel("cursor y (world px)")
    axa.set_title(f"ep{epi}  2D executed path  (cov={cov:.3f})")
    axa.legend(fontsize=7, loc="best")
    axa.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=axa, label="timestep t", fraction=0.046, pad=0.04)

    # ---- panel (b): x(t), y(t) ----
    axb = axes[r][1]
    axb.plot(t, a[:, 0], color="tab:blue", lw=1.2, label="x(t)")
    axb.plot(t, a[:, 1], color="tab:orange", lw=1.2, label="y(t)")
    for bt in range(0, T + 1, K):
        axb.axvline(bt, color="gray", ls="--", lw=0.7, alpha=0.6)
    axb.set_xlabel("timestep t")
    axb.set_ylabel("cursor coord (world px)")
    axb.set_title(f"ep{epi}  x(t),y(t)  (dashed=chunk bnd every {K})")
    axb.legend(fontsize=8, loc="best")

fig.suptitle("chunk-A executed (predicted) cursor trajectory — chunk_openloop rollout",
             fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("SAVED_PNG", OUT)
