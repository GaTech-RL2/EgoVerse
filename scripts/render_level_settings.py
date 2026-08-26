"""Render the initial 'setting' (obstacles + T-block + goal + pusher) for all 30
PushShapes obstacle levels at a random seed, into one labeled grid PNG so the
levels can be eyeballed for solvability BEFORE any training/eval. Also calls
verify_level_solvable per level if available."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Tsimulation.pushshapes.env import PushShapesEnv
try:
    from Tsimulation.pushshapes.obstacles import verify_level_solvable
except Exception:
    verify_level_solvable = None

SEED_BASE = int(os.environ.get("SEED_BASE", "777"))
OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)

results = []
for lvl in range(30):
    env = PushShapesEnv(object_shape="T", pusher_shape="circle",
                        obstacle_level=lvl, render_mode="rgb_array")
    seed = SEED_BASE + lvl
    obs, info = env.reset(seed=seed)
    frame = env.render()
    cov0 = float(info.get("coverage", 0.0))
    solv = None
    if verify_level_solvable is not None:
        try:
            solv = verify_level_solvable(lvl)
        except Exception:
            solv = None
    # also save the per-level frame individually
    plt.imsave(os.path.join(OUT, f"level_{lvl:02d}_seed{seed}.png"), frame)
    results.append((lvl, seed, frame, cov0, solv))
    env.close()
    print(f"level {lvl:2d} seed {seed} cov0 {cov0:.3f} solvable {solv}", flush=True)

fig, axes = plt.subplots(5, 6, figsize=(21, 18))
for ax, (lvl, seed, frame, cov0, solv) in zip(axes.flat, results):
    ax.imshow(frame)
    t = f"L{lvl}  seed {seed}  cov0={cov0:.2f}"
    if solv is not None:
        t += f"  solv={solv}"
    ax.set_title(t, fontsize=9)
    ax.axis("off")
fig.suptitle(
    f"PushShapes 30 obstacle levels — random init (object=T, pusher=circle, SEED_BASE={SEED_BASE})",
    fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(OUT, "level_settings_0-29_grid.png")
fig.savefig(out, dpi=110)
print("SAVED", out, flush=True)
