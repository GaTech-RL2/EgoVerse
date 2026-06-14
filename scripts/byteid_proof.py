"""Prove the env patch is byte-identical on the DEFAULT path.

Run a fixed-seed scripted rollout with default args (pusher_shape='circle',
pusher_radius=None -> 15.0) under (A) the PATCHED working-tree Tsimulation and
(B) the ORIGINAL git-HEAD Tsimulation extracted to a temp package, and assert
every obs field and every rendered image is bit-identical.

We import the two env modules from two different package roots so both live in
the same process for an apples-to-apples diff.
"""
import os, sys, subprocess, tempfile, importlib.util, shutil
import numpy as np

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ---- extract ORIGINAL (git HEAD) Tsimulation into a temp dir ----
orig_root = tempfile.mkdtemp(prefix="orig_tsim_")
# copy the whole working-tree Tsimulation, then overwrite the 3 patched files
# with their git-HEAD versions -> a clean ORIGINAL package.
shutil.copytree(os.path.join(REPO, "Tsimulation"), os.path.join(orig_root, "Tsimulation"))
for rel in ("pushshapes/env.py", "pushshapes/shapes.py", "pushshapes/render.py"):
    blob = subprocess.check_output(
        ["git", "-C", REPO, "show", f"HEAD:Tsimulation/{rel}"]
    )
    with open(os.path.join(orig_root, "Tsimulation", rel), "wb") as f:
        f.write(blob)

def load_env(root, tag):
    # import Tsimulation.pushshapes.env from `root` under a unique module alias
    sys.path.insert(0, root)
    # purge any cached Tsimulation modules so the chosen root wins
    for m in list(sys.modules):
        if m == "Tsimulation" or m.startswith("Tsimulation."):
            del sys.modules[m]
    import Tsimulation.pushshapes.env as envmod
    import importlib
    importlib.reload(envmod)
    sys.path.pop(0)
    return envmod.PushShapesEnv

def scripted_rollout(EnvCls, seed, n_steps=60):
    from Tsimulation.collect.scripted_collect import scripted_action
    env = EnvCls(object_shape="T", pusher_shape="circle", image_size=96)
    obs, _ = env.reset(seed=seed)
    frames = [obs["image"].copy()]
    states = [np.concatenate([obs["agent_pos"], obs["object_pose"]]).copy()]
    for _ in range(n_steps):
        a = scripted_action(
            agent_xy=np.asarray(obs["agent_pos"], float),
            object_xy=np.asarray(obs["object_pose"], float)[:2],
            goal_xy=np.asarray(obs["goal_pose"], float)[:2],
            world_size=512.0,
        )
        obs, r, term, trunc, info = env.step(np.asarray(a, float))
        frames.append(obs["image"].copy())
        states.append(np.concatenate([obs["agent_pos"], obs["object_pose"]]).copy())
        if term:
            break
    return np.stack(frames), np.stack(states)

# IMPORTANT: load ORIGINAL first, run it, then PATCHED — capturing arrays before
# the module swap (the scripted_collect import resolves against whichever root is active).
SEEDS = [0, 1, 2, 7, 42]
print("=== BYTE-IDENTICAL PROOF (default circle, pusher_radius=None) ===")
all_ok = True
for seed in SEEDS:
    EnvOrig = load_env(orig_root, "orig")
    f_o, s_o = scripted_rollout(EnvOrig, seed)
    EnvPatched = load_env(REPO, "patched")
    f_p, s_p = scripted_rollout(EnvPatched, seed)
    same_f = (f_o.shape == f_p.shape) and np.array_equal(f_o, f_p)
    same_s = (s_o.shape == s_p.shape) and np.array_equal(s_o, s_p)
    fdiff = -1 if same_f else int(np.abs(f_o.astype(int) - f_p.astype(int)).max()) if f_o.shape == f_p.shape else 9999
    sdiff = 0.0 if same_s else float(np.abs(s_o - s_p).max()) if s_o.shape == s_p.shape else 9999
    ok = same_f and same_s
    all_ok &= ok
    print(f"  seed={seed:3d}  frames_equal={same_f} (maxpxdiff={fdiff})  states_equal={same_s} (maxdiff={sdiff})  T_o={len(f_o)} T_p={len(f_p)}  {'OK' if ok else 'MISMATCH'}")

print(f"\nDEFAULT-PATH BYTE-IDENTICAL: {all_ok}")

# ---- sanity: circle_small actually produces a SMALLER pusher footprint ----
print("\n=== circle_small renders a smaller disk (sanity) ===")
EnvPatched = load_env(REPO, "patched2")
import Tsimulation.pushshapes.shapes as shp
print("  PUSHER_RADII:", shp.PUSHER_RADII)
def red_area(EnvCls, pusher_shape):
    env = EnvCls(object_shape="T", pusher_shape=pusher_shape, image_size=96)
    obs, _ = env.reset(seed=3)
    img = obs["image"]
    R=img[...,0].astype(int);G=img[...,1].astype(int);B=img[...,2].astype(int)
    return int(((R>150)&(G<120)&(B<120)).sum())
a_big = red_area(EnvPatched, "circle")
a_small = red_area(EnvPatched, "circle_small")
print(f"  circle red px={a_big}   circle_small red px={a_small}   smaller={a_small < a_big}")
shutil.rmtree(orig_root, ignore_errors=True)
