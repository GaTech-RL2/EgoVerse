#!/usr/bin/env python
"""Diagnostic: compare k-center greedy vs a count-equalizing (stratified)
selection on the SAME obs0 pool, on JOINT 4-d coverage and marginal grids.
Read-only — does not write the dataset; informs the method choice.
"""
import json
from pathlib import Path
import numpy as np

OUTDIR = Path("/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts")
SCAN = json.loads((OUTDIR / "nc3_scan.json").read_text())["records"]
N = 200
WORLD = 512.0
SEED = 0


def grid_counts(xs, ys, n=4):
    edges = np.linspace(0, WORLD, n + 1)
    cx = np.clip(np.digitize(xs, edges) - 1, 0, n - 1)
    cy = np.clip(np.digitize(ys, edges) - 1, 0, n - 1)
    g = np.zeros((n, n), int)
    for a, b in zip(cy, cx):
        g[a, b] += 1
    return g


def cv(g):
    f = g.flatten().astype(float)
    return f.std() / f.mean()


pool = [r for r in SCAN if r["obstacle_level"] == 0]
keys = ["tstart_x", "tstart_y", "goal_x", "goal_y"]
P = np.array([[r[k] for k in keys] for r in pool], float)
mins, maxs = P.min(0), P.max(0)
Pn = (P - mins) / (maxs - mins)


def kcenter(X, k, seed):
    sel = [seed]; d = np.linalg.norm(X - X[seed], axis=1)
    for _ in range(k - 1):
        nx = int(np.argmax(d)); sel.append(nx)
        d = np.minimum(d, np.linalg.norm(X - X[nx], axis=1))
    return sel


# k-center
centroid = Pn.mean(0); seed = int(np.argmin(np.linalg.norm(Pn - centroid, axis=1)))
kc = kcenter(Pn, N, seed)

# count-equalizing stratified: 4-d coarse grid (2 per dim -> 16 cells), round-robin
# fill cells as evenly as possible, picking within-cell points to maximize spread.
def stratified(Pn, k, nb=2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.clip((Pn * nb).astype(int), 0, nb - 1)
    cell = idx[:, 0] * nb**3 + idx[:, 1] * nb**2 + idx[:, 2] * nb + idx[:, 3]
    from collections import defaultdict
    members = defaultdict(list)
    for i, c in enumerate(cell):
        members[c].append(i)
    cells = list(members.keys())
    for c in cells:
        rng.shuffle(members[c])
    chosen = []
    ptr = {c: 0 for c in cells}
    while len(chosen) < k:
        progressed = False
        for c in cells:
            if ptr[c] < len(members[c]) and len(chosen) < k:
                chosen.append(members[c][ptr[c]]); ptr[c] += 1; progressed = True
        if not progressed:
            break
    return chosen

st = stratified(Pn, N, nb=2, seed=SEED)


def report(name, sel):
    s = [pool[i] for i in sel]
    tx = np.array([r["tstart_x"] for r in s]); ty = np.array([r["tstart_y"] for r in s])
    gx = np.array([r["goal_x"] for r in s]); gy = np.array([r["goal_y"] for r in s])
    gt = grid_counts(tx, ty); gg = grid_counts(gx, gy)
    # joint 4-d coverage with 2 bins/dim = 16 cells
    idx = np.clip((((np.array([[r[k] for k in keys] for r in s], float)) - mins) / (maxs - mins) * 2).astype(int), 0, 1)
    jc = idx[:, 0] * 8 + idx[:, 1] * 4 + idx[:, 2] * 2 + idx[:, 3]
    jcounts = np.bincount(jc, minlength=16)
    print(f"{name}: Tstart CV={cv(gt):.3f} empty={int((gt==0).sum())} | "
          f"goal CV={cv(gg):.3f} empty={int((gg==0).sum())} | "
          f"JOINT16 CV={jcounts.std()/jcounts.mean():.3f} empty_cells={int((jcounts==0).sum())}/16 "
          f"min={jcounts.min()} max={jcounts.max()}")

report("k-center  ", kc)
report("stratified", st)
# full pool joint for reference
idxp = np.clip((Pn * 2).astype(int), 0, 1)
jcp = idxp[:, 0]*8 + idxp[:, 1]*4 + idxp[:, 2]*2 + idxp[:, 3]
jp = np.bincount(jcp, minlength=16)
print(f"FULL pool : JOINT16 distribution = {jp.tolist()}  CV={jp.std()/jp.mean():.3f}")
