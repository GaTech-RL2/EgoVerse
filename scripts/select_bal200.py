#!/usr/bin/env python
"""Select a BALANCED 200-episode subset of new_circle_3 via farthest-point
(k-center greedy) sampling on the normalized 4-d vector
[Tstart_x, Tstart_y, goal_x, goal_y].

Pool restricted to obstacle_level==0 (766 episodes).
Seed: deterministic — start from the episode closest to the distribution
centroid (in normalized space).

Outputs (all under SCRATCH dir, a shared flash path):
  - bal200_selection.json  (episode list + metadata)
  - SELECTION.md           (method + evidence grids)
  - prints evidence grids for full-953, first200, balanced-200.
"""
import json
import sys
from pathlib import Path

import numpy as np
import zarr

DATA = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3")
FIRST200 = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3_first200")
OUTDIR = Path("/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts")
SCAN_JSON = OUTDIR / "nc3_scan.json"
SEL_JSON = OUTDIR / "bal200_selection.json"
SEED = 0
N_SELECT = 200
GRID = 4
WORLD = 512.0


def parse_obstacle_level(attrs):
    td = attrs.get("task_description", None)
    if td is None:
        return None
    try:
        return int(json.loads(td)["env_args"]["obstacle_level"])
    except Exception:
        return None


def scan(eps):
    recs = []
    for ep in eps:
        g = zarr.open_group(str(ep), mode="r")
        attrs = dict(g.attrs)
        s0 = np.asarray(g["observations.state"][0])
        g0 = np.asarray(g["goal_pose"][0])
        recs.append({
            "name": ep.name,
            "tstart_x": float(s0[2]), "tstart_y": float(s0[3]),
            "tstart_theta": float(s0[4]),
            "goal_x": float(g0[0]), "goal_y": float(g0[1]),
            "goal_theta": float(g0[2]),
            "obstacle_level": parse_obstacle_level(attrs),
            "total_frames": attrs.get("total_frames", None),
        })
    return recs


def grid_counts(xs, ys, lo=0.0, hi=WORLD, n=GRID):
    """4x4 occupancy over [lo,hi]^2. Returns (n,n) int array, row=y-bin, col=x-bin."""
    edges = np.linspace(lo, hi, n + 1)
    cx = np.clip(np.digitize(xs, edges) - 1, 0, n - 1)
    cy = np.clip(np.digitize(ys, edges) - 1, 0, n - 1)
    grid = np.zeros((n, n), dtype=int)
    for a, b in zip(cy, cx):
        grid[a, b] += 1
    return grid


def fmt_grid(grid):
    lines = []
    for row in grid:
        lines.append(" ".join(f"{v:4d}" for v in row))
    return "\n".join(lines)


def grid_stats(grid):
    flat = grid.flatten().astype(float)
    mn, mx = int(flat.min()), int(flat.max())
    mean = flat.mean()
    cv = flat.std() / mean if mean > 0 else float("nan")
    nempty = int((flat == 0).sum())
    return mn, mx, mean, cv, nempty


def kcenter_greedy(X, k, seed_idx):
    """Farthest-point / k-center greedy on rows of X (already normalized).
    Deterministic given seed_idx. Returns list of selected row indices."""
    n = X.shape[0]
    selected = [seed_idx]
    # min distance from each point to the current selected set
    d = np.linalg.norm(X - X[seed_idx], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        selected.append(nxt)
        nd = np.linalg.norm(X - X[nxt], axis=1)
        d = np.minimum(d, nd)
    return selected


def main():
    np.random.seed(SEED)
    eps = sorted(DATA.glob("episode_*.zarr"))
    print(f"Total episodes: {len(eps)}", flush=True)
    recs = scan(eps)
    SCAN_JSON.write_text(json.dumps({"records": recs}, indent=2))

    by_name = {r["name"]: r for r in recs}

    # obstacle-level breakdown
    lvl = {}
    for r in recs:
        lvl[r["obstacle_level"]] = lvl.get(r["obstacle_level"], 0) + 1
    print(f"obstacle_level counts: {dict(sorted(lvl.items()))}", flush=True)

    # ---- POOL: obstacle-level 0 only ----
    pool = [r for r in recs if r["obstacle_level"] == 0]
    print(f"obs0 pool size: {len(pool)}", flush=True)

    # 4-d feature, normalized to [0,1] by dataset (full obs0 pool) min/max per dim
    feat_keys = ["tstart_x", "tstart_y", "goal_x", "goal_y"]
    P = np.array([[r[k] for k in feat_keys] for r in pool], dtype=float)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    rng = np.where((maxs - mins) > 1e-9, maxs - mins, 1.0)
    Pn = (P - mins) / rng

    # seed = episode closest to centroid in normalized space
    centroid = Pn.mean(axis=0)
    seed_idx = int(np.argmin(np.linalg.norm(Pn - centroid, axis=1)))
    print(f"seed episode (centroid-nearest): {pool[seed_idx]['name']}", flush=True)
    print(f"  norm feat min/max per dim: min={mins.tolist()} max={maxs.tolist()}", flush=True)

    sel_local = kcenter_greedy(Pn, N_SELECT, seed_idx)
    sel_recs = [pool[i] for i in sel_local]
    sel_names = [r["name"] for r in sel_recs]
    assert len(set(sel_names)) == N_SELECT, "duplicate selection!"

    # ---------- EVIDENCE GRIDS ----------
    def arrs(records, key_x, key_y):
        return (np.array([r[key_x] for r in records]),
                np.array([r[key_y] for r in records]))

    # full 953
    full_recs = recs
    # first200 (by folder membership)
    first200_names = sorted(p.name for p in FIRST200.iterdir() if p.name.endswith(".zarr"))
    first200_recs = [by_name[n] for n in first200_names if n in by_name]

    groups = {
        "FULL (953)": full_recs,
        "NAIVE first200": first200_recs,
        "BALANCED 200": sel_recs,
    }

    report_lines = []
    grids_for_json = {}
    for gname, grecs in groups.items():
        tx, ty = arrs(grecs, "tstart_x", "tstart_y")
        gx, gy = arrs(grecs, "goal_x", "goal_y")
        g_t = grid_counts(tx, ty)
        g_g = grid_counts(gx, gy)
        st_t = grid_stats(g_t)
        st_g = grid_stats(g_g)
        block = []
        block.append(f"### {gname}  (n={len(grecs)})")
        block.append("")
        block.append("T-start (x,y) 4x4 occupancy [rows=y-bin top->bottom, cols=x-bin]:")
        block.append("```")
        block.append(fmt_grid(g_t))
        block.append("```")
        block.append(f"  min={st_t[0]} max={st_t[1]} mean={st_t[2]:.1f} CV={st_t[3]:.3f} empty_bins={st_t[4]}/16")
        block.append("")
        block.append("Goal (x,y) 4x4 occupancy:")
        block.append("```")
        block.append(fmt_grid(g_g))
        block.append("```")
        block.append(f"  min={st_g[0]} max={st_g[1]} mean={st_g[2]:.1f} CV={st_g[3]:.3f} empty_bins={st_g[4]}/16")
        block.append("")
        report_lines.extend(block)
        print("\n".join(block), flush=True)
        grids_for_json[gname] = {
            "n": len(grecs),
            "tstart_grid": g_t.tolist(), "tstart_stats": {"min": st_t[0], "max": st_t[1], "mean": st_t[2], "cv": st_t[3], "empty": st_t[4]},
            "goal_grid": g_g.tolist(), "goal_stats": {"min": st_g[0], "max": st_g[1], "mean": st_g[2], "cv": st_g[3], "empty": st_g[4]},
        }

    # ---------- WRITE JSON ----------
    SEL_JSON.write_text(json.dumps({
        "method": "k-center greedy (farthest-point) on normalized 4-d [Tstart_x,Tstart_y,goal_x,goal_y]",
        "seed": SEED,
        "seed_episode": pool[seed_idx]["name"],
        "n_select": N_SELECT,
        "pool": "obstacle_level==0",
        "pool_size": len(pool),
        "norm_mins": mins.tolist(),
        "norm_maxs": maxs.tolist(),
        "selected": sel_names,
        "grids": grids_for_json,
    }, indent=2))
    print(f"\nWrote {SEL_JSON}", flush=True)

    # ---------- SELECTION.md content (to be written into the dataset dir) ----------
    md = []
    md.append("# new_circle_3_bal200 — Balanced 200-episode subset")
    md.append("")
    md.append("## Method")
    md.append("")
    md.append("- **Source**: `/coc/flash7/paphiwetsa3/datasets/new_circle_3` (953 episodes).")
    md.append("- **Pool**: restricted to **obstacle_level == 0** (766 episodes). Levels 1–12 (187 eps) excluded.")
    md.append("  - obstacle_level read from each episode's `zarr.json` attrs `task_description.env_args.obstacle_level`; it matches the filename `obsN` token for every episode.")
    md.append("- **Selection**: farthest-point / **k-center greedy** on the per-episode 4-d vector")
    md.append("  `[Tstart_x, Tstart_y, goal_x, goal_y]`, each dim normalized to [0,1] by the obs0-pool per-dim min/max.")
    md.append("  - **Tstart** = object (T) pose at frame 0 = `observations.state[0, 2:4]` (state layout `[agent_x, agent_y, obj_x, obj_y, obj_theta]`, confirmed in `egomimic/eval/eval_sim.py:_state_to_init`).")
    md.append("  - **goal** = `goal_pose[0, 0:2]`.")
    md.append(f"- **Seed**: {SEED} (deterministic). Greedy starts from the episode nearest the distribution centroid in normalized space: `{pool[seed_idx]['name']}`.")
    md.append(f"- **norm mins** = {mins.tolist()}")
    md.append(f"- **norm maxs** = {maxs.tolist()}")
    md.append("")
    md.append("## Evidence: 4x4 occupancy grids (counts per bin) over the 512x512 world")
    md.append("")
    md.extend(report_lines)
    md.append("## Selected episodes (200)")
    md.append("")
    for n in sorted(sel_names):
        md.append(f"- {n}")
    (OUTDIR / "bal200_SELECTION.md").write_text("\n".join(md))
    print(f"Wrote {OUTDIR / 'bal200_SELECTION.md'}", flush=True)


if __name__ == "__main__":
    main()
