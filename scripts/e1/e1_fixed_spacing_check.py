#!/usr/bin/env python3
"""Codec check of fixed_spacing on real ABC skirts chunks (cmd_ee_pose, 200-frame windows)."""
import os, sys, numpy as np, zarr
from scipy.spatial.transform import Rotation as R
from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length
from egomimic.rldb.zarr.e1_arc_tokenizer import ARM_LAYOUT, TokenizeBimanualArcLengthE1
root, hashes_f, n_eps, per_ep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
hs = [l.strip() for l in open(hashes_f) if l.strip()][:n_eps]
D, M, dt, W, H = 0.40, 100, 1 / 30, 200, 39
toks = {("mean", False): TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="mean"),
        ("mean", True): TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="mean", fixed_spacing=True),
        ("logdur", False): TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="logdur"),
        ("logdur", True): TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="logdur", fixed_spacing=True)}
rng = np.random.default_rng(0); res = {k: {"e": [], "r": [], "nv": [], "plateau_ok": 0, "n": 0} for k in toks}
partial = 0; total = 0
for h in hs:
    g = zarr.open_group(f"{root}/{h}", mode="r"); T = int(dict(g.attrs)["total_frames"])
    cols = []
    for side in ("left", "right"):
        x = np.asarray(g[f"{side}.cmd_ee_pose"][:T], dtype=np.float64); q = x[:, 3:7] / np.maximum(np.linalg.norm(x[:, 3:7], axis=1, keepdims=True), 1e-9)
        cols += [x[:, :3], R.from_quat(q[:, [1, 2, 3, 0]]).as_euler("ZYX"), np.asarray(g[f"{side}.cmd_gripper"][:T], dtype=np.float64).reshape(-1, 1)]
    ep = np.concatenate(cols, axis=1)
    if T <= W + 5: continue
    for s in rng.integers(0, T - W, size=per_ep):
        chunk = ep[s:s + W]; gt = chunk[:100]
        for k, tk in toks.items():
            a = tk.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
            Mw = M if tk.wide else M
            dec = tk.detokenize(a, action_horizon=100); clocks = tk.clock_at_waypoints(a)
            for j, (xo, _, _, _) in enumerate(ARM_LAYOUT):
                wp = a[:M, xo:xo + 3]; cum = cumulative_arc_length(wp); gt_cum = cumulative_arc_length(chunk[:, xo:xo + 3])
                res[k]["e"].append(np.sqrt(np.mean(np.sum((dec[:H, xo:xo + 3] - gt[:H, xo:xo + 3]) ** 2, axis=1))))
                s_chk = min(float(cum[-1]), float(gt_cum[-1])) * 0.999
                if s_chk > 1e-3:
                    t_true = float(np.interp(s_chk, gt_cum, np.arange(len(gt_cum)))) * dt; t_pred = float(np.interp(s_chk, cum, clocks[j]))
                    res[k]["r"].append(t_pred / max(t_true, 1e-6))
                if k[1]:
                    nv = int((np.diff(cum) > 1e-9).sum()) + 1; res[k]["nv"].append(nv)
                    res[k]["plateau_ok"] += bool(np.allclose(wp[nv - 1:], wp[nv - 1])); res[k]["n"] += 1
                if k == ("mean", False): total += 1; partial += float(gt_cum[-1]) < D
print(f"{total} arm-tokens; partial (window reach < D): {100*partial/total:.1f} %")
for k, r in res.items():
    e, rt = np.array(r["e"]), np.array(r["r"])
    extra = f" | n_valid median {np.median(r['nv']):.0f}, q10 {np.quantile(r['nv'],.1):.0f}; plateau rows verified {r['plateau_ok']}/{r['n']}" if k[1] else ""
    print(f"  {k[0]:7s} fixed={k[1]!s:5s} codec E_time({H}f) RMS {np.sqrt(np.mean(e**2))*100:.2f} cm median {np.median(e)*100:.2f}; clock ratio median {np.median(rt):.3f} (q10 {np.quantile(rt,.1):.2f} q90 {np.quantile(rt,.9):.2f}){extra}")
