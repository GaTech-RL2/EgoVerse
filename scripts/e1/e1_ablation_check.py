#!/usr/bin/env python3
"""CPU checks for the tempo-ablation tokenizer modes on real fold chunks.

usage: e1_ablation_check.py <episode_root> <n_eps> <chunks_per_ep> <hash_list>

Per chunk (200 frames, wrist pose, both arms) it tokenizes with
  profile        - the E1 Arc+Vel row (A0)
  logdur         - #1+#2 (A1)
  logdur_smooth  - #1+#2+#4 (the tokenizer half of A2; #3 is a sampler)
and reports codec round-trip E_time over H frames, the clock ratio at the
token's span, d_clock at the waypoints, the logdur column statistics, and the
jitter-inflation ratio (raw / smoothed arc length) split by chunk tempo.
"""
import sys
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length
from egomimic.rldb.zarr.e1_arc_tokenizer import ARM_LAYOUT, LOGDUR_CLIP, TokenizeBimanualArcLengthE1, lowpass_positions
from egomimic.rldb.zarr.e1_anchor_sampler import episode_frame_speed

root, n_eps, per_ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
hashes = [l.strip() for l in open(sys.argv[4]) if l.strip()][:n_eps]
D, M, dt, W, H = 0.40, 100, 1 / 30, 200, 33
toks = {
    "profile": TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="profile"),
    "logdur": TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="logdur"),
    "logdur_smooth": TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="path", velocity_mode="logdur", progress_smooth_hz=3.0),
}
rng = np.random.default_rng(0)
res = {k: {"e_time": [], "ratio": [], "d_clock": [], "anchor": []} for k in toks}
row0, rows, clipped, infl, tempo = [], [], 0, [], []
n = 0
for h in hashes:
    g = zarr.open_group(f"{root}/{h}", mode="r")
    cols = []
    for side in ("left", "right"):
        x = np.asarray(g[f"{side}.obs_wrist_pose"][:], dtype=np.float64)
        q = x[:, 3:7] / np.linalg.norm(x[:, 3:7], axis=1, keepdims=True)
        cols += [x[:, :3], R.from_quat(q[:, [1, 2, 3, 0]]).as_euler("ZYX"), np.zeros((len(x), 1))]
    ep = np.concatenate(cols, axis=1)
    spd = episode_frame_speed(f"{root}/{h}")
    for s in rng.integers(0, len(ep) - W, size=per_ep):
        chunk = ep[s : s + W]
        gt = chunk[:100]
        tempo.append(float(np.median(spd[s : s + 100])))
        for k, tk in toks.items():
            a = tk.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
            assert a.shape == (M, 16), a.shape
            dec = tk.detokenize(a, action_horizon=100)
            clocks = tk.clock_at_waypoints(a)
            for j, (xo, _, _, _) in enumerate(ARM_LAYOUT):
                res[k]["e_time"].append(np.sqrt(np.mean(np.sum((dec[:H, xo:xo + 3] - gt[:H, xo:xo + 3]) ** 2, axis=1))))
                res[k]["anchor"].append(np.linalg.norm(dec[0, xo:xo + 3] - gt[0, xo:xo + 3]))
                wp = a[:, xo:xo + 3]
                cum = cumulative_arc_length(wp)
                gt_pos = chunk[:, xo:xo + 3] if tk.progress_smooth_hz is None else lowpass_positions(chunk[:, xo:xo + 3], tk.progress_smooth_hz, 30.0)
                gt_cum = cumulative_arc_length(gt_pos)
                s_chk = min(float(cum[-1]), float(gt_cum[-1])) * 0.999
                if s_chk > 1e-3:
                    t_true = float(np.interp(s_chk, gt_cum, np.arange(len(gt_cum)))) * dt
                    t_pred = float(np.interp(s_chk, cum, clocks[j]))
                    res[k]["ratio"].append(t_pred / max(t_true, 1e-6))
                    u = np.linspace(0, s_chk, M)
                    res[k]["d_clock"].append(np.sqrt(np.mean((np.interp(u, cum, clocks[j]) - np.interp(u, gt_cum, np.arange(len(gt_cum))) * dt) ** 2)))
                if k == "logdur":
                    row0.append(a[0, 14 + j]); rows.append(a[1:, 14 + j]); clipped += int(np.sum(np.abs(a[1:, 14 + j]) >= LOGDUR_CLIP - 1e-9))
        # jitter inflation: raw vs 3 Hz-smoothed arc length over the first 100 frames, per arm
        for xo, _, _, _ in ARM_LAYOUT:
            p = chunk[:100, xo:xo + 3]
            lr, ls = cumulative_arc_length(p)[-1], cumulative_arc_length(lowpass_positions(p, 3.0, 30.0))[-1]
            infl.append(lr / max(ls, 1e-6))
        n += 1
print(f"{n} chunks x 2 arms; H = {H} frames")
for k, r in res.items():
    e, rt, dc, an = map(np.array, (r["e_time"], r["ratio"], r["d_clock"], r["anchor"]))
    print(f"  {k:14s} codec E_time RMS {np.sqrt(np.mean(e**2))*100:.2f} cm (median {np.median(e)*100:.2f}); "
          f"clock ratio median {np.median(rt):.3f} (q10 {np.quantile(rt,.1):.2f}, q90 {np.quantile(rt,.9):.2f}); "
          f"d_clock RMS {np.sqrt(np.mean(dc**2)):.3f} s (median {np.median(dc):.3f}); anchor offset max {an.max()*1000:.2f} mm")
rows = np.concatenate(rows)
print(f"  logdur row0 (log slowness s/m): median {np.median(row0):.2f} -> {np.exp(-np.median(row0)):.3f} m/s; q10/q90 {np.quantile(row0,.1):.2f}/{np.quantile(row0,.9):.2f}")
print(f"  logdur rows1+ (log rel. duration): mean {rows.mean():.3f}, sd {rows.std():.3f}, q01/q99 {np.quantile(rows,.01):.2f}/{np.quantile(rows,.99):.2f}, clipped {clipped}/{rows.size}")
tempo, infl = np.array(tempo), np.array(infl).reshape(-1, 2).mean(1)
slow, fast = infl[tempo <= np.median(tempo)], infl[tempo > np.median(tempo)]
print(f"  jitter inflation L_raw/L_smooth(3 Hz) over 100 frames: all {np.median(infl):.3f}; slow half {np.median(slow):.3f}; fast half {np.median(fast):.3f} "
      f"(tempo median split at {np.median(tempo):.3f} m/s)")
ok = all(np.median(res[k]["ratio"]) > 0.9 and np.median(res[k]["ratio"]) < 1.1 for k in toks) and clipped < 0.01 * rows.size
print("ABLATION_CHECK_OK" if ok else "ABLATION_CHECK_FAIL")
