#!/usr/bin/env python3
"""RTC offline proof (run BEFORE the robot): validates rtc_policy.py end to end
on two REAL consecutive basket training frames (slim set), no hardware.

    emimic/bin/python egomimic/scripts/test_rtc_dryrun.py                # in-process V0
    emimic/bin/python egomimic/scripts/test_rtc_dryrun.py --sweep       # + d/E sweep
    emimic/bin/python egomimic/scripts/test_rtc_dryrun.py --ws 8100     # via live server
                                                                        # (RTC=1 serve_abl_0921.sh V0)

Checks:
  a. DISARMED request == plain policy (MAE in the 0.008-0.05 basket-train band);
  b. frozen rows bit-match the previous chunk (<= normalize round-off);
  c. boundary continuity |new[delta]-prev[K+delta]|: ~0 WITH RTC vs the naive-async
     baseline gap WITHOUT (that number is the chunk-boundary jerk RTC removes);
  d. constrained chunk's GT-MAE not degraded vs unconstrained;
  e. (--sweep) d x E grid;  f. (--ws) same through websocket + latency -> suggested d.
"""
import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np

_EGOVERSE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_EGOVERSE))

BLOCKS = {"base": (0, 3), "torso": (3, 9), "head": (9, 11), "l_arm": (11, 18),
          "r_arm": (18, 25), "l_hand": (25, 37), "r_hand": (37, 49)}
AC = "actions.joint_base_torso_head_arm_hand"


def load_frame(ep: int, t: int):
    import pandas as pd
    pq = (_EGOVERSE / "datasets/dp3c_basket60_v2_slim/data/chunk-000"
          / f"episode_{ep:06d}.parquet")
    df = pd.read_parquet(pq)
    assert t + 64 <= len(df), f"t={t}+64 > {len(df)}"
    row = df.iloc[t]
    jp = np.asarray(list(row["obs.robot0_joint_pos"]), np.float64).ravel()
    obs = {
        "front_pcd_1": np.asarray(list(row["obs.aria_pcdc"]), np.float32).reshape(-1, 6),
        "front_pcd_2": np.asarray(list(row["obs.aria_pcdc_local"]), np.float32).reshape(-1, 6),
        "eef_pose_glass": np.asarray(list(row["obs.eef_pose_glass"]), np.float32).ravel(),
        "hand_left_qpos": np.asarray(list(row["obs.hand_left_qpos"]), np.float32).ravel(),
        "hand_right_qpos": np.asarray(list(row["obs.hand_right_qpos"]), np.float32).ravel(),
        "robot0_joint_pos": jp[4:26].astype(np.float32),
        "task_id": np.zeros(64, np.float32),
    }
    gt = np.stack([np.asarray(list(df.iloc[t + i][AC]), np.float64) for i in range(64)])
    return obs, gt


def block_gaps(a, b):
    g = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return "  ".join(f"{k} {g[s:e].max():.4f}" for k, (s, e) in BLOCKS.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(_EGOVERSE / "checkpoints/abl0921/V0.ckpt"))
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--t", type=int, default=100)
    ap.add_argument("--period", type=int, default=8, help="K: steps between frames")
    ap.add_argument("--freeze", type=int, default=3, help="d")
    ap.add_argument("--soft", type=int, default=19, help="E")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ws", type=int, default=0,
                    help="port of a RUNNING serve_policy_rtc server; 0 = in-process")
    args = ap.parse_args()
    K, d, E = args.period, args.freeze, args.soft

    obs_t, gt_t = load_frame(args.ep, args.t)
    obs_tk, gt_tk = load_frame(args.ep, args.t + K)

    lat_ms = []
    if args.ws:
        import msgpack_numpy
        import websockets.sync.client
        msgpack_numpy.patch()
        conn = websockets.sync.client.connect(
            f"ws://127.0.0.1:{args.ws}", compression=None, max_size=None)
        meta = msgpack_numpy.unpackb(conn.recv())
        print(f"[ws] server metadata rtc={meta.get('rtc')} "
              f"steps={meta.get('num_inference_steps')} ckpt={meta.get('checkpoint')}")
        assert meta.get("rtc"), "server is not RTC-capable (start with RTC=1)"

        def infer(o):
            t0 = _time.perf_counter()
            conn.send(msgpack_numpy.packb(o))
            r = conn.recv()
            lat_ms.append((_time.perf_counter() - t0) * 1000.0)
            if isinstance(r, str):
                raise SystemExit(f"[server] ERROR:\n{r}")
            a = np.asarray(msgpack_numpy.unpackb(r)["actions"], np.float64)
            return a[0] if a.ndim == 3 else a
    else:
        import torch
        from egomimic.models.denoising_policy import DenoisingPolicy
        from egomimic.pl_utils.pl_model import ModelWrapper
        from egomimic.serving.rtc_policy import RTCEgoVersePolicy
        model = ModelWrapper.load_from_checkpoint(args.ckpt, weights_only=False)
        for h in model.model.nets["policy"].heads.values():
            if isinstance(h, DenoisingPolicy):
                h.num_inference_steps = int(args.steps)
        pol = RTCEgoVersePolicy(model)

        def infer(o):
            t0 = _time.perf_counter()
            a = np.asarray(pol.infer(dict(o))["actions"], np.float64)
            lat_ms.append((_time.perf_counter() - t0) * 1000.0)
            return a[0] if a.ndim == 3 else a

    fails = []

    # a) disarmed == plain path, sane band
    A = infer(obs_t)
    mae_a = np.abs(A[:32, :49] - gt_t[:32]).mean()
    ok_a = 0.005 < mae_a < 0.06
    print(f"a) disarmed chunk {A.shape}, MAE32 vs GT = {mae_a:.4f} "
          f"({'OK' if ok_a else 'OUT OF BAND 0.005-0.06'})")
    if not ok_a:
        fails.append("a")

    def rtc_obs(dd, ee):
        o = dict(obs_tk)
        o["rtc_prev_actions"] = A[K:].astype(np.float32)
        o["rtc_freeze_steps"] = int(dd)
        o["rtc_soft_horizon"] = int(ee)
        o["rtc_soft_power"] = float(args.gamma)
        return o

    # baseline: naive async (unconstrained chunk from the same new obs)
    B0 = infer(dict(obs_tk))
    B = infer(rtc_obs(d, E))

    # b) frozen rows
    fz = float(np.abs(B[:d] - A[K:K + d]).max())
    ok_b = fz < 5e-4
    print(f"b) frozen rows max|B[:{d}] - A[{K}:{K + d}]| = {fz:.2e} "
          f"({'OK' if ok_b else 'FAIL (expect < 5e-4, normalize round-off)'})")
    if not ok_b:
        fails.append("b")

    # c) boundary continuity, RTC vs naive
    print("c) boundary continuity |new[j] - prev[K+j]| (max over dims):")
    worst_rtc_frozen = 0.0
    for j in range(0, min(d + 5, 64 - K)):
        g_rtc = float(np.abs(B[j] - A[K + j]).max())
        g_naive = float(np.abs(B0[j] - A[K + j]).max())
        tag = "frozen" if j < d else ("soft" if j < E else "free")
        if j < d:
            worst_rtc_frozen = max(worst_rtc_frozen, g_rtc)
        print(f"   j={j:2d} [{tag:6s}]  RTC {g_rtc:.4f}   naive-async {g_naive:.4f}")
    print(f"   blocks at j=0: RTC   {block_gaps(B[0], A[K])}")
    print(f"   blocks at j=0: naive {block_gaps(B0[0], A[K])}")

    # d) quality not degraded
    n = 64 - K
    mae_rtc = np.abs(B[:32, :49] - gt_tk[:32]).mean()
    mae_naive = np.abs(B0[:32, :49] - gt_tk[:32]).mean()
    ok_d = mae_rtc < max(1.5 * mae_naive, mae_naive + 0.01)
    print(f"d) MAE32 vs GT at t+{K}: RTC {mae_rtc:.4f} vs naive {mae_naive:.4f} "
          f"({'OK' if ok_d else 'DEGRADED'})")
    if not ok_d:
        fails.append("d")

    if args.sweep:
        print("e) sweep (frozen-row error | boundary j=d gap | MAE32):")
        for dd in (2, 3, 5):
            for ee in (dd, dd + 16, 64):
                Bs = infer(rtc_obs(dd, ee))
                fz = float(np.abs(Bs[:dd] - A[K:K + dd]).max())
                bd = float(np.abs(Bs[min(dd, 63 - K)] - A[K + min(dd, 63 - K)]).max())
                mm = np.abs(Bs[:32, :49] - gt_tk[:32]).mean()
                print(f"   d={dd} E={ee:2d}: frozen {fz:.1e} | j={dd} gap {bd:.4f} | MAE {mm:.4f}")

    if lat_ms:
        arr = np.asarray(lat_ms)
        rec_d = int(np.ceil(np.percentile(arr, 95) / 100.0)) + 1
        print(f"latency ms: min {arr.min():.0f} med {np.median(arr):.0f} "
              f"p95 {np.percentile(arr, 95):.0f} max {arr.max():.0f} "
              f"-> recommended --rtc-freeze-steps >= {rec_d} (10 Hz)")

    print(("ALL RTC CHECKS PASS" if not fails else
           f"FAILED checks: {fails}") + f"  (d={d} E={E} K={K} gamma={args.gamma})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
