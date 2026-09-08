#!/usr/bin/env python3
"""dp3cmix dry-run: feed one example frame (dp3cmix_example_*.npz) to a served
policy and score against gt_actions_49_future from the same (never-trained) frame.

    emimic/bin/python egomimic/scripts/test_dp3cmix_dryrun.py \
        --npz ai_docs/assets_rect_lut/dp3cmix_example_1024.npz --port 8020

Reference on THIS frame: dp3c_dual = 0.078 (hard val frame; the 0.005-0.01 band is\nfor train-episode replays and does not apply). v4 emits 128x49 -> scored on all rows.
"""
import argparse
import numpy as np

BLOCKS = {"base": (0, 3), "torso": (3, 9), "head": (9, 11), "l_arm": (11, 18),
          "r_arm": (18, 25), "l_hand": (25, 37), "r_hand": (37, 49)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--repeats", type=int, default=3, help="re-infer to show sampling noise")
    ap.add_argument("--proprio-dim", type=int, default=0, help="0=auto (22, dp3c_dual convention)")
    args = ap.parse_args()

    import msgpack_numpy
    import websockets.sync.client
    msgpack_numpy.patch()

    z = np.load(args.npz, allow_pickle=True)
    print(f"[npz] {args.npz}: {sorted(z.files)}")
    if "meta" in z.files:
        print(f"[npz] meta: {str(z['meta'])[:300]}")

    def get(k):
        return np.asarray(z[k]) if k in z.files else None

    n6 = get("obs.aria_pcdc").reshape(-1, 6).astype(np.float32)
    l6 = get("obs.aria_pcdc_local").reshape(-1, 6).astype(np.float32)
    jp = get("obs.robot0_joint_pos").astype(np.float32).ravel()
    obs = {
        "front_pcd_1": n6,
        "front_pcd_2": l6,
        "eef_pose_glass": get("obs.eef_pose_glass").astype(np.float32).ravel(),
        "hand_left_qpos": get("obs.hand_left_qpos").astype(np.float32).ravel(),
        "hand_right_qpos": get("obs.hand_right_qpos").astype(np.float32).ravel(),
    }
    gt = get("gt_actions_49_future").astype(np.float64)

    conn = websockets.sync.client.connect(
        f"ws://{args.host}:{args.port}", compression=None, max_size=None)
    meta = msgpack_numpy.unpackb(conn.recv())
    print(f"[server] metadata: {meta}")
    horizon = int(meta.get("action_horizon") or 32)

    # dp3c_dual contract: 22-D no-wheel proprio = position[4:26]. Override with --proprio-dim 26.
    want = args.proprio_dim or int(meta.get("robot0_joint_pos_dim") or 22)
    obs["robot0_joint_pos"] = jp[4:26] if (want == 22 and len(jp) == 26) else jp
    tid = np.zeros(64, dtype=np.float32); tid[1] = 1.0
    obs["task_id"] = tid
    for k, v in obs.items():
        print(f"    {k}: {np.asarray(v).shape}")

    maes = []
    for r in range(args.repeats):
        conn.send(msgpack_numpy.packb(obs))
        resp = conn.recv()
        if isinstance(resp, str):
            raise SystemExit(f"[server] ERROR:\n{resp}")
        act = np.asarray(msgpack_numpy.unpackb(resp)["actions"], dtype=np.float64)
        if act.ndim == 3:
            act = act[0]
        n = min(len(act), len(gt))
        d = np.abs(act[:n] - gt[:n])
        maes.append(d.mean())
        if r == 0:
            print(f"[dry] chunk {act.shape} scored on first {n} gt rows "
                  f"(gt has {len(gt)}; horizon {horizon})")
            print("      per-block MAE: " + "  ".join(
                f"{k} {d[:, s:e].mean():.4f}" for k, (s, e) in BLOCKS.items()))
    conn.close()
    m = np.mean(maes)
    # Band calibrated 2026-09-08 ON THIS EXAMPLE FRAME: it is a HARD val frame —
    # fully-trained dp3c_dual scores 0.078 on it (0.067 over all of val ep23), so the
    # guide's 0.005-0.01 (train-episode replays) does NOT apply here.
    verdict = ("EXCELLENT (beats the trained dp3c_dual reference 0.078)" if m < 0.06 else
               "OK — champion-parity band on this hard val frame (dp3c_dual: 0.078)" if m < 0.11 else
               "BAD — check contract (even dp3c_dual only reaches 0.078 here)")
    print(f"[dry] MAE over {args.repeats} samples: {m:.4f}  (min {min(maes):.4f} max {max(maes):.4f})  -> {verdict}")


if __name__ == "__main__":
    main()
