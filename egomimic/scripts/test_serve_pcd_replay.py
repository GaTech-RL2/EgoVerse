#!/usr/bin/env python3
"""
Replay-through-server plumbing test for point-cloud / depth (and RGB) policies.

Step 1 of the sim-eval protocol in ``pcd_policy_deployment_guide.md`` §5:
feed RECORDED episode observations through the live serving path and compare
predicted action chunks against the logged actions. Every silent contract bug
this project has produced (cloud routed as image, wrong N, 26-D vs 22-D
proprio, dropped depth) fails loudly or shows up as garbage MAE here — with
zero hardware risk.

Metadata-driven: obs keys come from the server's camera_keys / proprio_keys
and are looked up as ``obs.<key>`` columns in the episode parquet, so the SAME
script tests DP3 (front_pcd_1), Adapt3R (front_img_1 + aria_depth) and RGB
checkpoints. No egomimic imports — runs in the emimic venv or the rby1 env.

Usage (server already running):
  cd ~/RB_Y1_workspace/EgoVerse && source emimic/bin/activate
  python egomimic/scripts/serve_policy.py \
      --checkpoint checkpoints/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/checkpoints/epoch_epoch=1299.ckpt \
      --port 8000 &
  python egomimic/scripts/test_serve_pcd_replay.py \
      --dataset datasets/rby1_teleop_pcd1024_glass --episode 0 --max-steps 60

Outputs (in --out-dir, default logs/pcd_replay_<ts>/): metrics.txt,
per-block MAE table, first-action trajectory plot, chunk overlay plot.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 49-D action layout (policy_hardware_deployment_handoff.md §1).
ACTION_BLOCKS = {
    "base": (0, 3),
    "torso": (3, 9),
    "head": (9, 11),
    "l_arm": (11, 18),
    "r_arm": (18, 25),
    "l_hand": (25, 37),
    "r_hand": (37, 49),
}

# Server obs key -> dataset column names to try, in order. The trainer renames
# raw recorder keys (obs.aria_pcd -> front_pcd_1, obs.aria_image ->
# front_img_1) the same way the RGB stack always has.
DATASET_KEY_ALIASES = {
    # 0823 dual fleet: front_pcd_1 = GLOBAL stream (aria_pcd / coloured
    # aria_pcdc), front_pcd_2 = LOCAL eef-ball stream in eef coords
    # (aria_pcd_local / aria_pcdc_local). eef_pose_glass is a plain proprio
    # key and resolves by name.
    "front_pcd_1": ["front_pcd_1", "aria_pcd", "aria_pcdc"],
    "front_pcd_2": ["front_pcd_2", "aria_pcd_local", "aria_pcdc_local"],
    "front_img_1": ["front_img_1", "aria_image"],
    "aria_depth": ["aria_depth"],
}

# Channel count for flat pcd columns, decided by the RESOLVED column name:
# aria_pcdc* stores xyzrgb (1024*6 = 6144 floats), everything else xyz.
# Size alone cannot decide — 6144 is divisible by 3 as well.
def _pcd_channels(colname: str) -> int:
    return 6 if "pcdc" in colname else 3

# Keys NOT advertised in the server's camera_keys/proprio_keys — they come
# from HPT.depth_key_map ({front_img_1: aria_depth}) or extrinsics_key_map
# ({front_img_1: eef_T}) and take their own paths in
# egoverse_policy._obs_to_batch. Building obs from metadata alone therefore
# replays an Adapt3R policy with NO DEPTH and no error (`if key in obs` just
# skips it). eef_T is the opposite: the serving GUARD raises when an eef-frame
# checkpoint doesn't receive it. Send each whenever the dataset has the
# column; a policy that doesn't want them ignores the extra key.
# name -> shape to restore for flat columns (LeRobot stores 1-D features).
EXTRA_OBS_SPECS = {
    "aria_depth": (224, 224),   # 50176 floats, metres
    "eef_T": (16,),             # flattened row-major 4x4 T_eef_rect
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="", help="LeRobot dataset root")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--action-key", default="joint_base_torso_head_arm_hand",
                    help="actions.<key> column (auto-falls back to the single "
                         "actions.* column if absent).")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="Frames to replay (0 = whole episode).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Replay every k-th frame.")
    ap.add_argument("--out-dir", default="",
                    help="Output dir (default logs/pcd_replay_<timestamp>).")
    ap.add_argument("--proprio-dim", type=int, default=0,
                    help="Force robot0_joint_pos width (0 = auto from server "
                         "metadata / dataset).")
    ap.add_argument("--from-recording", default="",
                    help="Replay a server-side recording session dir "
                         "(serve_policy.py --save-inputs-dir) instead of a "
                         "dataset: re-sends the recorded obs and scores the "
                         "returned actions against the RECORDED actions. "
                         "Proves the recording is faithful; doubles as a "
                         "serving regression test after code syncs.")
    ap.add_argument("--drop-depth", action="store_true",
                    help="Deliberately withhold aria_depth. NEGATIVE CONTROL: "
                         "for a depth policy the MAE must get clearly worse. "
                         "If it does not, depth is not reaching the encoder "
                         "and every 'depth helps/hurts' conclusion is void.")
    ap.add_argument("--require-depth", action="store_true",
                    help="Fail if aria_depth could not be sent. Use for every "
                         "Adapt3R checkpoint: the server silently skips a "
                         "missing depth key, so an image-only replay of a "
                         "depth policy otherwise looks merely 'worse', not "
                         "broken.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Dataset (LeRobot V2 parquet, direct read)
# --------------------------------------------------------------------------- #
class EpisodeTable:
    def __init__(self, root: Path, episode: int, action_key: str):
        import pyarrow.parquet as pq

        info = json.loads((root / "meta" / "info.json").read_text())
        tpl = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        chunk = episode // int(info.get("chunks_size", 1000))
        path = root / tpl.format(episode_chunk=chunk, episode_index=episode)
        if not path.exists():
            raise SystemExit(f"Episode parquet not found: {path}")
        self.table = pq.read_table(path.as_posix())
        self.n = self.table.num_rows
        self.features = {
            k: tuple(v.get("shape") or ())
            for k, v in (info.get("features") or {}).items()
            if isinstance(v, dict)
        }
        self.fps = float(info.get("fps", 10.0) or 10.0)

        act_col = f"actions.{action_key}"
        if act_col not in self.table.column_names:
            candidates = [c for c in self.table.column_names if c.startswith("actions.")]
            if len(candidates) == 1:
                print(f"[dataset] actions.{action_key} absent; using {candidates[0]}")
                act_col = candidates[0]
            else:
                raise SystemExit(
                    f"{act_col} not in parquet. actions.* columns: {candidates}"
                )
        self.actions = np.stack(
            [self._cell(self.table.column(act_col), i, np.float64) for i in range(self.n)]
        )
        self._cache: dict[str, np.ndarray] = {}
        print(f"[dataset] {path.name}: {self.n} frames, actions {self.actions.shape}, "
              f"fps {self.fps}")
        obs_cols = [c for c in self.table.column_names if c.startswith("obs.")]
        print(f"[dataset] obs columns: {obs_cols}")

    def _cell(self, col, i: int, dtype) -> np.ndarray:
        cell = col[i]
        v = cell.as_py() if hasattr(cell, "as_py") else cell
        return np.asarray(v, dtype=dtype)

    def obs_column(self, key: str) -> np.ndarray | None:
        """(n, *shape) array for ``obs.<key>``; images decoded to BGR uint8."""
        if key in self._cache:
            return self._cache[key]
        name = f"obs.{key}"
        if name not in self.table.column_names:
            return None
        col = self.table.column(name)
        rows = []
        for i in range(self.n):
            cell = col[i]
            v = cell.as_py() if hasattr(cell, "as_py") else cell
            if isinstance(v, dict) and ("bytes" in v or "path" in v):
                rows.append(self._decode_image(v))
                continue
            if isinstance(v, (bytes, bytearray)):
                rows.append(self._decode_image({"bytes": bytes(v)}))
                continue
            arr = np.asarray(v, dtype=np.float32)
            feat = self.features.get(name)
            if feat and len(feat) >= 2 and arr.ndim == 1 and arr.size == int(np.prod(feat)):
                arr = arr.reshape(feat)
            rows.append(arr)
        out = np.stack(rows)
        self._cache[key] = out
        return out

    @staticmethod
    def _decode_image(d: dict) -> np.ndarray:
        from PIL import Image

        raw = d.get("bytes")
        pil = Image.open(io.BytesIO(raw))
        arr = np.array(pil)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr[..., ::-1].copy()  # RGB -> BGR (server flips back)


# --------------------------------------------------------------------------- #
_WARNED_OMITTED: set[str] = set()


def _lookup_column(ep: EpisodeTable, key: str):
    """(column, resolved_name) for a server obs key, following the trainer's
    renames; (None, None) when absent."""
    for name in DATASET_KEY_ALIASES.get(key, [key]):
        col = ep.obs_column(name)
        if col is not None:
            return col, name
    return None, None


def build_obs(ep: EpisodeTable, i: int, metadata: dict, proprio_dim: int,
              require_depth: bool = False, drop_depth: bool = False) -> dict:
    cam_keys = list(dict.fromkeys(metadata.get("camera_keys") or []))
    proprio_keys = list(dict.fromkeys(metadata.get("proprio_keys") or []))
    obs: dict = {}
    missing: list[str] = []
    for k in cam_keys + proprio_keys:
        if k == "task_id":
            continue
        if k == "robot0_joint_pos":
            # 22-D no-wheel contract (pcd guide §1): prefer the explicit
            # no-wheel column, else slice a 26-D vector to [4:26]. Override
            # with --proprio-dim 26 only if a future model truly wants 26.
            want = proprio_dim or int(metadata.get("robot0_joint_pos_dim") or 22)
            col = ep.obs_column("robot0_joint_pos_no_wheel")
            if col is not None and want == 22:
                val = np.asarray(col[i], dtype=np.float32).reshape(-1)
            else:
                full = ep.obs_column("robot0_joint_pos")
                if full is None:
                    missing.append(k)
                    continue
                v1 = np.asarray(full[i], dtype=np.float32).reshape(-1)
                val = v1[4:26] if (want == 22 and v1.shape[0] == 26) else v1
            obs[k] = np.ascontiguousarray(val, dtype=np.float32)
            continue
        col, colname = _lookup_column(ep, k)
        if col is None:
            missing.append(k)
            continue
        val = col[i]
        if isinstance(val, np.ndarray) and val.dtype != np.uint8:
            val = np.ascontiguousarray(val, dtype=np.float32)
        if "pcd" in k and isinstance(val, np.ndarray) and val.ndim == 1:
            # Datasets store the cloud flattened (e.g. [3072] xyz or [6144]
            # xyzrgb); the wire contract is (N, C) float32 — the serving
            # router accepts C in (3, 6). Channel count comes from the
            # resolved column name (size alone is ambiguous: 6144 % 3 == 0).
            ch = _pcd_channels(colname)
            if val.size % ch != 0:
                raise SystemExit(
                    f"{k} ({colname}) has {val.size} values, not /{ch}")
            val = val.reshape(-1, ch)
        obs[k] = val

    # Metadata-invisible keys (see EXTRA_OBS_SPECS). Passed through
    # UNNORMALIZED — depth is back-projected with metric intrinsics and eef_T
    # is consumed as a rigid transform, so any rescale here silently corrupts
    # geometry.
    for k, shape in ({} if drop_depth else EXTRA_OBS_SPECS).items():
        if k in obs:
            continue
        col, _cn = _lookup_column(ep, k)
        if col is None:
            continue
        val = np.asarray(col[i], dtype=np.float32)
        want = int(np.prod(shape))
        if val.size != want:
            raise SystemExit(
                f"{k} has {val.size} values, expected {want} {shape}")
        val = val.reshape(shape)
        obs[k] = np.ascontiguousarray(val, dtype=np.float32)
        if k not in _WARNED_OMITTED:
            _WARNED_OMITTED.add(k)
            if k == "aria_depth":
                v = val[val > 0]
                print(f"[replay] depth key {k!r} SENT: {val.shape} valid "
                      f"{100.0 * v.size / val.size:.1f}%  median "
                      f"{np.median(v):.2f} m  p95 {np.percentile(v, 95):.2f} m")
            else:
                print(f"[replay] extra key {k!r} SENT: {val.shape} "
                      f"(eef<->cam dist {np.linalg.norm(val.reshape(4, 4)[:3, 3]):.3f} m)"
                      if k == "eef_T" else f"[replay] extra key {k!r} SENT: {val.shape}")
    if require_depth and "aria_depth" not in obs:
        cols = [c for c in ep.table.column_names if c.startswith("obs.")]
        raise SystemExit(
            f"--require-depth: 'aria_depth' not found in the "
            f"dataset.\nDataset obs columns: {cols}\n"
            "An Adapt3R checkpoint replayed without depth still returns "
            "actions — this guard exists so that never passes silently."
        )

    # Missing CAMERA keys are tolerable when another modality resolved:
    # metadata lists the shared stem's full camera set (e.g. cosmetic
    # front_img_1 next to front_pcd_1) and the serving path only consumes the
    # keys actually present in the obs dict. Omit + warn once. Anything else
    # missing (proprio, or ALL modalities) is the plumbing failure.
    resolved_modalities = [k for k in cam_keys if k in obs]
    fatal = [k for k in missing if k not in cam_keys] if resolved_modalities else missing
    omitted = [k for k in missing if k in cam_keys] if resolved_modalities else []
    for k in omitted:
        if k not in _WARNED_OMITTED:
            _WARNED_OMITTED.add(k)
            print(f"[replay] NOTE: omitting camera key {k!r} (no dataset column; "
                  f"serving skips absent cam keys). Resolved: {resolved_modalities}")
    if fatal:
        cols = [c for c in ep.table.column_names if c.startswith("obs.")]
        raise SystemExit(
            f"Server requests obs keys {fatal} that the dataset lacks.\n"
            f"Dataset obs columns: {cols}\n"
            "This IS the plumbing failure this test exists to catch — the "
            "dataset/policy pairing is wrong, or the serving metadata changed."
        )
    task = np.zeros(64, dtype=np.float32)
    task[1] = 1.0
    obs["task_id"] = task
    return obs


def gt_chunk(actions: np.ndarray, i: int, horizon: int) -> np.ndarray:
    end = min(actions.shape[0], i + horizon)
    chunk = actions[i:end]
    if chunk.shape[0] < horizon:  # pad by repeating the last row
        pad = np.repeat(chunk[-1:], horizon - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def per_block_mae(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (s, e) in ACTION_BLOCKS.items():
        if gt.shape[-1] >= e:
            out[name] = float(np.mean(np.abs(pred[..., s:e] - gt[..., s:e])))
    return out


def replay_recording(args) -> None:
    """Re-send a recorded session's inputs; score vs the recorded outputs."""
    import msgpack_numpy
    import websockets.sync.client
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from egomimic.serving.input_recorder import load_session

    msgpack_numpy.patch()
    sess = load_session(args.from_recording)
    n_total = len(sess["idx"])
    n = n_total if args.max_steps <= 0 else min(args.max_steps, n_total)
    obs_keys = [k for k in sess if k.startswith("obs/")]
    print(f"[recording] {args.from_recording}: {n_total} records, "
          f"{len(sess['connections'])} connection events, obs keys "
          f"{[k[4:] for k in obs_keys]}")
    uri = f"ws://{args.host}:{args.port}"
    conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
    _ = msgpack_numpy.unpackb(conn.recv())
    maes, t1s, rts = [], [], []
    for i in range(0, n, max(1, args.stride)):
        obs = {}
        for k in obs_keys:
            v = sess[k][i]
            if v is None:
                continue
            obs[k[4:]] = np.ascontiguousarray(v)
        t0 = time.time()
        conn.send(msgpack_numpy.packb(obs))
        resp = msgpack_numpy.unpackb(conn.recv())
        rts.append((time.time() - t0) * 1e3)
        pred = np.asarray(resp["actions"] if isinstance(resp, dict) else resp, np.float32)
        rec = np.asarray(sess["act/actions"][i], np.float32)
        maes.append(float(np.mean(np.abs(pred - rec))))
        t1s.append(float(np.mean(np.abs(pred.reshape(-1, 49)[0] - rec.reshape(-1, 49)[0]))))
    conn.close()
    print("\n" + "=" * 70)
    print(f"recording    : {args.from_recording}")
    print(f"frames       : {len(maes)} of {n_total}")
    print(f"round-trip   : median {np.median(rts):.0f} ms")
    print(f"pred vs RECORDED actions: full-chunk MAE {np.mean(maes):.6f}  "
          f"t1 {np.mean(t1s):.6f}")
    print("  reference: sending the SAME obs twice gives MAE ~3.5e-3 (dp3c_dual, "
          "measured 2026-08-26) — that is the flow-matching resample noise. A "
          "value at that level = faithful; clearly larger = the serving path "
          "changed or the recording is not what was sent.")


def main() -> None:
    args = parse_args()
    if args.from_recording:
        replay_recording(args)
        return
    if not args.dataset:
        raise SystemExit("--dataset is required unless --from-recording is given")
    import msgpack_numpy
    import websockets.sync.client

    msgpack_numpy.patch()

    root = Path(args.dataset)
    ep = EpisodeTable(root, args.episode, args.action_key)

    uri = f"ws://{args.host}:{args.port}"
    print(f"[server] connecting to {uri} …")
    conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
    metadata = msgpack_numpy.unpackb(conn.recv())
    print(f"[server] metadata: {metadata}")
    horizon = int(metadata.get("action_horizon") or 32)
    action_dim = int(metadata.get("action_dim") or ep.actions.shape[1])
    if ep.actions.shape[1] != action_dim:
        print(f"[WARN] dataset action dim {ep.actions.shape[1]} != server {action_dim}")

    n_steps = ep.n if args.max_steps <= 0 else min(args.max_steps, ep.n)
    steps = list(range(0, n_steps, max(1, args.stride)))

    out_dir = Path(args.out_dir or
                   f"logs/pcd_replay_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, gts, lat_ms = [], [], []
    t_all = time.time()
    for n, i in enumerate(steps):
        obs = build_obs(ep, i, metadata, args.proprio_dim, args.require_depth,
                        args.drop_depth)
        if n == 0:
            print("[replay] first obs payload:")
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: shape {v.shape} dtype {v.dtype}")
        t0 = time.perf_counter()
        conn.send(msgpack_numpy.packb(obs))
        resp = conn.recv()
        lat_ms.append((time.perf_counter() - t0) * 1e3)
        if isinstance(resp, str):
            raise SystemExit(f"[server] ERROR at frame {i}:\n{resp}")
        actions = np.asarray(msgpack_numpy.unpackb(resp)["actions"], dtype=np.float64)
        if actions.ndim == 3:
            actions = actions[0]
        preds.append(actions)
        gts.append(gt_chunk(ep.actions, i, actions.shape[0]))
        if (n + 1) % 10 == 0:
            print(f"[replay] {n + 1}/{len(steps)} frames "
                  f"({np.median(lat_ms):.0f} ms median round-trip)")
    conn.close()

    pred = np.stack(preds)   # (T, H, D)
    gt = np.stack(gts)       # (T, H, D)
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    # Scale-normalized: per-dim MAE / per-dim GT std over the replay window
    # (~1.0 would mean "as wrong as predicting a constant"; good tracks <<1).
    gt_std = gt.reshape(-1, gt.shape[-1]).std(axis=0)
    active = gt_std > 1e-6
    norm_mae = float(np.mean(
        np.abs(diff.reshape(-1, diff.shape[-1]))[:, active] / gt_std[active]
    ))
    first_mae = float(np.mean(np.abs(pred[:, 0] - gt[:, 0])))
    blocks = per_block_mae(pred, gt)

    lines = [
        f"dataset      : {root}  episode {args.episode}",
        f"frames       : {len(steps)} (stride {args.stride}) of {ep.n}",
        f"chunk        : horizon {pred.shape[1]}  dim {pred.shape[2]}",
        f"round-trip   : median {np.median(lat_ms):.0f} ms  p90 {np.percentile(lat_ms, 90):.0f} ms",
        f"full-chunk   : MSE {mse:.6f}   MAE {mae:.6f} rad",
        f"first-action : MAE {first_mae:.6f} rad",
        f"normalized   : MAE/GT-std {norm_mae:.3f}  (<<1 tracks, ~1 constant-level)",
        "per-block MAE (rad):",
    ] + [f"    {k:7s}: {v:.5f}" for k, v in blocks.items()]
    report = "\n".join(lines)
    print("\n" + "=" * 70 + "\n" + report)
    (out_dir / "metrics.txt").write_text(report + "\n")

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, H, D = pred.shape
    t_axis = np.array(steps)
    # 1. First-action trajectory vs GT for a representative dim per block.
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for ax, (name, (s, e)) in zip(axes.flat, ACTION_BLOCKS.items()):
        d = s + (e - s) // 2
        ax.plot(t_axis, gt[:, 0, d], "b-", label="GT", lw=1.5)
        ax.plot(t_axis, pred[:, 0, d], "r--", label="pred", lw=1.2)
        ax.set_title(f"{name} (dim {d})", fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(ACTION_BLOCKS):]:
        ax.axis("off")
    fig.suptitle(f"First-action tracking — MAE {first_mae:.4f} rad, "
                 f"norm {norm_mae:.3f} ({root.name} ep{args.episode})")
    fig.tight_layout()
    fig.savefig(out_dir / "first_action_tracking.png", dpi=110)
    plt.close(fig)

    # 2. Chunk overlays every ~T/6 steps for one arm dim.
    d_show = ACTION_BLOCKS["r_arm"][0] + 3 if D >= 25 else min(D - 1, 3)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_axis, gt[:, 0, d_show], "g-", lw=2, label="GT trajectory")
    for n in range(0, T, max(1, T // 6)):
        x = t_axis[n] + np.arange(H) * args.stride
        ax.plot(x, pred[n, :, d_show], "r-", alpha=0.55, lw=1)
        ax.plot(x, gt[n, :, d_show], "b-", alpha=0.4, lw=1)
    ax.set_title(f"Chunk overlays, dim {d_show} (red pred / blue GT chunks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "chunk_overlays.png", dpi=110)
    plt.close(fig)

    np.savez_compressed(out_dir / "replay_raw.npz",
                        pred=pred, gt=gt, steps=t_axis, lat_ms=np.asarray(lat_ms))
    print(f"\n[replay] wrote {out_dir}/ (metrics.txt, PNGs, replay_raw.npz) "
          f"in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
