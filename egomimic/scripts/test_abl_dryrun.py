#!/usr/bin/env python3
"""abl0921 acceptance test (spec §6): per-checkpoint dry-run BEFORE the robot.

In-process (no websocket): loads each checkpoint like serve_policy does, sets
num_inference_steps, prints the checkpoint-DRIVEN obs contract (spec §2:
cam/proprio keys read from the checkpoint, ghost keys fed zeros), builds the
example-frame obs through this variant's §3 pipeline transform, infers, and
scores first-32 MAE vs the stored GT future actions.

    emimic/bin/python egomimic/scripts/test_abl_dryrun.py                 # all 26
    emimic/bin/python egomimic/scripts/test_abl_dryrun.py --variants V0 H5 G6

Reading the result:
  * The example frame is the HARD val frame — fully-trained dp3c_dual scores
    0.078 on it; basket-family ~0.1-0.2 is the sane band (spec §6).
  * A variant wildly off (>2x its neighbours) = §3 config mismatch — fix
    before deploying; it will not announce itself any other way.
  * exactness: EXACT = faithful re-crop from the example npz; APPROX = the
    npz (built with the V0 recipe) cannot reproduce this variant's crop
    (e.g. slab EXTENSIONS, un-greying) — elevated MAE there is expected and
    the LIVE pipeline (ABL= preset) is still correct.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

_EGOVERSE = Path(__file__).resolve().parents[2]
_SEW = Path("/home/aloha/RB_Y1_workspace/SEW-Geometric-Teleop")
sys.path.insert(0, str(_EGOVERSE))
sys.path.insert(0, str(_SEW))

VARIANTS = ["V0", "A1", "A2", "A3", "A4", "A5", "B2", "C3", "C4", "D1", "D3",
            "E4", "G1", "G4", "G6", "G7", "H3", "H4", "H5",
            "J1", "J2", "J3", "J3b", "J4", "J4b", "J5"]

# npz-level transform exactness (live pipeline is exact for ALL via ABL= preset)
EXACTNESS = {
    "A1": "APPROX (slab extension 0.05 m: content not in the npz)",
    "A2": "APPROX (slab extension to 3.0 m: content not in the npz)",
    "A3": "APPROX (1.0 m near-crop applied; 2.0-3.0 m content missing)",
    "A4": "EXACT crop / resampled",
    "A5": "EXACT crop / resampled",
    "B2": "EXACT crop / resampled",
    "C3": "APPROX (npz colours already eef-greyed; cannot un-grey)",
    "C4": "EXACT",
    "D1": "EXACT (2048 global from the _2048 npz, 1024 local)",
    "D3": "APPROX (FPS 1024->512 instead of full-pool->512)",
    "E4": "EXACT (rigid re-expression)",
}


def fps(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Plain farthest-point sampling indices (small n, numpy)."""
    m = len(pts)
    if m <= n:
        rng = np.random.default_rng(seed)
        return np.concatenate([np.arange(m), rng.choice(m, n - m)])
    sel = np.empty(n, dtype=np.int64)
    sel[0] = 0
    d = np.linalg.norm(pts - pts[0], axis=1)
    for i in range(1, n):
        sel[i] = int(d.argmax())
        d = np.minimum(d, np.linalg.norm(pts - pts[sel[i]], axis=1))
    return sel


def resample(p6: np.ndarray, n: int) -> np.ndarray:
    idx = fps(p6[:, :3].astype(np.float64), n)
    return p6[idx]


def load_frame(source: str, assets: Path):
    """(g6, l6, jp26, extras) from either the pp example npz or a basket
    training frame ('basket:<ep>:<t>'). Basket is the DEFAULT: the abl fleet
    is basket-trained, so a train frame gives a LOW, tight MAE band where a
    §3 config mismatch actually stands out (the dp3cmix npz is a PICK-PLACE
    frame — basket policies score ~0.3 there from task OOD alone)."""
    if source.startswith("basket"):
        import pandas as pd
        _, ep, t = (source.split(":") + ["0", "100"])[:3]
        pq = (_EGOVERSE / "datasets/dp3c_basket60_v2_slim/data/chunk-000"
              / f"episode_{int(ep):06d}.parquet")
        df = pd.read_parquet(pq)
        t = int(t)
        assert t + 64 <= len(df), f"frame {t}+64 > episode len {len(df)}"
        row = df.iloc[t]
        g6 = np.asarray(list(row["obs.aria_pcdc"]), np.float32).reshape(-1, 6)
        l6 = np.asarray(list(row["obs.aria_pcdc_local"]), np.float32).reshape(-1, 6)
        jp = np.asarray(list(row["obs.robot0_joint_pos"]), np.float64).ravel()
        gt = np.stack([np.asarray(list(df.iloc[t + i]["actions.joint_base_torso_head_arm_hand"]),
                                  np.float64) for i in range(64)])
        extras = {"eef_pose_glass": np.asarray(list(row["obs.eef_pose_glass"]), np.float32).ravel(),
                  "hand_left_qpos": np.asarray(list(row["obs.hand_left_qpos"]), np.float32).ravel(),
                  "hand_right_qpos": np.asarray(list(row["obs.hand_right_qpos"]), np.float32).ravel(),
                  "gt": gt, "tag": f"basket ep{ep} t{t} (TRAIN frame: expect LOW MAE)"}
        return g6, l6, jp, extras
    z = dict(np.load(assets / "dp3cmix_example_1024.npz", allow_pickle=True))
    g6 = np.asarray(z["obs.aria_pcdc"], np.float32).reshape(-1, 6)
    l6 = np.asarray(z["obs.aria_pcdc_local"], np.float32).reshape(-1, 6)
    jp = np.asarray(z["obs.robot0_joint_pos"], np.float64).ravel()
    extras = {"eef_pose_glass": np.asarray(z["obs.eef_pose_glass"], np.float32).ravel(),
              "hand_left_qpos": np.asarray(z["obs.hand_left_qpos"], np.float32).ravel(),
              "hand_right_qpos": np.asarray(z["obs.hand_right_qpos"], np.float32).ravel(),
              "gt": np.asarray(z["gt_actions_49_future"], np.float64),
              "tag": "dp3cmix pp example (OOD for basket ckpts: ~0.3 is normal)"}
    return g6, l6, jp, extras


def build_streams(variant: str, assets: Path, source: str):
    """Apply the variant's §3 transform to the source frame's streams."""
    from projects.rby1_teleop.utils.rby1_eef_frames import RBY1EefFK, T_DEV_RECT

    g6, l6, jp, extras = load_frame(source, assets)
    geo = RBY1EefFK().eef_geometry(jp[4:26])
    R_dev_rect = np.asarray(T_DEV_RECT, np.float64)[:3, :3]

    if variant == "D1":
        if not source.startswith("basket"):
            z2 = dict(np.load(assets / "dp3cmix_example_2048.npz", allow_pickle=True))
            g6 = np.asarray(z2["obs.aria_pcdc"], np.float32).reshape(-1, 6)   # 2048
        else:
            # basket frame only has 1024 global points: pad-duplicate to 2048
            rng = np.random.default_rng(0)
            g6 = np.concatenate([g6, g6[rng.choice(len(g6), 2048 - len(g6))]])
        return g6, l6, jp, extras                                            # local 1024
    if variant == "D3":
        return resample(g6, 512), resample(l6, 512), jp, extras
    if variant == "C4":
        g6, l6 = g6.copy(), l6.copy()
        g6[:, 3:] = 0.5
        l6[:, 3:] = 0.5
        return g6, l6, jp, extras
    if variant in ("A4", "A5"):
        r = 1.5 if variant == "A4" else 2.0
        p_rect = g6[:, :3].astype(np.float64) @ R_dev_rect   # inverse of Pg = P_rect @ R.T
        keep = (p_rect[:, 0] ** 2 + p_rect[:, 1] ** 2) < r * r
        return resample(g6[keep], len(g6)), l6, jp, extras
    if variant == "A3":
        p_rect = g6[:, :3].astype(np.float64) @ R_dev_rect
        keep = p_rect[:, 2] > 1.0
        if keep.sum() < 64:
            print(f"  [warn] A3 near-crop leaves {int(keep.sum())} pts; keeping all")
            return g6, l6, jp, extras
        return resample(g6[keep], len(g6)), l6, jp, extras
    if variant == "B2":
        # local stream is in the RIGHT-EEF frame; the training ball centre is
        # the RAW eef point, expressed in that frame.
        T = np.linalg.inv(geo["T_glass_eefR"])
        c = T[:3, :3] @ geo["p_eef_glass_raw"] + T[:3, 3]
        keep = np.linalg.norm(l6[:, :3].astype(np.float64) - c, axis=1) < 0.5
        if keep.sum() < 64:
            print(f"  [warn] B2 ball leaves {int(keep.sum())} pts; keeping all")
            return g6, l6, jp, extras
        return g6, resample(l6[keep], len(l6)), jp, extras
    if variant == "E4":
        Tbg = geo["T_base_glass"]
        g6 = g6.copy()
        g6[:, :3] = (g6[:, :3].astype(np.float64) @ Tbg[:3, :3].T + Tbg[:3, 3]).astype(np.float32)
        return g6, l6, jp, extras
    # V0 / A1 / A2 / C3 / G* / H* / J*: send the npz streams as-is
    return g6, l6, jp, extras


def schematic_dim(pol, key: str) -> int | None:
    """Expected flat dim for an obs key from the checkpoint's schematic."""
    try:
        df = pol._model.model.data_schematic.df
        rows = df[df["key_name"] == key]
        if len(rows):
            shape = rows.iloc[0]["shape"]
            if isinstance(shape, str):
                shape = tuple(int(s) for s in shape.strip("()[] ").split(",") if s.strip())
            return int(np.prod(shape))
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default=str(_EGOVERSE / "checkpoints/abl0921"))
    ap.add_argument("--assets", default=str(_EGOVERSE / "ai_docs/assets_rect_lut"))
    ap.add_argument("--source", default="basket:0:100",
                    help="'basket:<ep>:<t>' (slim-set TRAIN frame, default) or "
                         "'dp3cmix' (pp example npz — OOD for basket ckpts)")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--steps", type=int, default=16, help="num_inference_steps")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--contracts-out",
                    default=str(_EGOVERSE / "checkpoints/abl0921/contracts.json"))
    args = ap.parse_args()

    import torch
    from egomimic.models.denoising_policy import DenoisingPolicy
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.serving.egoverse_policy import EgoVersePolicy

    assets = Path(args.assets)
    results, contracts = [], {}
    for v in args.variants:
        ck = Path(args.ckpt_dir) / f"{v}.ckpt"
        if not ck.exists():
            print(f"[{v}] MISSING {ck}")
            continue
        print(f"\n=== {v} ({ck.name}) — {EXACTNESS.get(v, 'EXACT (V0 pipeline)')} ===")
        model = ModelWrapper.load_from_checkpoint(str(ck), weights_only=False)
        if getattr(model.model, "diffusion", False) and args.steps > 0:
            for head in model.model.nets["policy"].heads.values():
                if isinstance(head, DenoisingPolicy):
                    head.num_inference_steps = int(args.steps)
        pol = EgoVersePolicy(model)
        cam_keys, pro_keys = list(pol._cam_keys), list(pol._proprio_keys)
        print(f"  cam_keys={cam_keys}")
        print(f"  proprio_keys={pro_keys}")
        print(f"  horizon={pol._action_horizon} action_dim={pol._action_dim}")
        contracts[v] = {"cam_keys": cam_keys, "proprio_keys": pro_keys,
                        "action_horizon": int(pol._action_horizon),
                        "action_dim": int(pol._action_dim),
                        "num_inference_steps": int(args.steps)}

        g6, l6, jp, extras = build_streams(v, assets, args.source)
        if v == args.variants[0]:
            print(f"  frame: {extras['tag']}")
        obs = {}
        skipped_cams = []
        for k in dict.fromkeys(cam_keys):
            if k == "front_pcd_1":
                obs[k] = g6
            elif k == "front_pcd_2":
                obs[k] = l6
            else:
                # ghost camera key from a config merge (e.g. front_img_1):
                # the serving path skips keys absent from the payload, so the
                # faithful dry-run OMITS them rather than feeding garbage.
                skipped_cams.append(k)
        if skipped_cams:
            print(f"  [info] ghost camera keys omitted (as in live serving): {skipped_cams}")
        for k in pro_keys:
            dim = schematic_dim(pol, k)
            if k == "robot0_joint_pos":
                obs[k] = (jp[4:26] if dim == 22 and len(jp) == 26 else jp).astype(np.float32)
            elif k in extras:
                obs[k] = extras[k]
            elif k == "task_id":
                # spec §2: ghost key from a config merge — zeros pass through
                obs[k] = np.zeros(dim or 64, np.float32)
            else:
                print(f"  [warn] GHOST proprio key {k} (dim {dim}) -> zeros")
                obs[k] = np.zeros(dim or 1, np.float32)
        for k, val in obs.items():
            d = schematic_dim(pol, k)
            got = int(np.prod(np.asarray(val).shape))
            if d is not None and k in pro_keys and got != d:
                print(f"  [warn] {k}: sending {got} but schematic says {d}")

        gt = extras["gt"]
        maes = []
        ok_shape = True
        for r in range(args.repeats):
            out = pol.infer(obs)
            act = np.asarray(out["actions"], np.float64)
            if act.ndim == 3:
                act = act[0]
            if r == 0:
                ok_shape = act.shape == (64, 49)
                print(f"  chunk {act.shape}  (expect (64, 49): {'OK' if ok_shape else 'MISMATCH!'})")
            n = min(32, len(act), len(gt))
            maes.append(np.abs(act[:n, :49] - gt[:n, :49]).mean())
        m = float(np.mean(maes))
        print(f"  MAE(first {n}) over {args.repeats} samples: {m:.4f} "
              f"(min {min(maes):.4f} max {max(maes):.4f})")
        results.append((v, m, ok_shape))
        contracts[v]["dryrun_mae32"] = m
        del pol, model
        gc.collect()
        torch.cuda.empty_cache()

    Path(args.contracts_out).write_text(json.dumps(contracts, indent=2))
    print(f"\ncontracts + scores -> {args.contracts_out}")

    if results:
        med = float(np.median([m for _, m, _ in results]))
        print("\n===== SUMMARY =====")
        print("  band (calibrated 2026-09-21): basket TRAIN frame ~0.008-0.05 "
              "for a correctly-configured variant; dp3cmix pp npz ~0.3 (task "
              "OOD). A variant >2x the median = §3 config mismatch.")
        for v, m, oks in sorted(results, key=lambda t: t[1]):
            flags = []
            if not oks:
                flags.append("SHAPE MISMATCH")
            if m > 2 * med:
                flags.append("SUSPECT (>2x median — check the §3 config)")
            if v in EXACTNESS and "APPROX" in EXACTNESS[v]:
                flags.append("approx-input")
            print(f"  {v:4s}  MAE32 {m:.4f}  {' | '.join(flags)}")
        print(f"  median {med:.4f}")


if __name__ == "__main__":
    main()
