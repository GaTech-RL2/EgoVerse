"""Convert RoboTwin 2.0 episodes to EgoVerse Zarr (the reusable-corpus path).

Companion to the in-memory shim (``egomimic/ricl/robotwin_data.py``): where the
shim feeds RoboTwin straight into RICL training, this converter writes RoboTwin
demos as first-class EgoVerse Zarr stores via the supported ``ZarrWriter`` so they
can flow through the standard ``MultiDataset`` pipeline and cotrain alongside the
eva/human corpus later.

It labels episodes as ``eva_bimanual`` (no new embodiment registration) and emits
exactly the zarr keys ``Eva._get_keymap("cartesian_pi")`` reads, in the
**cartesian end-effector** representation:

    images.front_1 / images.left_wrist / images.right_wrist  (JPEG)
    left.obs_ee_pose  / right.obs_ee_pose   (T, 7)  XYZWXYZ  [x,y,z,qw,qx,qy,qz]
    left.cmd_ee_pose  / right.cmd_ee_pose   (T, 7)  == obs pose (chunked at load)
    left.obs_gripper  / right.obs_gripper   (T, 1)
    left.cmd_gripper  / right.cmd_gripper   (T, 1)

RoboTwin's ``endpose`` is the SAPIEN ``pose.p + pose.q`` = ``[x,y,z,qw,qx,qy,qz]``,
which is already EgoVerse's XYZWXYZ order (no quaternion reordering). ``cmd_*`` is
set equal to ``obs_*`` — the same "observed pose IS the action target, chunked at
load time" convention the aria human keymap uses — so the action chunk is the
future end-effector trajectory.

Requires the ``endpose`` group: the cartesian path needs end-effector poses. If an
episode only has ``joint_action`` (no ``endpose``), conversion raises with a clear
message (FK joints->ee, or a joint-space Zarr embodiment, are follow-ups — the
joint-space shim path is unaffected).

NOTE: Eva's ``cartesian`` transform re-expresses poses relative to *EVA's* fixed
camera extrinsics, which is not RoboTwin's true camera geometry; obs and cmd are
transformed by the same rigid extrinsic so relative motion is preserved, which is
sufficient for the shim-first plan. (A dedicated RoboTwin extrinsic/embodiment is a
later refinement if these Zarrs are used for serious cotraining.)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

EMBODIMENT = "eva_bimanual"
_CAM_TO_ZARR = {
    "head_camera": "images.front_1",
    "left_camera": "images.left_wrist",
    "right_camera": "images.right_wrist",
}


def _decode_cam(group: h5py.Group, cam: str, image_hw: tuple[int, int]) -> np.ndarray:
    """Decode a RoboTwin ``observation/<cam>/rgb`` JPEG dataset -> (T,H,W,3) RGB."""
    raw = group[f"{cam}/rgb"][()]
    H, W = image_hw
    out = np.empty((len(raw), H, W, 3), dtype=np.uint8)
    for i, b in enumerate(raw):
        img = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"failed to decode {cam} frame {i}")
        if tuple(img.shape[:2]) != (H, W):
            img = cv2.resize(img, (W, H))
        out[i] = img  # cv2 round-trips channel order -> renderer RGB preserved
    return out


def _read_instruction(task_dir: str, episode_name: str) -> str:
    path = os.path.join(task_dir, "instructions", f"{episode_name}.json")
    try:
        with open(path) as f:
            instr = json.load(f)
        seen = instr.get("seen") or instr.get("unseen") or []
        if seen:
            return str(seen[0])
    except FileNotFoundError:
        pass
    return ""


def convert_episode(
    hdf5_path: str,
    out_zarr: str,
    *,
    task_name: str,
    task_description: str = "",
    image_hw: tuple[int, int] = (224, 224),
    fps: int = 25,
) -> str:
    """Convert one RoboTwin ``episode*.hdf5`` to an EgoVerse Zarr store."""
    with h5py.File(hdf5_path, "r") as f:
        if "endpose" not in f:
            raise ValueError(
                f"{hdf5_path}: no 'endpose' group — the cartesian Zarr path needs "
                "end-effector poses. Re-collect RoboTwin with data_type.endpose=True, "
                "add FK (joints->ee), or use the joint-space shim (robotwin_data.py)."
            )
        ep = f["endpose"]
        left_pose = np.asarray(ep["left_endpose"][()], dtype=np.float64)  # (T,7)
        right_pose = np.asarray(ep["right_endpose"][()], dtype=np.float64)
        left_grip = np.asarray(ep["left_gripper"][()], dtype=np.float32).reshape(-1, 1)
        right_grip = np.asarray(ep["right_gripper"][()], dtype=np.float32).reshape(
            -1, 1
        )
        obs = f["observation"]
        images = {
            zarr_key: _decode_cam(obs, cam, image_hw)
            for cam, zarr_key in _CAM_TO_ZARR.items()
            if cam in obs
        }

    T = left_pose.shape[0]
    if not (right_pose.shape[0] == T == left_grip.shape[0] == right_grip.shape[0]):
        raise ValueError(f"{hdf5_path}: inconsistent frame counts")
    for k, v in images.items():
        if len(v) != T:
            raise ValueError(f"{hdf5_path}: {k} has {len(v)} frames, expected {T}")

    numeric = {
        # observed pose == commanded pose target; cmd is chunked into the future
        # trajectory at load time via the keymap horizon (the aria "obs as action").
        "left.obs_ee_pose": left_pose,
        "right.obs_ee_pose": right_pose,
        "left.cmd_ee_pose": left_pose.copy(),
        "right.cmd_ee_pose": right_pose.copy(),
        "left.obs_gripper": left_grip,
        "right.obs_gripper": right_grip,
        "left.cmd_gripper": left_grip.copy(),
        "right.cmd_gripper": right_grip.copy(),
    }
    annotations = [(task_description, 0, T)] if task_description else None

    writer = ZarrWriter(
        episode_path=out_zarr,
        embodiment=EMBODIMENT,
        fps=fps,
        task_name=task_name,
        task_description=task_description,
        annotations=annotations,
    )
    writer.write(numeric_data=numeric, image_data=images)
    return out_zarr


def convert_task(
    root: str,
    out_dir: str,
    *,
    image_hw: tuple[int, int] = (224, 224),
    fps: int = 25,
    limit: int | None = None,
) -> list[str]:
    """Convert every ``<task>/data/episode*.hdf5`` under ``root`` to Zarr stores in
    ``out_dir``, named ``<group>__<episode>.zarr``. Returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    data_dirs = sorted(glob.glob(os.path.join(root, "**", "data"), recursive=True))
    for data_dir in data_dirs:
        eps = sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5")))
        if not eps:
            continue
        task_dir = os.path.dirname(data_dir)
        rel = os.path.relpath(task_dir, root)
        gid = "root" if rel == "." else rel.replace(os.sep, "__")
        for ep in eps:
            ep_name = os.path.splitext(os.path.basename(ep))[0]
            desc = _read_instruction(task_dir, ep_name)
            out_zarr = os.path.join(out_dir, f"{gid}__{ep_name}.zarr")
            convert_episode(
                ep,
                out_zarr,
                task_name=gid,
                task_description=desc,
                image_hw=image_hw,
                fps=fps,
            )
            written.append(out_zarr)
            print(f"[zarr] {ep} -> {out_zarr}", flush=True)
            if limit is not None and len(written) >= limit:
                return written
    if not written:
        raise FileNotFoundError(f"no <task>/data/episode*.hdf5 found under {root!r}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", required=True, help="RoboTwin task root (parent of <task>/data)"
    )
    p.add_argument("--out", required=True, help="output dir for .zarr stores")
    p.add_argument("--limit", type=int, default=None, help="max episodes to convert")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()
    written = convert_task(
        args.root,
        args.out,
        image_hw=(args.height, args.width),
        fps=args.fps,
        limit=args.limit,
    )
    print(f"wrote {len(written)} zarr stores -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
