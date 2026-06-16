"""Fetch a small RoboTwin 2.0 slice (or synthesize a fixture) for the RICL bring-up.

RoboTwin 2.0 ships ~100k expert trajectories across 50 bimanual tasks on
HuggingFace (``TianxingChen/RoboTwin2.0``). For the EgoVerse RICL integration we
pull a slice — a few episodes (or a full task) — to (a) drive the HDF5->Zarr
converter and (b) feed the in-memory shim + RICL collate. This module does two
things:

- ``download_robotwin_slice`` : download + unzip one per-(task, embodiment, setting)
  archive from HF (``dataset/<task>/<embodiment>_<setting>_<N>.zip``). ``aloha-agilex``
  is the bimanual embodiment; ``clean`` (50 demos, ~230 MB) is the smallest.
- ``make_synthetic_robotwin_task`` : write a schema-faithful *synthetic* task dir
  (no network) so the converter and unit tests always have valid input without
  pulling a real archive.

Raw RoboTwin episode layout this targets (verified against a real
``aloha-agilex_clean_50`` archive + ``policy/pi05/scripts/process_data.py``)::

    <task_dir>/data/episode{i}.hdf5
        /joint_action/left_arm       (T, A)  float64   joint angles (A=6 for aloha-agilex)
        /joint_action/left_gripper   (T,)    float64   aperture
        /joint_action/right_arm      (T, A)  float64
        /joint_action/right_gripper  (T,)    float64
        /joint_action/vector         (T, 2A+2) float64 [l_arm, l_grip, r_arm, r_grip]
        /observation/{head,left,right,front}_camera/rgb (T,) S{maxlen} JPEG bytes (240x320)
        /endpose/left_endpose        (T, 7)  float64   [x,y,z,qw,qx,qy,qz] (SAPIEN p+q)
        /endpose/left_gripper        (T,)    float64
        /endpose/right_endpose       (T, 7)  float64
        /endpose/right_gripper       (T,)    float64
    <task_dir>/instructions/episode{i}.json  {"seen": [...], "unseen": [...]}

``endpose`` is optional in real RoboTwin (only when ``data_type.endpose`` is set;
the released aloha-agilex sets DO include it). The synthetic fixture defaults to
6-DOF arms (14-D state, like aloha-agilex) and always includes ``endpose``.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import h5py
import numpy as np

HF_REPO_ID = "TianxingChen/RoboTwin2.0"
HF_REPO_TYPE = "dataset"


# --------------------------------------------------------------------------- #
# Synthetic fixture (no network) — schema-faithful tiny RoboTwin task
# --------------------------------------------------------------------------- #
def _encode_jpeg_bytes(img_rgb: np.ndarray) -> bytes:
    """JPEG-encode an HWC RGB uint8 frame to bytes (cv2, RoboTwin-style).

    cv2.imencode/imdecode round-trips the array byte-for-byte regardless of
    channel-order interpretation, so a frame stored here decodes back identically
    (see module note in ``robotwin_data.image``)."""
    ok, buf = cv2.imencode(".jpg", img_rgb)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _store_jpeg_dataset(group: h5py.Group, name: str, frames_rgb: np.ndarray) -> None:
    """Store (T,H,W,3) RGB frames as a fixed-length ``S{maxlen}`` JPEG byte dataset
    (matching RoboTwin's ``observation/<cam>/rgb`` layout)."""
    encoded = [_encode_jpeg_bytes(f) for f in frames_rgb]
    maxlen = max(len(b) for b in encoded)
    padded = [b.ljust(maxlen, b"\0") for b in encoded]
    group.create_dataset(name, data=np.array(padded), dtype=f"S{maxlen}")


def _random_unit_quats(n: int, rng: np.random.Generator) -> np.ndarray:
    """n random unit quaternions in (w, x, y, z) order (SAPIEN convention)."""
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0  # canonical hemisphere (w >= 0)
    return q


def make_synthetic_robotwin_episode(
    hdf5_path: str,
    *,
    n_frames: int = 60,
    img_hw: tuple[int, int] = (56, 56),
    arm_dof: int = 6,
    with_endpose: bool = True,
    seed: int = 0,
) -> None:
    """Write one schema-faithful synthetic RoboTwin ``episode{i}.hdf5``.

    ``arm_dof`` defaults to 6 (the real aloha-agilex bimanual embodiment -> 14-D
    state, grippers at slots 6/13)."""
    rng = np.random.default_rng(seed)
    H, W = img_hw
    T = n_frames

    # Smooth-ish joint trajectories so action=next-state chunks are non-trivial.
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    left_arm = (0.3 * np.sin(2 * np.pi * (t + rng.random((1, arm_dof))))).astype(
        np.float32
    )
    right_arm = (0.3 * np.cos(2 * np.pi * (t + rng.random((1, arm_dof))))).astype(
        np.float32
    )
    left_grip = (0.5 + 0.5 * np.sin(2 * np.pi * t[:, 0])).astype(np.float32)
    right_grip = (0.5 + 0.5 * np.cos(2 * np.pi * t[:, 0])).astype(np.float32)
    vector = np.concatenate(
        [left_arm, left_grip[:, None], right_arm, right_grip[:, None]], axis=1
    ).astype(np.float32)  # (T, 2*arm_dof + 2)

    # Distinct per-frame imagery so DINOv2 / fake embeddings differ across frames.
    frames = rng.integers(0, 256, size=(T, H, W, 3), dtype=np.uint8)

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        ja = f.create_group("joint_action")
        ja.create_dataset("left_arm", data=left_arm)
        ja.create_dataset("left_gripper", data=left_grip)
        ja.create_dataset("right_arm", data=right_arm)
        ja.create_dataset("right_gripper", data=right_grip)
        ja.create_dataset("vector", data=vector)

        obs = f.create_group("observation")
        for cam in ("head_camera", "left_camera", "right_camera"):
            cam_grp = obs.create_group(cam)
            _store_jpeg_dataset(cam_grp, "rgb", frames)

        if with_endpose:
            ep = f.create_group("endpose")
            lpos = (0.1 * rng.standard_normal((T, 3))).astype(np.float32)
            rpos = (0.1 * rng.standard_normal((T, 3))).astype(np.float32)
            lq = _random_unit_quats(T, rng)
            rq = _random_unit_quats(T, rng)
            ep.create_dataset("left_endpose", data=np.concatenate([lpos, lq], axis=1))
            ep.create_dataset("left_gripper", data=left_grip)
            ep.create_dataset("right_endpose", data=np.concatenate([rpos, rq], axis=1))
            ep.create_dataset("right_gripper", data=right_grip)


def make_synthetic_robotwin_task(
    root: str,
    task: str = "pick_cube",
    *,
    n_episodes: int = 4,
    n_frames: int = 60,
    img_hw: tuple[int, int] = (56, 56),
    arm_dof: int = 6,
    with_endpose: bool = True,
    seed: int = 0,
) -> str:
    """Create ``<root>/<task>/{data,instructions}`` with ``n_episodes`` synthetic
    episodes. Returns the task dir (a single RICL "group")."""
    task_dir = os.path.join(root, task)
    data_dir = os.path.join(task_dir, "data")
    instr_dir = os.path.join(task_dir, "instructions")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(instr_dir, exist_ok=True)
    instr = {
        "seen": [f"perform the {task.replace('_', ' ')} task"],
        "unseen": [f"complete the {task.replace('_', ' ')} demonstration"],
    }
    for i in range(n_episodes):
        make_synthetic_robotwin_episode(
            os.path.join(data_dir, f"episode{i}.hdf5"),
            n_frames=n_frames,
            img_hw=img_hw,
            arm_dof=arm_dof,
            with_endpose=with_endpose,
            seed=seed + i,
        )
        with open(os.path.join(instr_dir, f"episode{i}.json"), "w") as fp:
            json.dump(instr, fp, indent=2)
    return task_dir


# --------------------------------------------------------------------------- #
# HuggingFace task-archive download (download + unzip)
# --------------------------------------------------------------------------- #
def download_robotwin_slice(
    out_dir: str,
    task: str = "beat_block_hammer",
    embodiment: str = "aloha-agilex",
    setting: str = "clean",
    extract: bool = True,
) -> str:
    """Download + unzip one RoboTwin task archive from HuggingFace.

    RoboTwin 2.0 ships raw demos as per-(task, embodiment, setting) zip archives at
    ``dataset/<task>/<embodiment>_<setting>_<N>.zip`` (e.g.
    ``dataset/beat_block_hammer/aloha-agilex_clean_50.zip``). ``aloha-agilex`` is the
    **bimanual** embodiment; ``clean`` is the smallest set (50 demos). The archive
    expands to the raw ``data/episode*.hdf5`` + ``instructions/`` layout the shim and
    converter consume.

    Returns the extraction dir (point ``--root`` at it; groups are discovered
    within). Raises with a clear message if the archive can't be found/listed so
    callers can fall back to ``make_synthetic_robotwin_task``. NOTE: even the
    smallest archive is ~230 MB.
    """
    import zipfile

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE)
    prefix = f"dataset/{task}/{embodiment}_{setting}_"
    cands = sorted(f for f in files if f.startswith(prefix) and f.endswith(".zip"))
    if not cands:
        tasks = sorted(
            {
                f.split("/")[1]
                for f in files
                if f.startswith("dataset/") and f.count("/") >= 2
            }
        )
        raise ValueError(
            f"no archive {prefix}*.zip in {HF_REPO_ID}; available tasks e.g. "
            f"{tasks[:8]} (embodiment: aloha-agilex/arx-x5/franka/piper/ur5; "
            f"setting: clean/randomized)"
        )

    def _n_demos(f: str) -> int:
        try:
            return int(f.rsplit("_", 1)[1].split(".")[0])
        except Exception:
            return 1 << 30

    zip_repo_path = min(cands, key=_n_demos)  # smallest set (clean_50 before _500)
    print(f"[download] {HF_REPO_ID}: {zip_repo_path}", flush=True)
    local_zip = hf_hub_download(
        HF_REPO_ID, zip_repo_path, repo_type=HF_REPO_TYPE, local_dir=out_dir
    )
    if not extract:
        return local_zip
    extract_dir = os.path.join(out_dir, "extracted", f"{task}__{embodiment}_{setting}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(extract_dir)
    print(f"[download] extracted -> {extract_dir}", flush=True)
    return extract_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["synthetic", "hf"],
        default="synthetic",
        help="synthetic: write a tiny local fixture (no network); "
        "hf: download a real slice from HuggingFace",
    )
    p.add_argument("--out", required=True, help="output root dir")
    p.add_argument("--task", default=None, help="RoboTwin task name")
    p.add_argument(
        "--embodiment", default="aloha-agilex", help="hf mode: bimanual=aloha-agilex"
    )
    p.add_argument("--setting", default="clean", help="hf mode: clean | randomized")
    p.add_argument("--episodes", type=int, default=4, help="synthetic: episodes")
    p.add_argument("--frames", type=int, default=60, help="frames/episode (synthetic)")
    args = p.parse_args()

    if args.mode == "synthetic":
        task_dir = make_synthetic_robotwin_task(
            args.out,
            task=args.task or "pick_cube",
            n_episodes=args.episodes,
            n_frames=args.frames,
        )
        print(f"wrote synthetic task -> {task_dir}", flush=True)
    else:
        task_dir = download_robotwin_slice(
            args.out,
            task=args.task or "beat_block_hammer",
            embodiment=args.embodiment,
            setting=args.setting,
        )
        print(f"downloaded slice -> {task_dir}", flush=True)


if __name__ == "__main__":
    main()
