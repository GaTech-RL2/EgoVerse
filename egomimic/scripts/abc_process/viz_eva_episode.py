#!/usr/bin/env python
"""Visualize an EVA (ABC-130k) zarr episode -> mp4.

Reads the camera frames straight out of the zarr store (no model / no GPU) and
writes a video. Optionally tiles multiple cameras side-by-side and burns in the
task description + frame index.

    ./emimic/bin/python egomimic/scripts/abc_process/viz_eva_episode.py \
        --episode 00022b62-0a07-4ac7-bb74-9b4da47f6c6e \
        --cams front_1 left_wrist right_wrist --out eva_ep.mp4

--episode accepts a bare hash (resolved under --folder, which defaults to the
dataset dir from hydra_configs/paths/default.yaml: $EGOVERSE_DATASET_DIR or
/coc/flash7/scratch/egoverseS3ZarrDataset) or a full .zarr path.

--draw-axes overlays, on the front_1 tile, the per-arm EE pose triads AND the
world-frame origin triad (X=red, Y=green, Z=blue), projected with the SAME
math the training pipeline uses: world pose -> cam frame via
PoseCoordinateFrameTransform against EXTRINSICS[--extrinsics-key], then pixels
via INTRINSICS[--intrinsics-key]. The front_1 tile is stretched to 640x480
first (the space that K and the abc_fold_viz solvePnP fit live in). Use
--still-frames to also dump overlaid PNGs of specific frames.
"""

import argparse
import os

import numpy as np
import zarr
from PIL import Image

from egomimic.scripts.zarr_episode_viz import (
    decode,
    draw_triad,
    episode_path,
    project,
    write_episode_video,
)
from egomimic.utils.paths import default_dataset_dir

DEFAULT_DATASET_DIR = default_dataset_dir()


# Viz space the overlay lives in: front_1 stretched to 640x480, matching both
# the pipeline's _resize_image_keys and the clicker/solvePnP fit.
VIZ_W, VIZ_H = 640, 480

# Viz-only calibrations, kept local to this script (training reads intrinsics
# per-episode from zarr.json and bakes Eva.EXTRINSICS; nothing else in the repo
# needs these). Keys that exist on the Eva class are added lazily from it in
# AxisOverlay (see _registries) so this module stays import-light.
#   eva K: ABC-130k RealSense top camera (640x480 space)
#   yam K: rectified stereo front cam, calibration-derived, 640x480 space
#   abc_fold_viz: solvePnP world->cam fit over ABC fold-clothes clicks (one
#     shared world frame for both arms); yam: per-arm base->cam hand-eye calib
#   eva (extrinsics, from Eva.EXTRINSICS): the per-arm base->cam bake training
#     uses; aria (intrinsics, from Eva.INTRINSICS): the class fallback K,
#     640x480 space
INTRINSICS = {
    "eva": np.array(
        [[436.26, 0.0, 310.1, 0.0], [0.0, 435.13, 241.93, 0.0], [0.0, 0.0, 1.0, 0.0]]
    ),
    "yam": np.array(
        [
            [250.15807185802416, 0.0, 267.5409836065574, 0.0],
            [0.0, 260.01789323969166, 152.72727272727272, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
}
_ABC_FOLD_VIZ = np.array(
    [
        [-0.00540728648402, -0.91864395735192, 0.39504941573641, -0.07230219027034],
        [-0.99924104589933, -0.01027605228132, -0.03757306135427, -0.23692835817344],
        [0.03857581422213, -0.39495275966919, -0.91789118319482, 0.97783070025080],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
EXTRINSICS = {
    "abc_fold_viz": {"left": _ABC_FOLD_VIZ, "right": _ABC_FOLD_VIZ},
    "yam": {
        "left": np.array(
            [
                [0.0437291, -0.85821391, 0.5114261, 0.08690521],
                [-0.99754573, -0.00948836, 0.06937217, -0.25327518],
                [-0.05468356, -0.5132045, -0.85652253, 0.83357606],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "right": np.array(
            [
                [0.04251619, -0.84174054, 0.53820557, 0.04959916],
                [-0.98836195, 0.04331659, 0.14582295, 0.27905066],
                [-0.14605832, -0.53814174, -0.83010266, 0.87148985],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    },
}
# Intrinsics keys whose K lives in the stretched 640x480 space (the pipeline's
# _resize_image_keys target); YAM's K is already in the stored image space.
_STRETCHED_INTRINSICS = {"eva", "aria"}


def _registries():
    """Local viz calibrations + the values that live on the Eva class."""
    from egomimic.rldb.embodiment.eva import Eva

    return (
        {**INTRINSICS, "aria": Eva.INTRINSICS},
        {**EXTRINSICS, "eva": Eva.EXTRINSICS},
    )


class AxisOverlay:
    """Project EE / world triads into front_1 pixels via the pipeline transform."""

    def __init__(self, extrinsics_key: str, intrinsics_key: str = "eva"):
        from egomimic.rldb.zarr.action_chunk_transforms import (
            PoseCoordinateFrameTransform,
        )
        from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyzwxyz_to_matrix

        intrinsics, extrinsics = _registries()
        self._to_mat = _xyzwxyz_to_matrix
        self.K = intrinsics[intrinsics_key]
        # EVA's K lives in the pipeline's stretched 640x480 space; YAM's K is
        # already in the stored image space -> draw at native resolution there.
        self.stretch = intrinsics_key in _STRETCHED_INTRINSICS
        extr = extrinsics[extrinsics_key]
        # cam<-world pose target per arm, exactly as the Eva transform list bakes it.
        self.extr_pose = {
            s: _matrix_to_xyzwxyz(np.asarray(extr[s], dtype=np.float64)[None])[0]
            for s in ("left", "right")
        }
        # One shared world frame (e.g. abc_fold_viz) vs per-arm base frames (yam).
        self.shared_base = np.allclose(extr["left"], extr["right"])
        self._t = PoseCoordinateFrameTransform(
            target_world="extr",
            pose_world="pose",
            transformed_key_name="out",
            mode="xyzwxyz",
        )
        self.key = extrinsics_key

    def cam_pose_matrix(self, world_pose_xyzwxyz, side: str) -> np.ndarray:
        """World-frame pose (7,) -> 4x4 cam<-ee matrix (pipeline-identical)."""
        out = self._t.transform(
            {
                "extr": self.extr_pose[side],
                "pose": np.asarray(world_pose_xyzwxyz, dtype=np.float64),
            }
        )["out"]
        return self._to_mat(np.asarray(out, dtype=np.float64)[None])[0]

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        return project(self.K, pts_cam)

    def draw_triad(self, img, M_cam_pose, length: float, label: str, thickness=2):
        draw_triad(img, self.K, M_cam_pose, length, label, thickness)

    def overlay(self, front_native, lpose_w, rpose_w):
        """front_1 native frame + world EE poses -> 640x480 RGB with triads.

        Returns (img, info) where info holds the projected EE origin pixels.
        """
        import cv2

        if self.stretch:
            img = cv2.resize(
                front_native, (VIZ_W, VIZ_H), interpolation=cv2.INTER_LINEAR
            )
        else:
            # PIL-decoded arrays are read-only; cv2 draws in place -> copy.
            img = front_native.copy()
        img = np.ascontiguousarray(img)
        h = img.shape[0]
        info = {}
        # Base-frame triad(s): one shared world frame, or per-arm bases (yam).
        ident = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        bases = (
            [("left", "world")]
            if self.shared_base
            else [("left", "L-base"), ("right", "R-base")]
        )
        for side, lab in bases:
            self.draw_triad(
                img, self.cam_pose_matrix(ident, side), 0.15, lab, thickness=3
            )
        for side, pose_w, lab in (
            ("left", lpose_w, "L-ee"),
            ("right", rpose_w, "R-ee"),
        ):
            # ABC episodes zero-pad the tail frames -> zero-norm quat; skip them.
            if np.linalg.norm(np.asarray(pose_w, dtype=np.float64)[3:7]) < 1e-6:
                info[side] = np.array([np.nan, np.nan])
                continue
            M = self.cam_pose_matrix(pose_w, side)
            self.draw_triad(img, M, 0.08, lab)
            info[side] = self.project(M[:3, 3][None])[0]
        cv2.putText(
            img,
            f"axes X=red Y=green Z=blue  extr={self.key}",
            (6, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="episode hash or full .zarr path")
    ap.add_argument(
        "--folder",
        default=DEFAULT_DATASET_DIR,
        help="zarr dataset dir a bare --episode hash is resolved under "
        "(default: $EGOVERSE_DATASET_DIR or the lab dataset dir)",
    )
    ap.add_argument(
        "--cams", nargs="+", default=["front_1"], help="front_1 left_wrist right_wrist"
    )
    ap.add_argument(
        "--out", default=None, help="output mp4 (default: ./eva_<hash>.mp4)"
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--draw-axes",
        action="store_true",
        help="overlay EE + world XYZ triads on front_1 (see module docstring)",
    )
    ap.add_argument(
        "--extrinsics-key",
        default="abc_fold_viz",
        help="EXTRINSICS key for the world->cam bake "
        "(abc_fold_viz | yam | eva; default: abc_fold_viz)",
    )
    ap.add_argument(
        "--intrinsics-key",
        default="eva",
        help="INTRINSICS key for projection (eva | yam | aria)",
    )
    ap.add_argument(
        "--still-frames",
        type=int,
        nargs="*",
        default=[],
        help="also dump these frame indices as overlaid PNGs next to --out",
    )
    args = ap.parse_args()

    path = episode_path(args.episode, args.folder)
    g = zarr.open_group(path, mode="r")
    a = dict(g.attrs)
    fps = int(a.get("fps", 30))
    n = int(a.get("total_frames", 0))
    print(
        f"[viz] {path}\n  task={a.get('task_description') or a.get('task_name')!r} "
        f"frames={n} fps={fps} cams={args.cams}"
    )

    overlay = None
    lobs = robs = None
    if args.draw_axes:
        if "front_1" not in args.cams:
            args.cams = ["front_1"] + args.cams
        overlay = AxisOverlay(args.extrinsics_key, args.intrinsics_key)
        lobs, robs = g["left.obs_ee_pose"], g["right.obs_ee_pose"]
        print(
            f"[viz] axis overlay on front_1: extrinsics_key={args.extrinsics_key}, "
            f"K=INTRINSICS['{args.intrinsics_key}'], stretch={overlay.stretch}"
        )

    cam_arrs = {c: g[f"images.{c}"] for c in args.cams}
    # Arrays are zero-padded past attrs total_frames -> cap at the real length.
    nf = min(
        next(iter(cam_arrs.values())).shape[0], n or 10**9, args.max_frames or 10**9
    )
    out = args.out or f"eva_{os.path.basename(path).replace('.zarr', '')}.mp4"
    stills = sorted(set(i for i in args.still_frames if 0 <= i < nf))

    def render(i):
        tiles = []
        for c in args.cams:
            f = decode(cam_arrs[c][i])
            if overlay is not None and c == "front_1":
                f, info = overlay.overlay(f, np.asarray(lobs[i]), np.asarray(robs[i]))
                if i in stills:
                    print(
                        f"[viz] frame {i}: EE pixels L={np.round(info['left'], 1)} "
                        f"R={np.round(info['right'], 1)}"
                    )
            tiles.append(f)
        h = min(t.shape[0] for t in tiles)
        tiles = [
            t[:h]
            if t.shape[0] == h
            else np.asarray(
                Image.fromarray(t).resize((int(t.shape[1] * h / t.shape[0]), h))
            )
            for t in tiles
        ]
        return np.concatenate(tiles, axis=1)

    write_episode_video(
        out, fps, nf, render, (a.get("task_description") or "")[:60], stills
    )


if __name__ == "__main__":
    main()
