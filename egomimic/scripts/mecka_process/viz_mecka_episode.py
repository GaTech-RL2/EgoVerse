#!/usr/bin/env python
"""Visualize a Mecka (egocentric human) zarr episode -> mp4 with frame overlays.

Egocentric analog of abc_process/viz_eva_episode.py --draw-axes: projects into
the moving head camera (per-frame ``obs_head_pose`` is the world->cam target,
exactly the headframe transform training uses) with MECKA_INTRINSICS.

Overlays, per frame:
  - L/R wrist triads (X=red Y=green Z=blue) from ``obs_ee_pose`` — shows the
    hand-tracking wrist-frame convention (diagonal to hand geometry, mirrored
    between hands, unlike EVA/YAM tool frames).
  - hand keypoints (gray dots) + a WHITE line wrist-root -> middle-MCP =
    "knuckle-forward", the physical reference to compare the triads against.
  - the WORLD origin triad (thick, length 0.3m): origin sits on the floor under
    the operator's head at episode start, Z up (gravity-aligned).

    ./emimic/bin/python egomimic/scripts/mecka_process/viz_mecka_episode.py \
        --episode 696d0f031244464a4aac1b8f --still-frames 150 450 750

--episode accepts a bare hash (resolved under --folder, which defaults to the
dataset dir from hydra_configs/paths/default.yaml: $EGOVERSE_DATASET_DIR or
/coc/flash7/scratch/egoverseS3ZarrDataset) or a full .zarr path.
"""

import argparse
import os

import numpy as np
import zarr

from egomimic.scripts.zarr_episode_viz import (
    decode,
    draw_triad,
    episode_path,
    project,
    write_episode_video,
)
from egomimic.utils.paths import default_dataset_dir

DEFAULT_DATASET_DIR = default_dataset_dir()

# MediaPipe-style indices (verified on-data: kp0 roots all four finger chains,
# adjacent MCPs 5/9/13/17 are ~2.5cm apart). NOT Aria ordering.
WRIST_ROOT, MIDDLE_MCP = 0, 9


class MeckaOverlay:
    """World-frame geometry -> head-camera pixels via the training headframe math."""

    def __init__(self):
        # Intrinsics live on the embodiment module; imported lazily so --help
        # stays cheap.
        from egomimic.rldb.embodiment.human import MECKA_INTRINSICS
        from egomimic.rldb.zarr.action_chunk_transforms import (
            PoseCoordinateFrameTransform,
        )
        from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

        self.K = MECKA_INTRINSICS
        self._to_mat = _xyzwxyz_to_matrix
        self._t_pose = PoseCoordinateFrameTransform(
            target_world="head",
            pose_world="pose",
            transformed_key_name="out",
            mode="xyzwxyz",
        )
        self._t_xyz = PoseCoordinateFrameTransform(
            target_world="head",
            pose_world="pose",
            transformed_key_name="out",
            mode="xyz",
        )

    def cam_pose_matrix(self, head_pose, world_pose):
        out = self._t_pose.transform(
            {
                "head": np.asarray(head_pose, np.float64),
                "pose": np.asarray(world_pose, np.float64),
            }
        )["out"]
        return self._to_mat(np.asarray(out, np.float64)[None])[0]

    def cam_xyz(self, head_pose, world_xyz):
        """(N,3) world points -> (N,3) cam-frame points."""
        return np.stack(
            [
                self._t_xyz.transform(
                    {
                        "head": np.asarray(head_pose, np.float64),
                        "pose": np.asarray(p, np.float64),
                    }
                )["out"]
                for p in np.asarray(world_xyz, np.float64).reshape(-1, 3)
            ]
        )

    def project(self, pts_cam):
        return project(self.K, pts_cam)

    def draw_triad(self, img, M, length, label, thickness=2):
        draw_triad(img, self.K, M, length, label, thickness, font_scale=0.4)

    def overlay(self, img, head, lpose, rpose, lkp, rkp):
        import cv2

        # PIL-decoded arrays are read-only; cv2 draws in place -> copy.
        img = np.ascontiguousarray(img.copy())
        # world origin triad (floor under head-at-start, Z up)
        if np.linalg.norm(np.asarray(head)[3:7]) > 1e-6:
            Mw = self.cam_pose_matrix(head, np.array([0, 0, 0, 1, 0, 0, 0.0]))
            self.draw_triad(img, Mw, 0.3, "world", thickness=3)
            for pose, kp, lab in ((lpose, lkp, "L-wrist"), (rpose, rkp, "R-wrist")):
                if (
                    np.linalg.norm(np.asarray(pose)[3:7]) < 1e-6
                    or np.abs(kp).sum() < 1e-9
                ):
                    continue
                # keypoints + knuckle-forward reference line
                pc = self.cam_xyz(head, kp)
                px = self.project(pc)
                for p in px:
                    if not np.isnan(p).any():
                        cv2.circle(
                            img, tuple(np.round(p).astype(int)), 2, (200, 200, 200), -1
                        )
                a, b = px[WRIST_ROOT], px[MIDDLE_MCP]
                if not (np.isnan(a).any() or np.isnan(b).any()):
                    cv2.line(
                        img,
                        tuple(np.round(a).astype(int)),
                        tuple(np.round(b).astype(int)),
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                # the wrist-frame triad the pipeline actually uses
                M = self.cam_pose_matrix(head, pose)
                self.draw_triad(img, M, 0.07, lab)
        cv2.putText(
            img,
            "axes X=red Y=green Z=blue | white line = knuckle-forward",
            (6, img.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img


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
        "--out", default=None, help="output mp4 (default: ./mecka_<hash>_axes.mp4)"
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
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
    print(f"[viz] {path}\n  task={(a.get('task_description') or '')[:80]!r} fps={fps}")

    imgs = g["images.front_1"]
    head = np.asarray(g["obs_head_pose"])
    lp, rp = np.asarray(g["left.obs_ee_pose"]), np.asarray(g["right.obs_ee_pose"])
    lk = np.asarray(g["left.obs_keypoints"]).reshape(len(head), 21, 3)
    rk = np.asarray(g["right.obs_keypoints"]).reshape(len(head), 21, 3)
    nf = min(imgs.shape[0], len(head), args.max_frames or 10**9)
    out = args.out or f"mecka_{os.path.basename(path).replace('.zarr', '')}_axes.mp4"
    stills = [i for i in args.still_frames if 0 <= i < nf]

    ov = MeckaOverlay()
    write_episode_video(
        out,
        fps,
        nf,
        lambda i: ov.overlay(decode(imgs[i]), head[i], lp[i], rp[i], lk[i], rk[i]),
        (a.get("task_description") or "")[:60],
        stills,
    )


if __name__ == "__main__":
    main()
