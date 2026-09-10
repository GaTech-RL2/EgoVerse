#!/usr/bin/env python
"""Visualize a Mecka (egocentric human) zarr episode -> mp4 with frame overlays.

Egocentric analog of abc_process/viz_eva_episode.py --draw-axes: projects into
the moving head camera (per-frame ``obs_head_pose`` is the world->cam target,
exactly the headframe transform training uses) with INTRINSICS["mecka"].

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
"""

import argparse
import io
import os

import numpy as np
import zarr


def decode(elem):
    while isinstance(elem, np.ndarray) and elem.dtype == object and elem.ndim == 0:
        elem = elem.item()
    if isinstance(elem, np.ndarray) and elem.ndim >= 2:
        return elem[..., :3].astype(np.uint8)
    raw = bytes(elem) if not isinstance(elem, (bytes, bytearray)) else elem
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


AXIS_COLORS = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}  # RGB
# MediaPipe-style indices (verified on-data: kp0 roots all four finger chains,
# adjacent MCPs 5/9/13/17 are ~2.5cm apart). NOT Aria ordering.
WRIST_ROOT, MIDDLE_MCP = 0, 9


class MeckaOverlay:
    """World-frame geometry -> head-camera pixels via the training headframe math."""

    def __init__(self):
        # Local repo keeps intrinsics on the embodiment classes (there is no
        # egomimicUtils.INTRINSICS registry like the remote's).
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
        px = np.full((len(pts_cam), 2), np.nan)
        front = pts_cam[:, 2] > 1e-6
        if front.any():
            p = np.concatenate([pts_cam[front], np.ones((front.sum(), 1))], axis=1)
            uv = self.K @ p.T
            px[front] = (uv[:2] / uv[2]).T
        return px

    def draw_triad(self, img, M, length, label, thickness=2):
        import cv2

        o = M[:3, 3]
        pts = np.stack([o] + [o + M[:3, k] * length for k in range(3)])
        px = self.project(pts)
        if np.isnan(px[0]).any():
            return
        oi = tuple(np.round(px[0]).astype(int))
        for k, ax in enumerate("xyz"):
            if np.isnan(px[k + 1]).any():
                continue
            ei = tuple(np.round(px[k + 1]).astype(int))
            cv2.line(img, oi, ei, AXIS_COLORS[ax], thickness, cv2.LINE_AA)
            cv2.putText(
                img,
                ax,
                (ei[0] + 3, ei[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                AXIS_COLORS[ax],
                1,
                cv2.LINE_AA,
            )
        cv2.circle(img, oi, 3, (255, 255, 255), -1)
        cv2.putText(
            img,
            label,
            (oi[0] + 5, oi[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

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
    ap.add_argument("--episode", required=True)
    ap.add_argument(
        "--folder", default="/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--still-frames", type=int, nargs="*", default=[])
    args = ap.parse_args()

    path = (
        args.episode
        if args.episode.endswith(".zarr")
        else os.path.join(args.folder, f"{args.episode}.zarr")
    )
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
    out = (
        args.out
        or f"/workspace/EgoVerse/mecka_{os.path.basename(path).replace('.zarr','')}_axes.mp4"
    )
    stills = sorted(set(i for i in args.still_frames if 0 <= i < nf))

    ov = MeckaOverlay()
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    for i in range(nf):
        frame = ov.overlay(decode(imgs[i]), head[i], lp[i], rp[i], lk[i], rk[i])
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        d.text(
            (6, 6),
            f"{i}/{nf}  {(a.get('task_description') or '')[:60]}",
            fill=(255, 255, 0),
        )
        frame = np.asarray(im)
        if i in stills:
            still = f"{os.path.splitext(out)[0]}_f{i}.png"
            Image.fromarray(frame).save(still)
            print(f"[viz] still -> {still}")
        writer.append_data(frame)
    writer.close()
    print(f"[viz] wrote {nf} frames -> {out}")


if __name__ == "__main__":
    main()
