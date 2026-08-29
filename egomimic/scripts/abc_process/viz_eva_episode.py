#!/usr/bin/env python
"""Visualize a local EVA (ABC-130k) zarr episode -> mp4.

Reads the camera frames straight out of the zarr store (no model / no GPU) and
writes a video. Optionally tiles multiple cameras side-by-side and burns in the
task description + frame index.

    ./emimic/bin/python egomimic/scripts/abc_process/viz_eva_episode.py \
        --episode 00022b62-0a07-4ac7-bb74-9b4da47f6c6e \
        --cams front_1 left_wrist right_wrist --out /workspace/EgoVerse/eva_ep.mp4

--episode accepts a bare hash (resolved under --folder) or a full .zarr path.

--draw-axes overlays, on the front_1 tile, the per-arm EE pose triads AND the
world-frame origin triad (X=red, Y=green, Z=blue), projected with the SAME
math the training pipeline uses: world pose -> cam frame via
PoseCoordinateFrameTransform against EXTRINSICS[--extrinsics-key], then pixels
via INTRINSICS["eva"]. The front_1 tile is stretched to 640x480 first (the
space that K and the abc_fold_viz solvePnP fit live in). Use --still-frames to
also dump overlaid PNGs of specific frames.
"""

import argparse
import io
import os

import numpy as np
import zarr


def decode(elem):
    """An image stored in the object array -> HxWx3 uint8.

    Frames are JPEG bytes wrapped in a 0-d object ndarray (sometimes nested), so
    unwrap to the raw bytes and decode with PIL (imageio can't sniff a bare BytesIO).
    """
    while isinstance(elem, np.ndarray) and elem.dtype == object and elem.ndim == 0:
        elem = elem.item()
    if isinstance(elem, np.ndarray) and elem.ndim >= 2:
        return elem[..., :3].astype(np.uint8)
    raw = bytes(elem) if not isinstance(elem, (bytes, bytearray)) else elem
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


# Viz space the overlay lives in: front_1 stretched to 640x480, matching both
# the pipeline's _resize_image_keys and the clicker/solvePnP fit.
VIZ_W, VIZ_H = 640, 480
AXIS_COLORS = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}  # RGB

# Inlined from the remote (wristframe-6d) repo's egomimicUtils registries —
# the local repo has no INTRINSICS/EXTRINSICS dicts (training reads intrinsics
# per-episode from zarr.json; these viz calibrations exist only on the remote
# branch).
#   eva K: ABC-130k RealSense top camera (640x480 space)
#   yam K: rectified stereo front cam, calibration-derived, 640x480 space
#   abc_fold_viz: solvePnP world->cam fit over ABC fold-clothes clicks (one
#     shared world frame for both arms); yam: per-arm base->cam hand-eye calib
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


class AxisOverlay:
    """Project EE / world triads into front_1 pixels via the pipeline transform."""

    def __init__(self, extrinsics_key: str, intrinsics_key: str = "eva"):
        from egomimic.rldb.zarr.action_chunk_transforms import (
            PoseCoordinateFrameTransform,
        )
        from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyzwxyz_to_matrix

        self._to_mat = _xyzwxyz_to_matrix
        self.K = INTRINSICS[intrinsics_key]
        # EVA's K lives in the pipeline's stretched 640x480 space; YAM's K is
        # already in the stored image space -> draw at native resolution there.
        self.stretch = intrinsics_key == "eva"
        extr = EXTRINSICS[extrinsics_key]
        # cam<-world pose target per arm, exactly as Eva.get_transform_list bakes it.
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
        """(N,3) cam-frame points -> (N,2) pixels; NaN where behind the camera."""
        px = np.full((len(pts_cam), 2), np.nan)
        front = pts_cam[:, 2] > 1e-6
        if front.any():
            p = np.concatenate([pts_cam[front], np.ones((front.sum(), 1))], axis=1)
            uv = self.K @ p.T
            px[front] = (uv[:2] / uv[2]).T
        return px

    def draw_triad(self, img, M_cam_pose, length: float, label: str, thickness=2):
        """Draw an XYZ triad for a cam<-frame pose matrix onto img (RGB, 640x480)."""
        import cv2

        o = M_cam_pose[:3, 3]
        pts = np.stack([o] + [o + M_cam_pose[:3, k] * length for k in range(3)])
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
                0.45,
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
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

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
    ap.add_argument("--folder", default="/workspace/eva/abc130k_zarr")
    ap.add_argument(
        "--cams", nargs="+", default=["front_1"], help="front_1 left_wrist right_wrist"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--draw-axes",
        action="store_true",
        help="overlay EE + world XYZ triads on front_1 (see module docstring)",
    )
    ap.add_argument(
        "--extrinsics-key",
        default="abc_fold_viz",
        help="EXTRINSICS key for the world->cam bake (default: abc_fold_viz)",
    )
    ap.add_argument(
        "--intrinsics-key",
        default="eva",
        help="INTRINSICS key for projection (eva | yam)",
    )
    ap.add_argument(
        "--still-frames",
        type=int,
        nargs="*",
        default=[],
        help="also dump these frame indices as overlaid PNGs next to --out",
    )
    args = ap.parse_args()

    path = (
        args.episode
        if args.episode.endswith(".zarr")
        else os.path.join(args.folder, f"{args.episode}.zarr")
    )
    g = zarr.open_group(path, mode="r")
    a = dict(g.attrs)
    fps = int(a.get("fps", 30))
    n = int(a.get("total_frames", 0))
    print(
        f"[viz] {path}\n  task={a.get('task_description') or a.get('task_name')!r} frames={n} fps={fps} cams={args.cams}"
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
    out = (
        args.out
        or f"/workspace/EgoVerse/eva_{os.path.basename(path).replace('.zarr','')}.mp4"
    )
    stills = sorted(set(i for i in args.still_frames if 0 <= i < nf))

    import imageio.v2 as imageio

    try:
        from PIL import Image, ImageDraw
    except Exception:
        Image = None
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    for i in range(nf):
        tiles = []
        for c in args.cams:
            f = decode(cam_arrs[c][i])
            if overlay is not None and c == "front_1":
                f, info = overlay.overlay(f, np.asarray(lobs[i]), np.asarray(robs[i]))
                if i in stills:
                    print(
                        f"[viz] frame {i}: EE pixels L={np.round(info['left'],1)} "
                        f"R={np.round(info['right'],1)}"
                    )
            tiles.append(f)
        h = min(t.shape[0] for t in tiles)
        tiles = (
            [
                t[:h]
                if t.shape[0] == h
                else np.asarray(
                    Image.fromarray(t).resize((int(t.shape[1] * h / t.shape[0]), h))
                )
                for t in tiles
            ]
            if Image
            else tiles
        )
        frame = np.concatenate(tiles, axis=1)
        if Image:
            im = Image.fromarray(frame)
            d = ImageDraw.Draw(im)
            d.text(
                (6, 6),
                f"{i}/{nf}  {(a.get('task_description') or '')[:60]}",
                fill=(255, 255, 0),
            )
            frame = np.asarray(im)
        if i in stills:
            still_path = f"{os.path.splitext(out)[0]}_f{i}.png"
            Image.fromarray(frame).save(still_path)
            print(f"[viz] still -> {still_path}")
        writer.append_data(frame)
    writer.close()
    print(f"[viz] wrote {nf} frames -> {out}")


if __name__ == "__main__":
    main()
