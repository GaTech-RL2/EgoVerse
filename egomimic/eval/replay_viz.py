"""GT / prediction replay video: play each action chunk twice over the frames it
spans, first drawing where the ground-truth hand is at each step, then where
the prediction puts it.

Chunks are anchored every ``H * stride`` frames and do not overlap; step ``k``
of the chunk anchored at frame ``a`` is drawn on frame ``a + k * stride``.

A chunk lives in its anchor frame's camera, and the head moves while it plays.
Each step is mapped into its frame's camera with a rigid fit of the GT
keypoints (``gt_a[k]`` in cam ``a`` against ``gt_f[0]`` in cam ``f``, the same
hand at the same instant), so it needs no head pose. That fit exists only for
the 126-D head-frame keypoint layout; any other layout is drawn uncompensated,
which is exact for Eva's fixed camera.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from egomimic.rldb.embodiment.embodiment import _intrinsics_from_batch, get_embodiment
from egomimic.utils.pose_utils import cam_frame_to_cam_pixels
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import _prepare_viz_image

_KP_WIDTH = 126


@dataclass
class ReplayClip:
    """Consecutive video frames with the GT and predicted chunk of each."""

    images: np.ndarray  # (N, H, W, 3) uint8
    gt: np.ndarray  # (N, T, D)
    pred: np.ndarray  # (N, T, D)
    intrinsics: list  # N x (K | None)

    def __len__(self):
        return len(self.images)

    @classmethod
    def from_batch(cls, viz_fn, predictions, batch):
        """Pull what ``viz_fn`` (a ``viz_gt_preds`` partial) would draw."""
        kw = viz_fn.keywords
        name = get_embodiment(batch["embodiment"][0].item()).lower()
        images = _to_numpy(batch[kw["image_key"]])
        return cls(
            images=np.stack([_prepare_viz_image(im) for im in images]),
            gt=_to_numpy(batch[kw["action_key"]]).astype(np.float32),
            pred=_to_numpy(predictions[f"{name}_{kw['action_key']}"]).astype(
                np.float32
            ),
            intrinsics=[_intrinsics_from_batch(batch, i) for i in range(len(images))],
        )

    @classmethod
    def concat(cls, clips):
        return cls(
            images=np.concatenate([c.images for c in clips]),
            gt=np.concatenate([c.gt for c in clips]),
            pred=np.concatenate([c.pred for c in clips]),
            intrinsics=[k for c in clips for k in c.intrinsics],
        )

    def tail(self, start):
        return ReplayClip(
            self.images[start:],
            self.gt[start:],
            self.pred[start:],
            self.intrinsics[start:],
        )


def _valid_points(x):
    return np.isfinite(x).all(-1) & (np.abs(x) < 1e8).all(-1)


def rigid_fit(src, dst):
    """Least-squares ``R, t`` with ``dst ~= src @ R.T + t`` over valid pairs."""
    ok = _valid_points(src) & _valid_points(dst)
    if ok.sum() < 3:
        return np.eye(3), np.zeros(3)
    src, dst = src[ok].astype(np.float64), dst[ok].astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - mu_s).T @ (dst - mu_d))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_d - mu_s @ R.T


def _apply_rigid(chunk, R, t):
    pts = chunk.reshape(*chunk.shape[:-1], -1, 3)
    return (pts @ R.T + t).reshape(chunk.shape).astype(np.float32)


def _label(image, text, color):
    h = image.shape[0]
    scale = max(0.5, h / 700)
    thick = max(1, int(h / 350))
    org = (int(h * 0.03), int(h * 0.03) + int(24 * scale))
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )
    return image


def _draw_skeleton(image, keypoints, intrinsics, edges, color):
    """Both hands' 21 MANO keypoints (``(126,)``, camera frame) with bones."""
    vis = np.ascontiguousarray(_prepare_viz_image(image)).copy()
    h, w = vis.shape[:2]
    pts = keypoints.reshape(2, -1, 3)
    ok = _valid_points(pts) & (pts[..., 2] > 0.01)
    safe = np.where(ok[..., None], pts, [0.0, 0.0, 1.0])
    px = cam_frame_to_cam_pixels(safe.reshape(-1, 3), intrinsics)[:, :2]
    px = np.round(px).astype(np.int32).reshape(2, -1, 2)
    ok &= (px[..., 0] >= 0) & (px[..., 0] < w) & (px[..., 1] >= 0) & (px[..., 1] < h)
    bone = tuple(int(0.6 * c + 0.4 * 255) for c in color)
    for hand in range(2):
        for i, j in edges:
            if ok[hand, i] and ok[hand, j]:
                cv2.line(
                    vis, tuple(px[hand, i]), tuple(px[hand, j]), bone, 2, cv2.LINE_AA
                )
        for i in np.flatnonzero(ok[hand]):
            cv2.circle(vis, tuple(px[hand, i]), 3, color, -1, cv2.LINE_AA)
    return vis


def render_replay(
    clip: ReplayClip,
    embodiment_cls,
    mode: str,
    stride: int = 1,
    trail: int = 5,
    viz_kwargs: dict | None = None,
):
    """Render every complete chunk of ``clip``.

    The 126-D keypoint layout draws each step as both hands' skeletons;
    any other layout draws the last ``trail`` steps with ``embodiment_cls.viz``.

    Returns ``(frames (M, H, W, 3) uint8, consumed)``; ``clip.tail(consumed)``
    holds the frames a later clip needs to complete the next chunk.
    """
    viz_kwargs = viz_kwargs or {}
    n, horizon = len(clip), clip.gt.shape[1]
    span = horizon * stride
    keypoints = clip.gt.shape[-1] == _KP_WIDTH

    out = []
    a = 0
    while a + span <= n:
        fits = []
        for k in range(horizon):
            f = a + k * stride
            if keypoints:
                fits.append(
                    rigid_fit(
                        clip.gt[a, k].reshape(-1, 3), clip.gt[f, 0].reshape(-1, 3)
                    )
                )

        for role, chunk, color, rgb in (
            ("GT", clip.gt[a], "Greens", (80, 220, 80)),
            ("PRED", clip.pred[a], "Reds", (240, 80, 80)),
        ):
            for k in range(horizon):
                f = a + k * stride
                K = clip.intrinsics[f]
                if keypoints:
                    im = _draw_skeleton(
                        clip.images[f],
                        _apply_rigid(chunk[k], *fits[k]),
                        K if K is not None else embodiment_cls.INTRINSICS,
                        embodiment_cls.FINGER_EDGES,
                        rgb,
                    )
                else:
                    im = embodiment_cls.viz(
                        clip.images[f],
                        chunk[max(0, k - trail) : k + 1],
                        mode=mode,
                        color=color,
                        intrinsics=K,
                        **viz_kwargs,
                    )
                im = _label(np.ascontiguousarray(im), f"{role}  {k + 1}/{horizon}", rgb)
                out.extend([im] * stride)
        a += span

    if not out:
        return np.zeros((0, *clip.images.shape[1:]), np.uint8), 0
    return np.stack(out), a
