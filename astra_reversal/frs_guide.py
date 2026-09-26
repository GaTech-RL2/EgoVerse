"""Calibrated gripper guide for the FRS reasoner, never for policy inputs."""

import numpy as np
from PIL import Image, ImageDraw

from .records import digest


def project_policy_pixels(points, camera_transform, image_size=224):
    """OpenCV projection followed by the LIBERO policy's horizontal reflection.

    robosuite's OpenGL array is bottom-up. Its camera utility already converts
    that to top-down OpenCV coordinates. policy_observation flips both original
    image axes, so only the additional horizontal reflection remains here.
    """
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(camera_transform, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or matrix.shape != (4, 4)
        or not np.isfinite(points).all()
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid camera calibration or world points")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    projected = homogeneous @ matrix.T
    if np.any(projected[:, 2] <= 0):
        raise ValueError("Gripper guide is behind the camera")
    xy = projected[:, :2] / projected[:, 2:3]
    xy[:, 0] = image_size - 1 - xy[:, 0]
    return xy


def gripper_guide(observation, env):
    """Use robot proprioception, camera calibration and known table height only."""
    from robosuite import macros
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("FRS guide requires the pinned OpenGL image convention")
    image = np.asarray(observation["observation/image"])
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("FRS guide requires the native 224px RGB policy view")
    eef = np.asarray(observation["observation/state"][:3], dtype=np.float64)
    native = getattr(env, "env", env)
    table_height = float(native.workspace_offset[2])
    table_point = eef.copy()
    table_point[2] = table_height
    calibration = get_camera_transform_matrix(env.sim, "agentview", 224, 224)
    points = np.stack((eef, table_point))
    xy = project_policy_pixels(points, calibration)
    probes = np.stack(
        (
            table_point,
            table_point + [0.01, 0, 0],
            table_point + [0, 0.01, 0],
            table_point + [0, 0, 0.01],
        )
    )
    probe_pixels = project_policy_pixels(probes, calibration)
    deltas = probe_pixels[1:] - probe_pixels[0]
    # This fixed benchmark camera looks along the world X axis. Derive signs
    # from the actual displayed view rather than silently assuming image-right
    # is +worldY. Forward is camera-nearer (table pixels move down the image).
    signs = [int(np.sign(deltas[0, 1])), int(np.sign(deltas[1, 0])), 1]
    if 0 in signs or deltas[2, 1] >= 0:
        raise ValueError("Unexpected FRS camera/controller axis convention")
    # The policy image is copied; the navigation overlay never reaches pi05.
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)
    drawing.line([tuple(p) for p in xy], fill=(35, 100, 255, 155), width=2)
    x, y = xy[0]
    drawing.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(35, 100, 255, 200))
    guided = np.asarray(Image.alpha_composite(base, overlay).convert("RGB")).copy()
    return guided, {
        "camera": "agentview",
        "image_convention": "opengl_then_flip_both_axes",
        "table_height": table_height,
        "world_points": points.tolist(),
        "camera_transform": calibration.tolist(),
        "policy_pixels_xy": xy.tolist(),
        "axis_probe_delta_pixels": deltas.tolist(),
        "camera_to_controller_signs": signs,
        "raw_sha256": digest(image),
        "guide_sha256": digest(guided),
        "policy_receives_guide": False,
    }
