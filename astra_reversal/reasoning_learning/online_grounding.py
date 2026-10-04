"""Estimate camera-axis correspondence from one rollout's already executed prefix.

The labeler receives real images alone. Proprioception enters only the subsequent
fit; neither the projection nor its validation uses simulator object geometry.
"""

import numpy as np

from astra_reversal.demo_skill_agent import png_wire
from astra_reversal.records import digest

from .visual_grounding import fit_projection

AT_STEP = 120
FIT_STEPS = tuple(range(0, 120, 10))
VALIDATION_STEPS = tuple(range(5, 120, 10))


def estimate(history, ask_pixels):
    required = FIT_STEPS + VALIDATION_STEPS
    if any(step not in history for step in required):
        raise ValueError("Projection requires every recorded prefix observation")
    receipt = {
        "source": "current_collection_episode_only",
        "source_observation_sha256": {s: digest(history[s]) for s in required},
        "environment_actions_added": 0,
        "used_for_policy_training": False,
    }

    def labels(steps):
        frames = [
            {"step": s, "image": png_wire(history[s]["observation/image"])}
            for s in steps
        ]
        proposal = ask_pixels(frames)
        selected = [
            row
            for row in proposal["frames"]
            if row["visible"] and row["confidence"] >= 0.6
        ]
        xyz, uv = [], []
        for row in selected:
            raw = history[row["step"]]
            height, width = np.asarray(raw["observation/image"]).shape[:2]
            xyz.append(np.asarray(raw["observation/state"], dtype=float)[:3])
            uv.append(np.asarray(row["uv"]) * [width - 1, height - 1])
        return selected, np.asarray(xyz), np.asarray(uv)

    train, xyz, uv = labels(FIT_STEPS)
    receipt["fit_label_steps"] = [r["step"] for r in train]
    if len(train) < 6:
        return {**receipt, "accepted": False, "reason": "insufficient_fit_labels"}
    projection = fit_projection(xyz, uv)
    receipt["projection"] = projection
    if not projection["accepted"]:
        return {**receipt, "accepted": False, "reason": projection["reason"]}
    test, test_xyz, test_uv = labels(VALIDATION_STEPS)
    receipt["validation_label_steps"] = [r["step"] for r in test]
    if len(test) < 6:
        return {
            **receipt,
            "accepted": False,
            "reason": "insufficient_validation_labels",
        }
    predicted = test_xyz @ np.asarray(projection["jacobian_pixels_per_meter"]).T
    predicted += np.asarray(projection["offset_pixels"])
    errors = np.linalg.norm(predicted - test_uv, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    positions = np.concatenate([xyz, test_xyz])
    return {
        **receipt,
        "accepted": rms <= 8.0,
        "reason": "passed" if rms <= 8.0 else "inaccurate_separate_frame_projection",
        "validation_errors_pixels": errors.tolist(),
        "validation_rms_pixels": rms,
        "supported_xyz_min": positions.min(0).tolist(),
        "supported_xyz_max": positions.max(0).tolist(),
    }


def context(receipt, state):
    if not receipt or not receipt["accepted"]:
        return None
    xyz = np.asarray(state, dtype=float)[:3]
    low, high = (
        np.asarray(receipt["supported_xyz_min"]),
        np.asarray(receipt["supported_xyz_max"]),
    )
    return {
        "source": "Visible gripper-center labels and robot XYZ from this rollout's earlier real prefix; no previous rollout calibration or privileged object poses.",
        "external_image_pixels_per_world_meter": receipt["projection"][
            "jacobian_pixels_per_meter"
        ],
        "offset_pixels": receipt["projection"]["offset_pixels"],
        "coordinate_definition": "External camera u right, v down in pixels; world XYZ meters. Approximate grasp-center projection, not the bottle center.",
        "supported_xyz_min": low.tolist(),
        "supported_xyz_max": high.tolist(),
        "current_xyz": xyz.tolist(),
        "extrapolation_outside_box_meters": np.maximum(
            np.maximum(low - xyz, xyz - high), 0
        ).tolist(),
        "fit_loo_rms_pixels": receipt["projection"]["loo_rms_pixels"],
        "separate_frame_rms_pixels": receipt["validation_rms_pixels"],
        "limitations": "Local affine approximation to VLM labels, not independent ground-truth calibration. Extrapolation may be inaccurate. It gives approximate axis directions, not destination depth, object-to-gripper offset, collision clearance, dynamics or candidate outcomes. Keep uncertainty where these missing quantities matter.",
    }
