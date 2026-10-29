"""Yam station inference for EgoVerse ``ModelWrapper`` checkpoints.

The checkpoint is self-contained (config tree + norm stats). The station only
declares what the training data cannot: the embodiment the model was trained
under, the camera mapping, and ``base_T_model``, which maps each arm base into
the model's proprio/action frame. Proprio and action widths select the
rotation layout (14 = xyz + ZYX euler + gripper per arm, 20 = xyz + 6D + gripper).
``eef_frame`` actions are relative to the measured end-effector pose.
"""

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from egomimic.robot.yam_station.interface import ARM_OFFSET, pose_matrix, pose_vector
from egomimic.robot.yam_station.teleop import rigid_transform


def validate_control_value(name, spec, value):
    kind = spec.get("type")
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"Inference control {name!r} requires a boolean")
    elif kind == "enum":
        choices = spec.get("choices")
        if (
            not isinstance(choices, (list, tuple))
            or not choices
            or any(not isinstance(choice, str) for choice in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ValueError(f"Inference control {name!r} needs unique string choices")
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Inference control {name!r} must be one of {choices}")
    elif kind in {"integer", "number"}:
        check = (
            (lambda v: type(v) is int)
            if kind == "integer"
            else (lambda v: type(v) in (int, float) and math.isfinite(v))
        )
        lower, upper = spec.get("min"), spec.get("max")
        if not check(lower) or not check(upper) or lower > upper:
            raise ValueError(f"Inference control {name!r} has invalid {kind} bounds")
        if not check(value) or not lower <= value <= upper:
            raise ValueError(
                f"Inference control {name!r} requires {kind} in [{lower}, {upper}]"
            )
        step = spec.get("step")
        if step is not None:
            if not check(step) or step <= 0:
                raise ValueError(f"Inference control {name!r} has invalid step")
            quotient = (value - lower) / step
            if not math.isclose(quotient, round(quotient), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Inference control {name!r} must use step {step}")
    else:
        raise ValueError(f"Inference control {name!r} has unsupported type {kind!r}")
    return value


class InvalidActionSample(ValueError):
    """One sampled plan is unusable; a fresh sample may still be valid."""


class EgoverseYamPolicy:
    action_type = "cartesian"

    def __init__(
        self,
        model,
        embodiment,
        base_T_model,
        camera_keys,
        action_frame,
        image_hw=None,
        prompt="",
        replan_every=30,
        sampling_steps=None,
        gripper_clip_tolerance=0.05,
        max_valid_samples=8,
    ):
        from egomimic.models.denoising_policy import DenoisingPolicy
        from egomimic.rldb.embodiment.embodiment import get_embodiment_id

        self.model = model
        self.embodiment = embodiment.lower()
        self.embodiment_id = get_embodiment_id(embodiment)
        norm = model.norm_stats
        self.action_key = model.ac_keys[self.embodiment_id]
        proprio = model.proprio_keys[self.embodiment_id]
        if len(proprio) != 1:
            raise ValueError(f"Expected one proprio key, the model has {proprio}")
        self.proprio_key = proprio[0]
        self.action_shape = tuple(norm.key_shape(self.action_key, self.embodiment_id))
        widths = {
            norm.key_shape(self.proprio_key, self.embodiment_id)[-1],
            self.action_shape[-1],
        }
        if len(self.action_shape) != 2 or len(widths) != 1 or widths - {14, 20}:
            raise ValueError(
                "Proprio and action must both be 14 (euler) or 20 (6D) wide"
            )
        self.width = widths.pop()
        if set(camera_keys.values()) != set(model.camera_keys[self.embodiment_id]):
            raise ValueError(
                f"camera_keys must cover the model's {model.camera_keys[self.embodiment_id]}"
            )
        if set(base_T_model) != set(ARM_OFFSET):
            raise ValueError("Both arms require an explicit base_T_model calibration")
        if action_frame not in ("eef_frame", "model_frame"):
            raise ValueError("action_frame must be eef_frame or model_frame")
        self.base_T_model = {a: rigid_transform(v) for a, v in base_T_model.items()}
        self.camera_keys, self.action_frame = dict(camera_keys), action_frame
        self.image_hw = None if image_hw is None else tuple(int(x) for x in image_hw)
        self.prompt = prompt
        self.gripper_clip_tolerance = float(gripper_clip_tolerance)
        if not 0 <= self.gripper_clip_tolerance <= 0.5:
            raise ValueError("gripper_clip_tolerance must be in [0, 0.5]")
        if type(max_valid_samples) is not int or not 1 <= max_valid_samples <= 16:
            raise ValueError("max_valid_samples must be an integer in [1, 16]")
        self.max_valid_samples = max_valid_samples
        self.adapter = None  # no decoder: the rollout's plan stats stay empty
        self._heads = [
            m for m in model.nets.modules() if isinstance(m, DenoisingPolicy)
        ]
        self._controls = {
            "replan_every": {
                "label": "Actions before replanning",
                "description": "Execute this prefix of each predicted chunk before predicting again.",
                "type": "integer",
                "min": 1,
                "max": self.action_shape[0],
                "step": 1,
            }
        }
        values = {"replan_every": replan_every}
        if self._heads:
            self._controls["sampling_steps"] = {
                "label": "Sampling steps",
                "description": "Flow/diffusion denoising steps per plan (default: the checkpoint's own).",
                "type": "integer",
                "min": 1,
                "max": 100,
                "step": 1,
            }
            values["sampling_steps"] = (
                self._heads[0].num_inference_steps
                if sampling_steps is None
                else sampling_steps
            )
        self.apply_inference_overrides(values)

    def inference_controls(self):
        return {
            name: {**spec, "value": getattr(self, name)}
            for name, spec in self._controls.items()
        }

    def apply_inference_overrides(self, overrides):
        unknown = set(overrides) - set(self._controls)
        if unknown:
            raise ValueError(f"Unknown inference control(s): {sorted(unknown)}")
        for name, value in overrides.items():
            validate_control_value(name, self._controls[name], value)
        for name, value in overrides.items():
            setattr(self, name, value)
            if name == "sampling_steps":
                for head in self._heads:
                    head.num_inference_steps = value
        return self.inference_controls()

    def execution_plan(self, prediction):
        return np.asarray(prediction)[: self.replan_every]

    def _key(self, name):
        """The dataset key a training batch carried for one model key name."""
        return (
            self.model.norm_stats.keyname_to_zarr_key(name, self.embodiment_id) or name
        )

    def _batch(self, obs):
        proprio = []
        for arm, offset in ARM_OFFSET.items():
            pose = np.linalg.inv(self.base_T_model[arm]) @ pose_matrix(
                obs["ee_poses"][offset : offset + 6]
            )
            rot = (
                Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX")
                if self.width == 14
                else np.r_[pose[:3, 0], pose[:3, 1]]
            )
            proprio.extend(np.r_[pose[:3, 3], rot, obs["joint_positions"][offset + 6]])
        values = {
            self._key(self.proprio_key): torch.tensor([proprio], dtype=torch.float32),
            # forward_eval crops its prediction to the reference action shape.
            self._key(self.action_key): torch.zeros((1, *self.action_shape)),
        }
        for camera, key in self.camera_keys.items():
            bgr = obs.get(camera)
            if (
                bgr is None
                or bgr.ndim != 3
                or bgr.shape[-1] != 3
                or bgr.dtype != np.uint8
            ):
                raise ValueError(f"Missing uint8 BGR observation: {camera}")
            image = (
                torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))
                .permute(2, 0, 1)[None]
                .float()
                / 255.0
            )
            if self.image_hw is not None:
                image = torch.nn.functional.interpolate(
                    image, self.image_hw, mode="bilinear", align_corners=False
                )
            values[self._key(key)] = image
        # Training batches arrive normalized (MultiDataset.__getitem__).
        values = self.model.norm_stats.normalize(values, self.embodiment_id)
        annotation_key = getattr(self.model, "annotation_key", None)
        if annotation_key is not None and self.prompt:
            values[annotation_key] = [[self.prompt]]
        return self.model.process_batch_for_training({self.embodiment: values})

    def _actions(self, native, obs):
        anchors = {
            arm: pose_matrix(obs["ee_poses"][offset : offset + 6])
            for arm, offset in ARM_OFFSET.items()
        }
        half, actions = self.width // 2, []
        for row in native:
            command = []
            for index, arm in enumerate(ARM_OFFSET):
                values = row[index * half : (index + 1) * half]
                tolerance = self.gripper_clip_tolerance
                if not -tolerance <= values[-1] <= 1 + tolerance:
                    raise InvalidActionSample(
                        "Predicted gripper opening is outside [0, 1]"
                    )
                if self.width == 14:
                    target = pose_matrix(values[:6])
                else:
                    a, b = values[3:6].astype(float), values[6:9].astype(float)
                    if np.linalg.norm(a) < 1e-8:
                        raise InvalidActionSample("Degenerate 6D rotation")
                    a /= np.linalg.norm(a)
                    b -= a * np.dot(a, b)
                    if np.linalg.norm(b) < 1e-8:
                        raise InvalidActionSample("Degenerate 6D rotation")
                    b /= np.linalg.norm(b)
                    target = np.eye(4)
                    target[:3, 3] = values[:3]
                    target[:3, :3] = np.column_stack([a, b, np.cross(a, b)])
                base = (
                    anchors[arm]
                    if self.action_frame == "eef_frame"
                    else self.base_T_model[arm]
                )
                command.extend(
                    np.r_[pose_vector(base @ target), np.clip(values[-1], 0, 1)]
                )
            actions.append(command)
        return np.asarray(actions, dtype=np.float64)

    @torch.no_grad()
    def predict(self, obs):
        batch = self._batch(obs)
        error = None
        for _ in range(self.max_valid_samples):
            native = self.model.forward_eval(batch)[
                f"{self.embodiment}_{self.action_key}"
            ]
            native = native.detach().float().cpu().numpy()
            if native.shape != (1, *self.action_shape) or not np.isfinite(native).all():
                raise ValueError(
                    f"Expected a finite (1, {self.action_shape}) prediction"
                )
            try:
                return self._actions(native[0], obs)
            except InvalidActionSample as caught:
                # Flow/diffusion sampling is stochastic: reject the whole plan
                # rather than clamping one actuator, then sample again.
                error = caught
        raise ValueError(
            f"Rejected all {self.max_valid_samples} sampled plan(s): {error}. "
            "No robot command was issued."
        )


def load_egoverse_policy(config):
    from egomimic.pl_utils.pl_model import ModelWrapper

    config = dict(config)
    device = torch.device(config.pop("device", "cuda:0"))
    wrapper = ModelWrapper.load_from_checkpoint(
        config.pop("checkpoint"), weights_only=False, map_location="cpu"
    )
    wrapper = wrapper.to(device).eval()
    wrapper.model.device = device
    return EgoverseYamPolicy(wrapper.model, **config)
