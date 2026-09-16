import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

import numpy as np
import torch

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    _viz_annotations,
    _viz_axes,
    _viz_rotation_txt,
    _viz_traj,
)


class EMBODIMENT(Enum):
    # All human demonstration data is one embodiment (HUMAN_*); the robot Eva is
    # the only non-human embodiment. There is NO vendor/source notion at the
    # embodiment level — the data source is recorded only in the SQL `lab` field.
    HUMAN_RIGHT_ARM = 1
    HUMAN_LEFT_ARM = 2
    HUMAN_BIMANUAL = 3
    EVA_RIGHT_ARM = 4
    EVA_LEFT_ARM = 5
    EVA_BIMANUAL = 6


EMBODIMENT_ID_TO_KEY = {member.value: member.name for member in EMBODIMENT}


def _intrinsics_from_batch(batch, i: int):
    """Return per-sample intrinsics from batch, or None if missing/NaN sentinel."""
    K = batch.get("intrinsics") if isinstance(batch, dict) else None
    if K is None:
        return None
    K_i = K[i]
    if isinstance(K_i, torch.Tensor):
        if torch.isnan(K_i).any():
            return None
        return K_i.detach().cpu().numpy()
    K_i = np.asarray(K_i)
    if np.isnan(K_i).any():
        return None
    return K_i


def get_embodiment(index):
    return EMBODIMENT_ID_TO_KEY.get(index, None)


# Human demo data written by the vendor-split registry carries vendor-tagged
# embodiment metadata (e.g. MECKA_BIMANUAL, SCALE_LEFT_ARM). Locally all human
# demonstration data is ONE embodiment (see the EMBODIMENT docstring; the
# source lives only in the SQL `lab` field), so those names collapse to
# HUMAN_*. Robot names (EVA_*) are never aliased.
HUMAN_VENDOR_PREFIXES = ("MECKA", "SCALE", "ARIA", "LIGHTWHEEL")


def canonical_embodiment_name(embodiment_name: str) -> str:
    """Upper-case EMBODIMENT member name, with legacy vendor prefixes
    (``MECKA_BIMANUAL`` ...) collapsed onto ``HUMAN_*``. Does not validate."""
    name = embodiment_name.upper()
    vendor, _, suffix = name.partition("_")
    if vendor in HUMAN_VENDOR_PREFIXES and suffix:
        return f"HUMAN_{suffix}"
    return name


def is_legacy_vendor_embodiment(embodiment_name: str) -> bool:
    """True for vendor-tagged human names that only resolve through aliasing."""
    return canonical_embodiment_name(embodiment_name) != embodiment_name.upper()


def get_embodiment_id(embodiment_name):
    return EMBODIMENT[canonical_embodiment_name(embodiment_name)].value


def _strip_pi_keymap_mode(cls, keymap_mode: str) -> str:
    """The ``*_pi`` keymap modes emitted PaliGemma camera names from the
    dataset. Datasets now use one naming for every algo; the Pi wrapper
    renames onto openpi's slots (egomimic.models.preprocess_pi_obs). The suffix
    is still accepted so configs saved by earlier runs (eval, resume) rebuild."""
    if not keymap_mode.endswith("_pi"):
        return keymap_mode
    base_mode = keymap_mode[: -len("_pi")]
    warnings.warn(
        f"keymap_mode '{keymap_mode}' is deprecated for {cls.__name__}; using "
        f"'{base_mode}'. Pi maps the dataset camera keys onto base_0_rgb / "
        "*_wrist_0_rgb itself.",
        FutureWarning,
        stacklevel=3,
    )
    return base_mode


# Rotation representations that are discontinuous on SO(3) -- Euler angles wrap
# at +-pi and gimbal-lock, quaternions double-cover -- so per-dim normalization
# and regression losses on them are ill-defined. Training targets the continuous
# 6D representation everywhere; these modes survive only at the data/rollout
# boundary and for checkpoints that predate it.
LEGACY_ROTATION_MODES = {
    "cartesian": "cartesian_6d",
    "cartesian_padded": "cartesian_6d",
    "cartesian_wristframe_ypr": "cartesian_wristframe_6d",
    "cartesian_wristframe_quat": "cartesian_wristframe_6d",
    "keypoints_headframe_ypr": "keypoints_headframe_6d",
    "keypoints_headframe_quat": "keypoints_headframe_6d",
    "keypoints_wristframe_ypr": "keypoints_wristframe_6d",
    "keypoints_wristframe_quat": "keypoints_wristframe_6d",
}


def _reject_legacy_rotation(cls, mode: str, allow_legacy_rotation: bool) -> None:
    replacement = LEGACY_ROTATION_MODES.get(mode)
    if replacement is None or allow_legacy_rotation:
        return
    raise ValueError(
        f"{cls.__name__} transform_list mode '{mode}' trains on a discontinuous "
        f"rotation; use '{replacement}' instead. Reading data written before the "
        "6D conversion (viz, rollout, a legacy checkpoint) needs an explicit "
        "allow_legacy_rotation=True."
    )


class Embodiment(ABC):
    """Base embodiment class. An embodiment is responsible for defining the transform pipeline that converts between the raw data in the dataset and the canonical representation used by the model."""

    INTRINSICS = None
    EXTRINSICS = None
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list() -> list[Transform]:
        """Returns the list of transforms that convert between the raw data in the dataset and the canonical representation used by the model."""
        raise NotImplementedError

    @classmethod
    def viz_transformed_batch(
        cls,
        batch,
        mode=Literal["traj", "traj+rotation", "axes", "annotations"],
        viz_batch_key="actions_cartesian",
        image_key=None,
        transform_list=None,
        **kwargs,
    ):
        """Visualizes a batch of transformed data."""
        if transform_list is not None:
            batch = cls.apply_transform(batch, transform_list)
        image_key = image_key or cls.VIZ_IMAGE_KEY
        mode = (mode or "traj").lower()
        B = batch[image_key].shape[0]
        image = _to_numpy(batch[image_key][0])
        if (
            hasattr(batch[viz_batch_key], "shape")
            and batch[viz_batch_key].shape[0] == B
        ):
            viz_data = _to_numpy(batch[viz_batch_key][0])
        else:
            viz_data = batch[viz_batch_key]
        return cls.viz(
            image=image,
            viz_data=viz_data,
            mode=mode,
            intrinsics=_intrinsics_from_batch(batch, 0),
            **kwargs,
        )

    @classmethod
    def viz(
        cls,
        image,
        viz_data,
        mode=Literal["traj", "traj+rotation", "axes", "annotations"],
        intrinsics=None,
        **kwargs,
    ):
        K = intrinsics if intrinsics is not None else cls.INTRINSICS
        if mode == "traj":
            return _viz_traj(
                image=image,
                actions=viz_data,
                intrinsics=K,
                **kwargs,
            )
        if mode == "traj+rotation":
            vis = _viz_traj(
                image=image,
                actions=viz_data,
                intrinsics=K,
                **kwargs,
            )
            return _viz_rotation_txt(
                image=vis,
                actions=viz_data,
                **kwargs,
            )
        if mode == "axes":
            return _viz_axes(
                image=image,
                actions=viz_data,
                intrinsics=K,
                **kwargs,
            )
        if mode == "annotations":
            return _viz_annotations(
                image=image,
                annotations=viz_data,
                **kwargs,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: ('traj', 'traj+rotation', 'axes', 'annotations')."
        )

    @classmethod
    def get_keymap(cls, keymap_mode: str, norm_mode: bool = False, annotation_key=None):
        """Returns a dictionary mapping from the raw keys in the dataset to the canonical keys used by the model."""
        key_map = cls._get_keymap(keymap_mode)
        if annotation_key is not None and not norm_mode:
            key_map[annotation_key] = {
                "key_type": "annotation_keys",
                "zarr_key": annotation_key,
            }
        if norm_mode:
            to_delete = [
                k
                for k, v in key_map.items()
                if v.get("key_type") in ("camera_keys", "annotation_keys")
            ]
            for k in to_delete:
                del key_map[k]
        return key_map

    @abstractmethod
    def _get_keymap(cls, keymap_mode: str):
        raise NotImplementedError

    @classmethod
    def viz_gt_preds(
        cls,
        predictions,
        batch,
        image_key,
        action_key,
        annotation_key=None,
        mode=Literal["traj", "traj+rotation", "axes", "keypoints"],
        gt_alpha=1.0,
        pred_alpha=0.7,
        **kwargs,
    ):
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()

        pred_actions = predictions[f"{embodiment_name}_{action_key}"]

        images = batch[image_key]
        actions = batch[action_key]
        if annotation_key is not None:
            annotations = batch[annotation_key]
        ims_list = []
        images = _to_numpy(images)
        actions = _to_numpy(actions)
        pred_actions = _to_numpy(pred_actions)
        for i in range(images.shape[0]):
            image = images[i]
            action = actions[i]
            pred_action = pred_actions[i]
            K_i = _intrinsics_from_batch(batch, i)
            ims = cls.viz(
                image,
                action,
                mode=mode,
                color="Greens",
                alpha=gt_alpha,
                intrinsics=K_i,
                **kwargs,
            )
            ims = cls.viz(
                ims,
                pred_action,
                mode=mode,
                color="Reds",
                alpha=pred_alpha,
                intrinsics=K_i,
                **kwargs,
            )
            if annotation_key is not None:
                ims = cls.viz(ims, [annotations[i]], mode="annotations", **kwargs)
            ims_list.append(ims)
        ims = np.stack(ims_list, axis=0)
        return ims

    @classmethod
    def apply_transform(cls, batch, transform_list: list[Transform]):
        if transform_list:
            batch_size = None
            for v in batch.values():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    batch_size = v.shape[0]
                    break

            if batch_size is not None:
                # Apply transforms per-sample (matching how ZarrDataset applies them)
                results = []
                for i in range(batch_size):
                    sample = {}
                    for k, v in batch.items():
                        if (
                            isinstance(v, (np.ndarray, torch.Tensor))
                            and v.shape[0] == batch_size
                        ):
                            sample[k] = (
                                v[i].cpu().numpy()
                                if isinstance(v, torch.Tensor)
                                else v[i]
                            )
                        else:
                            continue

                    for transform in transform_list:
                        sample = transform.transform(sample)
                    results.append(sample)

                batch = {}
                for k in results[0]:
                    vals = [r[k] for r in results]
                    if isinstance(vals[0], np.ndarray):
                        batch[k] = np.stack(vals, axis=0)
                    elif isinstance(vals[0], torch.Tensor):
                        batch[k] = torch.stack(vals, dim=0)
                    else:
                        batch[k] = vals
            else:
                for transform in transform_list:
                    batch = transform.transform(batch)

        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(torch.float32)

        return batch
