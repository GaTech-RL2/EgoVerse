import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import torch

from egomimic.rldb.embodiment.registry import (
    EndEffectorSpec,
    PlatformSpec,
    load_aliases,
    load_embodiment_platforms,
    load_end_effectors,
    load_platforms,
)
from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    _viz_annotations,
    _viz_axes,
    _viz_rotation_txt,
    _viz_traj,
)


class EMBODIMENT(Enum):
    # IDs 1 through 3 identify human data. IDs 4 through 6 identify EVA data.
    # The SQL `lab` field identifies the data source.
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


def canonical_embodiment_name(embodiment_name: str) -> str:
    """Return the lowercase canonical form of an embodiment name.

    Args:
        embodiment_name: A current, deprecated, or unknown embodiment name.

    Returns:
        The current name for a configured alias. A current name takes priority
        over an alias. The function returns an unknown name in lowercase.
    """
    name = str(embodiment_name).lower()
    if name.upper() in EMBODIMENT.__members__:
        return name
    return load_aliases().get(name, name)


def get_embodiment_id(embodiment_name):
    """Return the stable integer ID for an embodiment name.

    Args:
        embodiment_name: A current name or an alias in
            ``registry/aliases.yaml``. The lookup ignores letter case.

    Returns:
        The integer value of the matching ``EMBODIMENT`` member.

    Raises:
        KeyError: If the canonical name is not an ``EMBODIMENT`` member.
    """
    return EMBODIMENT[canonical_embodiment_name(embodiment_name).upper()].value


#: Valid keys for ``ResolvedEmbodiment.end_effectors``.
SIDES = ("left", "right")


def _import_embodiment_class(path: str) -> type["Embodiment"]:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"embodiment_class {path!r} is not a dotted path")
    return getattr(importlib.import_module(module_name), attr)


@dataclass(frozen=True)
class ResolvedEmbodiment:
    """Store a platform and its selected end-effectors.

    Attributes:
        platform: The selected platform specification.
        end_effectors: A mapping from ``"left"`` or ``"right"`` to the
            end-effector specification for that side.
        embodiment_name: The canonical embodiment name, if resolution started
            from a name.
    """

    platform: PlatformSpec
    end_effectors: Mapping[str, EndEffectorSpec]
    embodiment_name: str | None = None

    @property
    def action_space(self) -> str:
        """Return the action space shared by all selected end-effectors.

        Raises:
            ValueError: If the selected end-effectors specify different action
                spaces.
        """
        spaces = {ee.action_space for ee in self.end_effectors.values()}
        if len(spaces) != 1:
            raise ValueError(
                f"{self.describe()}: end-effectors disagree on action_space "
                f"({sorted(spaces)}); an episode trains through exactly one head"
            )
        return spaces.pop()

    @property
    def arity(self) -> str | None:
        """Return the arm configuration this embodiment name selects.

        Returns:
            One of the platform's ``arity`` values, or ``None`` when
            resolution started from a morphology block with no name.
        """
        if self.embodiment_name is None:
            return None
        prefix = f"{self.platform.embodiment_prefix}_"
        if self.embodiment_name.startswith(prefix):
            return self.embodiment_name[len(prefix) :]
        return None

    @property
    def sides(self) -> tuple[str, ...]:
        """Return the arms this embodiment carries.

        A single-arm arity carries one side. Anything else carries both, which
        is what a morphology block with no name resolves to.
        """
        arity = self.arity
        if arity == "left_arm":
            return ("left",)
        if arity == "right_arm":
            return ("right",)
        return SIDES

    @property
    def embodiment_class(self) -> type["Embodiment"]:
        """Import and return the platform's configured ``Embodiment`` class.

        Raises:
            NotImplementedError: If the platform does not specify an
                ``embodiment_class`` value.
        """
        path = self.platform.embodiment_class
        if path is None:
            raise NotImplementedError(
                f"{self.describe()}: platform {self.platform.name!r} declares no "
                "`embodiment_class:` and there is no derived transform pipeline "
                "yet — add one to platforms.yaml"
            )
        return _import_embodiment_class(path)

    def keypoints(self, side: str):
        """Return the keypoint topology and valid slots for one side.

        Args:
            side: The ``"left"`` or ``"right"`` end-effector key.

        Raises:
            KeyError: If ``side`` is not in ``end_effectors``.
        """
        return self.end_effectors[side].keypoints

    def get_keymap(self, *args, **kwargs):
        return self.embodiment_class.get_keymap(*args, **kwargs)

    def get_transform_list(self, *args, **kwargs):
        return self.embodiment_class.get_transform_list(*args, **kwargs)

    def describe(self) -> str:
        sides = ", ".join(
            f"{s}={self.end_effectors[s].name}" for s in sorted(self.end_effectors)
        )
        return f"{self.embodiment_name or self.platform.name} ({sides})"


class Embodiment(ABC):
    """Define dataset transforms and visualization for an embodiment."""

    INTRINSICS = None
    EXTRINSICS = None
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @classmethod
    def resolve(cls, spec) -> ResolvedEmbodiment:
        """Resolve an embodiment name or morphology mapping.

        A name selects its platform and the platform's default end-effector for
        both sides. A morphology mapping has this form::

            {"platform": "eva_x5",
             "end_effector": {"left": "eva_parallel_jaw",
                              "right": "eva_parallel_jaw"}}

        Args:
            spec: A current or deprecated embodiment name, or a morphology
                mapping. In a morphology mapping, ``end_effector`` can be one
                name for both sides or a mapping of side names to end-effectors.

        Returns:
            The selected platform and end-effector specifications.

        Raises:
            TypeError: If ``spec`` is not a string or mapping.
            ValueError: If a platform, side, or end-effector name is invalid.
        """
        if isinstance(spec, Mapping):
            return cls._resolve_morphology(spec)
        if isinstance(spec, str):
            return cls._resolve_name(spec)
        raise TypeError(
            "resolve() takes an embodiment name or a morphology mapping, got "
            f"{type(spec).__name__}"
        )

    @classmethod
    def _resolve_name(cls, embodiment_name: str) -> ResolvedEmbodiment:
        name = canonical_embodiment_name(embodiment_name)
        platform = load_embodiment_platforms().get(name)
        if platform is None:
            raise ValueError(
                f"embodiment {embodiment_name!r} is not owned by any platform in "
                "registry/platforms.yaml; known: "
                f"{sorted(load_embodiment_platforms())}"
            )
        end_effectors = load_end_effectors()
        default = end_effectors[platform.default_end_effector]
        return ResolvedEmbodiment(
            platform=platform,
            end_effectors={side: default for side in SIDES},
            embodiment_name=name,
        )

    @classmethod
    def _resolve_morphology(cls, morphology: Mapping) -> ResolvedEmbodiment:
        platforms = load_platforms()
        platform_name = morphology.get("platform")
        platform = platforms.get(platform_name)
        if platform is None:
            raise ValueError(
                f"morphology.platform {platform_name!r} is not in "
                f"registry/platforms.yaml; known: {sorted(platforms)}"
            )

        end_effectors = load_end_effectors()
        declared = morphology.get("end_effector", platform.default_end_effector)
        if isinstance(declared, str):
            declared = {side: declared for side in SIDES}
        if not isinstance(declared, Mapping) or not declared:
            raise ValueError(
                "morphology.end_effector must be an end-effector name or a "
                f"{{side: name}} mapping, got {declared!r}"
            )

        resolved = {}
        for side, ee_name in declared.items():
            if side not in SIDES:
                raise ValueError(
                    f"morphology.end_effector has unknown side {side!r}; "
                    f"expected one of {list(SIDES)}"
                )
            if ee_name not in end_effectors:
                raise ValueError(
                    f"morphology.end_effector[{side!r}] = {ee_name!r} is not in "
                    f"registry/end_effectors.yaml; known: {sorted(end_effectors)}"
                )
            resolved[side] = end_effectors[ee_name]

        return ResolvedEmbodiment(
            platform=platform,
            end_effectors=resolved,
            embodiment_name=morphology.get("embodiment"),
        )

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
                image, action, mode=mode, color="Greens", alpha=gt_alpha,
                intrinsics=K_i, **kwargs
            )
            ims = cls.viz(
                ims, pred_action, mode=mode, color="Reds", alpha=pred_alpha,
                intrinsics=K_i, **kwargs
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
