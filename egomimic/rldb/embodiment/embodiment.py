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
from egomimic.utils.pose_utils import _split_action_pose
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    ColorPalette,
    _viz_annotations,
    _viz_axes,
    _viz_keypoints,
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
    DEXMATE_BIMANUAL = 7
    YAM_BIMANUAL = 8


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
        """Return the arity suffix encoded in ``embodiment_name``.

        Returns:
            The text after ``<platform embodiment_prefix>_``, or ``None`` when
            the resolved value has no matching embodiment name.
        """
        if self.embodiment_name is None:
            return None
        prefix = f"{self.platform.embodiment_prefix}_"
        if self.embodiment_name.startswith(prefix):
            return self.embodiment_name[len(prefix) :]
        return None

    @property
    def sides(self) -> tuple[str, ...]:
        """Return side candidates implied by the encoded arity.

        ``left_arm`` and ``right_arm`` select one side. All other values return
        both candidates; callers resolving a morphology mapping must still
        filter candidates not present in ``end_effectors``.
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

    def viz(self, image, viz_data, mode="keypoints", **kwargs):
        """Draw with the resolved per-side slot ownership, including robots."""
        cls = self.embodiment_class if self.platform.embodiment_class else Embodiment
        if mode == "keypoints":
            kwargs.setdefault("keypoint_specs", {s: ee.keypoints for s, ee in self.end_effectors.items()})
        return cls.viz(image, viz_data, mode=mode, **kwargs)

    def describe(self) -> str:
        sides = ", ".join(
            f"{s}={self.end_effectors[s].name}" for s in sorted(self.end_effectors)
        )
        return f"{self.embodiment_name or self.platform.name} ({sides})"


class Embodiment(ABC):
    """Define dataset transforms and visualization for an embodiment."""

    # Canonical MANO 21-keypoint topology: 0=wrist, 1-4 thumb, 5-8 index, ...
    FINGER_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # index
        (0, 9), (9, 10), (10, 11), (11, 12),    # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    ]
    FINGER_COLORS = {
        "thumb": (255, 100, 100),
        "index": (100, 255, 100),
        "middle": (100, 100, 255),
        "ring": (255, 255, 100),
        "pinky": (255, 100, 255),
    }
    FINGER_EDGE_RANGES = [
        ("thumb", 0, 4),
        ("index", 4, 8),
        ("middle", 8, 12),
        ("ring", 12, 16),
        ("pinky", 16, 20),
    ]
    DOT_COLOR = (255, 165, 0)

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
        arity = name.removeprefix(f"{platform.embodiment_prefix}_")
        sides = SIDES if arity == "bimanual" else (arity.removesuffix("_arm"),)
        return ResolvedEmbodiment(
            platform=platform,
            end_effectors={
                side: end_effectors[platform.default_for_side(side)] for side in sides
            },
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
        name = morphology.get("embodiment")
        named = cls._resolve_name(name) if name else None
        if named is not None and named.platform.name != platform.name:
            raise ValueError("morphology.platform disagrees with embodiment")
        active = set()
        for arity in platform.arity:
            active.update(SIDES if arity == "bimanual" else (arity.removesuffix("_arm"),))
        sides = named.sides if named is not None else tuple(s for s in SIDES if s in active)
        declared = morphology.get("end_effector", platform.default_end_effector)
        if isinstance(declared, str):
            declared = {side: declared for side in sides}
        elif "end_effector" not in morphology and isinstance(declared, Mapping):
            declared = {side: declared[side] for side in sides}
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

        if named is not None and set(resolved) != set(sides):
            raise ValueError(
                f"morphology.end_effector must agree with {name!r} active sides {sides}"
            )
        selected_arity = "bimanual" if set(resolved) == set(SIDES) else f"{next(iter(resolved))}_arm"
        if selected_arity not in platform.arity:
            raise ValueError(f"morphology.end_effector selects unsupported arity {selected_arity!r}")

        return ResolvedEmbodiment(
            platform=platform,
            end_effectors=resolved,
            embodiment_name=named.embodiment_name if named is not None else None,
        )

    @classmethod
    def from_attrs(cls, attrs: Mapping) -> ResolvedEmbodiment:
        """Resolve episode morphology with its name's active-side constraint."""
        morphology = attrs.get("morphology")
        if morphology is not None:
            if not isinstance(morphology, Mapping):
                raise ValueError("morphology must be a mapping")
            return cls.resolve({**morphology, "embodiment": attrs.get("embodiment")})
        return cls.resolve(attrs.get("embodiment"))

    @classmethod
    def split_action_pose(cls, actions):
        """Split a supported Cartesian layout into per-side XYZ and YPR.

        The base implementation accepts ``[L xyz ypr, R xyz ypr]`` with 12
        columns and ``[L xyz ypr gripper, R xyz ypr gripper]`` with 14 columns.
        It omits the gripper columns from the returned blocks. Subclasses can
        override this method for platform-specific layouts.

        Args:
            actions: An array with one of the supported widths on its last
                axis.

        Returns:
            ``(left_xyz, left_ypr, right_xyz, right_ypr)``. Each block preserves
            the input's leading dimensions and has a final width of three.

        Raises:
            ValueError: If the final axis is neither 12 nor 14 columns wide.
        """
        return _split_action_pose(actions)

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
        if mode == "keypoints":
            specs = kwargs.pop("keypoint_specs", None)
            keypoint_spec = kwargs.pop("keypoint_spec", None)
            finger_edges = kwargs.pop("finger_edges", None)
            finger_edge_ranges = kwargs.pop("finger_edge_ranges", None)
            sizes = {spec.n_slots for spec in (specs or {}).values()}
            if len(sizes) > 1:
                raise ValueError("both hands must use the same keypoint topology size")
            n_kp = next(iter(sizes), 21 if keypoint_spec is None else keypoint_spec.n_slots)
            valid_slots = None if keypoint_spec is None else keypoint_spec.valid
            if specs is not None:
                valid_slots = {side: specs[side].valid if side in specs else () for side in SIDES}
            color = kwargs.get("color", None)
            if color is not None and ColorPalette.is_valid(color):
                n = len(cls.FINGER_COLORS)
                colors = {
                    finger: ColorPalette.to_rgb(color, value=(i + 1) / (n + 1))
                    for i, finger in enumerate(cls.FINGER_COLORS)
                }
                dot_color = ColorPalette.to_rgb(color, value=0.7)
            else:
                colors = cls.FINGER_COLORS
                dot_color = cls.DOT_COLOR
            return _viz_keypoints(
                image=image,
                actions=viz_data,
                intrinsics=K,
                n_kp=n_kp,
                valid_slots=valid_slots,
                edges=finger_edges if finger_edges is not None else cls.FINGER_EDGES,
                edge_ranges=(
                    finger_edge_ranges
                    if finger_edge_ranges is not None
                    else cls.FINGER_EDGE_RANGES
                ),
                colors=colors,
                dot_color=dot_color,
                **kwargs,
            )
        # Pass the bound classmethod so subclass-specific layouts are honored.
        split_pose = cls.split_action_pose
        if mode == "traj":
            return _viz_traj(
                image=image,
                actions=viz_data,
                intrinsics=K,
                split_pose=split_pose,
                **kwargs,
            )
        if mode == "traj+rotation":
            vis = _viz_traj(
                image=image,
                actions=viz_data,
                intrinsics=K,
                split_pose=split_pose,
                **kwargs,
            )
            return _viz_rotation_txt(
                image=vis,
                actions=viz_data,
                split_pose=split_pose,
                **kwargs,
            )
        if mode == "axes":
            return _viz_axes(
                image=image,
                actions=viz_data,
                intrinsics=K,
                split_pose=split_pose,
                **kwargs,
            )
        if mode == "annotations":
            return _viz_annotations(
                image=image,
                annotations=viz_data,
                **kwargs,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: ('traj', 'traj+rotation', 'axes', 'annotations', 'keypoints')."
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
