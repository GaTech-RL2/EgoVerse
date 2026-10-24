"""
ZarrWriter: General-purpose Zarr episode writer.

This module provides a reusable writer for creating Zarr v3 episode stores
compatible with the ZarrEpisode reader.
"""

import glob
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import simplejpeg
import zarr
from zarr.core.dtype import VariableLengthBytes

from egomimic.rldb.zarr.episode_rules import (
    SCHEMA_VERSION,
    camera_errors,
    embodiment_error,
    fps_error,
    matrices_to_jsonable,
    null_intrinsics_allowed,
)
from egomimic.rldb.zarr.required_metadata import (
    REQUIRED_METADATA_FIELDS,
    MissingMetadata,
    check_writer_metadata,
    objects_list,
    vendor_recording_id_error,
)

logger = logging.getLogger(__name__)

# attrs the writer derives itself; metadata_override may not name them.
_WRITER_OWNED_ATTRS = frozenset(
    {
        "embodiment",
        "total_frames",
        "fps",
        "task_description",
        "features",
        "intrinsics",
        "extrinsics",
        "schema_version",
        "vendor_recording_id",
        *REQUIRED_METADATA_FIELDS,
    }
)


class ZarrWriter:
    """
    General-purpose writer for Zarr v3 episode stores.

    Creates episodes compatible with the ZarrEpisode reader, handling both
    numeric and image data with intelligent chunking and optional sharding.
    """

    JPEG_QUALITY = 85  # Fixed JPEG quality for image compression

    def __init__(
        self,
        episode_path: str | Path,
        *,
        embodiment: str,
        intrinsics: "dict | MissingMetadata",
        task_name: "str | MissingMetadata",
        operator: "str | MissingMetadata",
        lab: "str | MissingMetadata",
        scene: "str | MissingMetadata",
        objects: "str | list[str] | MissingMetadata",
        vendor_recording_id: str,
        fps: int = 30,
        task_description: str = "",
        annotations: list[tuple[str, int, int]] | None = None,
        chunk_timesteps: int = 100,
        extrinsics: dict | None = None,
    ):
        """
        Initialize ZarrWriter.

        The camera and metadata arguments are keyword-only with no default, so
        forgetting one is a TypeError at the call site; their values are
        checked by write() before anything touches disk. To edit an existing
        episode (annotations, removing a key) use the module-level
        write_annotations / remove_key / key_exists instead.

        Args:
            episode_path: Path to episode .zarr directory.
            embodiment: Embodiment identifier (e.g. "human_bimanual"); must be
                a valid EMBODIMENT name (CONTRIBUTING_DATA.md §9).
            fps: Frames per second for playback (default: 30).
            task_description: Task description.
            annotations: List of (text, start_idx, end_idx) tuples describing language annotations.
            chunk_timesteps: Number of timesteps per chunk for numeric arrays (default: 100).
            intrinsics: REQUIRED: {camera_key: 3x4 K matrix} dict
                (single-camera = one entry, e.g. {"front_1": K}). Stored under the
                "intrinsics" key in zarr.json metadata. Only an embodiment whose
                source has no camera calibration (dexmate_bimanual) may pass
                MissingMetadata(); it is stored as null.
            extrinsics: Optional camera extrinsics. None (egocentric human data) or
                a non-empty dict of 4x4 transforms; robots key per-arm
                (e.g. {"left": world<-cam, "right": world<-cam}). Stored under the
                "extrinsics" key in zarr.json metadata.
            task_name, operator, lab, scene, objects: REQUIRED episode
                metadata; rules in CONTRIBUTING_DATA.md §6.3. objects may be a
                list or a comma-separated str and is stored as a list.
                task_name, operator and lab are always required. scene and
                objects may be MissingMetadata() only if the source data has no
                value; it is stored as null.
            vendor_recording_id: REQUIRED: your ID for the raw recording this
                episode was cut from (CONTRIBUTING_DATA.md §6.6).
        """
        self.episode_path = Path(episode_path)
        self.vendor_recording_id = vendor_recording_id
        self.fps = fps
        self.embodiment = embodiment
        self.task_description = task_description
        self.annotations = annotations if annotations is not None else []
        self.chunk_timesteps = chunk_timesteps
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.required_metadata = {
            "task_name": task_name,
            "operator": operator,
            "lab": lab,
            "scene": scene,
            "objects": objects,
        }
        self._features: dict[str, dict[str, Any]] = {}

    def write(
        self,
        numeric_data: dict[str, np.ndarray] | None = None,
        image_data: dict[str, np.ndarray] | None = None,
        pre_encoded_image_data: dict[str, tuple[np.ndarray, list[int]]] | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> Path:
        """
        Validate the episode configuration, then write it to the Zarr store.

        Args:
            numeric_data: Dictionary of numeric arrays (state, actions, etc.).
                All arrays must have same length along axis 0.
            image_data: Dictionary of image arrays with shape (T, H, W, 3).
                Images will be JPEG-compressed.
            pre_encoded_image_data: Dictionary mapping key to (encoded_array, image_shape).
                encoded_array is np.ndarray(dtype=object) of JPEG bytes.
                image_shape is [H, W, 3]. Skips internal JPEG encoding.
            metadata_override: Extra keys to add to zarr.attrs. Must not name
                a key the writer sets itself (embodiment, fps, intrinsics, the
                required metadata fields, ...); pass those as ZarrWriter
                arguments instead.

        Returns:
            Path to the written episode.

        Raises:
            ValueError: If the data, cameras, embodiment, fps, required
                metadata (§6.3) or metadata_override are invalid. Nothing is
                written, and an existing episode at episode_path is left as it
                was, when write() raises.
        """
        _remove_stale_siblings(self.episode_path)
        self._validate()
        numeric_data = numeric_data or {}
        image_data = image_data or {}
        pre_encoded_image_data = pre_encoded_image_data or {}
        self._features = {}

        if not numeric_data and not image_data and not pre_encoded_image_data:
            raise ValueError(
                "Must provide at least one of numeric_data, image_data, "
                "or pre_encoded_image_data"
            )
        self._validate_inputs(image_data, pre_encoded_image_data, metadata_override)

        # Infer total_frames from data
        all_lengths: list[int] = []
        for arr in numeric_data.values():
            all_lengths.append(len(arr))
        for arr in image_data.values():
            all_lengths.append(len(arr))
        for enc_arr, _shape in pre_encoded_image_data.values():
            all_lengths.append(len(enc_arr))

        if len(set(all_lengths)) > 1:
            raise ValueError(f"Inconsistent frame counts across arrays: {all_lengths}")

        total_frames = all_lengths[0]
        if total_frames == 0:
            raise ValueError("Episode has zero frames")

        # Padded to a whole number of chunks for sharding alignment.
        padded_frames = -(-total_frames // self.chunk_timesteps) * self.chunk_timesteps

        self.episode_path.parent.mkdir(parents=True, exist_ok=True)

        # Build in a sibling temp dir and swap it in at the end, so a failure
        # never leaves a half-written episode or wipes an existing one.
        tmp_path = _sibling(self.episode_path, "tmp")
        try:
            store = zarr.open(str(tmp_path), mode="w", zarr_format=3)

            for key, arr in numeric_data.items():
                self._write_numeric_array(store, key, arr, padded_frames)

            for key, arr in image_data.items():
                self._write_image_array(store, key, arr, padded_frames)

            for key, (enc_arr, img_shape) in pre_encoded_image_data.items():
                self._write_pre_encoded_image_array(
                    store, key, enc_arr, img_shape, padded_frames
                )

            self._features["annotations"] = _write_annotation_array(
                store, self.annotations, "annotations"
            )

            store.attrs.update(self._build_metadata(total_frames, metadata_override))
            _swap_in(tmp_path, self.episode_path)
        except BaseException:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        return self.episode_path

    @staticmethod
    def _validate_inputs(
        image_data: dict[str, np.ndarray],
        pre_encoded_image_data: dict[str, tuple[np.ndarray, list[int]]],
        metadata_override: dict[str, Any] | None,
    ) -> None:
        """Checks that need the data, run before anything touches disk."""
        for key, arr in image_data.items():
            if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                raise ValueError(
                    f"Image array '{key}' must be uint8 with shape (T, H, W, 3), "
                    f"got {arr.dtype} {arr.shape}"
                )
        for key, (_enc, img_shape) in pre_encoded_image_data.items():
            if len(img_shape) != 3 or img_shape[-1] != 3:
                raise ValueError(
                    f"Pre-encoded image '{key}' shape must be [H, W, 3], "
                    f"got {img_shape}"
                )
        if metadata_override:
            owned = sorted(_WRITER_OWNED_ATTRS & metadata_override.keys())
            if owned:
                raise ValueError(
                    f"metadata_override must not set {owned}: the writer sets "
                    "these itself. Pass them as ZarrWriter arguments (total_frames "
                    "and features come from the data)."
                )
            try:
                json.dumps(metadata_override)
            except TypeError as e:
                raise ValueError(
                    f"metadata_override is not JSON-serializable: {e}"
                ) from e

    def _validate(self) -> None:
        """Check embodiment, fps, cameras, required metadata and the target path."""
        errors = [
            e
            for e in (embodiment_error(self.embodiment), fps_error(self.fps))
            if e is not None
        ]
        no_intrinsics = isinstance(self.intrinsics, MissingMetadata)
        errors += camera_errors(
            None if no_intrinsics else self.intrinsics,
            self.extrinsics,
            allow_null_intrinsics=no_intrinsics
            and null_intrinsics_allowed(self.embodiment),
        )
        if (err := vendor_recording_id_error(self.vendor_recording_id)) is not None:
            errors.append(err)
        if errors:
            raise ValueError("; ".join(errors))
        if self.episode_path.is_symlink() or self.episode_path.is_file():
            raise ValueError(
                f"{self.episode_path} exists and is not an episode directory"
            )
        check_writer_metadata(self.required_metadata)
        for field, value in self.required_metadata.items():
            if isinstance(value, MissingMetadata):
                logger.warning("Writing %s without %s", self.episode_path, field)
        if no_intrinsics:
            logger.warning("Writing %s without intrinsics", self.episode_path)

    def _write_numeric_array(
        self, store: zarr.Group, key: str, arr: np.ndarray, padded_frames: int
    ) -> None:
        """
        Write a numeric array to the Zarr store.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            arr: Numeric array with shape (T, ...).
            padded_frames: Target frame count after padding (for sharding alignment).
        """
        num_frames = len(arr)

        # Store original shape and dtype before padding for metadata
        original_shape = arr.shape[1:]  # Shape excluding time dimension
        dtype_str = str(arr.dtype)

        # Pad array if needed
        if padded_frames > num_frames:
            pad_len = padded_frames - num_frames
            pad_shape = (pad_len,) + arr.shape[1:]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

        # Use chunk_timesteps for frames per chunk
        frames_per_chunk = min(self.chunk_timesteps, padded_frames)
        frames_per_chunk = max(1, frames_per_chunk)

        # Chunk shape: (frames, ...) - keep other dimensions intact
        chunk_shape = (frames_per_chunk,) + arr.shape[1:]

        store.create_array(
            key,
            data=arr,
            chunks=chunk_shape,
            shards=arr.shape,
        )

        # Track shape and dtype for metadata
        dimension_names = [f"dim_{i}" for i in range(len(original_shape))]
        self._features[key] = {
            "dtype": dtype_str,
            "shape": list(original_shape),
            "names": dimension_names,
        }

    def _write_image_array(
        self, store: zarr.Group, key: str, image_arr: np.ndarray, padded_frames: int
    ) -> None:
        """
        Write an image array to the Zarr store with JPEG compression.

        Images are always chunked 1 per timestep for efficient random access,
        regardless of chunk_timesteps setting.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            image_arr: Image array with shape (T, H, W, 3).
            padded_frames: Target frame count after padding (for sharding alignment).
        """
        # Encode each frame as JPEG
        num_frames = len(image_arr)

        # Encode to padded_frames length (pad with duplicate of last frame if needed)
        encoded = np.empty((padded_frames,), dtype=object)
        for i in range(padded_frames):
            # Use last frame for padding
            frame_idx = min(i, num_frames - 1)
            img = image_arr[frame_idx]
            jpeg_bytes = simplejpeg.encode_jpeg(
                img, quality=self.JPEG_QUALITY, colorspace="RGB"
            )
            encoded[i] = jpeg_bytes

        # Images are always chunked 1 per timestep, regardless of chunk_timesteps
        chunk_shape = (1,)

        store.create_array(
            key,
            shape=encoded.shape,
            chunks=chunk_shape,
            shards=encoded.shape,
            dtype=VariableLengthBytes(),
        )

        # Assign data after creation (required for VariableLengthBytes)
        store[key][:] = encoded

        # Track shape for metadata
        self._features[key] = {
            "dtype": "jpeg",
            "shape": list(image_arr.shape[1:]),  # [H, W, 3]
            "names": ["height", "width", "channel"],
        }

    def _write_pre_encoded_image_array(
        self,
        store: zarr.Group,
        key: str,
        encoded_arr: np.ndarray,
        image_shape: list[int],
        padded_frames: int,
    ) -> None:
        """
        Write already-JPEG-encoded image data to the Zarr store.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            encoded_arr: Object array of JPEG bytes with shape (T,).
            image_shape: Original image dimensions [H, W, 3].
            padded_frames: Target frame count after padding.
        """
        num_frames = len(encoded_arr)

        if padded_frames > num_frames:
            padded = np.empty((padded_frames,), dtype=object)
            padded[:num_frames] = encoded_arr
            last_jpeg = encoded_arr[-1]
            for i in range(num_frames, padded_frames):
                padded[i] = last_jpeg
            encoded_arr = padded

        chunk_shape = (1,)

        store.create_array(
            key,
            shape=encoded_arr.shape,
            chunks=chunk_shape,
            shards=encoded_arr.shape,
            dtype=VariableLengthBytes(),
        )

        store[key][:] = encoded_arr

        self._features[key] = {
            "dtype": "jpeg",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    def _build_metadata(
        self, total_frames: int, metadata_override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Build episode metadata dictionary.

        Args:
            metadata_override: Optional overrides to apply.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "embodiment": self.embodiment,
            "total_frames": total_frames,
            "fps": self.fps,
            "task_description": self.task_description,
            "vendor_recording_id": self.vendor_recording_id,
            "features": self._features,
        }

        metadata["intrinsics"] = (
            None
            if isinstance(self.intrinsics, MissingMetadata)
            else matrices_to_jsonable(self.intrinsics)
        )
        if self.extrinsics is not None:
            metadata["extrinsics"] = matrices_to_jsonable(self.extrinsics)

        for field in REQUIRED_METADATA_FIELDS:
            value = self.required_metadata[field]
            if isinstance(value, MissingMetadata):
                value = None
            elif field == "objects":
                value = objects_list(value)
            metadata[field] = value

        # write() has already rejected overrides that name a writer-owned key.
        if metadata_override:
            metadata.update(metadata_override)

        return metadata

    @staticmethod
    def create_and_write(
        episode_path: str | Path,
        numeric_data: dict[str, np.ndarray] | None = None,
        image_data: dict[str, np.ndarray] | None = None,
        pre_encoded_image_data: dict[str, tuple[np.ndarray, list[int]]] | None = None,
        *,
        embodiment: str,
        intrinsics: "dict | MissingMetadata",
        task_name: "str | MissingMetadata",
        operator: "str | MissingMetadata",
        lab: "str | MissingMetadata",
        scene: "str | MissingMetadata",
        objects: "str | list[str] | MissingMetadata",
        vendor_recording_id: str,
        fps: int = 30,
        task_description: str = "",
        annotations: list[tuple[str, int, int]] | None = None,
        chunk_timesteps: int = 100,
        extrinsics: dict | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> Path:
        """Build a ZarrWriter and write the episode in one call.

        Arguments are as in ZarrWriter.__init__ and ZarrWriter.write; all
        validation happens in write(). Returns the episode path.
        """
        return ZarrWriter(
            episode_path,
            embodiment=embodiment,
            intrinsics=intrinsics,
            task_name=task_name,
            operator=operator,
            lab=lab,
            scene=scene,
            objects=objects,
            vendor_recording_id=vendor_recording_id,
            fps=fps,
            task_description=task_description,
            annotations=annotations,
            chunk_timesteps=chunk_timesteps,
            extrinsics=extrinsics,
        ).write(
            numeric_data=numeric_data,
            image_data=image_data,
            pre_encoded_image_data=pre_encoded_image_data,
            metadata_override=metadata_override,
        )


def _sibling(episode_path: Path, kind: str) -> Path:
    return episode_path.with_name(f"{episode_path.name}.{kind}-{os.getpid()}")


def _remove_stale_siblings(episode_path: Path) -> None:
    """Drop temp/old dirs left by a writer that was killed mid-write (OOM,
    SIGKILL), whose own cleanup never ran. Assumes one writer per episode."""
    name = glob.escape(episode_path.name)
    for stale in episode_path.parent.glob(f"{name}.tmp-*"):
        _remove(stale)
    olds = sorted(
        episode_path.parent.glob(f"{name}.old-*"),
        key=lambda p: p.stat().st_mtime,
    )
    # Killed between the two renames in _swap_in: the .old copy is the episode.
    if olds and not episode_path.exists():
        os.replace(olds.pop(), episode_path)
    for stale in olds:
        _remove(stale)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _swap_in(new: Path, target: Path) -> None:
    """Replace ``target`` with ``new``; the old episode stays on disk until the
    new one is in place."""
    if not target.exists():
        os.replace(new, target)
        return
    old = _sibling(target, "old")
    os.replace(target, old)
    try:
        os.replace(new, target)
    except BaseException:
        os.replace(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _write_annotation_array(
    store: zarr.Group,
    annotations: list[tuple[str, int, int]],
    annotation_key: str,
) -> dict[str, Any]:
    """Write (text, start_idx, end_idx) annotations as JSON-encoded bytes.

    Returns the array's ``features`` entry.
    """
    encoded = np.array(
        [
            json.dumps(
                {
                    "text": text,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            for text, start_idx, end_idx in annotations
        ],
        dtype=object,
    )

    n_annotations = len(annotations)
    # Both the chunk and the SHARD edge must be >= 1 even when there are no
    # annotations: zarr v3 rejects a zero-length chunk grid, so an episode
    # written with shards=(0,) cannot be enumerated (group.keys() raises
    # "integer chunk edge length must be >= 1") even though the array itself
    # reads back fine. shape=(0,) with a >=1 edge is the correct encoding of
    # "empty" -- no chunk objects are written either way.
    store.create_array(
        annotation_key,
        shape=encoded.shape,
        chunks=(max(1, n_annotations),),
        shards=(max(1, n_annotations),),
        dtype=VariableLengthBytes(),
    )
    if n_annotations > 0:
        store[annotation_key][:] = encoded

    return {
        "dtype": "json",
        "shape": [n_annotations],
        "names": ["json"],
        "format": "annotation_v1",
    }


# ── Editing an existing episode ─────────────────────────────────────────────
# These work on an episode that is already written, so they need none of the
# metadata a ZarrWriter requires. Each keeps zarr.attrs["features"] in sync
# with the arrays (the episode validator checks that).


def key_exists(episode_path: str | Path, key: str) -> bool:
    """Whether ``key`` is an array/group in the episode store."""
    return key in zarr.open(str(episode_path), mode="r", zarr_format=3)


def remove_key(episode_path: str | Path, key: str) -> None:
    """Delete ``key`` from the episode store and from its features.

    Raises KeyError if the episode has no such key.
    """
    store = zarr.open_group(str(episode_path), mode="a", zarr_format=3)
    _remove_key(store, key)


def _remove_key(store: zarr.Group, key: str) -> None:
    if key not in store:
        raise KeyError(f"{key!r} is not in {store.store}")
    del store[key]
    features = dict(store.attrs.get("features", {}))
    if features.pop(key, None) is not None:
        store.attrs["features"] = features


def write_annotations(
    episode_path: str | Path,
    annotations: list[tuple[str, int, int]],
    annotation_key: str = "annotations",
    mode: Literal["w", "a"] = "a",
) -> None:
    """Write language annotations into an existing episode.

    Args:
        episode_path: Path to the episode .zarr directory.
        annotations: List of (text, start_idx, end_idx) tuples.
        annotation_key: Array key to write (default "annotations").
        mode: "w" replaces an existing ``annotation_key``; "a" only adds a new
            key and raises if it already exists.
    """
    store = zarr.open_group(str(episode_path), mode="a", zarr_format=3)
    if annotation_key in store:
        if mode != "w":
            raise ValueError(
                f"{annotation_key!r} already exists in {episode_path}; "
                'pass mode="w" to replace it'
            )
        _remove_key(store, annotation_key)
    feature = _write_annotation_array(store, annotations, annotation_key)
    features = dict(store.attrs.get("features", {}))
    features[annotation_key] = feature
    store.attrs["features"] = features
