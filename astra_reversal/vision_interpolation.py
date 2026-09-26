"""Camera-aligned VEI/VLI for the pinned, frozen LeRobot PI05 prefix.

VEI replaces projected image tokens by a convex current/donor mixture. VLI does
the same after decoder blocks 0..16; a block's own K/V is already built, so the
edit affects subsequent caches. Donors are single verified STANDARD training
frames, captured under the *current target instruction*, never demonstration
means. Direct visual writes exclude text, state, padding and other cameras;
later attention can propagate a change across modalities.

This separate module intentionally leaves the existing adapter/text-bank source
identity unchanged. Prefix preparation and temporary hooks must run serially
on a policy. No function changes model parameters or runs environment actions.
"""

import copy
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Real

import numpy as np

from . import interpolation_conditioning as text_ops
from .image_donor_bank import CAMERAS, DonorImage, ImageDonorLibrary
from .interpolation_catalog import DATASET_REPO, DATASET_REVISION
from .lerobot_policy import _interpolation_prefix_layout, prepare_velocity
from .policy_adapter import Condition
from .records import digest, file_sha256, to_numpy

OPERATORS = ("vei", "vli", "vei_vli")
CAMERA_FEATURES = {
    "observation/image": "observation.images.image",
    "observation/wrist_image": "observation.images.image2",
}
PROJECTOR_BOUNDARY = "native_embed_image_after_vision_tower_and_multimodal_projector"


def _alpha(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Visual alpha must be finite and in [0, 1]")
    return float(value)


def _cameras(cameras):
    cameras = tuple(cameras)
    if (
        not cameras
        or len(set(cameras)) != len(cameras)
        or any(camera not in CAMERAS for camera in cameras)
    ):
        raise ValueError("Select unique canonical observed cameras")
    return tuple(camera for camera in CAMERAS if camera in cameras)


def _frozen(policy):
    if policy.policy.training or any(
        p.requires_grad for p in policy.policy.parameters()
    ):
        raise ValueError("Vision interpolation requires an eval-mode frozen policy")


def _pair_metadata(pair):
    if not isinstance(pair, dict) or set(pair) != set(CAMERAS):
        raise ValueError("A donor must contain exactly the two canonical cameras")
    metadata = {}
    shared = None
    for camera in CAMERAS:
        image = pair[camera]
        if not isinstance(image, DonorImage) or image.camera != camera:
            raise TypeError("Resolve each camera with the verified image donor library")
        row = image.provenance
        if (
            row.get("dataset_repo") != DATASET_REPO
            or row.get("dataset_revision") != DATASET_REVISION
            or row["pixels_sha256"] != digest(image.pixels)
        ):
            raise ValueError("Donor pixels must bind the pinned STANDARD dataset")
        identity = {
            "donor_id": image.donor_id,
            **{
                key: row.get(key)
                for key in (
                    "library_id",
                    "sample_sha256",
                    "source_id",
                    "prompt",
                    "episode_index",
                    "frame_index",
                    "phase",
                )
            },
        }
        if any(value is None for value in identity.values()):
            raise ValueError("Donor lacks single-frame training provenance")
        if shared is not None and identity != shared:
            raise ValueError(
                "Camera donors must be the same paired demonstration frame"
            )
        shared = identity
        metadata[camera] = row
    return {**shared, "cameras": metadata}


def _compatibility(policy, native):
    config = policy.model.paligemma_with_expert.paligemma.config.vision_config
    size, patch = config.image_size, config.patch_size
    if (
        type(size) is not int
        or type(patch) is not int
        or not 0 < patch <= size
        or size % patch
    ):
        raise ValueError("Unsupported native square patch-grid configuration")
    return {
        "native": native,
        "vision_operator_source_sha256": file_sha256(__file__),
        "vision_image_size": size,
        "vision_patch_size": patch,
        "vision_grid": [size // patch, size // patch],
        "image_features": list(policy.config.image_features),
        "camera_features": CAMERA_FEATURES,
        "projector_boundary": PROJECTOR_BOUNDARY,
        "layer_boundary": text_ops.CAPTURE_BOUNDARY,
    }


def _layout(policy, batch, embeddings, padding, attention, token_mask, compatibility):
    import torch

    start = _interpolation_prefix_layout(embeddings, padding, attention, token_mask)
    feature_keys = list(policy.config.image_features)
    present = [key for key in feature_keys if key in batch]
    missing = [key for key in feature_keys if key not in batch]
    if set(present) != set(CAMERA_FEATURES.values()):
        raise ValueError(
            "Expected the two canonical real cameras and only masked extras"
        )
    grid = compatibility["vision_grid"]
    count = math.prod(grid)
    if (
        start != count * len(feature_keys)
        or embeddings.shape[-1] != compatibility["native"]["width"]
    ):
        raise ValueError(
            "Native prefix is not the configured camera grids followed by text"
        )
    rows = {}
    for index, key in enumerate(present + missing):
        left, right = index * count, (index + 1) * count
        valid = key in present
        if not torch.all(padding[:, left:right] == valid).item():
            raise ValueError("Native image padding does not match camera presence")
        camera = next(
            (camera for camera, feature in CAMERA_FEATURES.items() if feature == key),
            None,
        )
        rows[camera or key] = {
            "feature_key": key,
            "start": left,
            "stop": right,
            "grid": grid,
            "valid": valid,
        }
    return {"text_start": start, "camera_slots": rows, "tokens_per_camera": count}


def _visual_values(values, layout):
    import torch

    return torch.cat(
        [
            values[
                :,
                layout["camera_slots"][camera]["start"] : layout["camera_slots"][
                    camera
                ]["stop"],
            ]
            for camera in CAMERAS
        ],
        dim=1,
    )


@dataclass(frozen=True, init=False)
class VisionLatentBank:
    """Immutable paired projected tokens and optional all-18-layer states.

    Arrays are in canonical camera order, then native row-major patch order.
    Metadata hashes the arrays; calling metadata() never exposes bank tensors.
    Timings are separate from bank_id, so recapturing identical values has the
    same scientific identity. Persistence may store these arrays plus metadata;
    reconstruct and compare bank_id before accepting a persisted bank.
    """

    embeddings: np.ndarray
    states: np.ndarray | None
    token_ids: np.ndarray
    token_mask: np.ndarray
    bank_id: str
    _metadata_json: str
    _capture_json: str

    def __init__(
        self, embeddings, states, token_ids, token_mask, provenance, *, capture=None
    ):
        embeddings, token_ids, token_mask = map(
            np.asarray, (embeddings, token_ids, token_mask)
        )
        if (
            embeddings.dtype != np.float32
            or embeddings.ndim != 3
            or embeddings.shape[0] != 1
            or min(embeddings.shape[1:]) < 1
            or not np.isfinite(embeddings).all()
            or token_ids.dtype != np.int64
            or token_ids.ndim != 2
            or token_ids.shape[0] != 1
            or token_mask.dtype != np.bool_
            or token_mask.shape != token_ids.shape
            or not token_mask.any()
        ):
            raise ValueError("Malformed projected vision bank or prompt tokens")
        compat = provenance.get("compatibility", {})
        native = compat.get("native", {})
        expected = (
            1,
            2 * math.prod(compat.get("vision_grid", [0])),
            native.get("width"),
        )
        if (
            embeddings.shape != expected
            or native.get("layer_count") != 18
            or token_ids.shape[1] != native.get("text_slots")
            or not isinstance(provenance.get("target_prompt"), str)
            or not provenance["target_prompt"].strip()
            or provenance.get("capture_boundary") != text_ops.CAPTURE_BOUNDARY
            or provenance.get("projector_boundary") != PROJECTOR_BOUNDARY
            or set(provenance.get("donor", {}).get("cameras", {})) != set(CAMERAS)
        ):
            raise ValueError("Vision bank lacks its native layout/prompt/donor binding")
        if states is not None:
            states = np.asarray(states)
            if (
                states.dtype != np.float32
                or states.shape != (18, *expected)
                or not np.isfinite(states).all()
            ):
                raise ValueError(
                    "VLI bank must contain 18 finite float32 paired visual states"
                )
        metadata = {
            "schema_version": "pi05-single-frame-vision-bank-1",
            "provenance": copy.deepcopy(provenance),
            "arrays": {},
            "captured_layer_indices": list(range(18)) if states is not None else [],
            "effective_layer_indices": list(range(17)) if states is not None else [],
        }
        for name, value in (
            ("embeddings", embeddings),
            ("states", states),
            ("token_ids", token_ids),
            ("token_mask", token_mask),
        ):
            object.__setattr__(
                self, name, None if value is None else text_ops._immutable_array(value)
            )
            metadata["arrays"][name] = (
                None
                if value is None
                else {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": digest(value),
                }
            )
        metadata["bank_id"] = digest(metadata)
        object.__setattr__(self, "bank_id", metadata["bank_id"])
        object.__setattr__(
            self,
            "_metadata_json",
            json.dumps(metadata, sort_keys=True, allow_nan=False),
        )
        object.__setattr__(
            self,
            "_capture_json",
            json.dumps(capture or {}, sort_keys=True, allow_nan=False),
        )

    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def provenance(self):
        return self.metadata()["provenance"]

    @property
    def capture(self):
        return json.loads(self._capture_json)

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for value in (self.embeddings, self.states, self.token_ids, self.token_mask)
            if value is not None
        )

    def validate_for(
        self, prompt, compatibility, token_ids, token_mask, *, require_layers
    ):
        if (
            self.provenance["target_prompt"] != prompt
            or self.provenance["compatibility"] != compatibility
        ):
            raise ValueError("Vision bank current prompt/native compatibility mismatch")
        if not np.array_equal(
            self.token_ids, to_numpy(token_ids)
        ) or not np.array_equal(self.token_mask, to_numpy(token_mask)):
            raise ValueError(
                "Vision bank token IDs/mask differ from the current target"
            )
        if require_layers and self.states is None:
            raise ValueError("VLI requires a bank captured with include_layers=True")


def capture_vision_bank(
    policy, donor_pair, *, prompt, include_layers=True, observation=None
):
    """One donor prefix capture (VEI-only skips decoder prefill entirely).

    No demonstration state is available or imported. A zero state placeholder
    is used unless observation is supplied. The verified plain OpenPI profile
    excludes state from prefix tokens; the state is not part of the bank key.
    """
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if type(include_layers) is not bool:
        raise ValueError("include_layers must be boolean")
    _frozen(policy)
    donor = _pair_metadata(donor_pair)
    started = time.perf_counter()
    raw_input = (
        {"observation/state": np.zeros(8, np.float32)}
        if observation is None
        else copy.deepcopy(observation)
    )
    raw_input.update({camera: donor_pair[camera].pixels for camera in CAMERAS})
    _, batch, _, layers, native = policy._interpolation_inputs(raw_input, prompt)
    compatibility = _compatibility(policy, native)
    captured, states, dtypes = {}, [], []

    def prefix(embeddings, padding, attention):
        layout = _layout(
            policy,
            batch,
            embeddings,
            padding,
            attention,
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            compatibility,
        )
        captured.update(
            layout=layout,
            embeddings=to_numpy(_visual_values(embeddings, layout)).astype(
                np.float32, copy=True
            ),
            prefix_padding_sha256=digest(to_numpy(padding)),
            prefix_attention_sha256=digest(to_numpy(attention)),
            prefix_positions_sha256=digest(to_numpy(torch.cumsum(padding, dim=1) - 1)),
            text_embedding_sha256=digest(
                to_numpy(embeddings[:, layout["text_start"] :])
            ),
            embedding_dtype=str(embeddings.dtype),
        )
        return embeddings

    def layer(index, hidden):
        values = _visual_values(hidden, captured["layout"])
        states.append(to_numpy(values).astype(np.float32, copy=True))
        dtypes.append(str(values.dtype))

    if include_layers:
        with text_ops.scoped_post_block_hooks(layers, layer):
            prepare_velocity(policy.policy, batch, prefix_transform=prefix)
    else:
        with torch.no_grad():
            images, masks = policy.policy._preprocess_images(batch)
            prefix(
                *policy.model.embed_prefix(
                    images,
                    masks,
                    batch[OBS_LANGUAGE_TOKENS],
                    batch[OBS_LANGUAGE_ATTENTION_MASK],
                )
            )
    bank = VisionLatentBank(
        captured.pop("embeddings"),
        np.stack(states) if include_layers else None,
        to_numpy(batch[OBS_LANGUAGE_TOKENS]),
        to_numpy(batch[OBS_LANGUAGE_ATTENTION_MASK]),
        {
            "target_prompt": prompt,
            "target_prompt_sha256": digest(prompt),
            "donor": donor,
            "compatibility": compatibility,
            "capture_boundary": text_ops.CAPTURE_BOUNDARY,
            "projector_boundary": PROJECTOR_BOUNDARY,
            "native_layer_dtypes": dtypes,
            "single_frame_not_mean": True,
            "donor_state_imported": False,
            "state_excluded_from_prefix": True,
            **captured,
        },
    )
    # Include the immutable copies and full array hashes in measured capture cost.
    object.__setattr__(
        bank,
        "_capture_json",
        json.dumps(
            {
                "wall_seconds": time.perf_counter() - started,
                "vision_encoder_forwards": len(policy.config.image_features),
                "prefix_forwards": int(include_layers),
                "velocity_evaluations": 0,
                "environment_actions": 0,
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    return bank


class VisionBankCache:
    """Optional bounded CPU LRU; callers may instead cache banks themselves."""

    def __init__(self, max_entries=3, max_bytes=240 * 1024**2):
        if (
            type(max_entries) is not int
            or max_entries < 1
            or type(max_bytes) is not int
            or max_bytes < 1
        ):
            raise ValueError("Positive integer cache bounds required")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._banks = OrderedDict()

    @property
    def nbytes(self):
        return sum(bank.nbytes for bank in self._banks.values())

    def get(
        self, policy, observation, prompt, library, donor_id, *, include_layers=True
    ):
        if type(include_layers) is not bool:
            raise ValueError("include_layers must be boolean")
        _frozen(policy)
        if not isinstance(library, ImageDonorLibrary):
            raise TypeError("Use the verified ImageDonorLibrary")
        pair = {camera: library.resolve(donor_id, camera) for camera in CAMERAS}
        key = (
            id(policy.model),
            digest(
                {
                    "model": policy.metadata,
                    "prompt": prompt,
                    "donor": _pair_metadata(pair),
                    "operator": file_sha256(__file__),
                }
            ),
        )
        existing = self._banks.pop(key, None)
        hit = existing is not None and (
            not include_layers or existing.states is not None
        )
        if hit:
            bank = existing
        else:
            # Bound retained banks before allocating the next capture.
            while len(self._banks) >= self.max_entries:
                self._banks.popitem(last=False)
            bank = capture_vision_bank(
                policy,
                pair,
                prompt=prompt,
                include_layers=include_layers,
                observation=observation,
            )
        if bank.nbytes > self.max_bytes:
            raise ValueError("One bank exceeds the declared cache byte limit")
        while self._banks and self.nbytes + bank.nbytes > self.max_bytes:
            self._banks.popitem(last=False)
        self._banks[key] = bank
        return bank, {
            "cache_hit": hit,
            "bank_id": bank.bank_id,
            "resident_banks": len(self._banks),
            "resident_bytes": self.nbytes,
            "capture": {
                "wall_seconds": 0.0,
                "vision_encoder_forwards": 0,
                "prefix_forwards": 0,
                "velocity_evaluations": 0,
                "environment_actions": 0,
            }
            if hit
            else bank.capture,
        }


def interpolate_visual(values, donor, layout, *, alpha, cameras=CAMERAS):
    """Convex edit at exactly matching camera/patch slots, without in-place writes."""
    import torch

    alpha, cameras = _alpha(alpha), _cameras(cameras)
    if (
        values.ndim != 3
        or values.shape[0] != 1
        or not values.is_floating_point()
        or not torch.isfinite(values).all().item()
    ):
        raise ValueError("Visual values must be finite floating [1,S,D]")
    mask = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    count = layout["tokens_per_camera"]
    for camera in cameras:
        row = layout["camera_slots"][camera]
        if (
            not row["valid"]
            or row["stop"] - row["start"] != count
            or not 0
            <= row["start"]
            < row["stop"]
            <= layout["text_start"]
            <= values.shape[1]
        ):
            raise ValueError("Visual camera slots overlap text or padded cameras")
        mask[:, row["start"] : row["stop"]] = True
    edited = values
    if alpha > 0:
        if (
            not isinstance(donor, torch.Tensor)
            or donor.shape != (1, 2 * count, values.shape[-1])
            or donor.dtype != values.dtype
            or donor.device != values.device
            or not torch.isfinite(donor).all().item()
        ):
            raise ValueError("Donor camera/grid/width/dtype/device mismatch")
        with torch.no_grad():
            for camera in cameras:
                row = layout["camera_slots"][camera]
                left, right = row["start"], row["stop"]
                offset = CAMERAS.index(camera) * count
                base, target = values[:, left:right], donor[:, offset : offset + count]
                if torch.equal(base, target):
                    continue
                if edited is values:
                    edited = values.clone()
                edited[:, left:right] = (
                    target if alpha == 1 else (1 - alpha) * base + alpha * target
                )
            if not torch.isfinite(edited).all().item():
                raise ValueError("Visual interpolation produced nonfinite values")
            if torch.equal(edited, values):
                edited = values
    before, after = to_numpy(values[mask]), to_numpy(edited[mask])
    delta = after.astype(np.float64) - before.astype(np.float64)
    protected = torch.equal(values[~mask], edited[~mask])
    if not protected:
        raise ValueError("Visual interpolation wrote protected prefix slots")
    return edited, {
        "has_effect": edited is not values,
        "alpha": alpha,
        "cameras": list(cameras),
        "selected_before_sha256": digest(before),
        "selected_after_sha256": digest(after),
        "protected_before_sha256": digest(to_numpy(values[~mask])),
        "protected_after_sha256": digest(to_numpy(edited[~mask])),
        "protected_slots_unchanged": protected,
        "direct_text_write_unchanged": torch.equal(
            values[:, layout["text_start"] :], edited[:, layout["text_start"] :]
        ),
        "delta_frobenius": float(np.linalg.norm(delta)),
        "delta_rms": float(np.sqrt(np.mean(delta**2))),
        "delta_max_abs": float(np.abs(delta).max()),
        "hidden_dtype": str(values.dtype),
    }


def prepare_vision_interpolated(
    policy,
    observation,
    observation_id,
    prompt,
    *,
    bank=None,
    alpha=0.0,
    operator="vei",
    cameras=CAMERAS,
    source_prompts=None,
    text_latents=None,
    language_alpha=0.5,
    language_operator="tli",
):
    """Build one fresh prefix cache, optionally composing existing text TLI.

    Language alpha follows unchanged paper-style TLI: .5 is neutral, delta is
    (1-2*a)*(T_A-T_B). Visual alpha is independently 0=native, 1=donor. The two
    writes use disjoint instruction/visual slots in the same post-block hook.
    """
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    alpha, cameras = _alpha(alpha), _cameras(cameras)
    if operator not in OPERATORS or language_operator != "tli":
        raise ValueError("Expected VEI/VLI visual operator and optional existing TLI")
    _frozen(policy)
    started = time.perf_counter()
    raw, batch, instruction, layers, native = policy._interpolation_inputs(
        observation, prompt
    )
    compatibility = _compatibility(policy, native)
    active_vli = alpha > 0 and operator in ("vli", "vei_vli")
    active_vei = alpha > 0 and operator in ("vei", "vei_vli")
    if alpha > 0:
        if not isinstance(bank, VisionLatentBank):
            raise TypeError("Nonzero visual interpolation requires a VisionLatentBank")
        bank.validate_for(
            prompt,
            compatibility,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            require_layers=active_vli,
        )
    language_alpha = _alpha(language_alpha)
    language_requested = source_prompts is not None
    if not language_requested and (text_latents is not None or language_alpha != 0.5):
        raise ValueError("Language intervention requires its A/B source prompts")
    active_tli = language_requested and language_alpha != 0.5
    sources, source_masks = [], []
    target_mask = torch.tensor(instruction, dtype=torch.bool, device=policy.device)
    if language_requested:
        source_prompts, language_alpha = text_ops.validate_request(
            source_prompts, language_alpha, "tli"
        )
    if active_tli:
        if not isinstance(text_latents, dict) or set(text_latents) != {"a", "b"}:
            raise ValueError(
                "Nonzero TLI requires text_latents={'a': bank_A, 'b': bank_B}"
            )
        sources = [
            policy._interpolation_inputs(observation, source)
            for source in source_prompts
        ]
        for label, source_prompt, source in zip(
            ("a", "b"), source_prompts, sources, strict=True
        ):
            text_bank = text_latents[label]
            if not isinstance(text_bank, text_ops.TextLatentBank):
                raise TypeError("TLI sources must be compatible TextLatentBank objects")
            text_bank.validate_for(
                prompt=source_prompt,
                compatibility=native,
                token_ids=source[1][OBS_LANGUAGE_TOKENS],
                token_mask=source[1][OBS_LANGUAGE_ATTENTION_MASK],
                instruction_mask=source[2],
            )
            if text_bank.states.shape != (18, 1, native["text_slots"], native["width"]):
                raise ValueError("Text bank native dimensions differ")
            source_masks.append(
                torch.tensor(source[2], dtype=torch.bool, device=policy.device)
            )
    provenance = {
        "operator": operator,
        "operator_version": 1,
        "operator_source_sha256": file_sha256(__file__),
        "formula": "(1-alpha)*current+alpha*same_camera_same_patch_donor",
        "alpha": alpha,
        "target_prompt": prompt,
        "observation_id": observation_id,
        "raw_condition_id": digest(raw),
        "compatibility": compatibility,
        "bank": bank.metadata() if alpha > 0 else None,
        "cameras": list(cameras),
        "state_excluded_from_prefix": True,
        "state_sha256": digest(to_numpy(batch["observation.state"])),
        "token_ids_sha256": digest(to_numpy(batch[OBS_LANGUAGE_TOKENS])),
        "token_mask_sha256": digest(to_numpy(batch[OBS_LANGUAGE_ATTENTION_MASK])),
        "vli_layer_indices": list(range(17)) if active_vli else [],
        "vli_layers": [],
        "language": {
            "operator": "tli" if language_requested else None,
            "alpha": language_alpha,
            "formula": "(1-2*alpha)*(T_A-T_B)",
            "active": active_tli,
            "source_prompts": list(source_prompts) if language_requested else None,
            "banks": {label: text_latents[label].metadata() for label in ("a", "b")}
            if active_tli
            else None,
            "layers": [],
            "layer_indices": list(range(17)) if active_tli else [],
        },
        "direct_visual_writes_exclude_text": True,
        "later_text_states_may_change_through_attention": True,
        "cache_rebuilt": True,
        "prefix_forwards": 1,
        "velocity_evaluations": 0,
    }
    layout = {}

    def prefix(embeddings, padding, attention):
        layout.update(
            _layout(
                policy,
                batch,
                embeddings,
                padding,
                attention,
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                compatibility,
            )
        )
        if alpha > 0 and layout != bank.provenance["layout"]:
            raise ValueError("Current camera/grid layout differs from captured donor")
        donor = (
            torch.tensor(
                bank.embeddings, device=embeddings.device, dtype=embeddings.dtype
            )
            if active_vei
            else None
        )
        result, metrics = interpolate_visual(
            embeddings, donor, layout, alpha=alpha if active_vei else 0, cameras=cameras
        )
        provenance.update(
            layout=copy.deepcopy(layout),
            vei=metrics,
            prefix_before_sha256=digest(to_numpy(embeddings)),
            prefix_after_sha256=digest(to_numpy(result)),
            prefix_padding_sha256=digest(to_numpy(padding)),
            prefix_attention_sha256=digest(to_numpy(attention)),
            prefix_positions_sha256=digest(to_numpy(torch.cumsum(padding, dim=1) - 1)),
            text_embedding_before_sha256=digest(
                to_numpy(embeddings[:, layout["text_start"] :])
            ),
            text_embedding_after_sha256=digest(
                to_numpy(result[:, layout["text_start"] :])
            ),
            masks_and_positions_unchanged=True,
        )
        return result

    def layer(index, hidden):
        if index == 17:
            return None
        result = hidden
        if active_vli:
            donor = torch.tensor(
                bank.states[index], device=hidden.device, dtype=hidden.dtype
            )
            result, metrics = interpolate_visual(
                hidden, donor, layout, alpha=alpha, cameras=cameras
            )
            provenance["vli_layers"].append({"layer_index": index, **metrics})
        if active_tli:
            start = layout["text_start"]
            base = hidden[:, start:]
            values = [
                torch.tensor(
                    text_latents[label].states[index],
                    dtype=base.dtype,
                    device=base.device,
                )
                for label in ("a", "b")
            ]
            edited, metrics = text_ops.interpolate_text(
                base,
                target_mask,
                values[0],
                source_masks[0],
                values[1],
                source_masks[1],
                alpha=language_alpha,
                operator="tli",
            )
            before_text = result
            if edited is not base:
                result = torch.cat((result[:, :start], edited), dim=1)
            metrics.update(
                layer_index=index,
                direct_vision_write_unchanged=torch.equal(
                    before_text[:, :start], result[:, :start]
                ),
            )
            provenance["language"]["layers"].append(metrics)
        return result

    if active_vli or active_tli:
        with text_ops.scoped_post_block_hooks(layers, layer):
            velocity = prepare_velocity(policy.policy, batch, prefix_transform=prefix)
    else:
        velocity = prepare_velocity(policy.policy, batch, prefix_transform=prefix)
    effect = provenance["vei"]["has_effect"] or any(
        row["has_effect"]
        for row in provenance["vli_layers"] + provenance["language"]["layers"]
    )
    provenance.update(
        has_effect=effect, native_target_equivalent=not effect, hooks_removed=True
    )
    condition_id = (
        digest({"raw_condition_id": digest(raw), "representation": provenance})
        if effect
        else digest(raw)
    )
    provenance["condition_id"] = condition_id
    return Condition(
        condition_id,
        observation_id,
        prompt,
        raw,
        batch["observation.state"],
        velocity,
        time.perf_counter() - started,
    ), provenance


def weighted_vision_probe(
    policy, observation, prompt, donor_pair, *, text_latents=None, source_prompts=None
):
    """Allocated-CUDA-only evidence; fixed Euler10, no environment/provider calls.

    Returns checks and all measured action hashes/errors even if a numeric gate
    fails. Structural errors raise. This is interface evidence, not task success.
    Optional A/B banks add unchanged-existing-TLI vs composite parity checks.
    """
    import torch

    from .flow import error_metrics

    if torch.device(policy.device).type != "cuda":
        raise ValueError("The weighted vision probe requires an allocated CUDA device")
    _frozen(policy)
    started = time.perf_counter()
    versions = [
        (name, parameter._version)
        for name, parameter in policy.policy.named_parameters()
    ]
    noise = policy.tensor(
        np.random.default_rng(701)
        .normal(size=(1, policy.horizon, policy.action_dim))
        .astype(np.float32)
    )
    report = {
        "schema_version": "weighted-vision-interpolation-probe-1",
        "status": "failed",
        "device": str(policy.device),
        "gpu": torch.cuda.get_device_name(policy.device),
        "environment_actions": 0,
        "provider_calls": 0,
        "velocity_evaluations": 0,
        "solver": {"solver": "euler", "steps": 10},
        "noise_sha256": digest(to_numpy(noise)),
        "operator_source_sha256": file_sha256(__file__),
        "checkpoint": copy.deepcopy(policy.metadata),
        "checks": {},
        "solves": {},
        "scope": "weighted interface checks; no control-quality inference",
    }

    def solve(name, condition, provenance=None):
        result = policy.sample(condition, noise, steps=10, solver="euler")
        values = to_numpy(result.value)
        report["velocity_evaluations"] += result.velocity_evaluations
        report["solves"][name] = {
            "full_sha256": digest(values),
            "control_sha256": digest(values[..., :7]),
            "condition_id": condition.condition_id,
            "latency_seconds": result.latency_seconds,
            "velocity_evaluations": result.velocity_evaluations,
            "provenance": provenance,
        }
        return values

    observation_id = digest(observation)
    native = policy.prepare(observation, observation_id, prompt)
    base = solve("native", native)
    reference = policy.reference_actions(native, noise, steps=10)
    report["velocity_evaluations"] += 10
    native_error = error_metrics(
        reference, policy.output_transform({"actions": base[0]})["actions"]
    )
    report["native_sampler_parity"] = {
        **native_error,
        "tolerance": 1e-5,
        "velocity_evaluations": 10,
    }
    report["checks"]["native_sampler_parity"] = native_error["max_abs"] <= 1e-5
    bank = capture_vision_bank(policy, donor_pair, prompt=prompt, include_layers=True)
    report["bank"] = bank.metadata()
    report["capture"] = bank.capture
    for operator in ("vei", "vli"):
        for alpha in (0.0, 0.5):
            name = f"{operator}_{alpha}"
            condition, provenance = prepare_vision_interpolated(
                policy,
                observation,
                observation_id,
                prompt,
                bank=bank,
                alpha=alpha,
                operator=operator,
            )
            values = solve(name, condition, provenance)
            report["solves"][name]["full_error_from_native"] = error_metrics(
                base, values
            )
            control_error = error_metrics(base[..., :7], values[..., :7])
            report["solves"][name]["control_error_from_native"] = control_error
            report["checks"][name] = (
                np.array_equal(base, values)
                if alpha == 0
                else (provenance["has_effect"] and control_error["max_abs"] > 0)
            )
            rows = [provenance["vei"], *provenance["vli_layers"]]
            report["checks"][name + "_protected"] = all(
                row["protected_slots_unchanged"] and row["direct_text_write_unchanged"]
                for row in rows
            )
    donor_observation = {
        **copy.deepcopy(observation),
        **{camera: donor_pair[camera].pixels for camera in CAMERAS},
    }
    donor_id = digest(donor_observation)
    donor_native = solve(
        "same_donor_native", policy.prepare(donor_observation, donor_id, prompt)
    )
    for operator in ("vei", "vli"):
        condition, provenance = prepare_vision_interpolated(
            policy,
            donor_observation,
            donor_id,
            prompt,
            bank=bank,
            alpha=0.5,
            operator=operator,
        )
        values = solve(f"same_donor_{operator}", condition, provenance)
        report["checks"][f"same_donor_{operator}"] = (
            np.array_equal(donor_native, values) and not provenance["has_effect"]
        )
    if source_prompts is not None:
        condition, provenance = prepare_vision_interpolated(
            policy,
            observation,
            observation_id,
            prompt,
            alpha=0,
            operator="vli",
            source_prompts=source_prompts,
            text_latents=text_latents,
            language_alpha=0.5,
        )
        report["checks"]["composite_neutral"] = np.array_equal(
            base, solve("composite_neutral", condition, provenance)
        )
        text_condition, text_provenance = policy.prepare_interpolated(
            observation,
            observation_id,
            prompt,
            source_prompts=source_prompts,
            text_latents=text_latents,
            alpha=0,
            operator="tli",
        )
        text_native = solve("existing_tli", text_condition, text_provenance)
        condition, provenance = prepare_vision_interpolated(
            policy,
            observation,
            observation_id,
            prompt,
            alpha=0,
            operator="vli",
            source_prompts=source_prompts,
            text_latents=text_latents,
            language_alpha=0,
        )
        report["checks"]["existing_tli_composition_parity"] = np.array_equal(
            text_native, solve("composite_tli_only", condition, provenance)
        )
    report["checks"]["parameter_versions_unchanged"] = versions == [
        (name, parameter._version)
        for name, parameter in policy.policy.named_parameters()
    ]
    report["parameter_value_byte_hash_performed"] = False
    report["wall_seconds"] = time.perf_counter() - started
    report["status"] = "passed" if all(report["checks"].values()) else "failed"
    return report
