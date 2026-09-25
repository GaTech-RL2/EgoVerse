"""Tokenwise TEI and post-block TLI for the plain-language PI05 input profile.

The port aligns source instruction tokens to target instruction tokens in order,
zero-padding or truncating the source. BOS, newline, padding, and target masks
are preserved. This is an explicit port convention; it does not reproduce the
released implementation's hardcoded nine-token mask. TLI writes text residuals
after blocks 0..L-2 so the next block builds its K/V cache from the edited state.
Later attention can change image states even though the direct writes are text-only.
"""

import copy
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .records import digest, to_numpy

CAPTURE_BOUNDARY = "post_decoder_block_before_final_norm"
ALIGNMENT = "valid_instruction_tokens_left_aligned_zero_pad_or_truncate"
OPERATORS = ("tei", "tli", "tei_tli")


def validate_request(source_prompts, alpha, operator):
    if (
        not isinstance(source_prompts, (tuple, list))
        or len(source_prompts) != 2
        or any(not isinstance(p, str) or not p.strip() for p in source_prompts)
    ):
        raise ValueError("source_prompts must contain two nonempty strings A and B")
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be finite and in [0, 1]")
    return tuple(source_prompts), float(alpha)


def plain_instruction_mask(
    token_ids, token_mask, instruction_tokens, terminal_tokens, bos
):
    """Prove the plain-profile layout before selecting instruction-only slots."""
    ids, valid = np.asarray(token_ids), np.asarray(token_mask)
    expected = [bos, *instruction_tokens, *terminal_tokens]
    if (
        ids.dtype != np.int64
        or ids.ndim != 2
        or ids.shape[0] != 1
        or valid.dtype != np.bool_
        or valid.shape != ids.shape
        or not instruction_tokens
        or not terminal_tokens
        or len(expected) > ids.shape[1]
    ):
        raise ValueError("Unsupported plain instruction token layout")
    expected_mask = np.arange(ids.shape[1])[None] < len(expected)
    padded_ids = np.array([expected + [0] * (ids.shape[1] - len(expected))], np.int64)
    if not np.array_equal(ids, padded_ids) or not np.array_equal(valid, expected_mask):
        raise ValueError(
            "Native token IDs/mask differ from the verified plain instruction layout"
        )
    result = np.zeros_like(valid)
    result[:, 1 : 1 + len(instruction_tokens)] = True
    return result


def _immutable_array(value):
    array = np.ascontiguousarray(value)
    # Immutable bytes own the storage: setflags(write=True) cannot undo this.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, init=False)
class TextLatentBank:
    """Auditable native post-block states, including all L captured layers.

    Persist the four arrays in NPZ and ``metadata()`` in JSON. To load, construct
    this class with the saved arrays and metadata['provenance'], then require
    the reconstructed bank_id to equal the saved bank_id. A demonstration mean
    may zero noninstruction slots; the operator never reads those slots.
    """

    states: np.ndarray
    token_ids: np.ndarray
    token_mask: np.ndarray
    instruction_mask: np.ndarray
    bank_id: str
    _metadata_json: str

    def __init__(self, states, token_ids, token_mask, instruction_mask, provenance):
        arrays = [
            np.asarray(x) for x in (states, token_ids, token_mask, instruction_mask)
        ]
        states, token_ids, token_mask, instruction_mask = arrays
        if (
            states.dtype != np.float32
            or states.ndim != 4
            or states.shape[0] < 2
            or states.shape[1] != 1
            or min(states.shape[2:]) < 1
            or not np.isfinite(states).all()
            or token_ids.dtype != np.int64
            or token_ids.shape != states.shape[1:3]
            or token_mask.dtype != np.bool_
            or token_mask.shape != token_ids.shape
            or instruction_mask.dtype != np.bool_
            or instruction_mask.shape != token_ids.shape
            or not instruction_mask.any()
            or (instruction_mask & ~token_mask).any()
        ):
            raise ValueError(
                "Bank requires finite float32 [L,1,S,D] states and aligned int64 IDs/boolean masks"
            )
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("source_prompt"), str)
            or not provenance["source_prompt"].strip()
            or not isinstance(provenance.get("compatibility"), dict)
            or not provenance["compatibility"]
            or provenance.get("capture_boundary") != CAPTURE_BOUNDARY
        ):
            raise ValueError(
                "Bank provenance must bind its source prompt, native compatibility and capture boundary"
            )
        metadata = {
            "schema_version": 1,
            "provenance": copy.deepcopy(provenance),
            "arrays": {},
            "captured_layer_indices": list(range(states.shape[0])),
            "effective_layer_indices": list(range(states.shape[0] - 1)),
        }
        names = ("states", "token_ids", "token_mask", "instruction_mask")
        for name, array in zip(names, arrays, strict=True):
            object.__setattr__(self, name, _immutable_array(array))
            metadata["arrays"][name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": digest(array),
            }
        metadata["bank_id"] = digest(metadata)
        object.__setattr__(self, "bank_id", metadata["bank_id"])
        object.__setattr__(
            self,
            "_metadata_json",
            json.dumps(metadata, sort_keys=True, allow_nan=False),
        )

    @property
    def provenance(self):
        return self.metadata()["provenance"]

    def metadata(self):
        return json.loads(self._metadata_json)

    def validate_for(
        self, *, prompt, compatibility, token_ids, token_mask, instruction_mask
    ):
        if self.provenance["source_prompt"] != prompt:
            raise ValueError(
                "Text latent bank source prompt does not match its A/B source"
            )
        if self.provenance["compatibility"] != compatibility:
            raise ValueError(
                "Text latent bank native model/profile compatibility mismatch"
            )
        for name, current in (
            ("token_ids", token_ids),
            ("token_mask", token_mask),
            ("instruction_mask", instruction_mask),
        ):
            if not np.array_equal(getattr(self, name), to_numpy(current)):
                raise ValueError(
                    f"Text latent bank {name} does not match the source prompt"
                )


def _validate_text(values, mask, name):
    import torch

    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 3
        or values.shape[0] != 1
        or min(values.shape[1:]) < 1
        or not values.is_floating_point()
        or not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.shape != values.shape[:2]
        or mask.device != values.device
        or not mask.any().item()
        or not torch.isfinite(values).all().item()
    ):
        raise ValueError(
            f"{name} requires finite [1,S,D] values and a nonempty boolean instruction mask"
        )


def align_instruction(source, source_mask, target, target_mask):
    """Align only valid instruction positions; return a zero-padded target shape."""
    import torch

    _validate_text(source, source_mask, "Source")
    _validate_text(target, target_mask, "Target")
    if (
        source.shape[-1] != target.shape[-1]
        or source.device != target.device
        or source.dtype != target.dtype
    ):
        raise ValueError("Source and target text width/device/dtype must match")
    source_positions = torch.nonzero(source_mask[0], as_tuple=False).flatten()
    target_positions = torch.nonzero(target_mask[0], as_tuple=False).flatten()
    count = min(len(source_positions), len(target_positions))
    aligned = torch.zeros_like(target)
    aligned[:, target_positions[:count]] = source[:, source_positions[:count]]
    mapping = {
        "rule": ALIGNMENT,
        "source_positions": source_positions.tolist(),
        "target_positions": target_positions.tolist(),
        "mapped_positions": list(
            zip(
                source_positions[:count].tolist(),
                target_positions[:count].tolist(),
                strict=True,
            )
        ),
        "source_instruction_tokens": len(source_positions),
        "target_instruction_tokens": len(target_positions),
        "zero_padded_tokens": max(0, len(target_positions) - count),
        "truncated_tokens": max(0, len(source_positions) - count),
    }
    return aligned, mapping


def text_change_metrics(before, after, instruction_mask):
    import torch

    base = before[instruction_mask].double()
    delta = after[instruction_mask].double() - base
    base_norm = torch.linalg.vector_norm(base).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    if not math.isfinite(base_norm) or not math.isfinite(delta_norm):
        raise ValueError("Interpolation produced a nonfinite text residual")
    return {
        "text_before_sha256": digest(to_numpy(before)),
        "text_after_sha256": digest(to_numpy(after)),
        "delta_frobenius": delta_norm,
        "delta_rms": delta_norm / math.sqrt(base.numel()),
        "relative_rms": delta_norm / base_norm
        if base_norm
        else (0.0 if delta_norm == 0 else None),
        "base_frobenius": base_norm,
        "has_effect": not torch.equal(before, after),
        "protected_text_unchanged": torch.equal(
            before[~instruction_mask], after[~instruction_mask]
        ),
    }


def interpolate_text(
    target, target_mask, source_a, mask_a, source_b, mask_b, *, alpha, operator
):
    """TEI replaces instruction embeddings; TLI adds a signed residual.

    Inputs are already native sqrt(width)-scaled embeddings for TEI and native
    post-block hidden states for TLI. Arithmetic uses the target native dtype.
    There is no clipping or normalization of the paper-form TLI residual.
    """
    import torch

    _, alpha = validate_request(("a", "b"), alpha, operator)
    if operator == "tei_tli":
        raise ValueError(
            "Apply TEI at embeddings and TLI at post-block states separately"
        )
    _validate_text(target, target_mask, "Target")
    if operator == "tli" and alpha == 0.5:
        return target, {
            **text_change_metrics(target, target, target_mask),
            "factor": 0.0,
        }
    a, mapping_a = align_instruction(source_a, mask_a, target, target_mask)
    b, mapping_b = align_instruction(source_b, mask_b, target, target_mask)
    with torch.no_grad():
        if operator == "tei":
            if alpha == 0 or torch.equal(a, b):
                values = a
            elif alpha == 1:
                values = b
            else:
                values = (1 - alpha) * a + alpha * b
        else:
            values = target + (1 - 2 * alpha) * (a - b)
        edited = target.clone()
        edited[target_mask] = values[target_mask]
        if not torch.isfinite(edited).all().item():
            raise ValueError("Interpolation produced nonfinite text states")
        if torch.equal(edited, target):
            edited = target
    return edited, {
        **text_change_metrics(target, edited, target_mask),
        "mapping_a": mapping_a,
        "mapping_b": mapping_b,
        "factor": 1 - 2 * alpha if operator == "tli" else None,
    }


@contextmanager
def scoped_post_block_hooks(layers, callback):
    """Install ordered prefill-only hooks and remove them even after failure.

    callback(index, hidden) returns the replacement hidden state or None. It is
    called after the block has built its own K/V; its result affects later K/V.
    This context is deliberately not safe for concurrent forwards of one model.
    """
    handles, visited = [], []
    try:
        for index, layer in enumerate(layers):

            def hook(module, inputs, output, index=index):
                if not isinstance(output, tuple) or not output:
                    raise ValueError("Expected native decoder tuple output")
                if index != len(visited):
                    raise ValueError("Decoder layers did not run once in native order")
                visited.append(index)
                replacement = callback(index, output[0])
                if replacement is None or replacement is output[0]:
                    return output
                if (
                    replacement.shape != output[0].shape
                    or replacement.dtype != output[0].dtype
                    or replacement.device != output[0].device
                ):
                    raise ValueError(
                        "Post-block edit changed the native hidden-state layout"
                    )
                return (replacement, *output[1:])

            handles.append(layer.register_forward_hook(hook))
        yield visited
        if visited != list(range(len(layers))):
            raise ValueError("Native prefill did not visit every decoder layer")
    finally:
        for handle in handles:
            handle.remove()
