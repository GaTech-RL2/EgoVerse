"""Instruction boundaries, endpoint identity, and prefill cache semantics."""

import math

import numpy as np
import pytest
import torch

from astra_reversal.interpolation_conditioning import (
    CAPTURE_BOUNDARY,
    TextLatentBank,
    align_instruction,
    interpolate_text,
    plain_instruction_mask,
    scoped_post_block_hooks,
    validate_request,
)


def test_plain_mask_excludes_bos_newline_and_padding_and_fails_closed():
    ids = np.array([[2, 5, 6, 7, 10, 0, 0]], np.int64)
    valid = np.array([[True] * 5 + [False] * 2])
    mask = plain_instruction_mask(ids, valid, [5, 6, 7], [10], 2)
    np.testing.assert_array_equal(
        mask, [[False, True, True, True, False, False, False]]
    )
    corrupted = ids.copy()
    corrupted[0, 4] = 11
    with pytest.raises(ValueError, match="differ"):
        plain_instruction_mask(corrupted, valid, [5, 6, 7], [10], 2)
    with pytest.raises(ValueError, match="Unsupported"):
        plain_instruction_mask(ids, valid, [], [10], 2)


@pytest.mark.parametrize("alpha", [False, -0.1, 1.1, math.nan, math.inf, "0.5"])
def test_alpha_fails_closed(alpha):
    with pytest.raises(ValueError, match="alpha"):
        validate_request(("a", "b"), alpha, "tei")


def test_alignment_reports_source_and_target_positions_and_protects_other_slots():
    target = torch.tensor([[[90.0], [10.0], [91.0], [20.0], [30.0], [92.0]]])
    mask = torch.tensor([[False, True, False, True, True, False]])
    source = torch.tensor([[[99.0], [2.0], [98.0], [4.0]]])
    source_mask = torch.tensor([[False, True, False, True]])
    aligned, mapping = align_instruction(source, source_mask, target, mask)
    assert mapping["mapped_positions"] == [(1, 1), (3, 3)]
    assert mapping["zero_padded_tokens"] == 1
    assert mapping["truncated_tokens"] == 0
    torch.testing.assert_close(
        aligned.flatten(), torch.tensor([0.0, 2.0, 0.0, 4.0, 0.0, 0.0])
    )
    edited, metrics = interpolate_text(
        target,
        mask,
        source,
        source_mask,
        source,
        source_mask,
        alpha=0.37,
        operator="tei",
    )
    torch.testing.assert_close(
        edited.flatten(), torch.tensor([90.0, 2.0, 91.0, 4.0, 0.0, 92.0])
    )
    assert metrics["protected_text_unchanged"]
    _, truncated = align_instruction(target, mask, source, source_mask)
    assert truncated["truncated_tokens"] == 1


@pytest.mark.parametrize("alpha", [0.0, 0.17, 0.5, 0.93, 1.0])
def test_equal_tei_sources_are_exact_native_identity(alpha):
    generator = torch.Generator().manual_seed(7)
    base = torch.randn((1, 8, 9), generator=generator)
    mask = torch.tensor([[False, True, True, True, True, False, False, False]])
    result, metrics = interpolate_text(
        base, mask, base, mask, base, mask, alpha=alpha, operator="tei"
    )
    assert result is base
    assert metrics["delta_frobenius"] == 0.0
    assert not metrics["has_effect"]


def test_tei_endpoints_and_tli_sign_are_distinct_operators():
    base = torch.tensor([[[99.0], [10.0], [20.0], [88.0]]])
    a = torch.tensor([[[97.0], [4.0], [6.0], [77.0]]])
    b = torch.tensor([[[96.0], [2.0], [3.0], [66.0]]])
    mask = torch.tensor([[False, True, True, False]])
    for alpha, source in ((0, a), (1, b)):
        result, _ = interpolate_text(
            base, mask, a, mask, b, mask, alpha=alpha, operator="tei"
        )
        assert torch.equal(result[mask], source[mask])
        assert torch.equal(result[~mask], base[~mask])
    for alpha, expected in ((0, [12.0, 23.0]), (1, [8.0, 17.0])):
        result, metrics = interpolate_text(
            base, mask, a, mask, b, mask, alpha=alpha, operator="tli"
        )
        torch.testing.assert_close(result[mask].flatten(), torch.tensor(expected))
        assert metrics["factor"] == 1 - 2 * alpha
        assert torch.equal(result[~mask], base[~mask])
    identity, metrics = interpolate_text(
        base, mask, object(), object(), object(), object(), alpha=0.5, operator="tli"
    )
    assert identity is base  # A zero residual does not inspect the source banks.
    assert metrics["factor"] == metrics["delta_frobenius"] == 0


def bank_arrays():
    return (
        np.arange(3 * 1 * 5 * 2, dtype=np.float32).reshape(3, 1, 5, 2),
        np.array([[2, 3, 4, 10, 0]], np.int64),
        np.array([[True, True, True, True, False]]),
        np.array([[False, True, True, False, False]]),
    )


def bank_provenance():
    return {
        "source_prompt": "source a",
        "compatibility": {"checkpoint": "test-only"},
        "capture_boundary": CAPTURE_BOUNDARY,
    }


def test_bank_roundtrip_identity_and_immutability(tmp_path):
    arrays = bank_arrays()
    original = arrays[0].copy()
    provenance = bank_provenance()
    bank = TextLatentBank(*arrays, provenance)
    arrays[0][:] = -1
    provenance["compatibility"]["checkpoint"] = "wrong"
    bank.provenance["compatibility"]["checkpoint"] = "also wrong"
    np.testing.assert_array_equal(bank.states, original)
    with pytest.raises(ValueError):
        bank.states.setflags(write=True)
    np.savez(
        tmp_path / "bank.npz",
        **{
            key: getattr(bank, key)
            for key in ("states", "token_ids", "token_mask", "instruction_mask")
        },
    )
    with np.load(tmp_path / "bank.npz", allow_pickle=False) as saved:
        restored = TextLatentBank(**saved, provenance=bank.metadata()["provenance"])
    assert restored.bank_id == bank.bank_id
    assert bank.metadata()["effective_layer_indices"] == [0, 1]
    restored.validate_for(
        prompt="source a",
        compatibility={"checkpoint": "test-only"},
        token_ids=bank.token_ids,
        token_mask=bank.token_mask,
        instruction_mask=bank.instruction_mask,
    )
    with pytest.raises(ValueError, match="prompt"):
        restored.validate_for(
            prompt="source b",
            compatibility={},
            token_ids=bank.token_ids,
            token_mask=bank.token_mask,
            instruction_mask=bank.instruction_mask,
        )
    with pytest.raises(ValueError, match="compatibility"):
        restored.validate_for(
            prompt="source a",
            compatibility={"checkpoint": "wrong"},
            token_ids=bank.token_ids,
            token_mask=bank.token_mask,
            instruction_mask=bank.instruction_mask,
        )


@pytest.mark.parametrize(
    "corruption", ["nonfinite", "dtype", "mask", "shape", "boundary"]
)
def test_bank_rejects_corruption(corruption):
    arrays = list(bank_arrays())
    provenance = bank_provenance()
    if corruption == "nonfinite":
        arrays[0][0, 0, 0, 0] = np.nan
    elif corruption == "dtype":
        arrays[0] = arrays[0].astype(np.float64)
    elif corruption == "mask":
        arrays[3][0, -1] = True
    elif corruption == "shape":
        arrays[0] = arrays[0][:1]
    else:
        provenance["capture_boundary"] = "final_norm"
    with pytest.raises(ValueError):
        TextLatentBank(*arrays, provenance)


class CacheBlock(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.tensor(float(index + 1)), requires_grad=False
        )
        self.cached_input = None

    def forward(self, hidden):
        self.cached_input = hidden.clone()
        return (hidden + self.bias, "untouched attention output")


def test_post_block_hook_changes_next_cache_but_not_current_cache_or_parameters():
    layers = torch.nn.ModuleList([CacheBlock(i) for i in range(3)])
    original = torch.tensor([[[100.0], [10.0], [20.0], [99.0]]])
    parameter_bytes = [parameter.detach().clone() for parameter in layers.parameters()]
    seen = []

    def edit(index, hidden):
        seen.append(hidden.clone())
        if index == len(layers) - 1:
            return None
        result = hidden.clone()
        result[:, 1:3] += 4
        return result

    with scoped_post_block_hooks(layers, edit):
        value = original
        for layer in layers:
            output = layer(value)
            assert output[1] == "untouched attention output"
            value = output[0]
    assert torch.equal(layers[0].cached_input, original)
    assert torch.equal(layers[1].cached_input[:, [0, 3]], seen[0][:, [0, 3]])
    assert torch.equal(layers[1].cached_input[:, 1:3], seen[0][:, 1:3] + 4)
    assert torch.equal(value, seen[2])  # No ineffective final-block injection.
    assert all(not layer._forward_hooks for layer in layers)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameter_bytes, layers.parameters(), strict=True)
    )
    first_cache = layers[1].cached_input.clone()
    layers[1](torch.zeros_like(original))
    assert not torch.equal(first_cache, layers[1].cached_input)


def test_hook_cleanup_after_forward_failure_and_incomplete_prefill():
    layers = torch.nn.ModuleList([CacheBlock(i) for i in range(3)])

    def broken(index, hidden):
        raise RuntimeError("synthetic model failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        with scoped_post_block_hooks(layers, broken):
            layers[0](torch.ones(1, 2, 3))
    assert all(not layer._forward_hooks for layer in layers)
    with pytest.raises(ValueError, match="every decoder"):
        with scoped_post_block_hooks(layers, lambda index, hidden: None):
            layers[0](torch.ones(1, 2, 3))
    assert all(not layer._forward_hooks for layer in layers)
