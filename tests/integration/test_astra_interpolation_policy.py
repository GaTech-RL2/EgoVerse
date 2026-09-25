"""Actual patched Gemma prefill/cache tests with tiny synthetic frozen weights.

Uses the pinned native tokenizer/input profile and 18 real decoder blocks, but
small width and a deterministic cache-reading velocity. These CPU tests verify
hook/cache mechanics, not weighted-checkpoint behavior or control performance.
"""

# Optional pinned native dependencies are required before the following imports.
# ruff: noqa: E402

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")
native = pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
sentencepiece = pytest.importorskip("sentencepiece")
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from astra_reversal.interpolation_conditioning import TextLatentBank
from astra_reversal.lerobot_policy import FrozenLeRobotPI05
from astra_reversal.openpi_inputs import OpenPILiberoInputs
from astra_reversal.records import digest


class SmallPrefix(torch.nn.Module):
    forward = native.PaliGemmaWithExpertModel.forward
    embed_language_tokens = native.PaliGemmaWithExpertModel.embed_language_tokens

    def __init__(self, vocabulary_size):
        super().__init__()
        config = GemmaConfig(
            vocab_size=vocabulary_size,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=18,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            use_adarms=False,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        self.paligemma = torch.nn.Module()
        self.paligemma.language_model = GemmaModel(config)

    def embed_image(self, image):
        return image.mean((1, 2, 3))[:, None, None].expand(-1, 2, 8)


class SmallFlow(torch.nn.Module):
    embed_prefix = native.PI05Pytorch.embed_prefix
    _prepare_attention_masks_4d = native.PI05Pytorch._prepare_attention_masks_4d
    sample_actions = native.PI05Pytorch.sample_actions
    _rtc_enabled = native.PI05Pytorch._rtc_enabled

    def __init__(self, vocabulary_size):
        super().__init__()
        self.paligemma_with_expert = SmallPrefix(vocabulary_size)
        self.config = SimpleNamespace(chunk_size=10, max_action_dim=32, rtc_config=None)
        self.rtc_processor = None
        self.cache_calls = []

    def _apply_checkpoint(self, fn, *args):
        return fn(*args)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        self.cache_calls.append(past_key_values)
        signal = sum(value.mean() for value in past_key_values.value_cache)
        return 0.1 * x_t + signal + timestep[:, None, None]


class SmallPolicy(torch.nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.model = SmallFlow(vocabulary_size)

    def _preprocess_images(self, batch):
        images = [batch[f"observation.images.{name}"] for name in ("image", "image2")]
        return images, [torch.ones(1, dtype=torch.bool) for _ in images]


@pytest.fixture(scope="module")
def adapter():
    from astra_reversal.__main__ import enable_local_openpi

    enable_local_openpi()
    root = Path(__file__).resolve().parents[2]
    assets = Path(
        os.environ.get(
            "EGOVERSE_TEST_PI05_INPUT_ASSETS",
            root / "astra_reversal/.deps/reference/pi05_libero",
        )
    )
    tokenizer_path = assets / "paligemma_tokenizer.model"
    if not tokenizer_path.is_file():
        pytest.skip("Optional pinned tokenizer assets are not downloaded")
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    with torch.random.fork_rng():
        torch.manual_seed(71)
        policy = SmallPolicy(tokenizer.get_piece_size()).eval().requires_grad_(False)
    result = object.__new__(OpenPILiberoInputs)
    result.policy, result.model = policy, policy.model
    result.config = SimpleNamespace(
        input_features={"observation.state": SimpleNamespace(shape=(8,))},
        image_features={
            "observation.images.image": None,
            "observation.images.image2": None,
        },
    )
    result.metadata = {
        "input_profile": "openpi_libero",
        "artifact_sha256": {"synthetic": "test-only"},
        "input_profile_assets": {
            "tokenizer": digest(tokenizer_path.read_bytes().hex())
        },
    }
    result.device = torch.device("cpu")
    result.max_token_len, result.horizon, result.action_dim = 200, 10, 32
    result.sentencepiece = tokenizer
    result.norm_stats = {
        key: {name: np.asarray(value) for name, value in values.items()}
        for key, values in json.loads((assets / "norm_stats.json").read_text())[
            "norm_stats"
        ].items()
    }
    return result


@pytest.fixture
def observation():
    return {
        "observation/state": np.arange(8, dtype=np.float32) / 100,
        "observation/image": np.full((16, 16, 3), 20, np.uint8),
        "observation/wrist_image": np.full((16, 16, 3), 70, np.uint8),
    }


def run_velocity(adapter, condition):
    return condition.velocity(torch.zeros((1, 10, 32)), 0.5).clone()


def test_plain_profile_instruction_span_has_no_robot_state_tokens(adapter, observation):
    prompt = "pick up the red block"
    bank = adapter.capture_text_latents(observation, prompt, observation_id="original")
    changed = adapter.capture_text_latents(
        {**observation, "observation/state": np.ones(8, np.float32)},
        prompt,
        observation_id="changed",
    )
    assert bank.states.shape == (18, 1, 200, 8)
    assert bank.metadata()["effective_layer_indices"] == list(range(17))
    np.testing.assert_array_equal(bank.token_ids, changed.token_ids)
    np.testing.assert_array_equal(bank.instruction_mask, changed.instruction_mask)
    assert not bank.instruction_mask[0, 0]
    assert not bank.instruction_mask[~bank.token_mask].any()
    assert (bank.token_mask & ~bank.instruction_mask).sum() >= 2
    assert bank.provenance["raw_condition_id"] != changed.provenance["raw_condition_id"]


@pytest.mark.parametrize("alpha", [0.0, 0.31, 1.0])
def test_equal_tei_sources_match_native_cache_and_sampler_exactly(
    adapter, observation, alpha
):
    prompt = "pick up the red block"
    native_condition = adapter.prepare(observation, "obs", prompt)
    native_value = run_velocity(adapter, native_condition)
    edited, report = adapter.prepare_interpolated(
        observation,
        "obs",
        prompt,
        source_prompts=(prompt, prompt),
        alpha=alpha,
        operator="tei",
    )
    assert torch.equal(native_value, run_velocity(adapter, edited))
    assert edited.condition_id == native_condition.condition_id
    assert report["native_target_equivalent"]
    assert report["protected_embedding_slots_unchanged"]
    batch = adapter._preprocess({**observation, "prompt": prompt})
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, masks = adapter.policy._preprocess_images(batch)
    noise = torch.zeros((1, 10, 32))
    with torch.no_grad():
        reference = adapter.model.sample_actions(
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            noise=noise,
            num_steps=10,
        )
    actual = adapter.sample(edited, noise, steps=10)
    assert torch.equal(actual.value, reference)


def test_matching_mask_tei_endpoints_match_native_sources(adapter, observation):
    a, b = "pick up the red block", "pick up the blue block"
    source_a = adapter.capture_text_latents(observation, a)
    source_b = adapter.capture_text_latents(observation, b)
    np.testing.assert_array_equal(source_a.instruction_mask, source_b.instruction_mask)
    np.testing.assert_array_equal(source_a.token_mask, source_b.token_mask)
    for alpha, prompt in ((0.0, a), (1.0, b)):
        reference = run_velocity(adapter, adapter.prepare(observation, "obs", prompt))
        edited, report = adapter.prepare_interpolated(
            observation, "obs", a, source_prompts=(a, b), alpha=alpha, operator="tei"
        )
        assert torch.equal(reference, run_velocity(adapter, edited))
        assert report["vision_prefix_unchanged"]


def test_tli_midpoint_is_exact_native_and_nonzero_changes_next_cache_only(
    adapter, observation
):
    a, b = "pick up the red block", "pick up the blue block"
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in adapter.policy.named_parameters()
    }
    reference = adapter.prepare(observation, "obs", a)
    native_value = run_velocity(adapter, reference)
    native_cache = adapter.model.cache_calls[-1]
    midpoint, midpoint_report = adapter.prepare_interpolated(
        observation,
        "obs",
        a,
        source_prompts=(a, b),
        alpha=0.5,
        operator="tli",
        text_latents=None,
    )
    assert midpoint.condition_id == reference.condition_id
    assert torch.equal(native_value, run_velocity(adapter, midpoint))
    assert not midpoint_report["tli_active"]
    banks = {
        "a": adapter.capture_text_latents(observation, a),
        "b": adapter.capture_text_latents(observation, b),
    }
    edited, report = adapter.prepare_interpolated(
        observation,
        "obs",
        a,
        source_prompts=(a, b),
        alpha=0.0,
        operator="tli",
        text_latents=banks,
    )
    edited_value = run_velocity(adapter, edited)
    edited_cache = adapter.model.cache_calls[-1]
    assert not torch.equal(edited_value, native_value)
    assert torch.equal(native_cache.key_cache[0], edited_cache.key_cache[0])
    assert not torch.equal(native_cache.key_cache[1], edited_cache.key_cache[1])
    assert len(report["tli_layers"]) == 17
    assert all(
        row["protected_text_unchanged"] and row["direct_vision_write_unchanged"]
        for row in report["tli_layers"]
    )
    assert all(
        not layer._forward_hooks
        for layer in adapter.model.paligemma_with_expert.paligemma.language_model.layers
    )
    assert torch.equal(native_value, run_velocity(adapter, reference))
    assert all(
        torch.equal(parameters[name], parameter) and not parameter.requires_grad
        for name, parameter in adapter.policy.named_parameters()
    )


def test_swapped_bank_and_saved_state_profile_fail_before_interpolation(
    adapter, observation
):
    a, b = "pick up the red block", "pick up the blue block"
    bank = adapter.capture_text_latents(observation, a)
    with pytest.raises(ValueError, match="source prompt"):
        adapter.prepare_interpolated(
            observation,
            "obs",
            a,
            source_prompts=(a, b),
            alpha=0,
            operator="tli",
            text_latents={"a": bank, "b": bank},
        )
    reloaded = TextLatentBank(
        bank.states,
        bank.token_ids,
        bank.token_mask,
        bank.instruction_mask,
        bank.provenance,
    )
    assert reloaded.bank_id == bank.bank_id
    for key in ("adapter_source_sha256", "processor_source_sha256"):
        provenance = bank.provenance
        provenance["compatibility"][key] = "changed-source"
        incompatible = TextLatentBank(
            bank.states,
            bank.token_ids,
            bank.token_mask,
            bank.instruction_mask,
            provenance,
        )
        with pytest.raises(ValueError, match="compatibility"):
            adapter.prepare_interpolated(
                observation,
                "obs",
                a,
                source_prompts=(a, a),
                alpha=0,
                operator="tli",
                text_latents={"a": incompatible, "b": bank},
            )
    unsupported = object.__new__(FrozenLeRobotPI05)
    unsupported.metadata = {"input_profile": "checkpoint"}
    with pytest.raises(ValueError, match="State/Action"):
        unsupported.capture_text_latents(observation, a)


def test_combined_same_source_is_exact_native_identity(adapter, observation):
    prompt = "pick up the red block"
    bank = adapter.capture_text_latents(observation, prompt)
    native_condition = adapter.prepare(observation, "obs", prompt)
    expected = run_velocity(adapter, native_condition)
    condition, report = adapter.prepare_interpolated(
        observation,
        "obs",
        prompt,
        source_prompts=(prompt, prompt),
        alpha=0.31,
        operator="tei_tli",
        text_latents={"a": bank, "b": bank},
    )
    assert torch.equal(run_velocity(adapter, condition), expected)
    assert condition.condition_id == native_condition.condition_id
    assert not report["has_effect"]
    assert all(not row["has_effect"] for row in report["tli_layers"])
