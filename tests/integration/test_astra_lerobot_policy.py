"""Native processor and upstream sampler integration; no downloads or large model.

Run in the pinned LeRobot runtime. A small deterministic velocity keeps these
regressions fast; the separate weighted smoke report tests the actual checkpoint.
"""

# Imports below importorskip intentionally require the optional native runtime.
# ruff: noqa: E402

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
native = pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
from lerobot.configs.policies import PreTrainedConfig
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from astra_reversal.lerobot_policy import (
    FrozenLeRobotPI05,
    load_native_model,
    load_processors,
)
from astra_reversal.records import to_numpy

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/astra/lerobot_pi05"


class Prefix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace()),
            tie_weights=lambda: None,
        )

    def forward(self, *, inputs_embeds, **kwargs):
        return None, {"value": inputs_embeds[0].mean(dim=(1, 2))}


class TinyFlow(torch.nn.Module):
    sample_actions = native.PI05Pytorch.sample_actions
    _rtc_enabled = native.PI05Pytorch._rtc_enabled
    _prepare_attention_masks_4d = native.PI05Pytorch._prepare_attention_masks_4d

    def __init__(self, config):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.1))
        self.config, self.rtc_processor = config, None
        self.paligemma_with_expert = Prefix()
        self.calls = []

    def embed_prefix(self, images, masks, tokens, token_masks):
        value = tokens.float().mean(1) / 100 + images[0].mean((1, 2, 3))
        return (
            value[:, None, None],
            token_masks[:, :1],
            torch.zeros_like(token_masks[:, :1]),
        )

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        self.calls.append((past_key_values, timestep.clone(), torch.is_grad_enabled()))
        return (
            0.1 * x_t
            + timestep[:, None, None]
            + past_key_values["value"][:, None, None]
        )


class TinyPolicy(torch.nn.Module):
    _rtc_enabled = native.PI05Policy._rtc_enabled
    _preprocess_images = native.PI05Policy._preprocess_images
    prepare_action = native.PI05Policy.prepare_action
    predict_action_chunk = native.PI05Policy.predict_action_chunk

    def __init__(self, config):
        super().__init__()
        config.validate_features()
        self.config = config
        self.model = TinyFlow(config)

    def _fix_pytorch_state_dict_keys(self, state, config):
        return state


@pytest.fixture
def adapter(monkeypatch):
    # Local synthetic vocabulary exercises the actual tokenizer processor and
    # state budgeting without claiming parity with the gated production tokenizer.
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "Task:": 2,
        "move": 3,
        "cup,": 4,
        "State:": 5,
        "Action:": 6,
        "128": 7,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: fast)
    config = PreTrainedConfig.from_pretrained(FIXTURE)
    config.device = "cpu"
    before, after = load_processors(FIXTURE, "cpu")
    return FrozenLeRobotPI05(TinyPolicy(config), before, after, {"test_only": True})


@pytest.fixture
def observation():
    return {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.full((256, 256, 3), 255, dtype=np.uint8),
    }


def test_saved_processors_and_full_action_padding(adapter, observation):
    actions = np.linspace(-0.5, 0.5, 350, dtype=np.float32).reshape(50, 7)
    encoded = adapter.input_transform(
        {**observation, "prompt": "move cup", "actions": actions}
    )
    assert encoded["actions"].shape == (50, 32)
    np.testing.assert_array_equal(encoded["actions"][:, :7], actions)
    assert not encoded["actions"][:, 7:].any()
    np.testing.assert_array_equal(adapter.output_transform(encoded)["actions"], actions)
    batch = adapter._preprocess({**observation, "prompt": "move cup"})
    assert batch["observation.images.image2"].shape == (1, 3, 256, 256)
    assert batch["observation.images.image2"].min() == 1


def test_upstream_sampler_parity_and_reverse_velocity(adapter, observation):
    condition = adapter.prepare(observation, "obs", "move cup")
    noise = adapter.noise(np.random.default_rng(0))
    generated = adapter.sample(condition, noise, steps=5)
    upstream = adapter.reference_actions(condition, noise, steps=5)
    np.testing.assert_allclose(to_numpy(generated.value)[0, :, :7], upstream, atol=1e-6)
    reverse = adapter.invert(condition, generated.value, steps=5)
    assert reverse.value.shape == (1, 50, 32)
    assert not torch.equal(reverse.value[:, :, 7:], generated.value[:, :, 7:])
    calls = adapter.model.calls[-5:]
    np.testing.assert_allclose([x[1].item() for x in calls], [0, 0.2, 0.4, 0.6, 0.8])
    assert all(x[0] is calls[0][0] and not x[2] for x in calls)
    assert not any(p.requires_grad for p in adapter.policy.parameters())


def test_condition_changes_do_not_overwrite_prior_prefix(adapter, observation):
    first = adapter.prepare(observation, "obs", "move cup")
    noise = adapter.noise(np.random.default_rng(0))
    before = first.velocity(noise, 0.5).clone()
    second = adapter.prepare(
        {**observation, "observation/image": observation["observation/wrist_image"]},
        "obs2",
        "move cup",
    )
    assert first.condition_id != second.condition_id
    assert not torch.equal(before, second.velocity(noise, 0.5))
    torch.testing.assert_close(before, first.velocity(noise, 0.5))


def test_prompt_budget_includes_state_tokens(adapter, observation):
    length = adapter.prompt_length("move cup", observation)
    assert length > 32
    adapter.max_token_len = length
    prompt, omitted = adapter.assemble_prompt(
        "move cup", "move " * 30, ["constraint " * 30], observation=observation
    )
    assert prompt == "move cup" and len(omitted) == 2
    with pytest.raises(ValueError, match="truncated"):
        adapter.prepare(observation, "obs", "move " * 300)


def test_weight_loading_is_strict(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_bytes((FIXTURE / "config.json").read_bytes())
    monkeypatch.setattr(native, "PI05Policy", TinyPolicy)
    save_file({"anchor": torch.tensor(0.3)}, tmp_path / "model.safetensors")
    loaded = load_native_model(tmp_path)
    assert loaded.model.anchor.item() == pytest.approx(0.3)
    assert not loaded.model.anchor.requires_grad
    save_file({"unrelated": torch.tensor(0.3)}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_native_model(tmp_path)


@pytest.fixture
def openpi_inputs(adapter):
    from astra_reversal.__main__ import enable_local_openpi
    from astra_reversal.openpi_inputs import use_openpi_libero_inputs

    assets = Path(
        os.environ.get(
            "EGOVERSE_TEST_PI05_INPUT_ASSETS",
            ROOT / "astra_reversal/.deps/reference/pi05_libero",
        )
    )
    if not (assets / "paligemma_tokenizer.model").exists():
        pytest.skip("Optional pinned OpenPI input assets are not downloaded")
    enable_local_openpi()
    return use_openpi_libero_inputs(adapter, assets)


def upstream_definition(relative_file, class_name, method_name=None):
    """Execute only the actual upstream definition, avoiding its JAX imports."""
    import ast
    import logging

    source = ROOT / "external/openpi/src/openpi" / relative_file
    tree = ast.parse(source.read_text())
    node = next(
        x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name
    )
    if method_name:
        node = next(
            x
            for x in node.body
            if isinstance(x, ast.FunctionDef) and x.name == method_name
        )
    scope = {"np": np, "logging": logging, "NormStats": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[method_name or class_name]


def test_openpi_profile_matches_upstream_tokenization_and_normalization(
    openpi_inputs, observation
):
    reference_class = upstream_definition("models/tokenizer.py", "PaligemmaTokenizer")
    reference = object.__new__(reference_class)
    reference._tokenizer = openpi_inputs.sentencepiece
    reference._max_len = 200
    raw = {**observation, "prompt": "  put_the cup\non the plate  "}
    raw["actions"] = np.linspace(-0.75, 0.75, 70, dtype=np.float32).reshape(10, 7)
    batch = openpi_inputs._preprocess(raw)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    expected_tokens, expected_mask = reference.tokenize(raw["prompt"], state=None)
    np.testing.assert_array_equal(
        to_numpy(batch[OBS_LANGUAGE_TOKENS])[0], expected_tokens
    )
    np.testing.assert_array_equal(
        to_numpy(batch[OBS_LANGUAGE_ATTENTION_MASK])[0], expected_mask
    )
    normalize = upstream_definition("transforms.py", "Normalize", "_normalize_quantile")
    unnormalize = upstream_definition(
        "transforms.py", "Unnormalize", "_unnormalize_quantile"
    )
    encoded = openpi_inputs.input_transform(raw)
    for key, value in (
        ("state", raw["observation/state"]),
        ("actions", raw["actions"]),
    ):
        stats = SimpleNamespace(**openpi_inputs.norm_stats[key])
        expected = normalize(None, value, stats).astype(np.float32)
        np.testing.assert_array_equal(encoded[key][..., : value.shape[-1]], expected)
        assert not encoded[key][..., value.shape[-1] :].any()
    expected = unnormalize(
        None, encoded["actions"], SimpleNamespace(**openpi_inputs.norm_stats["actions"])
    )
    decoded = openpi_inputs.output_transform(encoded)["actions"]
    np.testing.assert_array_equal(decoded, expected[:, :7])
    np.testing.assert_allclose(decoded, raw["actions"], atol=1e-7)
    assert batch["observation.images.image"].shape == (1, 3, 224, 224)


def test_openpi_profile_uses_native_sampler_without_state_tokens(
    openpi_inputs, observation
):
    condition = openpi_inputs.prepare(observation, "obs", "move cup")
    noise = openpi_inputs.noise(np.random.default_rng(0))
    assert noise.shape == (1, 10, 32)
    sampled = openpi_inputs.sample(condition, noise, steps=5)
    decoded = openpi_inputs.output_transform({"actions": to_numpy(sampled.value)[0]})[
        "actions"
    ]
    upstream = openpi_inputs.reference_actions(condition, noise, steps=5)
    np.testing.assert_allclose(decoded, upstream, atol=1e-6)
    changed = {**observation, "observation/state": np.ones(8, dtype=np.float32)}
    assert openpi_inputs.prompt_length(
        "move cup", observation
    ) == openpi_inputs.prompt_length("move cup", changed)
