"""CPU synthetic prefix/cache evidence; no checkpoint or control-quality claim."""

import copy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal import vision_interpolation as vision
from astra_reversal.image_donor_bank import CAMERAS, DonorImage, ImageDonorLibrary
from astra_reversal.interpolation_catalog import DATASET_REPO, DATASET_REVISION
from astra_reversal.lerobot_policy import FrozenLeRobotPI05
from astra_reversal.records import digest, to_numpy

TOKENS = "observation.language.tokens"
MASK = "observation.language.attention_mask"


class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        # Attention-like cross-slot propagation distinguishes direct slot writes
        # from changes to later text representations and K/V inputs.
        return (hidden + 0.07 * torch.tanh(self.mix(hidden.mean(1, keepdim=True))),)


class Prefix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = torch.nn.Module()
        self.paligemma.config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=224, patch_size=112)
        )
        language = torch.nn.Module()
        language.embed_tokens = torch.nn.Embedding(32, 4)
        language.layers = torch.nn.ModuleList([DecoderBlock() for _ in range(18)])
        language.config = SimpleNamespace(_attn_implementation="eager")
        self.paligemma.language_model = language
        self.projector = torch.nn.Linear(3, 4, bias=False)
        self.caches = []

    def embed_image(self, image):
        patches = torch.nn.functional.avg_pool2d(image, 112).flatten(2).transpose(1, 2)
        return self.projector(patches)

    def embed_language_tokens(self, tokens):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(self, **kwargs):
        hidden = kwargs["inputs_embeds"][0]
        caches = []
        for block in self.paligemma.language_model.layers:
            caches.append(hidden.clone())  # Its K/V is built before its output hook.
            hidden = block(hidden)[0]
        self.caches.append(caches)
        return hidden, caches


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = Prefix()

    def embed_prefix(self, images, masks, tokens, token_mask):
        embeddings = [self.paligemma_with_expert.embed_image(image) for image in images]
        pad = [mask[:, None].expand(-1, 4) for mask in masks]
        embeddings.append(self.paligemma_with_expert.embed_language_tokens(tokens) * 2)
        pad.append(token_mask)
        values, padding = torch.cat(embeddings, 1), torch.cat(pad, 1)
        return values, padding, torch.zeros_like(padding)

    def _prepare_attention_masks_4d(self, mask):
        return mask

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        # Nonuniform position weights detect accidental camera/grid reordering.
        weights = torch.arange(1, past_key_values[0].shape[1] + 1)[None, :, None]
        signal = sum((value * weights).mean() for value in past_key_values) / 50
        return 0.1 * x_t + signal + timestep[:, None, None]


class NativePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Flow()
        self.config = SimpleNamespace(
            image_features={
                **dict.fromkeys(vision.CAMERA_FEATURES.values()),
                "observation.images.empty": None,
            }
        )

    def _preprocess_images(self, batch):
        present = [key for key in self.config.image_features if key in batch]
        missing = [key for key in self.config.image_features if key not in batch]
        images = [batch[key] for key in present]
        images.extend([torch.full_like(images[0], -1) for _ in missing])
        masks = [torch.tensor([True]) for _ in present] + [
            torch.tensor([False]) for _ in missing
        ]
        return images, masks


class Tokenizer:
    def bos_id(self):
        return 1

    def encode(self, text, add_bos=False):
        if text == "\n":
            return [30]
        return [sum(word.encode()) % 28 + 2 for word in text.split()]


class Adapter(FrozenLeRobotPI05):
    def __init__(self):
        with torch.random.fork_rng():
            torch.manual_seed(44)
            self.policy = NativePolicy().eval().requires_grad_(False)
        self.model, self.config = self.policy.model, self.policy.config
        self.device = torch.device("cpu")
        self.horizon, self.action_dim = 10, 32
        self.sentencepiece = Tokenizer()
        self.metadata = {
            "input_profile": "openpi_libero",
            "artifact_sha256": {"synthetic": "not-a-weighted-checkpoint"},
            "adapter_source_sha256": "synthetic-fixture",
        }

    def _preprocess(self, observation):
        prompt = observation["prompt"].strip().replace("_", " ").replace("\n", " ")
        ids = [1, *self.sentencepiece.encode(prompt), 30]
        batch = {
            TOKENS: torch.tensor([ids + [0] * (8 - len(ids))]),
            MASK: torch.tensor([[True] * len(ids) + [False] * (8 - len(ids))]),
            "observation.state": torch.tensor(
                np.pad(observation["observation/state"], (0, 24))[None]
            ),
        }
        for camera, feature in vision.CAMERA_FEATURES.items():
            batch[feature] = (
                torch.tensor(np.array(observation[camera]))
                .permute(2, 0, 1)[None]
                .float()
                / 255
            )
        return batch


@pytest.fixture
def adapter(monkeypatch):
    constants = ModuleType("lerobot.utils.constants")
    constants.OBS_LANGUAGE_TOKENS = TOKENS
    constants.OBS_LANGUAGE_ATTENTION_MASK = MASK
    modeling = ModuleType("lerobot.policies.pi05.modeling_pi05")
    modeling.make_att_2d_masks = lambda padding, attention: (
        padding[:, None, :] & padding[:, :, None]
    )
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield Adapter()
    torch.set_num_threads(previous)


@pytest.fixture
def observation():
    return {
        CAMERAS[0]: np.full((224, 224, 3), 30, np.uint8),
        CAMERAS[1]: np.full((224, 224, 3), 80, np.uint8),
        "observation/state": np.arange(8, dtype=np.float32) / 10,
    }


def donor_pair(index=0):
    result = {}
    for camera_index, camera in enumerate(CAMERAS):
        pixels = np.full((224, 224, 3), 130 + 10 * camera_index + index, np.uint8)
        pixels[112:, :112] += 40
        row = {
            "library_id": "1" * 64,
            "sample_sha256": digest(index),
            "pixels_sha256": digest(pixels),
            "donor_id": f"synthetic-{index}",
            "camera": camera,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "source_id": "13",
            "prompt": "synthetic training fixture",
            "episode_index": 0,
            "frame_index": index,
            "phase": {"numerator": index, "denominator": 4},
        }
        result[camera] = DonorImage(row["donor_id"], camera, pixels, row)
    return result


def run(adapter, condition):
    return to_numpy(
        adapter.sample(condition, torch.zeros(1, 10, 32), steps=10).value
    ).copy()


def model_hash(adapter):
    return digest(
        {name: to_numpy(value) for name, value in adapter.policy.state_dict().items()}
    )


@pytest.mark.parametrize("operator", ["vei", "vli"])
def test_input_skill_sequence_compiles_through_visual_hooks(
    adapter, observation, operator
):
    from astra_reversal.demo_skill_conditioning import InputSkillConditioner

    from .test_demo_skills import Bank, stage

    bank = Bank()
    bank.sources = copy.deepcopy(bank.sources)
    for source in bank.sources.values():
        source["episode_index"] = 0
    compiler = InputSkillConditioner(adapter, bank, {})
    choice = {**stage(vision_operator=operator, alpha=0.6), "frame": 12}
    raw_hash, weights = digest(observation), model_hash(adapter)
    native, _, _ = compiler.prepare(observation, "target", {**choice, "alpha": 0})
    edited, receipt, effective = compiler.prepare(observation, "target", choice)
    assert not np.allclose(run(adapter, native), run(adapter, edited))
    assert digest(effective) == raw_hash == digest(observation)
    assert model_hash(adapter) == weights
    assert receipt["setting"]["frame"] == 12 and compiler.capture_count == 1
    compiler.prepare(observation, "target", choice)
    assert compiler.capture_count == 1
    compiler.prepare(observation, "target", {**choice, "frame": 13})
    assert compiler.capture_count == 2


def test_selected_camera_patch_endpoint_and_protected_values():
    values = torch.arange(64, dtype=torch.float32).reshape(1, 16, 4)
    donor = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4) * -1
    layout = {
        "text_start": 12,
        "tokens_per_camera": 4,
        "camera_slots": {
            camera: {"start": 4 * i, "stop": 4 * (i + 1), "valid": True}
            for i, camera in enumerate(CAMERAS)
        },
    }
    original = values.clone()
    edited, report = vision.interpolate_visual(
        values, donor, layout, alpha=1, cameras=(CAMERAS[1],)
    )
    assert torch.equal(edited[:, 4:8], donor[:, 4:8])
    assert torch.equal(edited[:, :4], original[:, :4])
    assert torch.equal(
        edited[:, 8:], original[:, 8:]
    )  # Includes padded camera and all text.
    assert torch.equal(values, original)
    assert report["protected_slots_unchanged"] and report["direct_text_write_unchanged"]
    assert report["protected_before_sha256"] == report["protected_after_sha256"]
    zero, _ = vision.interpolate_visual(values, object(), layout, alpha=0)
    assert zero is values


@pytest.mark.parametrize(
    "alpha", [-0.01, 1.01, True, float("nan"), float("inf"), "0.5"]
)
def test_invalid_alpha_fails_before_native_forward(adapter, observation, alpha):
    with pytest.raises(ValueError, match="alpha"):
        vision.prepare_vision_interpolated(
            adapter, observation, "obs", "target", alpha=alpha
        )
    assert not adapter.model.paligemma_with_expert.caches


def test_capture_has_exact_pair_grid_and_immutable_states_without_state_leak(
    adapter, observation
):
    pair = donor_pair()
    bank = vision.capture_vision_bank(
        adapter, pair, prompt="target", observation=observation
    )
    changed = copy.deepcopy(observation)
    changed["observation/state"][:] = -10
    other = vision.capture_vision_bank(
        adapter, pair, prompt="target", observation=changed
    )
    assert bank.bank_id == other.bank_id
    assert bank.states.shape == (18, 1, 8, 4)
    assert bank.metadata()["effective_layer_indices"] == list(range(17))
    assert (
        bank.provenance["layout"]["camera_slots"]["observation.images.empty"]["valid"]
        is False
    )
    assert bank.capture["prefix_forwards"] == 1
    assert bank.capture["velocity_evaluations"] == 0
    assert bank.provenance["donor_state_imported"] is False
    with pytest.raises(ValueError):
        bank.states.setflags(write=True)
    with pytest.raises(ValueError):
        bank.embeddings[0, 0, 0] = 99
    metadata = bank.metadata()
    metadata["provenance"]["target_prompt"] = "tamper"
    assert bank.provenance["target_prompt"] == "target"


def test_vei_capture_skips_all_language_blocks(adapter):
    bank = vision.capture_vision_bank(
        adapter, donor_pair(), prompt="target", include_layers=False
    )
    assert bank.states is None
    assert bank.capture["prefix_forwards"] == 0
    assert not adapter.model.paligemma_with_expert.caches


@pytest.mark.parametrize("operator", ["vei", "vli", "vei_vli"])
def test_zero_alpha_ignores_bank_and_preserves_native_full_chunk_and_cache(
    adapter, observation, operator
):
    original = model_hash(adapter)
    native = adapter.prepare(observation, "obs", "target")
    expected = run(adapter, native)
    edited, report = vision.prepare_vision_interpolated(
        adapter, observation, "obs", "target", bank=object(), alpha=0, operator=operator
    )
    np.testing.assert_array_equal(run(adapter, edited), expected)
    assert native.condition_id == edited.condition_id
    assert report["native_target_equivalent"]
    assert report["bank"] is None
    np.testing.assert_array_equal(
        run(adapter, native), expected
    )  # Old closure's K/V remains valid.
    assert model_hash(adapter) == original


@pytest.mark.parametrize("operator", ["vei", "vli"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_same_donor_is_exact_identity_at_every_strength(
    adapter, observation, operator, alpha
):
    pair = donor_pair()
    bank = vision.capture_vision_bank(adapter, pair, prompt="target")
    observation.update({camera: pair[camera].pixels for camera in CAMERAS})
    expected = run(adapter, adapter.prepare(observation, "obs", "target"))
    condition, report = vision.prepare_vision_interpolated(
        adapter, observation, "obs", "target", bank=bank, alpha=alpha, operator=operator
    )
    np.testing.assert_array_equal(run(adapter, condition), expected)
    assert not report["has_effect"]


@pytest.mark.parametrize("operator", ["vei", "vli"])
def test_nonzero_edits_reach_velocity_at_correct_cache_boundary(
    adapter, observation, operator
):
    before = model_hash(adapter)
    pair = donor_pair()
    bank = vision.capture_vision_bank(adapter, pair, prompt="target")
    native = adapter.prepare(observation, "obs", "target")
    expected = run(adapter, native)
    caches = adapter.model.paligemma_with_expert.caches
    native_cache = caches[-1]
    condition, report = vision.prepare_vision_interpolated(
        adapter, observation, "obs", "target", bank=bank, alpha=0.5, operator=operator
    )
    edited_cache = caches[-1]
    assert not np.array_equal(run(adapter, condition)[..., :7], expected[..., :7])
    assert torch.equal(native_cache[0], edited_cache[0]) == (operator == "vli")
    if operator == "vli":
        assert not torch.equal(native_cache[1][:, :8], edited_cache[1][:, :8])
        assert torch.equal(native_cache[1][:, 8:], edited_cache[1][:, 8:])
        assert not torch.equal(native_cache[2][:, 12:], edited_cache[2][:, 12:])
        assert [row["layer_index"] for row in report["vli_layers"]] == list(range(17))
    for row in [report["vei"], *report["vli_layers"]]:
        assert row["protected_slots_unchanged"] and row["direct_text_write_unchanged"]
    np.testing.assert_array_equal(run(adapter, native), expected)
    assert model_hash(adapter) == before
    assert all(
        not block._forward_hooks
        for block in adapter.model.paligemma_with_expert.paligemma.language_model.layers
    )


def test_composition_matches_existing_tli_when_visual_disabled_and_has_disjoint_writes(
    adapter, observation
):
    prompts = ("red block", "blue cup")
    banks = {
        label: adapter.capture_text_latents(observation, prompt)
        for label, prompt in zip(("a", "b"), prompts, strict=True)
    }
    original, _ = adapter.prepare_interpolated(
        observation,
        "obs",
        "target",
        source_prompts=prompts,
        text_latents=banks,
        alpha=0,
        operator="tli",
    )
    expected = run(adapter, original)
    condition, report = vision.prepare_vision_interpolated(
        adapter,
        observation,
        "obs",
        "target",
        alpha=0,
        operator="vli",
        source_prompts=prompts,
        text_latents=banks,
        language_alpha=0,
    )
    np.testing.assert_array_equal(run(adapter, condition), expected)
    assert all(
        row["direct_vision_write_unchanged"] for row in report["language"]["layers"]
    )
    bank = vision.capture_vision_bank(adapter, donor_pair(), prompt="target")
    condition, report = vision.prepare_vision_interpolated(
        adapter,
        observation,
        "obs",
        "target",
        bank=bank,
        alpha=0.5,
        operator="vli",
        source_prompts=prompts,
        text_latents=banks,
        language_alpha=0,
    )
    assert not np.array_equal(run(adapter, condition), expected)
    assert len(report["vli_layers"]) == len(report["language"]["layers"]) == 17
    assert all(row["direct_text_write_unchanged"] for row in report["vli_layers"])
    assert all(
        row["direct_vision_write_unchanged"] and row["protected_text_unchanged"]
        for row in report["language"]["layers"]
    )


def test_neutral_composition_needs_no_text_or_visual_bank(adapter, observation):
    native = adapter.prepare(observation, "obs", "target")
    condition, report = vision.prepare_vision_interpolated(
        adapter,
        observation,
        "obs",
        "target",
        alpha=0,
        operator="vli",
        source_prompts=("a", "b"),
        text_latents=object(),
        language_alpha=0.5,
    )
    np.testing.assert_array_equal(run(adapter, condition), run(adapter, native))
    assert report["native_target_equivalent"]


@pytest.mark.parametrize("bad", ["prompt", "weights", "layout", "layers"])
def test_mismatched_banks_fail_closed(adapter, observation, bad):
    bank = vision.capture_vision_bank(
        adapter, donor_pair(), prompt="target", include_layers=bad != "layers"
    )
    prompt = "wrong target" if bad == "prompt" else "target"
    if bad == "weights":
        adapter.metadata["artifact_sha256"] = {"different": "weights"}
    if bad == "layout":
        adapter.config.image_features = dict(
            reversed(list(adapter.config.image_features.items()))
        )
    with pytest.raises(ValueError, match="compatibility|include_layers"):
        vision.prepare_vision_interpolated(
            adapter, observation, "obs", prompt, bank=bank, alpha=0.5, operator="vli"
        )


def test_cross_frame_pair_and_unpinned_dataset_rejected(adapter):
    pair = donor_pair()
    pair[CAMERAS[1]] = donor_pair(1)[CAMERAS[1]]
    with pytest.raises(ValueError, match="same paired"):
        vision.capture_vision_bank(adapter, pair, prompt="target")
    pair = donor_pair()
    image = pair[CAMERAS[0]]
    provenance = image.provenance
    provenance["dataset_repo"] = "ood-demonstrations"
    pair[CAMERAS[0]] = DonorImage(
        image.donor_id, image.camera, image.pixels, provenance
    )
    with pytest.raises(ValueError, match="STANDARD"):
        vision.capture_vision_bank(adapter, pair, prompt="target")


def test_failed_prefill_removes_all_hooks(adapter, observation, monkeypatch):
    layers = adapter.model.paligemma_with_expert.paligemma.language_model.layers
    bank = vision.capture_vision_bank(adapter, donor_pair(), prompt="target")

    def broken(hidden):
        raise RuntimeError("injected prefill failure")

    monkeypatch.setattr(layers[6], "forward", broken)
    with pytest.raises(RuntimeError, match="injected"):
        vision.prepare_vision_interpolated(
            adapter, observation, "obs", "target", bank=bank, alpha=0.5, operator="vli"
        )
    assert all(not block._forward_hooks for block in layers)


def test_cache_reuses_state_independent_capture_and_evicts_donor(adapter, observation):
    pairs = [donor_pair(index) for index in range(3)]
    images = {
        (image.donor_id, camera): image
        for pair in pairs
        for camera, image in pair.items()
    }
    library = ImageDonorLibrary({"library_id": "1" * 64}, images, [])
    cache = vision.VisionBankCache(max_entries=2)
    first, _ = cache.get(adapter, observation, "target", library, "synthetic-0")
    changed = copy.deepcopy(observation)
    changed["observation/state"] *= 3
    repeated, receipt = cache.get(adapter, changed, "target", library, "synthetic-0")
    assert repeated is first and receipt["cache_hit"]
    assert receipt["capture"]["prefix_forwards"] == 0
    cache.get(adapter, observation, "target", library, "synthetic-1")
    _, receipt = cache.get(adapter, observation, "target", library, "synthetic-2")
    assert (
        receipt["resident_banks"] == 2 and receipt["resident_bytes"] == 2 * first.nbytes
    )
    recaptured, receipt = cache.get(
        adapter, observation, "target", library, "synthetic-0"
    )
    assert not receipt["cache_hit"] and recaptured.bank_id == first.bank_id


def test_weighted_probe_refuses_local_cpu(adapter, observation):
    with pytest.raises(ValueError, match="allocated CUDA"):
        vision.weighted_vision_probe(adapter, observation, "target", donor_pair())
    assert not adapter.model.paligemma_with_expert.caches
