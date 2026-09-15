"""Lane C: the Qwen 3.5-0.8B VLM stem pair.

One VLM forward per batch feeds both the image stem (visual tokens) and the
text stem (text tokens of the same, image-conditioned forward):

* ``Qwen35VLMEncoder`` replaces the ResNet as ``encoder_specs.front_img_1``;
* ``Qwen35TextStem`` replaces ``QwenPerTokenEncoder`` as
  ``shared_stem_specs.annotation`` and holds a PLAIN reference to the encoder,
  so the VLM weights live in the policy exactly once;
* ``HPTModel.stem_process`` hands the batch's prompts to the encoder just
  before calling it.

The stub tests fake the HF model/processor so the token split, the mask and
the pretrained-weight bookkeeping are checked without a 1.9 GiB snapshot; the
skip-guarded tests repeat the shape/freeze claims against the real Qwen 3.5.
"""

from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.algo.hpt import HPTModel
from egomimic.models.hpt_nets import (
    Qwen35TextStem,
    Qwen35VLMEncoder,
    verify_pretrained_weights,
)

STUB_HIDDEN = 16
STUB_IMAGE_SIZE = (4, 8)  # -> (4/2) * (8/2) = 8 visual tokens
STUB_VISUAL_TOKENS = 8
IMAGE_TOKEN_ID = 900
TRUNK_EMBED = 32


# --------------------------------------------------------------------------
# a fake Qwen 3.5: deterministic hidden states, known image-token layout
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Whitespace tokenizer with a growing vocabulary (ids start at 1)."""

    def __init__(self) -> None:
        self.padding_side = "left"
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._ids:
            index = len(self._ids) + 1
            self._ids[word] = index
            self._words[index] = word
        return self._ids[word]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(w) for w in str(text).split()]}

    def decode(self, ids):
        return " ".join(self._words[i] for i in ids)


class _FakeProcessor:
    """Chat layout ``<img> <prompt>``; the image expands to one token per
    (2x2) patch of the frame, which is what pins the visual token count."""

    IMAGE = "<img>"
    STRIDE = 2

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        content = messages[0]["content"]
        assert content[0]["type"] == "image", "the image must come first"
        return f"{self.IMAGE} {content[1]['text']}".strip()

    def __call__(
        self,
        text,
        images,
        padding=True,
        return_tensors="pt",
        do_rescale=None,
        do_resize=None,
    ):
        assert do_rescale is False and do_resize is False
        images = list(images)
        n_vis = (images[0].shape[-2] // self.STRIDE) * (
            images[0].shape[-1] // self.STRIDE
        )
        rows = []
        for item in text:
            ids: list[int] = []
            for word in item.split():
                if word == self.IMAGE:
                    ids.extend([IMAGE_TOKEN_ID] * n_vis)
                else:
                    ids.append(self.tokenizer._id(word))
            rows.append(ids)
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):  # right padding, as the stem asks for
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention[i, : len(row)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "pixel_values": torch.stack(images).flatten(1),
        }


class _FakeInner(nn.Module):
    """``model.model``: the VLM without the lm_head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(1024, STUB_HIDDEN)
        self.layers = nn.ModuleList(
            [nn.Linear(STUB_HIDDEN, STUB_HIDDEN) for _ in range(2)]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        output_hidden_states=False,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        if pixel_values is not None:
            # make the image tokens depend on the pixels, so a wrong split shows
            image = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).float()
            hidden = hidden + image * pixel_values.mean(dim=-1)[:, None, None]
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        out = {"last_hidden_state": hidden}
        if output_hidden_states:
            out["hidden_states"] = tuple(states)
        return SimpleNamespace(**out)


class _FakeVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            text_config=SimpleNamespace(hidden_size=STUB_HIDDEN, num_hidden_layers=2),
            vision_config=SimpleNamespace(
                patch_size=_FakeProcessor.STRIDE, spatial_merge_size=1
            ),
        )
        self.model = _FakeInner()
        # a tied head: present on the module, absent from the checkpoint
        self.lm_head = nn.Linear(STUB_HIDDEN, 8, bias=False)
        self.lm_head.weight = nn.Parameter(torch.randn(8, STUB_HIDDEN))


def _write_stub_snapshot(tmp_path) -> str:
    """A snapshot dir holding the fake VLM's weights, plus one tensor the live
    model never builds (the real snapshot ships an unused ``mtp.*`` head)."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    reference = _FakeVLM()
    snapshot = tmp_path / "qwen35_stub"
    snapshot.mkdir()
    state = {k: v.contiguous() for k, v in reference.state_dict().items()}
    state.pop("lm_head.weight")  # tied in the real checkpoint too
    state["mtp.fc.weight"] = torch.zeros(2, 2)
    save_file(state, str(snapshot / "model.safetensors-00001-of-00001.safetensors"))
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    k: "model.safetensors-00001-of-00001.safetensors" for k in state
                }
            }
        ),
        encoding="utf-8",
    )
    return str(snapshot)


@pytest.fixture()
def stub_snapshot(tmp_path, monkeypatch):
    """Patch the two ``transformers`` entry points ``Qwen35VLMEncoder`` uses."""
    import transformers
    from safetensors.torch import load_file

    path = _write_stub_snapshot(tmp_path)

    def _load_model(load_path, dtype=None, **kwargs):
        model = _FakeVLM()
        state = load_file(
            os.path.join(load_path, "model.safetensors-00001-of-00001.safetensors")
        )
        state.pop("mtp.fc.weight", None)
        model.load_state_dict(state, strict=False)
        model.lm_head.weight = nn.Parameter(model.model.embed.weight[:8].clone())
        return model if dtype is None else model.to(dtype)

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: _FakeProcessor()
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        staticmethod(_load_model),
    )
    return path


def _cross_attn_specs(embed_dim: int = TRUNK_EMBED, latent: int = 4):
    return OmegaConf.create(
        {
            "random_horizon_masking": False,
            "cross_attn": {
                "crossattn_latent": latent,
                "crossattn_heads": 2,
                "crossattn_dim_head": 8,
                "crossattn_modality_dropout": 0.0,
                "modality_embed_dim": embed_dim,
            },
        }
    )


def _stub_encoder(snapshot: str, **kwargs) -> Qwen35VLMEncoder:
    kwargs.setdefault("image_size", STUB_IMAGE_SIZE)
    return Qwen35VLMEncoder(model_name=snapshot, dtype="float32", **kwargs)


# --------------------------------------------------------------------------
# 1. the config
# --------------------------------------------------------------------------

QWEN35_MODEL = "hpt_bc_mecka_6d_300M_qwen35"


@pytest.fixture()
def qwen35_cfg():
    from hydra import compose, initialize_config_module

    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(
            config_name="train_zarr_mecka_flagship_6d_hpt",
            overrides=[f"model={QWEN35_MODEL}"],
        )


def test_qwen35_config_wires_the_vlm_stem_pair(qwen35_cfg):
    rm = qwen35_cfg.model.robomimic_model
    assert (
        rm.encoder_specs.front_img_1._target_
        == "egomimic.models.hpt_nets.Qwen35VLMEncoder"
    )
    assert rm.encoder_specs.front_img_1.image_size == [352, 640]
    assert rm.encoder_specs.front_img_1.freeze is True
    assert rm.encoder_specs.front_img_1.trainable_layers == 0
    assert rm.encoder_specs.front_img_1.feature_layer == -1
    assert (
        rm.shared_stem_specs.annotation._target_
        == "egomimic.models.hpt_nets.Qwen35TextStem"
    )
    assert rm.shared_stem_specs.annotation.input_dim == 1024
    assert rm.shared_stem_specs.annotation.output_dim == 840
    assert rm.shared_stem_specs.annotation.vlm_modality == "front_img_1"
    # the image stem now takes the VLM's hidden width, not the ResNet's 840
    assert rm.shared_stem_specs.front_img_1.input_dim == 1024
    # the trunk layout is untouched
    assert rm.shared_obs_keys == ["front_img_1", "annotation"]
    for key in ("front_img_1", "annotation"):
        assert rm.shared_stem_specs[key].specs.cross_attn.crossattn_latent == 18
    assert rm.trunk.action_horizon == 64


def test_qwen35_config_drops_the_imagenet_normalize(qwen35_cfg):
    """The Qwen processor normalizes; a second ImageNet Normalize would put the
    frames outside the [0, 1] the encoder hands it."""
    rm = qwen35_cfg.model.robomimic_model
    targets = [t._target_ for t in rm.train_image_augs.transforms]
    assert targets == ["torchvision.transforms.ColorJitter"]
    assert rm.eval_image_augs is None


def test_qwen35_config_leaves_the_resnet_recipe_alone():
    from hydra import compose, initialize_config_module

    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(config_name="train_zarr_mecka_flagship_6d_hpt")
    rm = cfg.model.robomimic_model
    assert rm.encoder_specs.front_img_1._target_ == "egomimic.models.hpt_nets.ResNet"
    assert (
        rm.shared_stem_specs.annotation._target_
        == "egomimic.models.hpt_nets.QwenPerTokenEncoder"
    )
    assert rm.shared_stem_specs.front_img_1.input_dim == 840
    assert any(
        t._target_ == "torchvision.transforms.Normalize"
        for t in rm.train_image_augs.transforms
    )


# --------------------------------------------------------------------------
# 2. the token split, on the stub
# --------------------------------------------------------------------------


def test_encoder_pins_the_visual_token_count(stub_snapshot):
    encoder = _stub_encoder(stub_snapshot)
    assert encoder.num_visual_tokens == STUB_VISUAL_TOKENS
    # an image_size the patch grid cannot tile would make the count depend on
    # the processor's rounding, so it is rejected outright
    with pytest.raises(ValueError, match="divisible"):
        _stub_encoder(stub_snapshot, image_size=(5, 8))


def test_encoder_splits_visual_and_text_tokens(stub_snapshot):
    encoder = _stub_encoder(stub_snapshot).eval()
    images = torch.rand(2, 1, 1, 3, *STUB_IMAGE_SIZE)
    prompts = ["fold the towel", "pick up the red cup and put it down"]
    encoder.set_prompts(prompts)
    visual = encoder(images)

    assert visual.shape == (2, STUB_VISUAL_TOKENS, STUB_HIDDEN)
    assert torch.isfinite(visual).all()
    # the two samples saw different pixels, so their visual tokens differ
    assert not torch.allclose(visual[0], visual[1])

    text, mask = encoder.last_text_features, encoder.last_text_mask
    assert text.shape[0] == 2 and text.shape[-1] == STUB_HIDDEN
    # 3 vs 9 prompt words; no chat-template tokens in the stub layout
    assert mask.sum(dim=1).tolist() == [3, 9]
    assert text.shape[1] == 9
    # right padding: the real tokens are the leading ones
    assert mask[0].tolist() == [True] * 3 + [False] * 6
    assert (text[0, 3:] == 0).all()

    # ``set_prompts`` is per batch: the encoder refuses a stale/blank pairing
    encoder.set_prompts(prompts[:1])
    with pytest.raises(RuntimeError, match="prompts"):
        encoder(images)


def test_feature_layer_minus_one_is_the_last_hidden_state(stub_snapshot):
    last = _stub_encoder(stub_snapshot).eval()
    explicit = _stub_encoder(stub_snapshot, feature_layer=2).eval()  # 2 layers
    images = torch.rand(2, 1, 1, 3, *STUB_IMAGE_SIZE)
    prompts = ["fold the towel", "pick up the cup"]
    last.set_prompts(prompts)
    explicit.set_prompts(prompts)
    assert torch.allclose(last(images), explicit(images), atol=1e-6)


def test_text_stem_reads_the_encoder_and_masks_padding(stub_snapshot):
    """Short-vs-alone invariance: the padded row's latent must equal the one it
    gets on its own, which only holds if the mask reaches the cross-attention."""
    torch.manual_seed(3)
    encoder = _stub_encoder(stub_snapshot).eval()
    specs = _cross_attn_specs()
    stem = Qwen35TextStem(input_dim=STUB_HIDDEN, output_dim=TRUNK_EMBED, specs=specs)
    stem.init_cross_attn(specs.cross_attn)
    stem.attach_vlm(encoder)
    stem.eval()
    # the encoder must not be a registered child: the weights exist once
    assert not any(module is encoder for module in stem.modules())
    assert "_vlm" not in dict(stem.named_children())
    assert not any(k.startswith("_vlm") for k in stem.state_dict())

    images = torch.rand(2, 1, 1, 3, *STUB_IMAGE_SIZE)
    prompts = ["fold the towel", "pick up the red cup and put it down"]
    with torch.no_grad():
        encoder.set_prompts(prompts)
        encoder(images)
        batched = stem.compute_latent(prompts)

        # the same short prompt, alone in its own batch: no padding at all
        encoder.set_prompts(prompts[:1])
        encoder(images[:1])
        alone = stem.compute_latent(prompts[:1])

        # what an unmasked cross-attention would give for the padded row
        encoder.set_prompts(prompts)
        encoder(images)
        feats, _ = stem.forward_with_mask(prompts)
        tokens = stem.tokens.repeat(feats.shape[0], 1, 1)
        unmasked = stem.cross_attention(tokens, feats)

    masked_diff = (batched[0] - alone[0]).abs().max().item()
    unmasked_diff = (unmasked[0] - alone[0]).abs().max().item()
    print(
        f"\n[vlm-mask] max|short-in-padded-batch - short-alone|: "
        f"with mask {masked_diff:.3e} / without mask {unmasked_diff:.3e}"
    )
    assert masked_diff < 1e-5, masked_diff
    assert unmasked_diff > 1e-3, "the stub is degenerate: padding did not matter"


def test_text_stem_refuses_a_batch_the_encoder_did_not_see(stub_snapshot):
    encoder = _stub_encoder(stub_snapshot).eval()
    specs = _cross_attn_specs()
    stem = Qwen35TextStem(input_dim=STUB_HIDDEN, output_dim=TRUNK_EMBED, specs=specs)
    stem.init_cross_attn(specs.cross_attn)
    with pytest.raises(RuntimeError, match="no VLM encoder attached"):
        stem.compute_latent(["a"])
    stem.attach_vlm(encoder)
    with pytest.raises(RuntimeError, match="no text features"):
        stem.compute_latent(["a"])
    encoder.set_prompts(["a", "b"])
    encoder(torch.rand(2, 1, 1, 3, *STUB_IMAGE_SIZE))
    with pytest.raises(RuntimeError, match="2 rows of text features"):
        stem.compute_latent(["a"])


# --------------------------------------------------------------------------
# 3. the pretrained-weight bookkeeping
# --------------------------------------------------------------------------


def test_finalize_modules_leaves_the_vlm_untouched(stub_snapshot):
    """``_init_weights`` must not reach the VLM's Linear/LayerNorm layers."""
    encoder = _stub_encoder(stub_snapshot)
    before = {k: v.clone() for k, v in encoder.pretrained_state_dict().items()}
    model = HPTModel(embed_dim=TRUNK_EMBED, num_blocks=1, num_heads=2)
    model.init_encoder("front_img_1", encoder)
    model.stems["shared_front_img_1"] = nn.Linear(STUB_HIDDEN, TRUNK_EMBED)
    model.finalize_modules()
    after = encoder.pretrained_state_dict()
    assert set(after) == set(before)
    for key, value in before.items():
        assert torch.equal(after[key], value), key


def test_verify_pretrained_weights_covers_the_vlm(stub_snapshot):
    encoder = _stub_encoder(stub_snapshot)
    # the tied lm_head is not in the checkpoint; the unused mtp.* tensor is not
    # in the model. Both sides are reconciled, so the sets agree.
    live = set(encoder.pretrained_state_dict())
    reference = encoder.pretrained_reference_state_dict()
    assert reference is not None and set(reference) == live
    assert "model.lm_head.weight" not in live
    assert "model.mtp.fc.weight" not in live
    assert "model.model.embed.weight" in live

    checked = verify_pretrained_weights(encoder, verbose=False)
    assert len(checked) == 1 and next(iter(checked.values()))

    with torch.no_grad():
        encoder.model.model.layers[0].weight.add_(1.0)
    with pytest.raises(RuntimeError, match="do not match their checkpoint"):
        verify_pretrained_weights(encoder, verbose=False)


def test_frozen_vlm_has_no_trainable_parameters(stub_snapshot):
    encoder = _stub_encoder(stub_snapshot)
    assert not any(p.requires_grad for p in encoder.parameters())
    encoder.train(True)
    assert not encoder.model.training, "a frozen VLM must stay in eval mode"
    unfrozen = _stub_encoder(stub_snapshot, trainable_layers=1)
    trainable = [n for n, p in unfrozen.named_parameters() if p.requires_grad]
    assert trainable == ["model.model.layers.1.weight", "model.model.layers.1.bias"]


def test_vlm_keeps_its_load_dtype_through_an_fp32_upcast(stub_snapshot):
    """``HPT.__init__`` runs ``nets.float()``; a 853M VLM must not follow it."""
    encoder = Qwen35VLMEncoder(
        model_name=stub_snapshot, dtype="bfloat16", image_size=STUB_IMAGE_SIZE
    )
    holder = nn.ModuleDict({"policy": encoder})
    holder.float()
    assert encoder.model.model.layers[0].weight.dtype == torch.bfloat16


# --------------------------------------------------------------------------
# 4. the real Qwen 3.5 (skipped when the snapshot is absent)
# --------------------------------------------------------------------------


def _qwen35_snapshot() -> str | None:
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    hits = sorted(
        glob.glob(os.path.join(cache, "models--Qwen--Qwen3.5-0.8B", "snapshots", "*"))
    )
    return hits[-1] if hits else None


QWEN35_SNAPSHOT = _qwen35_snapshot()


def _transformers_too_old() -> bool:
    import transformers

    major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
    return (major, minor) < (5, 2)


requires_qwen35 = pytest.mark.skipif(
    QWEN35_SNAPSHOT is None or _transformers_too_old(),
    reason="Qwen3.5-0.8B snapshot absent or transformers < 5.2",
)


@pytest.fixture(scope="module")
def real_encoder():
    if QWEN35_SNAPSHOT is None or _transformers_too_old():
        pytest.skip("Qwen3.5-0.8B snapshot absent or transformers < 5.2")
    return Qwen35VLMEncoder(model_name=QWEN35_SNAPSHOT).eval()


@requires_qwen35
def test_real_qwen35_shapes_and_freeze(real_encoder):
    encoder = real_encoder
    assert encoder.num_visual_tokens == 220
    assert encoder.hidden_size == 1024
    assert not any(p.requires_grad for p in encoder.parameters())

    images = torch.rand(2, 1, 1, 3, 360, 640)  # HPT frames, resized to 352x640
    prompts = ["fold the towel", "pick up the red cup and place it on the plate"]
    encoder.set_prompts(prompts)
    with torch.no_grad():
        visual = encoder(images)
    text, mask = encoder.last_text_features, encoder.last_text_mask
    print(
        f"\n[qwen35] visual {tuple(visual.shape)} text {tuple(text.shape)}"
        f" real text tokens {mask.sum(dim=1).tolist()}"
    )
    assert visual.shape == (2, 220, 1024)
    assert text.shape[0] == 2 and text.shape[-1] == 1024
    assert torch.isfinite(visual).all() and torch.isfinite(text).all()
    assert mask.sum(dim=1)[0] < mask.sum(dim=1)[1]  # shorter prompt, fewer tokens
    assert (text[0, mask.sum(dim=1)[0] :] == 0).all()  # right padded


@requires_qwen35
def test_real_qwen35_feature_layer_minus_one_is_the_last_hidden_state(real_encoder):
    """``feature_layer=-1`` must be the same tensor ``hidden_states[-1]`` is."""
    images = torch.rand(1, 1, 1, 3, 352, 640)
    prompts = ["fold the towel"]
    assert real_encoder.feature_layer == -1
    real_encoder.set_prompts(prompts)
    with torch.no_grad():
        visual = real_encoder(images)
    # hidden_states[24] of a 24-layer text stack is the same post-norm tensor
    depth = int(real_encoder.model.config.text_config.num_hidden_layers)
    real_encoder.feature_layer = depth
    try:
        real_encoder.set_prompts(prompts)
        with torch.no_grad():
            other = real_encoder(images)
    finally:
        real_encoder.feature_layer = -1
    assert torch.equal(visual, other)


@requires_qwen35
def test_real_qwen35_passes_the_weight_hash_check(real_encoder):
    checked = verify_pretrained_weights(real_encoder, verbose=True)
    assert len(checked) == 1
    print(f"\n[qwen35] verified sha256: {next(iter(checked.values()))}")


# --------------------------------------------------------------------------
# 5. one HPT training step with the stub VLM, through the real recipe
# --------------------------------------------------------------------------


def test_hpt_train_step_with_the_vlm_stem_pair(tmp_path, monkeypatch, stub_snapshot):
    from fixtures.recipes import Recipe, common_overrides, cpu_trainer_overrides
    from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures

    import egomimic.trainHydra as train_hydra
    from egomimic.models import hpt_nets

    hermetic_env(monkeypatch)
    recipe = Recipe(
        "train_zarr_cartesian", "mecka_all_6d", QWEN35_MODEL, "human_bimanual"
    )
    data, out, hashes = write_fixtures(tmp_path, "mecka")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(2)
        + [
            "model.robomimic_model.trunk.num_blocks=2",
            f"model.robomimic_model.head_specs.{recipe.embodiment}.model.nblocks=1",
            f"model.robomimic_model.head_specs.{recipe.embodiment}.num_inference_steps=2",
            f"model.robomimic_model.encoder_specs.front_img_1.model_name={stub_snapshot}",
            "model.robomimic_model.encoder_specs.front_img_1.dtype=float32",
            f"model.robomimic_model.encoder_specs.front_img_1.image_size=[{STUB_IMAGE_SIZE[0]},{STUB_IMAGE_SIZE[1]}]",
            f"model.robomimic_model.shared_stem_specs.front_img_1.input_dim={STUB_HIDDEN}",
            f"model.robomimic_model.shared_stem_specs.annotation.input_dim={STUB_HIDDEN}",
        ],
        out,
    )

    seen: list[int] = []
    original = hpt_nets.SimpleTransformer.forward

    def spy(self, tokens, *args, **kwargs):
        seen.append(tokens.shape[-2])
        return original(self, tokens, *args, **kwargs)

    monkeypatch.setattr(hpt_nets.SimpleTransformer, "forward", spy)

    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert loss == loss, "non-finite loss"
    assert objects["trainer"].global_step == 2
    # 64 action + 18 proprio + 18 image + 18 text
    assert seen and set(seen) == {64 + 18 + 18 + 18}, seen
