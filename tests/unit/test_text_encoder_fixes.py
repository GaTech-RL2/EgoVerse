"""Lane C: the HPT text-encoder fixes.

Three defects, one test file:

1. ``HPTModel.finalize_modules`` used ``self.apply(self._init_weights)``, which
   xavier-randomised the *pretrained* HF encoders inside the text stems (and
   would do the same to any future pretrained stem). It now skips anything a
   module declares through ``PretrainedWeights``.
2. ``HPT.__init__`` called ``finalize_modules`` twice.
3. ``QwenPerTokenEncoder.compute_latent`` never passed the tokenizer attention
   mask to its cross-attention, so padded positions - which are the learned
   constant ``proj.bias``, not zero, because the zeroing happens before the
   biased ``proj`` - got real attention weight.

Plus the new guard rail: the pretrained weights are hashed against their
on-disk snapshot once the model is built, and a mismatch fails the run.
"""

from __future__ import annotations

import glob
import inspect
import os

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.models.hpt_nets import (
    PolicyStem,
    PretrainedWeights,
    QwenPerTokenEncoder,
    apply_skipping_pretrained,
    hash_state_dict,
    verify_pretrained_weights,
)

EMBED_DIM = 32


def _cross_attn_specs(latent: int = 4) -> OmegaConf:
    return OmegaConf.create(
        {
            "random_horizon_masking": False,
            "cross_attn": {
                "crossattn_latent": latent,
                "crossattn_heads": 2,
                "crossattn_dim_head": 8,
                "crossattn_modality_dropout": 0.0,
                "modality_embed_dim": EMBED_DIM,
            },
        }
    )


# --------------------------------------------------------------------------
# a tiny stand-in for a pretrained stem
# --------------------------------------------------------------------------


class FakePretrainedStem(PretrainedWeights, nn.Module):
    """Carries the marker, an "encoder" that must never be touched, and a
    ``proj`` that HPT owns and therefore must still be initialised."""

    _hpt_pretrained_attrs = ("encoder",)

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 8))
        self.proj = nn.Linear(8, EMBED_DIM)
        with torch.no_grad():  # distinctive "checkpoint" values, no default zeros
            for param in self.encoder.parameters():
                param.normal_(mean=1.0, std=1.0)
        # pretend these came off disk
        self._reference = {
            k: v.detach().clone() for k, v in self.pretrained_state_dict().items()
        }

    def pretrained_reference_state_dict(self):
        return {k: v.clone() for k, v in self._reference.items()}


def _small_hpt_model(stems: dict):
    from egomimic.algo.hpt import HPTModel

    model = HPTModel(
        embed_dim=EMBED_DIM,
        num_blocks=1,
        num_heads=2,
        observation_horizon=1,
        action_horizon=2,
        token_postprocessing="action_token",
    )
    model.init_domain_stem("d", stems)
    return model


# --------------------------------------------------------------------------
# 1 / 2. the init pass skips pretrained weights and nothing else
# --------------------------------------------------------------------------


def test_apply_skipping_pretrained_matches_apply_when_nothing_is_pretrained():
    """Modules HPT builds itself must be initialised exactly as before: with no
    pretrained markers in the tree the new traversal is bit-identical to
    ``nn.Module.apply`` (same post-order, same RNG draws)."""
    from egomimic.algo.hpt import HPTModel

    def build():
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Linear(8, 8),
            nn.LayerNorm(8),
            nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4)),
        )

    def init(module):  # HPTModel._init_weights only ever looks at its argument
        HPTModel._init_weights(None, module)

    reference, candidate = build(), build()
    torch.manual_seed(1234)
    reference.apply(init)
    torch.manual_seed(1234)
    apply_skipping_pretrained(candidate, init)

    ref_state, cand_state = reference.state_dict(), candidate.state_dict()
    assert set(ref_state) == set(cand_state)
    for key in ref_state:
        assert torch.equal(ref_state[key], cand_state[key]), key


def test_finalize_modules_leaves_pretrained_weights_untouched():
    torch.manual_seed(0)
    stem = FakePretrainedStem()
    before_pretrained = {
        k: v.detach().clone() for k, v in stem.pretrained_state_dict().items()
    }
    before_proj = stem.proj.weight.detach().clone()

    model = _small_hpt_model({"annotation": stem})
    model.finalize_modules()

    after = stem.pretrained_state_dict()
    for key, value in before_pretrained.items():
        assert torch.equal(value, after[key]), f"pretrained tensor {key} was modified"

    # ... while the module HPT owns *was* re-initialised (xavier + zero bias).
    assert not torch.equal(before_proj, stem.proj.weight)
    assert torch.equal(stem.proj.bias, torch.zeros_like(stem.proj.bias))


def test_finalize_modules_initialises_everything_hpt_builds():
    """Every Linear/LayerNorm outside a pretrained subtree gets the xavier
    treatment (bias 0 / LayerNorm 1,0), trunk and head included."""
    torch.manual_seed(0)
    stem = FakePretrainedStem()
    model = _small_hpt_model({"annotation": stem})
    model.finalize_modules()

    skipped = {id(m) for m in stem.encoder.modules()}
    seen = 0
    for module in model.modules():
        if id(module) in skipped:
            continue
        if isinstance(module, nn.Linear) and module.bias is not None:
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
            seen += 1
        elif isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
            seen += 1
    assert seen > 5, "expected the trunk/stem modules to be visited"

    # the pretrained LayerNorm keeps its (non-default) loaded values
    ln = stem.encoder[1]
    assert not torch.equal(ln.bias, torch.zeros_like(ln.bias))


# --------------------------------------------------------------------------
# 3. finalize_modules is called once per HPT.__init__
# --------------------------------------------------------------------------


def test_finalize_modules_called_once_in_hpt_init():
    from egomimic.algo.hpt import HPT

    source = inspect.getsource(HPT.__init__)
    assert source.count("finalize_modules()") == 1, (
        "HPT.__init__ must call finalize_modules exactly once; a second call "
        "re-runs the init pass and re-draws the shared action tokens"
    )


# --------------------------------------------------------------------------
# 4. the weight hash check
# --------------------------------------------------------------------------


def test_verify_pretrained_weights_passes_on_a_clean_model():
    torch.manual_seed(0)
    stem = FakePretrainedStem()
    model = _small_hpt_model({"annotation": stem})
    model.finalize_modules()
    checked = verify_pretrained_weights(model, verbose=False)
    assert checked, "the marked stem should have been hashed"
    ((name, digest),) = checked.items()
    assert "annotation" in name
    assert digest == hash_state_dict(stem.pretrained_reference_state_dict())


def test_verify_pretrained_weights_raises_when_a_weight_is_perturbed():
    torch.manual_seed(0)
    stem = FakePretrainedStem()
    model = _small_hpt_model({"annotation": stem})
    model.finalize_modules()
    with torch.no_grad():
        stem.encoder[0].weight[0, 0] += 1e-3
    with pytest.raises(RuntimeError, match="do not match their checkpoint"):
        verify_pretrained_weights(model, verbose=False)


# --------------------------------------------------------------------------
# 5. the per-token text stem passes the attention mask
# --------------------------------------------------------------------------


class StubPerTokenEncoder(QwenPerTokenEncoder):
    """``QwenPerTokenEncoder`` with the HF model swapped for fixed features, so
    the mask plumbing is tested without RoPE/position effects."""

    def __init__(self, hidden: torch.Tensor, mask: torch.Tensor, specs) -> None:
        PolicyStem.__init__(self, specs=specs)
        self.hidden_size = hidden.shape[-1]
        self.output_dim = EMBED_DIM
        self.freeze_encoder = False  # no HF encoder to keep in eval mode
        self._snapshot_dir = ""
        self._load_dtype = torch.float32
        self.proj = nn.Linear(self.hidden_size, EMBED_DIM)
        self.stub = (hidden, mask)

    def _encode(self, prompts):
        return self.stub


def _padded_stub(seq_short: int = 3, seq_long: int = 7, hidden_dim: int = 16):
    torch.manual_seed(7)
    hidden = torch.randn(2, seq_long, hidden_dim)
    mask = torch.ones(2, seq_long, dtype=torch.long)
    mask[0, : seq_long - seq_short] = 0  # left padding, as the tokenizer pads
    return hidden, mask


def test_per_token_stem_masks_padding():
    hidden, mask = _padded_stub()
    specs = _cross_attn_specs()
    stem = StubPerTokenEncoder(hidden, mask, specs)
    stem.init_cross_attn(specs.cross_attn)
    stem.eval()

    with torch.no_grad():
        batched = stem.compute_latent(None)

        # the same (short) prompt encoded on its own: no padding at all
        real = mask[0].bool()
        stem.stub = (hidden[:1, real], mask[:1, real])
        alone = stem.compute_latent(None)

        # what the parent did: cross-attention over every position, mask ignored
        stem.stub = (hidden, mask)
        feat = stem(None)
        tokens = stem.tokens.repeat(feat.shape[0], 1, 1)
        unmasked = stem.cross_attention(tokens, feat)

    masked_diff = (batched[0] - alone[0]).abs().max().item()
    unmasked_diff = (unmasked[0] - alone[0]).abs().max().item()
    print(
        f"\n[mask] max|short-in-padded-batch - short-alone|: "
        f"with mask {masked_diff:.3e} / without mask (parent) {unmasked_diff:.3e}"
    )
    assert masked_diff < 1e-5, masked_diff
    assert (
        unmasked_diff > 1e-3
    ), "the stub is degenerate: padding made no difference even unmasked"


# --------------------------------------------------------------------------
# the same, with the real Qwen encoder (skipped when the snapshot is absent)
# --------------------------------------------------------------------------


def _qwen_snapshot() -> str | None:
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    hits = sorted(
        glob.glob(
            os.path.join(cache, "models--Qwen--Qwen3-Embedding-0.6B", "snapshots", "*")
        )
    )
    return hits[-1] if hits else None


QWEN_SNAPSHOT = _qwen_snapshot()
requires_qwen = pytest.mark.skipif(
    QWEN_SNAPSHOT is None, reason="Qwen3-Embedding-0.6B snapshot not in the HF cache"
)


@pytest.fixture(scope="module")
def qwen_stem():
    if QWEN_SNAPSHOT is None:
        pytest.skip("Qwen3-Embedding-0.6B snapshot not in the HF cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    specs = _cross_attn_specs()
    stem = QwenPerTokenEncoder(
        model_name=QWEN_SNAPSHOT, max_length=32, output_dim=EMBED_DIM, specs=specs
    )
    stem.init_cross_attn(specs.cross_attn)
    return stem


@requires_qwen
def test_finalize_modules_keeps_the_real_qwen_encoder_bit_identical(qwen_stem):
    before = hash_state_dict(qwen_stem.pretrained_state_dict())
    model = _small_hpt_model({"annotation": qwen_stem})
    model.finalize_modules()
    after = hash_state_dict(qwen_stem.pretrained_state_dict())
    reference = hash_state_dict(qwen_stem.pretrained_reference_state_dict())
    print(
        f"\n[qwen] sha256 after finalize_modules: {after}\n[qwen] snapshot: {reference}"
    )
    assert after == before
    assert after == reference
    # and the check that runs at train start agrees, including after the upcast
    model.float()
    verify_pretrained_weights(model, verbose=False)


@requires_qwen
def test_verify_pretrained_weights_catches_a_perturbed_qwen_weight(qwen_stem):
    param = next(iter(qwen_stem.encoder.parameters()))
    original = param.detach().clone()
    try:
        with torch.no_grad():
            param[0, 0] += 1.0
        with pytest.raises(RuntimeError, match="do not match their checkpoint"):
            verify_pretrained_weights(qwen_stem, verbose=False)
    finally:
        with torch.no_grad():
            param.copy_(original)


@requires_qwen
def test_real_qwen_per_token_stem_masks_padding(qwen_stem):
    stem = qwen_stem
    stem.eval()
    prompts = ["pick up the cup", "pick up the red cup and place it on the blue plate"]
    with torch.no_grad():
        feat, mask = stem.forward_with_mask(prompts)
        latent = stem.compute_latent(prompts)
        one_token = stem.tokens.repeat(1, 1, 1)
        # reference: the short row's real positions only, same features
        alone = stem.cross_attention(one_token, feat[0][mask[0]].unsqueeze(0))
        unmasked = stem.cross_attention(one_token, feat[:1])

    masked_diff = (latent[0] - alone[0]).abs().max().item()
    unmasked_diff = (unmasked[0] - alone[0]).abs().max().item()
    n_pad = int((~mask[0]).sum())
    print(
        f"\n[qwen-mask] {n_pad} padded positions; max|padded-row - real-tokens-only|: "
        f"with mask {masked_diff:.3e} / without mask (parent) {unmasked_diff:.3e}"
    )
    assert n_pad > 0, "prompts must differ in length for this test to mean anything"
    assert masked_diff < 1e-5, masked_diff
    assert unmasked_diff > 1e-3, unmasked_diff
