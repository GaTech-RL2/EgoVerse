"""Lane C step 2: the frozen-text-feature cache on the Qwen stems.

The annotation vocabulary is tiny compared with the number of training steps,
so a frozen text encoder recomputes the same ~600M-parameter forward over and
over. ``_Qwen3BaseEncoder`` now keeps a lazy in-memory cache of the per-token
hidden states, keyed by the exact prompt string, and re-assembles the batch
(left-padded, exactly like the tokenizer) from the cached rows.

The cache is only legal when the encoder is frozen; a trainable encoder must
stay online (its outputs change every step and need a graph).
"""

from __future__ import annotations

import os
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from fixtures.text_stems import EMBED_DIM, QWEN_SNAPSHOT, requires_qwen
from fixtures.text_stems import cross_attn_specs as _cross_attn_specs

from egomimic.models.hpt_nets import PolicyStem, QwenPerTokenEncoder

HIDDEN_DIM = 16


# --------------------------------------------------------------------------
# a stub encoder: deterministic per-prompt features, counts its forwards
# --------------------------------------------------------------------------

PROMPT_LENGTHS = {"fold the towel": 2, "pick up the cup": 3, "wipe the table": 5}


class CountingStubEncoder(QwenPerTokenEncoder):
    """``QwenPerTokenEncoder`` with the tokenizer + HF model replaced by a fixed
    prompt -> per-token-features table, so the cache can be tested without
    loading Qwen. ``_encode_online`` is what the cache is supposed to avoid;
    every call is counted, along with how many prompts it saw."""

    def __init__(
        self,
        specs,
        freeze: bool = True,
        max_entries: int | None = None,
    ) -> None:
        PolicyStem.__init__(self, specs=specs)
        self._text_feature_cache = OrderedDict()
        if max_entries is not None:
            self.TEXT_CACHE_MAX_ENTRIES = max_entries
        self.hidden_size = HIDDEN_DIM
        self.output_dim = EMBED_DIM
        self.freeze_encoder = freeze
        self.proj = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.encoder = nn.Identity()  # never called; `train()` keeps it in eval

        generator = torch.Generator().manual_seed(0)
        self._table = {
            prompt: torch.randn(length, HIDDEN_DIM, generator=generator)
            for prompt, length in PROMPT_LENGTHS.items()
        }
        self.calls = []  # one entry per _encode_online call: the prompt list

    def _encode_online(self, prompts):
        self.calls.append(list(prompts))
        rows = [self._table[p] for p in prompts]
        longest = max(row.shape[0] for row in rows)
        hidden = torch.zeros(len(rows), longest, HIDDEN_DIM)
        mask = torch.zeros(len(rows), longest, dtype=torch.long)
        for i, row in enumerate(rows):  # left padding, as padding_side="left"
            hidden[i, longest - row.shape[0] :] = row
            mask[i, longest - row.shape[0] :] = 1
        return hidden, mask


def _stub(**kwargs) -> CountingStubEncoder:
    specs = _cross_attn_specs()
    stem = CountingStubEncoder(specs, **kwargs)
    stem.init_cross_attn(specs.cross_attn)
    stem.eval()
    return stem


A, B, C = list(PROMPT_LENGTHS)


# --------------------------------------------------------------------------
# 1. misses are encoded once, in one sub-batch of unique strings
# --------------------------------------------------------------------------


def test_cache_encodes_each_unique_prompt_exactly_once():
    stem = _stub()
    with torch.no_grad():
        stem._encode([A, B])
        stem._encode([B, C, A])
        stem._encode([C])
        stem._encode([A, A, B, C])

    # one call per batch that had misses, each carrying only the unique misses
    assert stem.calls == [[A, B], [C]], stem.calls
    assert len(stem._text_feature_cache) == 3


def test_cached_output_equals_the_uncached_output():
    """Bit-identical, since the stub features do not depend on the batch."""
    batches = [[A, B], [B, C, A], [C], [A, A, C]]
    cached, online = _stub(), _stub(freeze=False)  # trainable: always online
    online.load_state_dict(cached.state_dict())  # same proj / cross-attn weights
    with torch.no_grad():
        for prompts in batches:
            hid_c, mask_c = cached._encode(prompts)
            hid_o, mask_o = online._encode(prompts)
            assert hid_c.shape == hid_o.shape
            assert torch.equal(mask_c.bool(), mask_o.bool())
            assert torch.equal(hid_c, hid_o), prompts
            # and through the whole stem, cross-attention included
            assert torch.equal(
                cached.compute_latent(prompts), online.compute_latent(prompts)
            )
    # the uncached stem re-encodes every batch (twice here: _encode +
    # compute_latent), the cached one only ever saw the misses
    assert online.calls == [b for b in batches for _ in range(2)]
    assert cached.calls == [[A, B], [C]]


# --------------------------------------------------------------------------
# 2. the re-assembled batch keeps the tokenizer's left padding
# --------------------------------------------------------------------------


def test_reassembled_batch_is_left_padded():
    stem = _stub()
    prompts = [C, A, B]  # lengths 5, 2, 3
    with torch.no_grad():
        stem._encode([A])  # warm one entry so the batch mixes hits and misses
        hidden, mask = stem._encode(prompts)

    lengths = [PROMPT_LENGTHS[p] for p in prompts]
    longest = max(lengths)
    assert hidden.shape == (len(prompts), longest, HIDDEN_DIM)
    assert mask.shape == (len(prompts), longest)
    for i, length in enumerate(lengths):
        pad = longest - length
        assert mask[i, :pad].sum() == 0, "padding must be on the LEFT"
        assert mask[i, pad:].min() == 1
        assert torch.equal(hidden[i, pad:], stem._table[prompts[i]])
        assert torch.equal(hidden[i, :pad], torch.zeros(pad, HIDDEN_DIM))


def test_cached_tensors_are_detached():
    stem = _stub(freeze=True)
    with torch.no_grad():
        hidden, _ = stem._encode([A, B])
    assert not hidden.requires_grad
    for entry in stem._text_feature_cache.values():
        assert not entry.requires_grad
        assert entry.grad_fn is None


# --------------------------------------------------------------------------
# 3. bypasses
# --------------------------------------------------------------------------


def test_trainable_encoder_bypasses_the_cache():
    stem = _stub(freeze=False)
    stem._encode([A, B])
    stem._encode([A, B])
    assert stem.calls == [[A, B], [A, B]]
    assert len(stem._text_feature_cache) == 0


def test_moving_the_stem_clears_the_cache():
    """The cached rows belong to the old device / dtype."""
    stem = _stub()
    with torch.no_grad():
        stem._encode([A, B])
        stem.float()
        assert len(stem._text_feature_cache) == 0
        stem._encode([A, B])
    assert stem.calls == [[A, B], [A, B]]


# --------------------------------------------------------------------------
# 4. LRU eviction
# --------------------------------------------------------------------------


def test_lru_eviction_at_max_entries():
    stem = _stub(max_entries=2)
    with torch.no_grad():
        stem._encode([A, B])  # cache: A, B
        stem._encode([A])  # hit; A becomes most-recent, B is the LRU victim
        stem._encode([C])  # evicts B
        assert set(stem._text_feature_cache) == {A, C}
        stem._encode([A, C])  # both still cached: no new forward
        assert stem.calls == [[A, B], [C]]
        stem._encode([B])  # B was evicted -> re-encoded
    assert stem.calls == [[A, B], [C], [B]]
    assert len(stem._text_feature_cache) == 2


# --------------------------------------------------------------------------
# 5. the real Qwen encoder (skipped when the snapshot is absent)
# --------------------------------------------------------------------------


QWEN_PROMPTS = [
    "fold the towel",
    "pick up the red cup and place it on the blue plate",
    "wipe",
    "open the drawer and take out the blue sponge, then close the drawer",
]


@pytest.fixture(scope="module")
def qwen_stems():
    """Two stems off the same snapshot: one cached (frozen), one always online
    (trainable; the tests run it under no_grad in eval mode)."""
    if QWEN_SNAPSHOT is None:
        pytest.skip("Qwen3-Embedding-0.6B snapshot not in the HF cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    specs = _cross_attn_specs()
    built = []
    for freeze in (True, False):
        stem = QwenPerTokenEncoder(
            model_name=QWEN_SNAPSHOT,
            max_length=32,
            output_dim=EMBED_DIM,
            freeze=freeze,
            specs=specs,
        )
        stem.init_cross_attn(specs.cross_attn)
        stem.eval()
        built.append(stem)
    # identical cross-attention / projection weights on both stems
    built[1].load_state_dict(built[0].state_dict())
    return tuple(built)


@requires_qwen
def test_real_qwen_cached_matches_online(qwen_stems):
    cached, online = qwen_stems
    forwards = []
    original = type(cached)._encode_online

    def counting(self, prompts):
        forwards.append(list(prompts))
        return original(self, prompts)

    with torch.no_grad():
        hid_on, mask_on = online._encode(QWEN_PROMPTS)
        lat_on = online.compute_latent(QWEN_PROMPTS)

        # Warm one prompt on its own first, so the batch below is a mix of a
        # hit (encoded with NO padding) and misses (encoded padded to each
        # other) - the case where cached and online padding really differ.
        cached._text_feature_cache.clear()
        cached._encode([QWEN_PROMPTS[0]])
        cached._encode_online = counting.__get__(cached, type(cached))
        hid_c1, mask_c1 = cached._encode(QWEN_PROMPTS)
        lat_c1 = cached.compute_latent(QWEN_PROMPTS)
        first_calls = list(forwards)
        n_first = len(forwards)
        forwards.clear()
        lat_c2 = cached.compute_latent(QWEN_PROMPTS)  # fully warm

    assert n_first == 1, "the misses must go through the encoder in ONE sub-batch"
    assert first_calls[0] == QWEN_PROMPTS[1:], "only the misses, unique, in order"
    assert forwards == [], "a warm cache must perform zero encoder forwards"
    assert torch.equal(mask_c1.bool(), mask_on.bool())
    assert hid_c1.shape == hid_on.shape
    assert torch.equal(lat_c1, lat_c2), "the cache must be deterministic once warm"

    # Only the REAL token positions are comparable: the cache pads with zeros
    # while the online path leaves the encoder's output for the pad tokens
    # there. Both are masked out downstream (forward_with_mask), so this is the
    # tensor the rest of the stem actually reads.
    real = mask_on.bool()
    hidden_diff = (hid_c1[real].float() - hid_on[real].float()).abs().max().item()
    latent_diff = (lat_c1 - lat_on).abs().max().item()
    print(
        f"\n[qwen-cache] max|cached - online| hidden {hidden_diff:.3e} "
        f"latent {latent_diff:.3e} (entries {len(cached._text_feature_cache)})"
    )
    # The cache encodes a miss with different left padding than the online
    # batch would, and Qwen's RoPE positions are absolute, so fp16 rounding
    # differs. The right yardstick is how much the ONLINE path itself moves
    # under the same change of padding: encode one prompt alone and compare it
    # with its row in the padded batch.
    with torch.no_grad():
        solo = online._encode([QWEN_PROMPTS[1]])[0][0].float()
    control = (solo - hid_on[1][mask_on[1].bool()].float()).abs().max().item()
    print(f"[qwen-cache] online-vs-online padding noise (fp16): {control:.3e}")
    assert hidden_diff <= max(2 * control, 1e-6), (hidden_diff, control)
    # absolute ceilings: hidden states run to |x| ~ 27 here, so fp16's ~1e-3
    # relative precision is worth a few 1e-2 in absolute terms.
    assert hidden_diff < 1e-1, hidden_diff
    assert latent_diff < 5e-3, latent_diff


def test_a_batch_wider_than_the_cache_still_serves_every_prompt():
    """The LRU trim runs after the miss pass, so a batch with more unique
    prompts than the capacity evicts its own earliest entries; reading them
    back must not KeyError."""
    stem = _stub(max_entries=2)
    with torch.no_grad():
        hidden, mask = stem._encode([A, B, C])
        online_hidden, online_mask = stem._encode_online([A, B, C])
    assert hidden.shape[0] == 3 and mask.shape[0] == 3
    assert len(stem._text_feature_cache) <= 2
    for i in range(3):
        real, real_on = mask[i].bool(), online_mask[i].bool()
        torch.testing.assert_close(
            hidden[i][real], online_hidden[i][real_on], atol=0, rtol=0
        )
