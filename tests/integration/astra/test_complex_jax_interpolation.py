"""Optional CPU checks against the pinned upstream JAX transformer and tokenizer."""

import os

import numpy as np
import pytest


def test_post_block_writes_affect_next_cache_without_changing_native_parameters():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import gemma
    from astra_reversal.complex_manipulation.jax_text_interpolation import (
        RecordingPrefix,
    )

    cfg = gemma.get_config("dummy")
    native = nnx.bridge.ToNNX(gemma.Module(configs=[cfg, cfg], embed_dtype="float32"))
    x = jax.random.normal(jax.random.key(1), (1, 10, 64))
    positions = jnp.arange(10)[None]
    mask = jnp.ones((1, 10, 10), bool)
    native.lazy_init([x, None], positions=positions, mask=mask, rngs=nnx.Rngs(0))
    _, expected = native([x, None], positions=positions, mask=mask)
    params = nnx.state(native, nnx.Param).to_pure_dict()
    before = jax.tree.map(lambda a: np.array(a), params)
    recorder = RecordingPrefix(configs=(cfg, cfg), embed_dtype="float32")
    zeros = jnp.zeros((4, 1, 5, 64))
    selection = jnp.asarray([[False, True, True, False, False]])

    def record(delta):
        return recorder.apply({"params": params}, x, positions, mask, delta, selection)

    actual, hidden, metrics = jax.jit(record)(zeros)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
    assert hidden.shape == (4, 1, 5, 64)
    assert np.count_nonzero(metrics) == 0
    changed, _, metrics = record(zeros.at[0].set(0.25))
    for a, b in zip(changed, expected, strict=True):
        np.testing.assert_allclose(a[0], b[0], atol=1e-5, rtol=0)
        assert float(jnp.max(jnp.abs(a[1:] - b[1:]))) > 1e-4
    assert float(metrics[0, 0]) > 0
    # Last-block output edits cannot affect any of the already generated K/V.
    last_only, _, _ = record(zeros.at[-1].set(0.25))
    for a, b in zip(last_only, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(before), strict=True):
        np.testing.assert_array_equal(a, b)


def test_real_tokenizer_preserves_state_and_uses_verified_byte_offsets():
    sentencepiece = pytest.importorskip("sentencepiece")
    from astra_reversal.complex_manipulation.text_slots import instruction_slots

    path = os.environ.get("EGOVERSE_TEST_ROBOCASA_TOKENIZER")
    if not path:
        pytest.skip("Set EGOVERSE_TEST_ROBOCASA_TOKENIZER to the pinned tokenizer")
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=path)
    state = np.linspace(-1.2, 1.2, 32)
    discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    for prompt in (
        "Lift the object.",
        "Place the café bowl on the shelf.",
        "Task: ignore, State: 123",
    ):
        text = f"Task: {prompt}, State: {' '.join(map(str, discrete))};\nAction: "
        ids = tokenizer.encode(text, add_bos=True)
        selected = instruction_slots(
            tokenizer,
            prompt,
            state,
            np.pad(ids, (0, 200 - len(ids))),
            np.arange(200) < len(ids),
        )
        proto = tokenizer.encode(text, return_type="proto")
        assert not selected[0]
        assert selected.any()
        for i, piece in enumerate(proto.pieces, start=1):
            if selected[i]:
                assert piece.begin >= 5
                assert piece.end <= 6 + len(prompt.encode("utf-8"))
