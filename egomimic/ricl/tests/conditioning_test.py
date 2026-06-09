"""Tests for egomimic/ricl/conditioning.py (the RICL prefix surgery logic).

Import-light: needs only numpy + torch (no openpi / no egomimic package import),
so it runs on a CPU-only box. Run under pytest, or standalone:

    python egomimic/ricl/tests/conditioning_test.py
"""

from __future__ import annotations

import numpy as np
import torch

try:  # prefer the package import (cluster / pytest)
    from egomimic.ricl import conditioning as cond
except Exception:  # fall back to loading the co-located file directly (CPU box)
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location(
        "ricl_conditioning", pathlib.Path(__file__).with_name("conditioning.py")
    )
    cond = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cond)


def test_discretize_vector_bounds():
    nb = 256
    assert cond.discretize_vector([-1.0], nb) == [0]
    assert cond.discretize_vector([1.0], nb) == [nb - 1]
    mid = cond.discretize_vector([0.0], nb)[0]
    assert nb // 2 - 2 <= mid <= nb // 2 + 1
    # clipping out-of-range
    assert cond.discretize_vector([-5.0, 5.0], nb) == [0, nb - 1]


def test_discretize_matches_pi_binning():
    # Replicate pi.py exactly: edges = linspace(-1,1,nb+1)[:-1]; digitize - 1
    nb = 64
    v = np.array([-0.9, -0.1, 0.0, 0.3, 0.95], dtype=np.float32)
    edges = np.linspace(-1.0, 1.0, nb + 1)[:-1]
    expected = (np.digitize(np.clip(v, -1, 1), edges) - 1).tolist()
    assert cond.discretize_vector(v, nb) == expected


def test_build_block_format_and_count():
    k, Ds, Ha, Da = 3, 4, 5, 4
    states = np.random.default_rng(0).standard_normal((k, Ds)).astype(np.float32)
    actions = np.random.default_rng(1).standard_normal((k, Ha, Da)).astype(np.float32)
    block = cond.build_retrieved_prompt_block(
        states, actions, prompt="lift", action_steps=1
    )
    # one in-context exemplar per demo, each in the query block format
    assert block.count("Task: lift") == k
    assert block.count("State: ") == k
    assert block.count("|") == k  # each demo terminated by '|'
    assert cond.PI05_ANCHOR in block  # ';\nAction: ' separates state from action


def test_build_block_valid_mask():
    k, Ds, Da = 4, 3, 3
    states = np.zeros((k, Ds), np.float32)
    actions = np.zeros((k, Da), np.float32)
    valid = np.array([1, 0, 1, 0])
    block = cond.build_retrieved_prompt_block(states, actions, valid, prompt="t")
    assert block.count("|") == 2  # only the two valid demos rendered
    # all invalid -> empty
    assert cond.build_retrieved_prompt_block(states, actions, np.zeros(k)) == ""


def test_build_block_action_steps_controls_length():
    k, Ds, Ha, Da = 1, 2, 6, 4
    states = np.zeros((k, Ds), np.float32)
    actions = np.random.default_rng(2).standard_normal((k, Ha, Da)).astype(np.float32)
    b1 = cond.build_retrieved_prompt_block(states, actions, action_steps=1)
    b3 = cond.build_retrieved_prompt_block(states, actions, action_steps=3)
    # action bins live between the 'Action: ' anchor and the terminating '|'
    n1 = len(b1.split("Action: ")[1].split("|")[0].split())
    n3 = len(b3.split("Action: ")[1].split("|")[0].split())
    assert n1 == Da and n3 == 3 * Da


def test_splice_into_prompt():
    anchor = cond.PI05_ANCHOR
    base = f"Task: pick up the cup, Embodiment: eva bimanual{anchor}"
    demo = cond.build_retrieved_prompt_block(
        np.zeros((1, 3), np.float32),
        np.zeros((1, 3), np.float32),
        prompt="pick up the cup",
    )
    out = cond.splice_retrieved_into_prompt(base, demo)
    assert out.endswith(anchor), "anchor must remain at the very end (query block)"
    assert out.startswith(
        demo
    ), "demos prepended -> in-context order is demos then query"
    assert base in out
    # empty block -> passthrough
    assert cond.splice_retrieved_into_prompt(base, "") == base


def _query_images(B=2, H=224, W=224, device="cpu"):
    keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    images = {k: torch.rand(B, 3, H, W, device=device) * 2 - 1 for k in keys}
    masks = {k: torch.ones(B, dtype=torch.bool, device=device) for k in keys}
    return images, masks


def test_augment_images_appends_k_keys():
    B, k = 2, 4
    images, masks = _query_images(B)
    n0 = len(images)
    retrieved = torch.randint(
        0, 256, (B, k, 224, 224, 3), dtype=torch.uint8
    )  # BHWC uint8
    rmask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    cond.augment_images_with_retrieved(images, masks, retrieved, rmask)
    assert len(images) == n0 + k
    for i in range(k):
        key = f"retrieved_{i}_base_0_rgb"
        assert key in images and key in masks
        assert images[key].shape == (B, 3, 224, 224)
        assert images[key].dtype == torch.float32
        assert (
            float(images[key].min()) >= -1.0001 and float(images[key].max()) <= 1.0001
        )
    # mask honored
    assert masks["retrieved_3_base_0_rgb"].tolist() == [False, False]
    # query keys untouched
    assert "base_0_rgb" in images and images["base_0_rgb"].shape == (B, 3, 224, 224)


def test_mock_model_consumes_and_grads_through_retrieved():
    """Mimic embed_prefix: iterate ALL images, mask+pool, run a linear, backprop.

    Confirms (a) the model sees 3 query + k retrieved images, (b) gradients flow
    to model params through the retrieved-image path, (c) loss depends on the
    retrieved images (zeroing them changes the loss).
    """
    torch.manual_seed(0)
    B, k = 2, 4

    def consume(images, image_masks):
        feats = []
        for key, img in images.items():
            m = image_masks[key].float().view(-1, 1, 1, 1)
            feats.append((img * m).mean(dim=(2, 3)))  # (B, C)
        return torch.stack(feats, dim=1)  # (B, n_images, C)

    images, masks = _query_images(B)
    retrieved = torch.rand(B, k, 3, 224, 224) * 2 - 1  # BCHW float
    cond.augment_images_with_retrieved(images, masks, retrieved)
    feats = consume(images, masks)
    assert feats.shape == (B, 3 + k, 3), feats.shape

    model = torch.nn.Linear((3 + k) * 3, 1)
    loss = model(feats.flatten(1)).pow(2).mean()
    loss.backward()
    assert model.weight.grad is not None
    assert torch.linalg.norm(model.weight.grad) > 0

    # dependency on retrieved content: zero retrieved vs random retrieved -> different loss
    images_z, masks_z = _query_images(B)
    cond.augment_images_with_retrieved(
        images_z, masks_z, torch.zeros(B, k, 3, 224, 224)
    )
    with torch.no_grad():
        loss_rand = model(consume(images, masks).flatten(1)).pow(2).mean()
        loss_zero = model(consume(images_z, masks_z).flatten(1)).pow(2).mean()
    assert abs(float(loss_rand) - float(loss_zero)) > 1e-8


def test_estimate_prompt_tokens_monotonic():
    a = cond.estimate_prompt_tokens(0, 1)
    b = cond.estimate_prompt_tokens(4, 1)
    c = cond.estimate_prompt_tokens(4, 4)
    assert a < b < c


def _run_all():
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    print(f"== conditioning_test: running {len(fns)} tests ==")
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print("ALL CONDITIONING TESTS PASSED")


if __name__ == "__main__":
    _run_all()
