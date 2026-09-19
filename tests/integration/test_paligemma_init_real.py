"""The real PaliGemma release, loaded into the real openpi prefix.

The unit tier proves the remapping and the coverage rules on a miniature module
tree; this is the part that can only be checked against the actual 3B
checkpoint -- that openpi's prefix and `paligemma-3b-pt-224` line up tensor for
tensor, that the values land unchanged, and that nothing outside the prefix
moves. It builds a 3.6B model and reads ~12 GB of weights, so it lives here.

  # the load itself: CPU, needs the checkpoint (or Hub access to fetch it)
  python -m pytest --integration tests/integration/test_paligemma_init_real.py -q
  # plus the flow-matching step, on a GPU node
  python -m pytest --integration -m gpu tests/integration/test_paligemma_init_real.py -q
"""

from __future__ import annotations

import glob

import pytest
import torch
from fixtures.train_harness import pi_unavailable
from safetensors import safe_open

from egomimic.models.paligemma_init import (
    CANONICAL_REPO,
    load_paligemma_weights,
    resolve_paligemma_dir,
)

pytestmark = pytest.mark.skipif(
    pi_unavailable() is not None, reason=pi_unavailable() or ""
)

PG = "paligemma_with_expert.paligemma."
EX = "paligemma_with_expert.gemma_expert."

# These start random and must still be random afterwards: the whole point of
# this init is that pi0.5's own parameters are NOT pretrained.
FROM_SCRATCH = (
    EX + "model.layers.0.self_attn.q_proj.weight",
    EX + "model.layers.17.mlp.up_proj.weight",
    EX + "model.layers.0.input_layernorm.dense.weight",
    EX + "model.norm.dense.weight",
    "action_in_proj.weight",
    "action_out_proj.weight",
    "time_mlp_in.weight",
    "time_mlp_out.weight",
)

# module parameter -> the checkpoint key it must come from, one per remapping rule
PRETRAINED = {
    PG
    + "model.language_model.layers.0.self_attn.q_proj.weight": "language_model.model.layers.0.self_attn.q_proj.weight",
    PG
    + "model.language_model.layers.17.mlp.down_proj.weight": "language_model.model.layers.17.mlp.down_proj.weight",
    PG + "model.language_model.norm.weight": "language_model.model.norm.weight",
    PG
    + "model.language_model.embed_tokens.weight": "language_model.model.embed_tokens.weight",
    PG
    + "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight": "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
    PG
    + "model.vision_tower.encoder.layers.26.mlp.fc2.bias": "vision_tower.vision_model.encoder.layers.26.mlp.fc2.bias",
    PG
    + "model.vision_tower.embeddings.patch_embedding.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
    PG
    + "model.multi_modal_projector.linear.weight": "multi_modal_projector.linear.weight",
}


@pytest.fixture(scope="module")
def checkpoint_dir() -> str:
    try:
        return resolve_paligemma_dir(CANONICAL_REPO)
    except Exception as exc:  # gated and offline, or no verified mirror reachable
        pytest.skip(f"cannot resolve {CANONICAL_REPO}: {type(exc).__name__}: {exc}")


def _flat(named) -> dict:
    """Parameters under their transformers-5 names, which the tables above use:
    below 5 the SigLIP tree keeps a ``vision_model.`` hop."""
    return {
        k.replace("vision_tower.vision_model.", "vision_tower."): v for k, v in named
    }


@pytest.fixture(scope="module")
def loaded(checkpoint_dir):
    """The real pi0.5 chassis with only its VLM prefix initialized."""
    import openpi.models.pi0_config
    import openpi.models_pytorch.pi0_pytorch as pi0t

    # Production reaches openpi only through `egomimic.algo.pi`, which applies
    # the transformers-5 shims at import. This builds PI0Pytorch directly, so
    # take the same entry point rather than an unpatched openpi.
    import egomimic.algo.pi  # noqa: F401

    config = openpi.models.pi0_config.Pi0Config(
        dtype="bfloat16",
        action_dim=32,
        action_horizon=100,
        max_token_len=180,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )
    torch.manual_seed(0)
    model = pi0t.PI0Pytorch(config)
    before = _flat((k, v.detach().clone()) for k, v in model.named_parameters())
    load_paligemma_weights(model, checkpoint_dir)
    return model, before, _flat(model.named_parameters())


def test_pi05s_own_parameters_are_left_at_their_init(loaded):
    _, before, after = loaded
    for name in FROM_SCRATCH:
        assert torch.equal(after[name], before[name]), f"{name} was overwritten"


def test_the_prefix_moved_off_its_init(loaded):
    _, before, after = loaded
    for name in PRETRAINED:
        assert not torch.equal(after[name], before[name]), f"{name} was not loaded"


def test_the_prefix_matches_the_checkpoint_bit_for_bit(loaded, checkpoint_dir):
    _, _, after = loaded
    wanted = set(PRETRAINED.values())
    found = {}
    for shard in sorted(glob.glob(checkpoint_dir + "/*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
                if key in wanted:
                    found[key] = handle.get_tensor(key)
    assert set(found) == wanted, f"checkpoint is missing {wanted - set(found)}"

    for name, key in PRETRAINED.items():
        param = after[name]
        reference = found[key]
        if reference.shape != param.shape:  # padded vocabulary
            reference = reference[: param.shape[0]]
        assert torch.equal(
            param, reference.to(param.dtype)
        ), f"value mismatch at {name}"


def test_openpis_mixed_precision_policy_survives_the_load(loaded):
    # The norms openpi keeps in fp32 must not be dragged to bf16 by the copy,
    # and the matmul weights must not be widened.
    _, _, after = loaded
    assert (
        after[PG + "model.language_model.layers.0.input_layernorm.weight"].dtype
        is torch.float32
    )
    assert after[PG + "model.language_model.norm.weight"].dtype is torch.float32
    assert (
        after[PG + "model.language_model.layers.0.self_attn.q_proj.weight"].dtype
        is torch.bfloat16
    )
    assert (
        after[PG + "model.vision_tower.encoder.layers.0.mlp.fc1.weight"].dtype
        is torch.bfloat16
    )
    # openpi's keep-list, on either transformers (openpi_compat on 5.x)
    for name in (
        "patch_embedding.weight",
        "patch_embedding.bias",
        "position_embedding.weight",
    ):
        assert (
            after[PG + "model.vision_tower.embeddings." + name].dtype is torch.float32
        )


# ------------------------------------------------------------------ a real step


def _observation(batch: int, device, max_token_len: int):
    """A minimal openpi observation: three camera slots, the wrists masked out
    the way human data leaves them, and a prompt of real token ids."""
    from egomimic.models.preprocess_pi_obs import PI_CAMERA_SLOTS, _SimpleObservation

    image = torch.rand(batch, 3, 224, 224, device=device) * 2 - 1
    present = {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    mask = torch.ones(batch, max_token_len, dtype=torch.bool, device=device)
    return _SimpleObservation(
        images={slot: image.clone() for slot in PI_CAMERA_SLOTS},
        image_masks={
            slot: torch.full((batch,), present[slot], dtype=torch.bool, device=device)
            for slot in PI_CAMERA_SLOTS
        },
        state=torch.zeros(batch, 32, device=device),
        tokenized_prompt=torch.randint(
            0, 1000, (batch, max_token_len), device=device, dtype=torch.long
        ),
        tokenized_prompt_mask=mask,
        token_ar_mask=mask.clone(),
        token_loss_mask=mask.clone(),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_one_flow_matching_step_trains_both_halves(loaded):
    """The point of the arm: a finite loss whose gradient reaches the frozen-in
    PaliGemma prefix AND the action expert that started from noise. A load that
    left half the model detached would still produce a loss here, so check the
    gradients, not just the number."""
    model, _, _ = loaded
    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    observation = _observation(1, device, model.config.max_token_len)
    actions = torch.randn(
        1, model.config.action_horizon, model.config.action_dim, device=device
    )

    torch.manual_seed(0)
    losses = model.forward(observation, actions)
    loss = (
        torch.stack(losses).mean()
        if isinstance(losses, (list, tuple))
        else losses.mean()
    )
    assert torch.isfinite(loss), f"non-finite flow-matching loss: {loss}"

    loss.backward()
    probes = {
        "prefix (pretrained)": model.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj.weight,
        "vision tower (pretrained)": model.paligemma_with_expert.paligemma.model.vision_tower.encoder.layers[
            0
        ].self_attn.q_proj.weight,
        "action expert (from scratch)": model.paligemma_with_expert.gemma_expert.model.layers[
            0
        ].self_attn.q_proj.weight,
        "action_out_proj (from scratch)": model.action_out_proj.weight,
    }
    for label, param in probes.items():
        assert param.grad is not None, f"no gradient reached {label}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient at {label}"
        assert param.grad.abs().sum() > 0, f"zero gradient at {label}"

    # and a step actually moves them
    before = {k: v.detach().clone() for k, v in probes.items()}
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    for label, param in probes.items():
        assert not torch.equal(param, before[label]), f"{label} did not move"
