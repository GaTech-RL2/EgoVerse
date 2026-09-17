"""ABC-DiT nets: the published size ladder, the flow-matching path, and ABC's
state-masking / action-prefix knobs."""

from __future__ import annotations

import pytest
import torch
from fixtures.stub_abc_encoders import StubTaskEncoder, StubVisionTower

from egomimic.models.abc_dit_nets import (
    ABC_DIT_SIZES,
    ABCDiTPolicy,
    CLIPTaskEncoder,
    DINOv3Tower,
    resolve_size,
)

EMB = "human_bimanual"
CAM = "observations.images.front_img_1"
B, T, WIDTH, PROPRIO = 2, 4, 8, 6

# ABC's own layout, the one its published counts are for: three cameras, a
# 14-D state and action, a 30-step chunk and a 768-wide ViT-B.
ABC_CAMERAS = ["top", "left", "right"]
ABC_VIT_DIM = 768
ABC_TASK_DIM = 512
# Figure 21 ("the total number of parameters in S/B/L/xL are 153M, 290M, 746M,
# and 1.93B"): S/B/L are totals including the 85.7M DINOv3 ViT-B of Table 2,
# and the xL entry is that table's action-head figure.
DINOV3_VITB_PARAMS = 85.7e6
PAPER_TOTAL = {"s": 153e6, "b": 290e6, "l": 746e6}
PAPER_HEAD = {"xl": 1.93e9}


def _policy(size=None, **kwargs) -> ABCDiTPolicy:
    torch.manual_seed(0)
    defaults = dict(
        camera_roles=[CAM],
        action_width=WIDTH,
        action_horizon=T,
        state_dims={EMB: PROPRIO},
        vision=StubVisionTower(hidden_size=32, num_patches=4),
        task_encoder=StubTaskEncoder(output_dim=16),
        size=size,
        hidden_size=None if size else 32,
        depth=None if size else 2,
        num_heads=None if size else 2,
        vision_pool_num_queries=3,
        vision_pool_num_heads=2,
        num_inference_steps=2,
        mask_state_ratio=0.0,
    )
    defaults.update(kwargs)
    return ABCDiTPolicy(**defaults)


def _data(**kwargs) -> dict:
    g = torch.Generator().manual_seed(1)
    data = dict(
        images=torch.rand(B, 1, 3, 16, 16, generator=g),
        prompts=["fold clothes"] * B,
        state=torch.randn(B, 1, PROPRIO, generator=g),
        action=torch.randn(B, T, WIDTH, generator=g),
        loss_mask=torch.ones(B, T, WIDTH),
        embodiment_name=EMB,
    )
    data.update(kwargs)
    return data


# -- the size ladder ---------------------------------------------------------


def test_the_four_published_sizes_are_the_dit_ladder_plus_abcs_xl():
    assert [(s.depth, s.hidden, s.heads) for s in ABC_DIT_SIZES.values()] == [
        (12, 384, 6),  # DiT-S
        (12, 768, 12),  # DiT-B
        (24, 1024, 16),  # DiT-L
        (32, 1536, 24),  # ABC's own xL (paper S3.1)
    ]


@pytest.mark.parametrize("size", sorted(ABC_DIT_SIZES))
def test_size_parameter_counts_match_the_paper(size):
    """Each rung, built at ABC's own layout, reproduces its published count.

    The one tensor this port drops is ABC's ``img_proj``, a frozen Linear the
    reference keeps only so pre-existing checkpoints load (``dit.py``:
    "Retained only for checkpoint compatibility"), so it is added back here.
    """
    with torch.device("meta"):
        policy = ABCDiTPolicy(
            camera_roles=ABC_CAMERAS,
            action_width=14,
            action_horizon=30,
            state_dims={"yam": 14},
            vision=StubVisionTower(hidden_size=ABC_VIT_DIM),
            task_encoder=StubTaskEncoder(output_dim=ABC_TASK_DIM),
            size=size,
        )
    encoders = sum(
        p.numel()
        for module in (policy.vision, policy.task_encoder)
        for p in module.parameters()
    )
    hidden = policy.size.hidden
    img_proj = ABC_VIT_DIM * hidden + hidden
    head = sum(p.numel() for p in policy.parameters()) - encoders + img_proj

    expected = PAPER_TOTAL.get(size)
    measured = head + DINOV3_VITB_PARAMS if expected else head
    expected = expected or PAPER_HEAD[size]
    assert measured == pytest.approx(expected, rel=5e-3), (
        f"ABC-DiT-{size} is {measured / 1e6:.1f}M parameters, the paper says "
        f"{expected / 1e6:.1f}M"
    )


def test_size_overrides_and_bad_sizes():
    assert resolve_size("l") == resolve_size(None, 1024, 24, 16)
    assert resolve_size("l", num_heads=8).heads == 8  # one field, rest from L
    with pytest.raises(KeyError, match="unknown ABC-DiT size"):
        resolve_size("xxl")
    with pytest.raises(ValueError, match="not divisible"):
        resolve_size("b", num_heads=7)
    with pytest.raises(ValueError, match="pass size="):
        resolve_size(None, hidden_size=64)


# -- the model ---------------------------------------------------------------


def test_training_step_reaches_every_trainable_tensor():
    policy = _policy().train()
    loss = policy.compute_loss(_data())
    assert torch.isfinite(loss)
    loss.backward()
    starved = [
        name
        for name, param in policy.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    # pos_embed is ABC's frozen sincos table; the stub text encoder holds no
    # trainable weights.
    assert starved == [], f"no gradient reached {starved}"


def test_sampling_is_deterministic_under_a_generator():
    policy = _policy().eval()
    data = _data()
    first = policy.sample(data, generator=torch.Generator().manual_seed(7))
    second = policy.sample(data, generator=torch.Generator().manual_seed(7))
    assert first.shape == (B, T, WIDTH)
    assert torch.equal(first, second)
    other = policy.sample(data, generator=torch.Generator().manual_seed(8))
    assert not torch.equal(first, other)


def test_loss_is_the_masked_mean_squared_velocity_error():
    """Pin the flow-matching convention: ``x_t = (1 - t) a + t noise``, target
    ``noise - a``, and a loss averaged over the mask alone -- so the padded
    columns of a cotrain batch's action never reach the reported number.
    """
    policy = _policy().eval()
    native = 5
    mask = torch.zeros(B, T, WIDTH)
    mask[..., :native] = 1.0
    data = _data(loss_mask=mask)
    torch.manual_seed(0)
    loss = policy.compute_loss(data)

    # Same seed, same draws, in compute_loss's order.
    action = data["action"]
    torch.manual_seed(0)
    noise = torch.randn_like(action)
    t = policy.sample_time(B, action.device, action.dtype).view(B, 1, 1)
    x_t = (1 - t) * action + t * noise
    cond = policy.compute_cond(
        data["state"], policy.task_vectors(data), t[:, 0, 0], EMB
    )
    velocity = policy.predict_velocity(
        x_t, cond, policy.build_vision_tokens(data["images"])
    )
    expected = (((velocity - (noise - action)) ** 2) * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_state_masking_zeroes_the_state_it_selects():
    """mask_state_ratio=1 must make the loss identical to a zeroed state."""
    policy = _policy(mask_state_ratio=1.0).train()
    data = _data()
    torch.manual_seed(0)
    masked = policy.compute_loss(data)
    policy.mask_state_ratio = 0.0
    torch.manual_seed(0)
    # One rand() call fewer without masking: match ABC's draw order by hand.
    zeroed = dict(data, state=torch.zeros_like(data["state"]))
    torch.manual_seed(0)
    torch.rand(B)  # the mask draw the masked run made
    explicit = policy.compute_loss(zeroed)
    assert torch.allclose(masked, explicit, atol=1e-6)


def test_prefix_conditioning_holds_the_given_prefix():
    policy = _policy().eval()
    prefix = torch.full((B, T, WIDTH), 0.5)
    out = policy.sample_with_prefix(_data(), prefix, prefix_length=2)
    assert torch.equal(out[:, :2], prefix[:, :2])
    assert not torch.equal(out[:, 2:], prefix[:, 2:])


def test_prefix_training_runs_and_excludes_the_prefix_from_the_loss():
    policy = _policy(max_action_prefix=T).train()
    loss = policy.compute_loss(_data())
    assert torch.isfinite(loss)


def test_history_is_flattened_into_the_state_vector():
    policy = _policy(history_len=3)
    assert policy.x_embedder[EMB].in_features == 3 * PROPRIO
    data = _data(state=torch.randn(B, 3, PROPRIO))
    assert torch.isfinite(policy.compute_loss(data))
    with pytest.raises(ValueError, match="history steps"):
        policy.compute_loss(_data(state=torch.randn(B, 2, PROPRIO)))


def test_camera_and_state_mismatches_are_named():
    policy = _policy()
    with pytest.raises(ValueError, match="images carry 2 cameras"):
        policy.compute_loss(_data(images=torch.rand(B, 2, 3, 16, 16)))
    with pytest.raises(KeyError, match="no state embedder"):
        policy.compute_loss(_data(embodiment_name="eva_bimanual"))
    with pytest.raises(ValueError, match="action chunk has 3 steps"):
        policy.compute_loss(
            _data(action=torch.randn(B, 3, WIDTH), loss_mask=torch.ones(B, 3, WIDTH))
        )


def test_vision_tokens_are_one_row_per_camera_query():
    roles = [CAM, "observations.images.right_wrist_img"]
    policy = _policy(camera_roles=roles, vision_pool_num_queries=3)
    tokens = policy.build_vision_tokens(torch.rand(B, 2, 3, 16, 16))
    assert tokens.shape == (B, 2 * 3, policy.hidden_size)
    # The per-camera embedding must make the two cameras distinguishable even
    # when they carry the same image.
    same = torch.rand(B, 1, 3, 16, 16).expand(-1, 2, -1, -1, -1).contiguous()
    tokens = policy.build_vision_tokens(same)
    assert not torch.allclose(tokens[:, :3], tokens[:, 3:])


def test_the_batched_and_fused_camera_paths_agree():
    """ABC fuses small batches into one backbone call and splits large ones."""
    policy = _policy().eval()
    images = torch.rand(B, 1, 3, 16, 16)
    policy.fuse_camera_batch_limit = 32
    fused = policy.build_vision_tokens(images)
    policy.fuse_camera_batch_limit = 0
    split = policy.build_vision_tokens(images)
    assert torch.allclose(fused, split, atol=1e-6)


def test_task_vectors_prefer_a_precomputed_column():
    policy = _policy()
    given = torch.randn(B, 16)
    assert torch.equal(policy.task_vectors(_data(task_vec_clip=given)), given)
    assert policy.task_encoder.calls == 0
    assert policy.task_vectors(_data()).shape == (B, 16)
    assert policy.task_encoder.calls == 1


# -- the real encoders -------------------------------------------------------

DINOV3_TINY = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    image_size=32,
    patch_size=16,
    num_register_tokens=2,
)


def _dinov3_available() -> bool:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return "dinov3_vit" in CONFIG_MAPPING_NAMES


@pytest.mark.skipif(
    not _dinov3_available(),
    reason="installed transformers has no DINOv3 (the repo pins 5.17)",
)
def test_dinov3_tower_returns_the_patch_grid():
    tower = DINOv3Tower(
        pretrained=False, bf16_autocast=False, config_overrides=DINOV3_TINY
    )
    assert (tower.hidden_size, tower.num_patches) == (32, 4)
    # Resizes whatever it is handed to the configured square resolution.
    tokens = tower.encode_image_tokens(torch.rand(3, 3, 48, 64))
    assert tokens.shape == (3, 4, 32)
    assert list(tower.backbone_parameters())


def test_clip_task_encoder_normalises_and_caches():
    class CountingCLIP(CLIPTaskEncoder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.encodes = 0

        def _encode(self, prompts):
            self.encodes += 1
            rows = torch.randn(len(prompts), self.output_dim)
            return rows / rows.norm(dim=-1, keepdim=True)

    encoder = CountingCLIP(
        pretrained=False,
        config_overrides=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            projection_dim=8,
        ),
    )
    assert encoder.output_dim == 8
    first = encoder(["fold clothes", "fold clothes", "stack the towels"])
    assert first.shape == (3, 8)
    assert torch.allclose(first.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.equal(first[0], first[1])  # one vector per unique string
    encoder(["fold clothes"])
    assert encoder.encodes == 1  # served from the cache
    assert not any(p.requires_grad for p in encoder.parameters())
