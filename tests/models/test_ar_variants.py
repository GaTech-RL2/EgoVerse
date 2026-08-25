"""Properties every AR variant must hold, whatever it predicts."""

import pytest
import torch

from egomimic.models.ar import ARVariantConfig, VARIANTS, build_variant


def cfg():
    return ARVariantConfig(image_dim=32, state_dim=6, action_dim=4,
                           d_model=64, n_layers=2, n_heads=4, horizon=10)


def batch(b=2, t=10):
    return (torch.randn(b, t, 32), torch.randn(b, t, 6), torch.randn(b, t, 4))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_action_shape_and_finite_loss(name):
    m = build_variant(name, cfg())
    img, st, ac = batch()
    out = m(img, st, ac)
    assert out["action_pred"].shape == (2, 10, 4)
    assert torch.isfinite(m.losses(out, st, ac)["loss"])


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_is_causal(name):
    """Perturbing the LAST step must not move any earlier prediction.

    This is the property the whole design rests on: if it fails, the model is
    reading the future and every downstream number is meaningless.
    """
    m = build_variant(name, cfg()).eval()
    img, st, ac = batch()
    with torch.no_grad():
        a = m(img, st, ac)["action_pred"]
        img2, st2, ac2 = img.clone(), st.clone(), ac.clone()
        img2[:, -1] += 5.0
        st2[:, -1] += 5.0
        ac2[:, -1] += 5.0
        b = m(img2, st2, ac2)["action_pred"]
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_feedback_flag_is_honoured(name):
    """Scheduled sampling changes the pass, EXCEPT for the teacher-forced control.

    joint_tf exists to isolate the feedback question, so if its output moved
    with `progress` the ablation would be comparing two things at once.
    """
    m = build_variant(name, cfg())
    m.train()
    img, st, ac = batch()
    torch.manual_seed(7)
    a = m(img, st, ac, progress=0.0)["action_pred"]
    torch.manual_seed(7)
    b = m(img, st, ac, progress=1.0)["action_pred"]
    moved = not torch.allclose(a, b, atol=1e-6)
    assert moved == m.allow_feedback


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_closed_loop_rollout(name):
    m = build_variant(name, cfg()).eval()
    img, st, ac = batch()
    r = m.rollout(img, st, ac, steps=5)
    assert r.shape == (2, 5, 4)
    assert torch.isfinite(r).all()


def test_state_idm_sees_no_action_tokens():
    """The IDM variant's claim is that it never leans on action history."""
    m = build_variant("state_idm", cfg())
    assert not m.spec.action
    assert m.uses_idm and m.action_head is None


def test_positional_table_spans_tokens_not_steps():
    """A variant with K tokens per step needs K x horizon positions.

    Sizing the table by TIMESTEPS silently truncates the window for every
    multi-token variant, which shows up as a quiet accuracy loss rather than
    an error.
    """
    c = cfg()
    for name in VARIANTS:
        m = build_variant(name, c)
        assert m.backbone.max_window >= c.horizon * m.spec.per_step
