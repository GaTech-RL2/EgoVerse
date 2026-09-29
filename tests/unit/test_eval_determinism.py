"""Deterministic evaluation (lane B).

Validating the same checkpoint twice must give the same prompts, the same
sampled actions and the same metrics. ``ModelWrapper.on_validation_start``
saves the training RNG state and seeds every global RNG from the rank;
``on_validation_end`` puts the training state back. So pass N and pass N+1
replay the same draws, ranks differ, and training's stream is untouched.

Plus the gradient-clipping change that rides along in this branch: the MAD
spike detector in ``ModelWrapper.on_after_backward`` is log-only (fixed
clipping is Lightning's ``trainer.gradient_clip_val``).

Everything here is CPU-only and builds the smallest object that still runs the
real code path: the real ``FMPolicy`` Euler sampler, the real
``HPTModel.forward`` head dispatch and the real ``HPT.forward_eval``, with only
the stems/trunk (``forward_features``) and the norm stats stubbed out.
"""

from __future__ import annotations

import random
import types
from collections import deque

import numpy as np
import pytest
import torch
import torch.nn as nn

from egomimic.algo.hpt import HPT, HPTModel
from egomimic.models.fm_policy import FMPolicy
from egomimic.pl_utils.pl_model import ModelWrapper

DOMAIN = "eva_bimanual"
EMB_ID = 6  # EMBODIMENT.EVA_BIMANUAL; get_embodiment(6).lower() == DOMAIN
AC_KEY = "actions_cartesian"
ANNOTATION_KEY = "annotations"
BATCH = 4
HORIZON = 3
ACTION_DIM = 4
FEAT_DIM = 5


# ---------------------------------------------------------------------------
# Fixtures: a minimal but real HPT
# ---------------------------------------------------------------------------
class _TinyVelocity(nn.Module):
    """Stands in for ConditionalUnet1D: v(x_t, t, cond), same call signature."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x_t, t, global_cond):
        return self.lin(x_t) * t.reshape(-1, 1, 1)


class _StubNormStats:
    def unnormalize(self, predictions, embodiment_id):
        return dict(predictions)


def _make_algo(annotation_key: str | None = ANNOTATION_KEY) -> HPT:
    """A real HPT wired to a real FMPolicy head, without stems/trunk/dataset.

    ``HPT.__init__`` needs norm stats, configs and a dataset; every attribute
    ``forward_eval`` / ``_build_prompts`` actually read is set here instead.
    """
    torch.manual_seed(0)
    head = FMPolicy(
        model=_TinyVelocity(ACTION_DIM),
        action_horizon=HORIZON,
        infer_ac_dims={DOMAIN: ACTION_DIM},
        num_inference_steps=2,
    )
    model = HPTModel.__new__(HPTModel)
    nn.Module.__init__(model)
    model.diffusion = True
    model.shared_action = False
    model.auxiliary_ac_keys = {}
    model.heads = nn.ModuleDict({DOMAIN: head})
    model.device = torch.device("cpu")
    features = (
        torch.arange(BATCH * FEAT_DIM, dtype=torch.float32).reshape(BATCH, FEAT_DIM)
        / 10.0
    )
    model.forward_features = lambda domain, data: (features, None)

    algo = HPT.__new__(HPT)
    algo.nets = nn.ModuleDict({"policy": model})
    algo.device = torch.device("cpu")
    algo.norm_stats = _StubNormStats()
    algo.camera_keys = {EMB_ID: []}
    algo.proprio_keys = {EMB_ID: []}
    algo.lang_keys = {EMB_ID: []}
    algo.ac_keys = {EMB_ID: AC_KEY}
    algo.auxiliary_ac_keys = {}
    algo.shared_ac_key = None
    algo.is_6dof = False
    algo.freeze_repr = False
    algo.freeze_depth = 8
    algo.annotation_key = annotation_key
    algo.annotation_modality = "annotation"
    algo.annotation_sampling_mode = "random"
    algo.default_prompt = "do the task"
    algo.train_image_augs = None
    algo.eval_image_augs = None
    algo.nets.eval()
    return algo


def _make_batch() -> dict:
    torch.manual_seed(1)
    return {
        EMB_ID: {
            AC_KEY: torch.randn(BATCH, HORIZON, ACTION_DIM),
            "pad_mask": torch.ones(BATCH, HORIZON, 1),
            "embodiment": torch.tensor([EMB_ID]),
        }
    }


def _annotation_batch(n_options: int = 4) -> dict:
    return {
        ANNOTATION_KEY: [
            [f"sample{i}_option{j}" for j in range(n_options)] for i in range(BATCH)
        ]
    }


RAW, BATCH_DATA = _annotation_batch(), _make_batch()


def _val_pass(algo: HPT, rank: int = 0, n_batches: int = 3) -> list[tuple]:
    """One validation pass through ModelWrapper's hooks: per batch, the
    prompts, the sampled actions and the val loss."""
    wrapper = types.SimpleNamespace(
        model=algo,
        device=torch.device("cpu"),
        global_rank=rank,
        _val_heads=lambda: {"valid": None},
    )
    ModelWrapper.on_validation_start(wrapper)
    out = []
    for _ in range(n_batches):
        prompts = algo._build_prompts(RAW, BATCH)
        preds = algo.forward_eval(BATCH_DATA)
        out.append((prompts, preds[f"{DOMAIN}_{AC_KEY}"], preds[f"{DOMAIN}_loss"]))
    ModelWrapper.on_validation_end(wrapper)
    return out


# ---------------------------------------------------------------------------
# 1. Seeded validation passes
# ---------------------------------------------------------------------------
def _train_steps():
    """What training draws between two validation passes."""
    torch.randn(10), random.random(), np.random.rand()


def test_two_validation_passes_are_identical():
    algo = _make_algo()
    first = _val_pass(algo)
    _train_steps()
    second = _val_pass(algo)
    for (p1, a1, l1), (p2, a2, l2) in zip(first, second):
        assert p1 == p2
        assert torch.equal(a1, a2)
        assert torch.equal(l1, l2)
    # ... and not because nothing is random: batches within a pass differ
    assert not torch.equal(first[0][1], first[1][1])
    assert len({p for prompts, _, _ in first for p in prompts}) > BATCH


def test_validation_leaves_the_training_rng_stream_alone():
    def draws():
        return torch.randn(3), random.random(), np.random.rand()

    def seed():
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)

    algo = _make_algo()
    seed()
    expected = draws()
    seed()
    _val_pass(algo)
    got = draws()
    assert torch.equal(got[0], expected[0])
    assert got[1:] == expected[1:]


def test_ranks_draw_different_noise():
    algo = _make_algo()
    assert not torch.equal(_val_pass(algo, rank=0)[0][1], _val_pass(algo, rank=1)[0][1])


# ---------------------------------------------------------------------------
# 2. Prompts
# ---------------------------------------------------------------------------
def test_eval_prompts_fall_back_to_default_on_empty():
    algo = _make_algo()
    raw = {ANNOTATION_KEY: [[], ["a", "b"], [], ["c"]]}
    prompts = algo._build_prompts(raw, 4)
    assert prompts[0] == algo.default_prompt
    assert prompts[2] == algo.default_prompt
    assert prompts[1] in {"a", "b"} and prompts[3] == "c"


def test_missing_annotation_key_uses_default_prompt():
    algo = _make_algo(annotation_key=None)
    assert algo._build_prompts({}, 3) == [algo.default_prompt] * 3


@pytest.mark.parametrize("mode", ["random", "first"])
def test_prompts_follow_the_sampling_mode(mode):
    algo = _make_algo()
    algo.annotation_sampling_mode = mode
    raw = _annotation_batch()

    random.seed(12345)
    got = algo._build_prompts(raw, BATCH)

    random.seed(12345)
    expected = []
    for options in raw[ANNOTATION_KEY]:
        if mode == "random":
            expected.append(options[random.randint(0, len(options) - 1)])
        else:
            expected.append(options[0])
    assert got == expected


# ---------------------------------------------------------------------------
# 3. Log-only MAD gradient detector
# ---------------------------------------------------------------------------
class _GradHookStub:
    """Just enough LightningModule for ModelWrapper.on_after_backward."""

    def __init__(self, param: nn.Parameter, history):
        self.enable_grad_norm = True
        self.grad_norm_mad_scale = ModelWrapper.grad_norm_mad_scale
        self.grad_norm_mad_min_count = ModelWrapper.grad_norm_mad_min_count
        self.grad_norm_mad_window = ModelWrapper.grad_norm_mad_window
        self.grad_norm_history = deque(history, maxlen=self.grad_norm_mad_window)
        self._param = param
        self.global_step = 4242
        self.trainer = types.SimpleNamespace(is_global_zero=True)
        self.logged = {}

    def parameters(self):
        return [self._param]

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def _spiking_stub(spike_norm: float = 50.0):
    param = nn.Parameter(torch.zeros(4))
    grad = torch.full((4,), spike_norm / 2.0)  # ||grad||_2 == spike_norm
    param.grad = grad
    history = [
        0.9 if i % 2 else 1.1 for i in range(ModelWrapper.grad_norm_mad_min_count)
    ]
    return _GradHookStub(param, history), param


def test_on_after_backward_never_modifies_gradients():
    stub, param = _spiking_stub()
    before = param.grad.clone()
    ModelWrapper.on_after_backward(stub)
    assert torch.equal(param.grad, before), "MAD detector rescaled the gradients"


def test_on_after_backward_still_flags_and_logs_the_spike():
    stub, _ = _spiking_stub()
    ModelWrapper.on_after_backward(stub)
    logged = stub.logged
    assert logged["Train/policy_grad_norms_mad_flag"] == 1.0
    assert logged["Train/policy_grad_norms_raw"] == pytest.approx(50.0, rel=1e-5)
    assert "Train/policy_grad_norms_mad_threshold" in logged


def test_flagged_steps_are_appended_to_the_history():
    """Log-only detector: every step goes into the rolling median/MAD window."""
    stub, _ = _spiking_stub()
    n_before = len(stub.grad_norm_history)
    ModelWrapper.on_after_backward(stub)
    assert len(stub.grad_norm_history) == n_before + 1
    assert stub.grad_norm_history[-1] == pytest.approx(50.0, rel=1e-5)
