"""Deterministic evaluation (lane B).

Validating the same checkpoint twice must give the same prompts, the same
sampled actions and the same metrics. Three pieces make that true:

* ``HPT._build_prompts`` picks annotations with a per-pass seeded RNG whenever
  the algo is in eval mode (``annotation_sampling_mode`` only steers training);
* ``HPT.forward_eval`` builds a ``torch.Generator`` seeded from
  ``EVAL_BASE_SEED + rank * EVAL_RANK_STRIDE + pass_counter`` and threads it
  down to the denoising head's sampler, and forks/seeds the global RNG around
  the BC val-loss call; ``ModelWrapper.on_validation_start`` rewinds the
  counter, so pass N and pass N+1 replay the same draws.

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
from pathlib import Path

import hydra
import pytest
import torch
import torch.nn as nn

import egomimic.hydra_configs
from egomimic.algo.hpt import EVAL_BASE_SEED, HPT, HPTModel
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
    algo.eval_base_seed = EVAL_BASE_SEED
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


def _prompt_pass(algo: HPT, raw_batch: dict, n_batches: int = 3) -> list[list[str]]:
    """One validation pass: N batches, counter advancing as forward_eval does."""
    algo.reset_eval_pass_counter()
    out = []
    for _ in range(n_batches):
        out.append(algo._build_prompts(raw_batch, BATCH))
        algo._eval_pass_counter += 1  # what forward_eval does after each batch
    return out


# ---------------------------------------------------------------------------
# 1. Prompts
# ---------------------------------------------------------------------------
def test_eval_prompts_are_reproducible_across_passes():
    algo = _make_algo()
    raw = _annotation_batch()
    first = _prompt_pass(algo, raw)
    second = _prompt_pass(algo, raw)
    assert first == second


def test_eval_prompts_keep_variety_and_are_not_always_first():
    algo = _make_algo()
    raw = _annotation_batch()
    prompts = [p for batch in _prompt_pass(algo, raw, n_batches=5) for p in batch]
    firsts = [options[0] for options in raw[ANNOTATION_KEY]]
    assert any(
        p not in firsts for p in prompts
    ), f"eval prompts collapsed to sample[0]: {prompts}"
    assert len(set(prompts)) > 1


def test_eval_prompts_ignore_sampling_mode():
    """`annotation_sampling_mode` is an algo-level knob; eval overrides it."""
    algo = _make_algo()
    raw = _annotation_batch()
    algo.annotation_sampling_mode = "random"
    as_random = _prompt_pass(algo, raw)
    algo.annotation_sampling_mode = "first"
    as_first = _prompt_pass(algo, raw)
    assert as_random == as_first
    assert as_first[0] != [options[0] for options in raw[ANNOTATION_KEY]]


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
def test_train_prompt_behaviour_is_unchanged(mode):
    """In train mode the parent's logic still applies verbatim."""
    algo = _make_algo()
    algo.annotation_sampling_mode = mode
    algo.nets.train()
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
# 2. Seeded flow sampling
# ---------------------------------------------------------------------------
def _eval_pass(algo: HPT, batch: dict, n_batches: int = 2) -> list[dict]:
    algo.reset_eval_pass_counter()
    return [algo.forward_eval(batch) for _ in range(n_batches)]


def _actions(preds: list[dict]) -> list[torch.Tensor]:
    return [p[f"{DOMAIN}_{AC_KEY}"] for p in preds]


def test_two_validation_passes_give_bit_identical_actions():
    algo = _make_algo()
    batch = _make_batch()
    first = _eval_pass(algo, batch)
    second = _eval_pass(algo, batch)
    for a, b in zip(_actions(first), _actions(second)):
        assert torch.equal(a, b)


def test_val_loss_is_identical_across_passes():
    algo = _make_algo()
    batch = _make_batch()
    first = _eval_pass(algo, batch)
    second = _eval_pass(algo, batch)
    for a, b in zip(first, second):
        assert torch.equal(a[f"{DOMAIN}_loss"], b[f"{DOMAIN}_loss"])


def test_batches_within_a_pass_use_different_noise():
    """Without the reset the counter has moved on, so the draws differ."""
    algo = _make_algo()
    batch = _make_batch()
    algo.reset_eval_pass_counter()
    first = algo.forward_eval(batch)
    second = algo.forward_eval(batch)  # no reset: pass counter is now 1
    assert not torch.equal(first[f"{DOMAIN}_{AC_KEY}"], second[f"{DOMAIN}_{AC_KEY}"])


def test_forward_eval_does_not_disturb_the_global_rng():
    """Eval must not shift the training RNG stream (fork_rng + generator)."""
    algo = _make_algo()
    batch = _make_batch()
    torch.manual_seed(7)
    before = torch.randn(3)
    torch.manual_seed(7)
    algo.reset_eval_pass_counter()
    algo.forward_eval(batch)
    after = torch.randn(3)
    assert torch.equal(before, after)


def test_pass_seed_depends_on_rank_and_counter(monkeypatch):
    algo = _make_algo()
    algo.reset_eval_pass_counter()
    monkeypatch.setenv("RANK", "0")
    rank0 = algo._eval_pass_seed()
    monkeypatch.setenv("RANK", "3")
    rank3 = algo._eval_pass_seed()
    assert rank0 != rank3
    algo._eval_pass_counter += 1
    assert algo._eval_pass_seed() == rank3 + 1


def test_val_loss_noise_does_not_collide_with_the_next_batch(monkeypatch):
    """A torch.Generator seeded with S draws the same stream as a global
    manual_seed(S), so a +1 offset would give batch i's val loss the very noise
    batch i+1 samples with."""
    from egomimic.algo.hpt import EVAL_LOSS_STRIDE

    assert EVAL_LOSS_STRIDE > 1000
    algo = _make_algo()
    algo.reset_eval_pass_counter()
    seeds = set()
    for _ in range(4):
        pass_seed = algo._eval_pass_seed()
        seeds.add(pass_seed)
        seeds.add(pass_seed + EVAL_LOSS_STRIDE)
        algo._eval_pass_counter += 1
    assert len(seeds) == 8, "a val-loss seed repeats a sampling seed"


def test_eval_base_seed_is_a_real_knob():
    """A second eval pass can be made to draw DIFFERENT noise on purpose."""
    algo = _make_algo()
    assert algo.eval_base_seed == EVAL_BASE_SEED
    algo.reset_eval_pass_counter()
    default = algo._eval_pass_seed()
    algo.eval_base_seed = 12345
    assert algo._eval_pass_seed() == default + 12345


def test_generator_is_threaded_to_the_head():
    """HPTModel.forward -> DenoisingPolicy.forward -> sample_action(generator)."""
    algo = _make_algo()
    model = algo.nets["policy"]
    data = {"action": torch.zeros(BATCH, HORIZON, ACTION_DIM)}
    gen = torch.Generator(device="cpu")

    gen.manual_seed(42)
    a = model.forward(DOMAIN, data, generator=gen)[DOMAIN]
    gen.manual_seed(42)
    b = model.forward(DOMAIN, data, generator=gen)[DOMAIN]
    gen.manual_seed(43)
    c = model.forward(DOMAIN, data, generator=gen)[DOMAIN]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_on_validation_start_resets_the_pass_counter():
    algo = _make_algo()
    algo.reset_eval_pass_counter()
    algo._eval_pass_counter = 17
    wrapper = types.SimpleNamespace(
        model=algo,
        device=torch.device("cpu"),
        _val_heads=lambda: {"valid": None},
    )
    ModelWrapper.on_validation_start(wrapper)
    assert algo._eval_pass_counter == 0


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


# ---------------------------------------------------------------------------
# 4. Trainer config knob
# ---------------------------------------------------------------------------
def test_trainer_default_exposes_gradient_clip_val(compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", [])
    assert "gradient_clip_val" in cfg.trainer
    assert cfg.trainer.gradient_clip_val is None  # no clipping until chosen
    assert cfg.trainer.gradient_clip_algorithm == "norm"


# Derived from the config group, not listed: a new trainer config must not be
# able to join the repo without this knob being checked.
TRAINER_CONFIGS = sorted(
    p.stem
    for p in (Path(egomimic.hydra_configs.__file__).parent / "trainer").glob("*.yaml")
)


@pytest.mark.parametrize("trainer_cfg", TRAINER_CONFIGS)
def test_every_trainer_config_keeps_the_knob(trainer_cfg, compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", [f"trainer={trainer_cfg}"])
    assert cfg.trainer.gradient_clip_val is None
    assert cfg.trainer.gradient_clip_algorithm == "norm"


def test_gradient_clip_override_reaches_the_trainer(tmp_path, compose_resolve):
    cfg = compose_resolve(
        "train_zarr_cartesian",
        [
            "trainer=default",
            "trainer.gradient_clip_val=1.0",
            "trainer.accelerator=cpu",
            "trainer.precision=32",
            "trainer.max_epochs=1",
            "trainer.min_epochs=1",
            f"paths.output_dir={tmp_path}",
        ],
    )
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False, callbacks=None)
    assert trainer.gradient_clip_val == 1.0
    assert trainer.gradient_clip_algorithm == "norm"


def test_lightning_module_does_not_override_gradient_clipping():
    """Nothing in ModelWrapper may intercept Lightning's clipping hook."""
    assert "configure_gradient_clipping" not in vars(ModelWrapper)
