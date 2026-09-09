import importlib.util
import math
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.eval.checkpoint_loading import (
    extract_pipeline_nets_state,
    strict_load_pipeline_checkpoint,
)
from egomimic.eval.energy_score import energy_score
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.models.denoising_nets import CrossTransformer, PaperConditionalUnet1D
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.pipeline.pushshapes import PlanarCommon5NativeDecoder
from egomimic.utils.ema_callback import EMACallback


def _load_planar_stages_without_shared_pipeline():
    """The isolated slice intentionally receives Stage from the preceding PR."""
    module_name = "egomimic.pipeline.core"
    previous = sys.modules.get(module_name)
    stub = types.ModuleType(module_name)
    stub.Stage = nn.Module
    stub.resolve_homogeneous_scalar = resolve_homogeneous_scalar
    sys.modules[module_name] = stub
    try:
        path = Path(__file__).parents[1] / "egomimic/pipeline/stages_planar.py"
        spec = importlib.util.spec_from_file_location("_planar_stages_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


class TinyDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.proj_u = nn.Linear(latent_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent, time, condition):
        del time
        return self.proj_d(torch.tanh(self.proj_u(latent) + condition[..., :4]))


def test_planar_flow_sampler_train_and_eval_contract():
    module = _load_planar_stages_without_shared_pipeline()
    sampler = module.PlanarFlowSampler(
        TinyDenoiser(3, 4),
        condition_input_dim=2,
        action_horizon=5,
        action_dims={"pushshapes_sim_u_socket": 5},
        latent_dim=3,
        condition_dim=4,
        decoder_hidden_dim=8,
        denoiser_hidden_dim=4,
        gradient_checkpointing=False,
        num_inference_steps=4,
    )
    batch = {
        "embodiment": torch.tensor([19, 19]),
        "condition": torch.randn(2, 2),
        "sampler/noise": torch.randn(2, 5, 3),
    }
    result = sampler(batch.copy())
    assert result["pred_action"].shape == (2, 5, 5)
    torch.testing.assert_close(
        sampler.last_integration_step_sizes.sum(dim=-1), torch.ones(2)
    )
    sampler.train()
    batches_seen = int(sampler.training_batches_seen)
    result = sampler.execute(batch.copy(), mode="inference")
    assert result["log/sampler_unroll_steps"] == 4
    assert int(sampler.training_batches_seen) == batches_seen
    with pytest.raises(ValueError, match="homogeneous"):
        sampler({**batch, "embodiment": torch.tensor([19, 20])})


def test_planar_decoder_has_compact_sequential_state_keys():
    module = _load_planar_stages_without_shared_pipeline()
    decoder = module.TokenwiseActionDecoder(
        latent_dim=3,
        hidden_dim=8,
        action_dim=5,
        num_layers=3,
    )
    assert set(decoder.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    }


def test_paper_unet_preserves_action_shape():
    model = PaperConditionalUnet1D(
        input_dim=5,
        global_cond_dim=12,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        n_groups=4,
    )
    output = model(torch.randn(2, 16, 5), torch.tensor([1, 2]), torch.randn(2, 12))
    assert output.shape == (2, 16, 5)


@pytest.mark.parametrize(
    ("time_conditioning", "embedding_width"),
    (("concat", 4), ("additive", 8)),
)
def test_cross_transformer_time_conditioning_preserves_expected_state_shapes(
    time_conditioning, embedding_width
):
    model = CrossTransformer(
        nblocks=1,
        cond_dim=6,
        hidden_dim=8,
        act_dim=4,
        act_seq=3,
        n_heads=2,
        dropout=0.0,
        mlp_layers=1,
        mlp_ratio=2,
        time_conditioning=time_conditioning,
    )
    assert model.proj_u.weight.shape == (embedding_width, 4)
    assert model.state_dict()["pos_emb"].shape == (1, 3, embedding_width)
    output = model(
        torch.randn(2, 3, 4),
        torch.rand(2),
        torch.randn(2, 2, 6),
    )
    assert output.shape == (2, 3, 4)


def test_cross_transformer_rejects_invalid_or_rank_deficient_conditioning():
    kwargs = {
        "nblocks": 1,
        "cond_dim": 6,
        "act_seq": 3,
        "n_heads": 1,
        "dropout": 0.0,
        "mlp_layers": 1,
        "mlp_ratio": 2,
    }
    with pytest.raises(ValueError, match="either 'concat' or 'additive'"):
        CrossTransformer(
            hidden_dim=8,
            act_dim=4,
            time_conditioning="unknown",
            **kwargs,
        )
    with pytest.raises(ValueError, match="even hidden_dim"):
        CrossTransformer(
            hidden_dim=7,
            act_dim=3,
            time_conditioning="concat",
            **kwargs,
        )
    with pytest.raises(ValueError, match="Rank-deficient denoiser input"):
        CrossTransformer(
            hidden_dim=8,
            act_dim=5,
            time_conditioning="concat",
            **kwargs,
        )
    additive = CrossTransformer(
        hidden_dim=8,
        act_dim=5,
        time_conditioning="additive",
        **kwargs,
    )
    assert additive.proj_u.weight.shape == (8, 5)


def test_energy_score_separates_accuracy_and_diversity():
    target = torch.zeros(2, 3, 5)
    samples = torch.stack((target, target + 1, target - 1))
    values = energy_score(samples, target)
    assert values["accuracy"] > 0
    assert values["diversity"] > values["accuracy"]
    torch.testing.assert_close(
        values["score"], values["accuracy"] - 0.5 * values["diversity"]
    )


def test_planar_evaluator_uses_natural_batch_metadata_and_nested_results():
    target = torch.zeros(2, 16, 5)
    batch = {
        "opaque-stream": {
            "actions": target,
            "embodiment": torch.tensor([19, 19]),
        }
    }

    class IdentityNormalizer:
        @staticmethod
        def unnormalize(values, embodiment_id):
            assert embodiment_id == 19
            return values

    evaluator = PlanarActionEval(energy_score_enabled=False)
    evaluator.bind_data_context(normalizer=IdentityNormalizer())
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source: {"pred_action": source_batch["actions"] + 1}
            for source, source_batch in grouped.items()
        }
    )
    logged = {}
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        )
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    assert logged["Valid/MSE"].item() == pytest.approx(1.0)
    assert logged["Valid/MSE/pushshapes_sim_u_socket"].item() == pytest.approx(1.0)


class _IdentityNormalizer:
    @staticmethod
    def unnormalize(values, _embodiment_id):
        return values


def _stochastic_evaluator_fixture(evaluator, target, predictions, logged=None):
    def forward_eval(_batch):
        prediction = torch.rand_like(target)
        predictions.append(prediction.clone())
        return {"opaque-stream": {"pred_action": prediction}}

    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    evaluator.model = SimpleNamespace(forward_eval=forward_eval)
    evaluator.trainer = SimpleNamespace(
        current_epoch=2,
        global_step=17,
        global_rank=0,
        lightning_module=SimpleNamespace(
            log_dict=(
                (lambda metrics, **_kwargs: logged.update(metrics))
                if logged is not None
                else (lambda *_args, **_kwargs: None)
            )
        ),
    )
    return {
        "opaque-stream": {
            "actions": target,
            "embodiment": torch.tensor([19]),
        }
    }


def test_planar_evaluator_restores_rng_after_ordinary_prediction():
    target = torch.zeros(1, 2, 5)
    evaluator = PlanarActionEval(energy_score_enabled=False)
    batch = _stochastic_evaluator_fixture(evaluator, target, [])
    torch.manual_seed(8675309)
    state = torch.random.get_rng_state().clone()

    evaluator.on_validation_step(batch, batch_idx=0)

    assert torch.equal(torch.random.get_rng_state(), state)


def test_planar_evaluator_default_seed_is_wired_and_repeatable():
    target = torch.zeros(1, 2, 5)
    predictions = []
    evaluator = PlanarActionEval(energy_score_enabled=False)
    batch = _stochastic_evaluator_fixture(evaluator, target, predictions)

    evaluator.on_validation_step(batch, batch_idx=0)
    torch.rand(11)
    evaluator.on_validation_step(batch, batch_idx=1)

    assert evaluator.deterministic_seed == 420042
    evaluator_config = OmegaConf.load(
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/evaluator/eval_planar_v2.yaml"
    )
    assert evaluator_config.deterministic_seed == evaluator.deterministic_seed
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0.0, atol=0.0)


def test_planar_evaluator_reuses_first_energy_sample_for_mse(tmp_path):
    target = torch.zeros(1, 2, 5)
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs/evaluator"
    evaluator = PlanarActionEval(
        seed_bank_path=str(config_dir / "energy_score_seed_bank_k32_v1.json"),
        seed_bank_sha256=(
            "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
        ),
        artifact_root=str(tmp_path),
    )
    predictions = []
    logged = {}
    batch = _stochastic_evaluator_fixture(evaluator, target, predictions, logged)
    torch.manual_seed(314159)
    state = torch.random.get_rng_state().clone()

    evaluator.on_validation_step(batch, batch_idx=0)

    assert len(predictions) == 32
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.testing.assert_close(
        logged["Valid/MSE"], predictions[0].square().mean()
    )
    artifact = tmp_path / "epoch-2-step-17/rank-0-batch-0.pt"
    payload = torch.load(artifact, weights_only=True)
    assert payload["deterministic_seed"] == evaluator.seeds[0] == 420042


def test_planar_evaluator_rejects_nonreusable_deterministic_seed(tmp_path):
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs/evaluator"
    with pytest.raises(ValueError, match="must equal the first Energy Score seed"):
        PlanarActionEval(
            seed_bank_path=str(config_dir / "energy_score_seed_bank_k32_v1.json"),
            seed_bank_sha256=(
                "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
            ),
            artifact_root=str(tmp_path),
            deterministic_seed=7,
        )


def test_planar_evaluator_wraps_native_rotation_error_across_pi():
    epsilon = 0.01

    def common_five(theta):
        return torch.tensor(
            [[[0.0, 0.0, math.cos(theta), math.sin(theta), 0.0]]],
            dtype=torch.float64,
        )

    target = common_five(math.pi - epsilon)
    prediction = common_five(-math.pi + epsilon)
    batch = {
        "opaque-stream": {
            "actions": target,
            "embodiment": torch.tensor([19]),
        }
    }

    class IdentityNormalizer:
        @staticmethod
        def unnormalize(values, embodiment_id):
            assert embodiment_id == 19
            return values

    evaluator = PlanarActionEval(
        energy_score_enabled=False,
        native_decoder=PlanarCommon5NativeDecoder(
            action_horizon=1,
            native_action_dim=3,
        ),
    )
    evaluator.bind_data_context(normalizer=IdentityNormalizer())
    evaluator.model = SimpleNamespace(
        forward_eval=lambda _batch: {
            "opaque-stream": {"pred_action": prediction}
        }
    )
    logged = {}
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        )
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    expected = (2.0 * epsilon) ** 2 / 3.0
    assert logged["Valid/Native_MSE/pushshapes_sim_u_socket"].item() == pytest.approx(
        expected
    )
    assert logged["Valid/Native_MSE"].item() == pytest.approx(expected)


def test_planar_evaluator_selects_native_decoder_per_embodiment():
    target = torch.zeros(1, 16, 5)
    batch = {
        "u": {"actions": target, "embodiment": torch.tensor([19])},
        "chain": {"actions": target, "embodiment": torch.tensor([20])},
    }
    evaluator = PlanarActionEval(
        energy_score_enabled=False,
        native_decoders={
            "pushshapes_sim_u_socket": PlanarCommon5NativeDecoder(16, 3),
            "pushshapes_sim_chain_gripper": PlanarCommon5NativeDecoder(16, 4),
        },
    )
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source: {"pred_action": source_batch["actions"]}
            for source, source_batch in grouped.items()
        }
    )
    logged = {}
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        )
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    assert logged["Valid/Native_MSE/pushshapes_sim_u_socket"] == 0
    assert logged["Valid/Native_MSE/pushshapes_sim_chain_gripper"] == 0


def test_strict_checkpoint_loader_overlays_ema_and_retains_online_buffers():
    class BufferedPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 1)
            self.register_buffer("running_scale", torch.tensor(3.0))

    class Algo:
        nets = nn.ModuleDict({"pipeline": BufferedPolicy()})

    algo = Algo()
    expected = OrderedDict(
        (key, value.clone()) for key, value in algo.nets.state_dict().items()
    )
    online = OrderedDict((f"nets.{key}", value + 1) for key, value in expected.items())
    parameter_keys = set(dict(algo.nets.named_parameters()))
    ema = OrderedDict(
        (f"nets.{key}", expected[key] + 2) for key in sorted(parameter_keys)
    )
    checkpoint = {"state_dict": online, "ema_state_dict": ema}
    assert set(extract_pipeline_nets_state(checkpoint)) == set(expected)
    strict_load_pipeline_checkpoint(algo, checkpoint, use_ema=True)
    for key, value in algo.nets.state_dict().items():
        offset = 2 if key in parameter_keys else 1
        torch.testing.assert_close(value, expected[key] + offset)
    with pytest.raises(ValueError):
        strict_load_pipeline_checkpoint(algo, {"state_dict": {}}, use_ema=False)


def test_strict_checkpoint_loader_rejects_inexact_or_conflicting_ema_keys():
    class Algo:
        nets = nn.ModuleDict({"pipeline": nn.Linear(2, 1)})

    algo = Algo()
    expected = algo.nets.state_dict()
    online = OrderedDict(
        (f"nets.{key}", value.clone()) for key, value in expected.items()
    )
    ema = OrderedDict(
        (f"nets.{key}", value.clone())
        for key, value in algo.nets.named_parameters()
    )

    missing = OrderedDict(ema)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="EMA parameter key mismatch"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": missing},
            use_ema=True,
        )

    extra = OrderedDict(ema)
    extra["nets.pipeline.unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError, match="EMA parameter key mismatch"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": extra},
            use_ema=True,
        )

def test_train_eval_pipeline_load_is_strict():
    from egomimic.pipeline.algo import PipelineAlgo
    from egomimic.trainHydra import _load_eval_checkpoint

    class Algo(PipelineAlgo):
        def __init__(self):
            self.nets = nn.ModuleDict({"pipeline": nn.Linear(2, 1)})

    algo = Algo()
    expected = OrderedDict(
        (key, value.clone()) for key, value in algo.nets.state_dict().items()
    )
    state_dict = OrderedDict(
        (f"nets.{key}", value + 1) for key, value in expected.items()
    )

    class PipelineWrapper:
        model = algo

    wrapper = PipelineWrapper()
    _load_eval_checkpoint(wrapper, {"state_dict": state_dict}, OmegaConf.create({}))
    for key, value in algo.nets.state_dict().items():
        torch.testing.assert_close(value, expected[key] + 1)

    incomplete = OrderedDict(state_dict)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="Pipeline checkpoint key mismatch"):
        _load_eval_checkpoint(
            wrapper,
            {"state_dict": incomplete},
            OmegaConf.create({}),
        )

    class NonPipelineWrapper:
        model = nn.Linear(2, 1)

    with pytest.raises(TypeError, match="must wrap PipelineAlgo"):
        _load_eval_checkpoint(
            NonPipelineWrapper(),
            {"state_dict": state_dict},
            OmegaConf.create({}),
        )


def test_eval_mode_removes_training_time_ema_callback():
    from egomimic.trainHydra import _callbacks_for_mode

    ema_callback = EMACallback()
    retained_callback = object()
    assert _callbacks_for_mode([ema_callback, retained_callback], "eval") == [
        retained_callback
    ]
    assert _callbacks_for_mode([ema_callback, retained_callback], "train") == [
        ema_callback,
        retained_callback,
    ]


def test_paper_ema_warmup_and_fail_closed_resume():
    callback = EMACallback(decay=0.9999, use_warmup=True, power=0.75)
    assert callback._schedule(1) == 0
    assert 0 < callback._schedule(2) < callback._schedule(100)
    with pytest.raises(RuntimeError):
        callback.on_load_checkpoint(None, None, {"global_step": 3})
