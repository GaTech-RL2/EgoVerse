from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from egomimic.models.ddim_scheduler import DDIMScheduler
from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_diffusion import MultiDomainDiffusionPolicyStage
from egomimic.pipeline.stages_sampler import FusedObsEncoder

CHAIN = "pushshapes_sim_chain_gripper"
USOCKET = "pushshapes_sim_u_socket"


def _tiny_policy(domain: str, action_dim: int, prediction_type: str = "epsilon"):
    return DiffusionPolicy(
        model=ConditionalUnet1D(
            input_dim=action_dim,
            cond_dim=5,
            ac_latent_seq=1,
            diffusion_step_embed_dim=16,
            down_dims=[8, 16],
            kernel_size=3,
            n_groups=4,
            cond_predict_scale=True,
        ),
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=8,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=prediction_type,
        ),
        action_horizon=4,
        infer_ac_dims={domain: action_dim},
        num_inference_steps=2,
    )


def _tiny_stage():
    return MultiDomainDiffusionPolicyStage(
        policies={
            CHAIN: _tiny_policy(CHAIN, 6),
            USOCKET: _tiny_policy(USOCKET, 4),
        },
        action_horizon=4,
        condition_input_dim=5,
    )


def test_multidomain_diffusion_training_uses_epsilon_loss_and_backpropagates():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = _tiny_stage().to(device).train()
    batch = {
        "condition": torch.randn(2, 5, device=device),
        "embodiment": CHAIN,
        "target": torch.randn(2, 4, 6, device=device),
    }

    out = stage(batch)

    assert stage.objective == "epsilon"
    assert "pred_action" not in out
    assert out["loss/diffusion_noise"].ndim == 0
    assert torch.isfinite(out["loss/diffusion_noise"])
    out["loss/diffusion_noise"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in stage.parameters()
    )


@pytest.mark.parametrize("domain,width", [(CHAIN, 6), (USOCKET, 4)])
def test_multidomain_diffusion_eval_samples_the_domain_width(domain, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = _tiny_stage().to(device).eval()
    with torch.inference_mode():
        out = stage(
            {
                "condition": torch.randn(2, 5, device=device),
                "embodiment": domain,
            }
        )

    assert out["pred_action"].shape == (2, 4, width)
    assert torch.isfinite(out["pred_action"]).all()
    assert out["log/diffusion_inference_steps"] == 2.0


def test_multidomain_diffusion_rejects_non_epsilon_scheduler():
    with pytest.raises(ValueError, match="epsilon noise objective"):
        MultiDomainDiffusionPolicyStage(
            policies={CHAIN: _tiny_policy(CHAIN, 6, prediction_type="sample")},
            action_horizon=4,
            condition_input_dim=5,
        )


def test_multidomain_diffusion_rejects_wrong_domain_target_width():
    stage = _tiny_stage().train()
    with pytest.raises(ValueError, match="must be"):
        stage(
            {
                "condition": torch.randn(2, 5),
                "embodiment": USOCKET,
                "target": torch.randn(2, 4, 6),
            }
        )


def test_diffusion_pipeline_contracts_are_mode_exact():
    encoder = FusedObsEncoder(torch.nn.Identity(), n_obs_steps=1)
    diffusion = _tiny_stage()
    pipeline = Pipeline([encoder, diffusion])

    assert encoder.contract("train") == (
        ("obs/*", "embodiment", "actions"),
        ("condition", "target"),
    )
    assert encoder.contract("rollout") == (
        ("obs/*", "embodiment"),
        ("condition",),
    )
    assert diffusion.contract("train") == (
        ("condition", "target", "embodiment"),
        ("loss/diffusion_noise", "log/*"),
    )
    assert diffusion.contract("rollout") == (
        ("condition", "embodiment"),
        ("pred_action", "log/*"),
    )

    train_runnable, train_excluded = pipeline.plan(
        ["obs/state_agent_obj", "embodiment", "actions"], mode="train"
    )
    rollout_runnable, rollout_excluded = pipeline.plan(
        ["obs/state_agent_obj", "embodiment"], mode="rollout"
    )
    assert train_runnable == [encoder, diffusion]
    assert train_excluded == []
    assert rollout_runnable == [encoder, diffusion]
    assert rollout_excluded == []

    _, missing_target = pipeline.plan(
        ["obs/state_agent_obj", "embodiment"], mode="train"
    )
    assert missing_target == [
        (encoder, ["actions"]),
        (diffusion, ["condition", "target"]),
    ]

    rollout_graph = pipeline.explain(
        ["obs/state_agent_obj", "embodiment"], mode="rollout"
    )
    assert "actions" not in rollout_graph
    assert "target" not in rollout_graph
    assert "pred_action" in rollout_graph


def test_stage_contract_rejects_unknown_mode():
    with pytest.raises(ValueError, match="train\\|rollout"):
        _tiny_stage().contract("validation")


@pytest.mark.parametrize(
    "experiment_name",
    [
        "pusht/pipeline_diffusion_usocket_chain_h16",
        "pusht/pipeline_diffusion_usocket_chain_h16_smoke",
    ],
)
def test_diffusion_flow_transfer_hydra_composition(experiment_name):
    config_dir = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment_name}"],
        )

    model = cfg.model.robomimic_model
    assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
    assert model._target_ == "egomimic.pipeline.algo.PipelineAlgo"
    assert list(model.domains) == [CHAIN, USOCKET]
    assert model.ac_keys[CHAIN] == "actions"
    assert model.ac_keys[USOCKET] == "actions"
    assert (
        model.rollout_adapters[CHAIN]._target_
        == "egomimic.pipeline.pushshapes.ChainGripperPointRolloutAdapter"
    )
    assert (
        model.rollout_adapters[USOCKET]._target_
        == "egomimic.pipeline.pushshapes.USocketRotVecRolloutAdapter"
    )

    targets = [stage._target_ for stage in model.stages]
    assert targets == [
        "egomimic.pipeline.stages_sampler.FusedObsEncoder",
        "egomimic.pipeline.stages_diffusion.MultiDomainDiffusionPolicyStage",
    ]
    dp_stage = model.stages[1]
    assert dp_stage.policies[CHAIN].infer_ac_dims[CHAIN] == 6
    assert dp_stage.policies[USOCKET].infer_ac_dims[USOCKET] == 4
    for domain in (CHAIN, USOCKET):
        policy = dp_stage.policies[domain]
        assert policy._target_ == "egomimic.models.diffusion_policy.DiffusionPolicy"
        assert policy.noise_scheduler.prediction_type == "epsilon"
        assert (
            policy.noise_scheduler._target_
            == "egomimic.models.ddim_scheduler.DDIMScheduler"
        )
        assert (
            policy.model._target_ == "egomimic.models.denoising_nets.ConditionalUnet1D"
        )

    chain_resolver = cfg.data.train_datasets[CHAIN].resolver
    assert set(cfg.data.train_datasets) == set(cfg.data.train_dataloader_params)
    assert set(cfg.data.valid_datasets) == set(cfg.data.valid_dataloader_params)
    assert str(chain_resolver.folder_path).endswith("/chain_gripper_3000_v2")
    assert chain_resolver.key_map.action_zarr_key == "actions"
    assert (
        chain_resolver.transform_list._target_
        == "egomimic.rldb.embodiment.pushshapes.get_chain_gripper_point_transform_list"
    )
    assert (
        cfg.data.train_datasets[USOCKET].resolver.transform_list._target_
        == "egomimic.rldb.embodiment.pushshapes.get_rotvec_transform_list"
    )

    if experiment_name.endswith("_smoke"):
        assert cfg.trainer.max_steps == 2
        assert cfg.trainer.val_check_interval == 1
        assert cfg.trainer.limit_val_batches == 1
        assert cfg.callbacks.model_checkpoint.every_n_train_steps == 1
        assert cfg.callbacks.model_checkpoint.save_last is True
        assert cfg.callbacks.model_checkpoint.save_on_train_epoch_end is False
