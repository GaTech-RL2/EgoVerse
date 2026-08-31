import hashlib
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.human_robot_overlay_eval import HumanRobotOverlayEval

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / (
    "egomimic/hydra_configs/model/bf/"
    "bf_pipeline_unite_usocket_chain_points_unite_b_per_emb_proprio_h16.yaml"
)
EXPERIMENT_PATH = ROOT / (
    "egomimic/hydra_configs/experiment/pusht/"
    "pipeline_unite_b_usocket_chain_newdata_val01_h16_per_emb_proprio.yaml"
)
SEED_BANK_PATH = (
    ROOT / "egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json"
)
SEED_BANK_SHA256 = "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
EXPECTED_PARAMETERS = 216_917_546


def test_unite_b_action_model_has_paper_scale_total_and_fixed_contracts():
    model = OmegaConf.load(MODEL_PATH)
    stages = model.robomimic_model.stages
    policy = stages[3]
    denoiser = policy.generative_encoder.denoising_module

    assert stages[2].num_tokens == 16
    assert stages[2].latent_dim == 128
    assert policy.num_inference_steps == 8
    assert policy.generative_encoder.latent_dim == 128
    assert policy.generative_encoder.denoiser_hidden_dim == 512
    assert denoiser.hidden_dim == 512
    assert denoiser.nblocks == 12
    assert denoiser.n_heads == 8
    for decoder in policy.decoders.values():
        assert decoder.hidden_dim == 128
        assert decoder.num_blocks == 6
        assert decoder.num_heads == 8

    modules = instantiate(stages)
    total = sum(
        parameter.numel() for module in modules for parameter in module.parameters()
    )
    trainable = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    assert total == EXPECTED_PARAMETERS
    assert trainable == total


def test_unite_b_energy_score_contract_is_domain_specific_and_hashed():
    experiment = OmegaConf.load(EXPERIMENT_PATH)
    energy = experiment.evaluator.energy_score
    assert energy.enabled is True
    assert energy.sample_count == 32
    assert dict(energy.action_dims) == {
        "pushshapes_sim_u_socket": 4,
        "pushshapes_sim_chain_gripper": 6,
    }
    assert hashlib.sha256(SEED_BANK_PATH.read_bytes()).hexdigest() == SEED_BANK_SHA256
    assert energy.seed_bank_sha256 == SEED_BANK_SHA256

    for embodiment, action_dim in energy.action_dims.items():
        validated = HumanRobotOverlayEval._validate_distance_blocks(
            embodiment,
            int(action_dim),
            OmegaConf.to_container(energy.distance_blocks[embodiment]),
        )
        assert sum(len(block["indices"]) for block in validated.values()) == action_dim
