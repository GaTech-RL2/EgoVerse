from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes import get_keymap_hpt


CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def _compose(experiment):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


def test_fold_recipe_keeps_the_126d_28_to_100_contract():
    cfg = _compose("fold/rh/pipeline_sampler_kp")
    sampler = cfg.model.robomimic_model.stages[2]
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.eta_min == 1e-5
    assert sampler.action_horizon == 100
    assert sampler.action_dims.human_bimanual == 126
    assert sampler.latent_dim == 128
    assert sampler.denoising_module.nblocks == 16
    assert sampler.denoising_module.hidden_dim == 512
    assert sampler.denoising_module.time_conditioning == "additive"
    assert cfg.data.train_dataloader_params.eva_bimanual.batch_size == 16
    assert cfg.data.train_dataloader_params.human_bimanual.batch_size == 16


def test_pusht_recipe_keeps_horizon_16_and_matching_optimizer_recipe():
    cfg = _compose("pusht/pipeline_sampler_pusht_h16")
    sampler = cfg.model.robomimic_model.stages[2]
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.model.optimizer.lr == 1e-4
    assert cfg.model.optimizer.weight_decay == 1e-4
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.eta_min == 1e-5
    assert sampler.action_horizon == 16
    assert sampler.num_inference_steps == 16
    assert dict(sampler.sampling_schedule[2001]) == {2: 0.8, 4: 0.15, 8: 0.05}
    assert cfg.data.train_dataloader_params.pushshapes_sim.batch_size == 16
    assert (
        cfg.data.train_dataloader_params.pushshapes_sim_small_circle.batch_size
        == 16
    )


def test_pusht_dataset_schema_is_registered_and_leak_free():
    assert get_embodiment_id("pushshapes_sim") == 15
    assert get_embodiment_id("pushshapes_sim_small_circle") == 17
    keymap = get_keymap_hpt(action_horizon=16)
    assert "horizon" not in keymap["front_img_1"]
    assert "horizon" not in keymap["state_agent_obj"]
    assert keymap["actions"]["horizon"] == 16
