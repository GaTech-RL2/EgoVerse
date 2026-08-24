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
    assert cfg.data.train_dataloader_params.pushshapes_sim_small_circle.batch_size == 16


def test_arc_length_recipe_keeps_tokenizer_model_and_rollout_horizons_aligned():
    cfg = _compose("fold/rh/pipeline_sampler_arc_length_nv")
    noise = cfg.model.robomimic_model.stages[1]
    sampler = cfg.model.robomimic_model.stages[2]
    adapter = cfg.model.robomimic_model.rollout_adapter
    human_data = cfg.data.train_datasets.human_bimanual.resolver
    eva_data = cfg.data.train_datasets.eva_bimanual.resolver

    assert cfg.model.robomimic_model.action_horizon == 26
    assert noise.action_horizon == 26
    assert sampler.action_horizon == 26
    assert sampler.denoising_module.act_seq == 26
    assert dict(sampler.action_dims) == {"eva_bimanual": 8, "human_bimanual": 8}
    shared_images = cfg.model.robomimic_model.stages[0].encoder.shared_encoder
    assert list(shared_images.obs_encoder.img_encoders) == [
        "observations.images.front_img_1"
    ]
    assert adapter.min_distance_unit == 0.40
    assert adapter.resampled_vector_length == 25
    assert adapter.action_horizon == 100
    assert human_data.key_map.keymap_mode == "arc_tokenizer_cartesian"
    assert eva_data.key_map.keymap_mode == "arc_tokenizer_cartesian"
    assert human_data.transform_list.min_distance_unit == 0.40
    assert eva_data.transform_list.min_distance_unit == 0.40
    assert human_data.transform_list.resampled_vector_length == 25
    assert eva_data.transform_list.resampled_vector_length == 25


def test_pusht_dataset_schema_is_registered_and_leak_free():
    assert get_embodiment_id("pushshapes_sim") == 15
    assert get_embodiment_id("pushshapes_sim_small_circle") == 17
    keymap = get_keymap_hpt(action_horizon=16)
    assert "horizon" not in keymap["front_img_1"]
    assert "horizon" not in keymap["state_agent_obj"]
    assert keymap["actions"]["horizon"] == 16


def test_legacy_human_episode_embodiments_resolve_to_collapsed_ids():
    human_bimanual = get_embodiment_id("human_bimanual")
    for vendor in ("aria", "mecka", "scale", "lightwheel"):
        assert get_embodiment_id(f"{vendor}_bimanual") == human_bimanual
