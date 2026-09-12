"""(vendor, algo) -> the REAL hydra configs that recipe trains with, plus the
override lists both test tiers share. Adding a vendor is one RECIPES row."""

from __future__ import annotations

from dataclasses import dataclass

LOCAL_RESOLVER = "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
HPT_CAMERAS = {
    "eva_bimanual": ("front_img_1", "right_wrist_img", "left_wrist_img"),
    "human_bimanual": ("front_img_1",),
}


@dataclass(frozen=True)
class Recipe:
    top: str  # top-level config name
    data: str  # data/ group entry
    model: str  # model/ group entry
    embodiment: str
    extra: tuple[str, ...] = ()


def _pi_human(model: str, stride: int) -> Recipe:
    # There is no shipped single-vendor human Pi data config that instantiates:
    # aria_pi/mecka_pi/scale_pi call Human.get_keymap without keymap_mode (see
    # KNOWN_BROKEN_INSTANTIATE in test_data_configs_compose). So the human Pi
    # rows use cotrain_pi_base's human domain with the eva domain nulled and the
    # vendor's stride. On this branch mecka and scale then differ from each other
    # only by model config name (pi0.5_bc_{aria,mecka,scale}.yaml are identical);
    # both rows stay so a future per-vendor divergence is covered.
    return Recipe(
        "train_zarr_cartesian_pi",
        "cotrain_pi_base",
        model,
        "human_bimanual",
        (
            "data.train_datasets.eva_bimanual=null",
            "data.valid_datasets.eva_bimanual=null",
            f"data.train_datasets.human_bimanual.resolver.transform_list.stride={stride}",
            "data.train_datasets.human_bimanual.mode=total",
            "data.valid_datasets.human_bimanual.mode=total",
        ),
    )


RECIPES: dict[tuple[str, str], Recipe] = {
    ("eva", "hpt"): Recipe(
        "train_zarr_cartesian", "eva", "hpt_bc_flow_eva", "eva_bimanual"
    ),
    ("aria", "hpt"): Recipe(
        "train_zarr_cartesian", "aria", "hpt_bc_flow_aria", "human_bimanual"
    ),
    ("mecka", "hpt"): Recipe(
        "train_zarr_cartesian", "mecka", "hpt_bc_flow_mecka", "human_bimanual"
    ),
    ("scale", "hpt"): Recipe(
        "train_zarr_cartesian", "scale", "hpt_bc_flow_scale", "human_bimanual"
    ),
    ("eva", "pi"): Recipe(
        "train_zarr_cartesian_pi", "eva_pi", "pi0.5_bc_eva", "eva_bimanual"
    ),
    ("aria", "pi"): _pi_human("pi0.5_bc_aria", 3),
    ("mecka", "pi"): _pi_human("pi0.5_bc_mecka", 1),
    ("scale", "pi"): _pi_human("pi0.5_bc_scale", 1),
}


def common_overrides(
    emb: str, data_dir, out_dir, *, batch_size: int, num_workers: int
) -> list[str]:
    return [
        f"paths.dataset_dir={data_dir}",
        f"paths.output_dir={out_dir}",
        f"data.train_datasets.{emb}.resolver._target_={LOCAL_RESOLVER}",
        f"data.train_datasets.{emb}.filters=null",
        f"data.train_dataloader_params.{emb}.batch_size={batch_size}",
        f"data.train_dataloader_params.{emb}.num_workers={num_workers}",
        f"data.valid_dataloader_params.{emb}.batch_size={batch_size}",
        f"data.valid_dataloader_params.{emb}.num_workers={num_workers}",
        "~logger",
        "~callbacks",
        "~evaluator",
        "+mmap_checkpoint=false",
        "norm_stats.save_cache_dir=null",
        f"norm_stats.num_workers={num_workers}",
    ]


def cpu_trainer_overrides(steps: int = 2) -> list[str]:
    return [
        "trainer=default",
        "trainer.precision=32",
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        f"trainer.limit_train_batches={steps}",
        "trainer.limit_val_batches=0",
        "trainer.check_val_every_n_epoch=1",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
    ]


def hpt_small_overrides(emb: str) -> list[str]:
    return [
        "model.robomimic_model.trunk.num_blocks=2",
        f"model.robomimic_model.head_specs.{emb}.model.nblocks=1",
        f"model.robomimic_model.head_specs.{emb}.num_inference_steps=2",
        *[
            f"+model.robomimic_model.encoder_specs.{c}.weights=null"
            for c in HPT_CAMERAS[emb]
        ],
    ]


def pi_cpu_overrides() -> list[str]:
    return [
        "model.robomimic_model.config.pytorch_weight_path=null",
        "model.robomimic_model.config.pytorch_training_precision=float32",
    ]
