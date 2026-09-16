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


def _pi(data: str, model: str, emb: str) -> Recipe:
    # data/eva.yaml already emits the continuous-6D (wrist-frame) layout Pi
    # requires, the same mode cotrain_pi_base / eva_pi set.
    extra = [f"+data.train_datasets.{emb}.resolver.key_map.annotation_key=annotations"]
    return Recipe("train_zarr_cartesian_pi", data, model, emb, tuple(extra))


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
    # Pi trains on the same per-vendor data configs as HPT: datasets emit one
    # camera naming and the Pi wrapper renames onto openpi's slots. The plain
    # vendor configs load no language field (that is a data choice the *_lang
    # configs make); opt in here so the prompt path is exercised, not the
    # default-prompt fallback.
    ("eva", "pi"): _pi("eva", "pi0.5_bc_eva", "eva_bimanual"),
    ("aria", "pi"): _pi("aria", "pi0.5_bc_aria", "human_bimanual"),
    ("mecka", "pi"): _pi("mecka", "pi0.5_bc_mecka", "human_bimanual"),
    ("scale", "pi"): _pi("scale", "pi0.5_bc_scale", "human_bimanual"),
    # FlowVLA on the plain mecka data: its human arm already emits the 144-D
    # keypoint action the flagship recipe trains on, so the CPU tier exercises
    # the backbone + DiT head without the flagship operator split.
    ("mecka", "flowvla"): Recipe(
        "train_zarr_cartesian",
        "mecka",
        "flowvla_resnet_text_mecka_kp",
        "human_bimanual",
        (
            "+data.train_datasets.human_bimanual.resolver.key_map.annotation_key=annotations",
        ),
    ),
}


def flowvla_small_overrides() -> list[str]:
    """Shrink FlowVLA to something a CPU can take two steps through.

    ``weights=null`` keeps torchvision off the network; swapping the text
    encoder's ``_target_`` keeps the real 596M Qwen3-Embedding out of a unit
    test (the stub takes ``**kwargs``, so the config's model_name / dtype /
    freeze / cache_* keys pass through harmlessly).
    """
    return [
        # FlowVLA concatenates EVERY proprio key (HPT instead selects them with
        # stem_specs, which is why hpt_bc_flow_mecka declares 144 on this same
        # data). data/mecka emits observations.state.keypoints (144) AND
        # observations.state.ee_pose (20); the flagship keypoint config turns
        # ee_pose off, so the real run is 144 and only the CPU tier is 164.
        "model.robomimic_model.dims.human_bimanual.proprio=164",
        "model.robomimic_model.backbone.hidden_size=32",
        "model.robomimic_model.backbone.num_layers=2",
        "model.robomimic_model.backbone.weights=null",
        "model.robomimic_model.backbone.text_encoder._target_="
        "fixtures.stub_text_encoder.StubTextEncoder",
        "model.robomimic_model.backbone.text_encoder.output_dim=32",
        "model.robomimic_model.head.dit_hidden=16",
        "model.robomimic_model.head.head_dim=8",
        "model.robomimic_model.head.num_register_tokens=2",
        "model.robomimic_model.head.num_inference_steps=2",
    ]


def common_overrides(
    emb: str,
    data_dir,
    out_dir,
    *,
    batch_size: int,
    num_workers: int,
    episode_hashes: tuple[str, ...] | None = None,
) -> list[str]:
    overrides = [
        f"paths.dataset_dir={data_dir}",
        f"paths.output_dir={out_dir}",
        # Keep the norm-stat cache per test: the fixtures' made-up hashes must
        # never share entries in the checkout's logs/cache.
        f"paths.cache_dir={out_dir}/cache",
        f"data.train_datasets.{emb}.resolver._target_={LOCAL_RESOLVER}",
        # Reset to null first: some real configs' filters dict (e.g. eva_pi's
        # embodiment-only lambda) is struct-locked to its own keys, so a plain
        # merge that introduces `episode_hashes` would fail. Force-adding onto
        # a null value has no such restriction.
        f"data.train_datasets.{emb}.filters=null",
    ]
    if episode_hashes:
        pins = ", ".join(episode_hashes)
        filters = f"{{_target_: egomimic.rldb.filters.DatasetFilter, episode_hashes: [{pins}]}}"
        overrides.append(f"++data.train_datasets.{emb}.filters={filters}")
    overrides += [
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
    return overrides


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
