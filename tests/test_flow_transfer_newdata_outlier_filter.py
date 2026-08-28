from pathlib import Path

import pytest
import zarr
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
EXCLUDED_EPISODE = "episode_T_chain_gripper_obs7_000050"
CHAIN_ROOTS = [
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
]


def _compose(experiment: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


@pytest.mark.parametrize(
    "experiment",
    [
        "pusht/pipeline_diffusion_usocket_chain_newdata_h16",
        "pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16",
    ],
)
def test_newdata_chain_filter_excludes_idle_episode_for_both_splits(
    experiment: str,
) -> None:
    cfg = _compose(experiment)

    for split in ("train_datasets", "valid_datasets"):
        chain = cfg.data[split].pushshapes_sim_chain_gripper
        assert list(chain.resolver.folder_paths) == CHAIN_ROOTS
        assert chain.resolver.key_map.action_horizon == 16

        episode_filter = instantiate(chain.filters)
        assert episode_filter.filter_lambdas == [
            f"lambda row: row.get('episode_hash') != '{EXCLUDED_EPISODE}'"
        ]
        assert episode_filter.matches({"episode_hash": EXCLUDED_EPISODE}) is False
        assert episode_filter.matches({"episode_hash": f"{EXCLUDED_EPISODE}.zarr"})
        assert episode_filter.matches({"episode_hash": "episode_keep"})


def test_local_resolver_strips_zarr_suffix_before_chain_filter(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chain_gripper_gen"
    zarr.open_group(str(root / f"{EXCLUDED_EPISODE}.zarr"), mode="w")
    zarr.open_group(str(root / "episode_keep.zarr"), mode="w")
    episode_filter = instantiate(
        {
            "_target_": "egomimic.rldb.filters.DatasetFilter",
            "filter_lambdas": [
                f"lambda row: row.get('episode_hash') != '{EXCLUDED_EPISODE}'"
            ],
        }
    )

    selected = LocalEpisodeResolver._get_local_filtered_paths(
        root, filters=episode_filter
    )

    assert [episode_hash for _, episode_hash in selected] == ["episode_keep"]
