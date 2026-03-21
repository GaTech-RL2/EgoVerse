import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.utils import DataSchematic

logger = logging.getLogger(__name__)


class _LegacyDataSchematic(DataSchematic):
    def infer_norm_from_dataset_legacy(self, dataset):
        """
        dataset: huggingface dataset or zarr dataset
        returns: dictionary of means and stds for proprio and action keys
        """
        norm_columns = []

        embodiment = dataset.embodiment
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_columns.extend(self.keys_of_type("proprio_keys", embodiment))
        norm_columns.extend(self.keys_of_type("action_keys", embodiment))

        logger.info(
            f"[NormStats] Starting norm inference for embodiment={embodiment}, "
            f"{len(norm_columns)} columns"
        )

        def get_zarr_data(ds, col):
            if hasattr(ds, "episode_reader"):
                if col in ds.episode_reader._store:
                    return ds.episode_reader._store[col][:]
                return None
            if hasattr(ds, "datasets"):
                data_list = []
                for child_dataset in ds.datasets.values():
                    res = get_zarr_data(child_dataset, col)
                    if res is not None:
                        data_list.append(res)
                if data_list:
                    return np.concatenate(data_list, axis=0)
            return None

        for column in norm_columns:
            if not self.is_key_with_embodiment(column, embodiment):
                continue
            column_name = self.keyname_to_zarr_key(column, embodiment)
            logger.info(f"[NormStats] Processing column={column_name}")

            column_data = get_zarr_data(dataset, column_name)

            if column_data is None:
                logger.warning(
                    f"Skipping {column_name}, data not found given dataset type"
                )
                continue

            if column_data.ndim not in (2, 3):
                raise ValueError(
                    f"Column {column} has shape {column_data.shape}, "
                    "expected 2 or 3 dims"
                )

            mean = np.mean(column_data, axis=0)
            std = np.std(column_data, axis=0)
            minv = np.min(column_data, axis=0)
            maxv = np.max(column_data, axis=0)
            median = np.median(column_data, axis=0)
            q1 = np.percentile(column_data, 1, axis=0)
            q99 = np.percentile(column_data, 99, axis=0)

            self.norm_stats[embodiment][column] = {
                "mean": torch.from_numpy(mean).float(),
                "std": torch.from_numpy(std).float(),
                "min": torch.from_numpy(minv).float(),
                "max": torch.from_numpy(maxv).float(),
                "median": torch.from_numpy(median).float(),
                "quantile_1": torch.from_numpy(q1).float(),
                "quantile_99": torch.from_numpy(q99).float(),
            }

        logger.info("[NormStats] Finished norm inference")


class _DummyNormDataset:
    def __init__(self) -> None:
        self.embodiment = "eva_bimanual"

        obs_base = np.arange(14, dtype=np.float32)
        action_dim = np.arange(14, dtype=np.float32).reshape(1, 14)
        action_time = np.arange(100, dtype=np.float32).reshape(100, 1)

        ee_pose = np.stack([obs_base + i for i in range(4)], axis=0)
        actions = np.stack(
            [action_time + action_dim + i for i in range(4)],
            axis=0,
        )

        self.episode_reader = type(
            "_EpisodeReader",
            (),
            {
                "_store": {
                    "observations.state.ee_pose": ee_pose,
                    "actions_cartesian": actions,
                }
            },
        )()

    def __len__(self) -> int:
        return self.episode_reader._store["observations.state.ee_pose"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "observations.state.ee_pose": self.episode_reader._store[
                "observations.state.ee_pose"
            ][idx],
            "actions_cartesian": self.episode_reader._store["actions_cartesian"][idx],
        }


def _expected_stats(array: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.from_numpy(np.mean(array, axis=0)).float(),
        "std": torch.from_numpy(np.std(array, axis=0)).float(),
        "min": torch.from_numpy(np.min(array, axis=0)).float(),
        "max": torch.from_numpy(np.max(array, axis=0)).float(),
        "median": torch.from_numpy(np.median(array, axis=0)).float(),
        "quantile_1": torch.from_numpy(np.percentile(array, 1, axis=0)).float(),
        "quantile_99": torch.from_numpy(np.percentile(array, 99, axis=0)).float(),
    }


def _print_norm_stats(
    stats: dict[int, dict[str, dict[str, torch.Tensor]]], embodiment_id: int
) -> None:
    for key, key_stats in stats[embodiment_id].items():
        print(f"{key}:")
        for stat_name, value in key_stats.items():
            print(f"  {stat_name}: shape={tuple(value.shape)}")
            print(f"  {stat_name} value=\n{value}")


def test_infer_norm_from_dataset_legacy_matches_current_on_dummy_dataset() -> None:
    dataset = _DummyNormDataset()

    sample = dataset[0]
    assert sample["observations.state.ee_pose"].shape == (14,)
    assert sample["actions_cartesian"].shape == (100, 14)
    print(
        f"observations.state.ee_pose sample shape: {sample['observations.state.ee_pose'].shape}"
    )
    print(f"actions_cartesian sample shape: {sample['actions_cartesian'].shape}")

    schematic_dict = {
        "eva_bimanual": {
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
        }
    }
    viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}

    legacy_schematic = _LegacyDataSchematic(schematic_dict, viz_img_key)
    current_schematic = DataSchematic(schematic_dict, viz_img_key)

    legacy_schematic.infer_norm_from_dataset_legacy(dataset)
    current_schematic.infer_norm_from_dataset(
        dataset,
        dataset.embodiment,
        sample_frac=1.0,
        seed=0,
        batch_size=len(dataset),
        num_workers=0,
    )

    embodiment_id = get_embodiment_id(dataset.embodiment)
    expected = {
        "observations.state.ee_pose": _expected_stats(
            dataset.episode_reader._store["observations.state.ee_pose"]
        ),
        "actions_cartesian": _expected_stats(
            dataset.episode_reader._store["actions_cartesian"]
        ),
    }

    print("legacy norm stats")
    _print_norm_stats(legacy_schematic.norm_stats, embodiment_id)
    print("current norm stats")
    _print_norm_stats(current_schematic.norm_stats, embodiment_id)

    assert set(legacy_schematic.norm_stats[embodiment_id]) == set(expected)
    assert set(current_schematic.norm_stats[embodiment_id]) == set(expected)

    for key, expected_stats in expected.items():
        legacy_stats = legacy_schematic.norm_stats[embodiment_id][key]
        current_stats = current_schematic.norm_stats[embodiment_id][key]

        for stat_name, expected_value in expected_stats.items():
            torch.testing.assert_close(legacy_stats[stat_name], expected_value)
            torch.testing.assert_close(current_stats[stat_name], expected_value)
            torch.testing.assert_close(
                legacy_stats[stat_name], current_stats[stat_name]
            )


def _create_zarr_episode(path: Path, data: dict[str, np.ndarray], total_frames: int):
    """Write a minimal zarr episode to disk."""
    store = zarr.open_group(str(path), mode="w")
    store.attrs["total_frames"] = total_frames
    for key, arr in data.items():
        store.create_array(key, data=arr, chunks=arr.shape)


class _FakeZarrDataset:
    """Mimics ZarrDataset with episode_path attribute."""

    def __init__(self, episode_path: Path):
        self.episode_path = episode_path


class _FakeMultiDataset:
    """Mimics MultiDataset with .datasets dict and __getitem__/__len__."""

    def __init__(self, episode_paths: list[Path], all_data: dict[str, np.ndarray]):
        self.datasets = {
            f"ep_{i}": _FakeZarrDataset(p) for i, p in enumerate(episode_paths)
        }
        self._all_data = all_data
        self._n = all_data[next(iter(all_data))].shape[0]

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._all_data.items()}


def test_infer_norm_from_episodes_matches_dataloader() -> None:
    """Bulk zarr reader should produce identical stats to the DataLoader path."""
    rng = np.random.RandomState(42)
    n_episodes = 5
    frames_per_ep = 40
    total = n_episodes * frames_per_ep

    ee_all = rng.randn(total, 14).astype(np.float32)
    actions_all = rng.randn(total, 14).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ep_paths = []
        for i in range(n_episodes):
            s, e = i * frames_per_ep, (i + 1) * frames_per_ep
            ep_path = Path(tmpdir) / f"episode_{i:03d}.zarr"
            _create_zarr_episode(
                ep_path,
                {
                    "observations.state.ee_pose": ee_all[s:e],
                    "actions_cartesian": actions_all[s:e],
                },
                total_frames=frames_per_ep,
            )
            ep_paths.append(ep_path)

        concat_data = {
            "observations.state.ee_pose": ee_all,
            "actions_cartesian": actions_all,
        }
        dataset = _FakeMultiDataset(ep_paths, concat_data)

        schematic_dict = {
            "eva_bimanual": {
                "observations.state.ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "observations.state.ee_pose",
                },
                "actions_cartesian": {
                    "key_type": "action_keys",
                    "zarr_key": "actions_cartesian",
                },
            }
        }
        viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}

        bulk_schematic = DataSchematic(schematic_dict, viz_img_key)
        loader_schematic = DataSchematic(schematic_dict, viz_img_key)

        bulk_schematic.infer_norm_from_episodes(dataset, "eva_bimanual")

        loader_schematic.infer_norm_from_dataset(
            dataset,
            "eva_bimanual",
            sample_frac=1.0,
            seed=0,
            batch_size=total,
            num_workers=0,
        )

        embodiment_id = get_embodiment_id("eva_bimanual")
        expected = {
            "observations.state.ee_pose": _expected_stats(ee_all),
            "actions_cartesian": _expected_stats(actions_all),
        }

        for key, exp in expected.items():
            bulk_stats = bulk_schematic.norm_stats[embodiment_id][key]
            loader_stats = loader_schematic.norm_stats[embodiment_id][key]

            for stat_name, expected_value in exp.items():
                torch.testing.assert_close(
                    bulk_stats[stat_name],
                    expected_value,
                    msg=f"Bulk vs expected mismatch: {key}.{stat_name}",
                )
                torch.testing.assert_close(
                    loader_stats[stat_name],
                    expected_value,
                    msg=f"Loader vs expected mismatch: {key}.{stat_name}",
                )
                torch.testing.assert_close(
                    bulk_stats[stat_name],
                    loader_stats[stat_name],
                    msg=f"Bulk vs loader mismatch: {key}.{stat_name}",
                )

    print("PASSED: infer_norm_from_episodes matches infer_norm_from_dataset")
