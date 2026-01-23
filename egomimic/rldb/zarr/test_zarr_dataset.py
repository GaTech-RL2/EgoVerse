"""
Simple test script for ZarrDataset implementation.

Creates a mock dataset and tests basic functionality.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr

from egomimic.rldb.zarr import ZarrDataset, ZarrDatasetMetadata, RLDBZarrDataset


def create_mock_zarr_dataset(root: Path, num_episodes: int = 3, frames_per_episode: int = 10):
    """Create a mock Zarr dataset for testing."""
    root = Path(root)

    # Create directory structure
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    # Create info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "EVA_BIMANUAL",
        "total_episodes": num_episodes,
        "total_frames": num_episodes * frames_per_episode,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {
            "train": list(range(num_episodes - 1)),
            "valid": [num_episodes - 1],
        },
        "data_path": "data/episode_{episode_index:06d}/chunk-{chunk_idx:03d}.zarr",
        "video_path": None,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
            "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        },
    }

    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Create stats.json (empty for now)
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump({}, f)

    # Create tasks.jsonl
    with open(root / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Test task"}) + "\n")

    # Create episodes.jsonl
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx in range(num_episodes):
            ep_data = {
                "episode_index": ep_idx,
                "tasks": ["Test task"],
                "length": frames_per_episode,
            }
            f.write(json.dumps(ep_data) + "\n")

    # Create episode data
    global_idx = 0
    for ep_idx in range(num_episodes):
        ep_dir = root / "data" / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Create episode meta
        (ep_dir / "meta").mkdir(exist_ok=True)
        with open(ep_dir / "meta" / "episode_info.json", "w") as f:
            json.dump({"episode_index": ep_idx, "length": frames_per_episode}, f)

        # Create chunk-000.zarr
        chunk_path = ep_dir / "chunk-000.zarr"
        store = zarr.open(str(chunk_path), mode="w")

        # Generate mock data
        timestamps = np.arange(frames_per_episode, dtype=np.float32) / 30.0
        frame_indices = np.arange(frames_per_episode, dtype=np.int64)
        episode_indices = np.full(frames_per_episode, ep_idx, dtype=np.int64)
        indices = np.arange(global_idx, global_idx + frames_per_episode, dtype=np.int64)
        task_indices = np.zeros(frames_per_episode, dtype=np.int64)
        actions = np.random.randn(frames_per_episode, 7).astype(np.float32)
        states = np.random.randn(frames_per_episode, 14).astype(np.float32)

        # Write arrays
        store.create_dataset("timestamp", shape=timestamps.shape, data=timestamps)
        store.create_dataset("frame_index", shape=frame_indices.shape, data=frame_indices)
        store.create_dataset("episode_index", shape=episode_indices.shape, data=episode_indices)
        store.create_dataset("index", shape=indices.shape, data=indices)
        store.create_dataset("task_index", shape=task_indices.shape, data=task_indices)
        store.create_dataset("action", shape=actions.shape, data=actions)
        store.create_dataset("observation.state", shape=states.shape, data=states)

        global_idx += frames_per_episode

    return root


def test_zarr_dataset_metadata():
    """Test ZarrDatasetMetadata loading."""
    print("\n=== Testing ZarrDatasetMetadata ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_mock_zarr_dataset(Path(tmpdir) / "test_dataset")

        meta = ZarrDatasetMetadata(repo_id="test", root=root)

        print(f"  fps: {meta.fps}")
        print(f"  total_episodes: {meta.total_episodes}")
        print(f"  total_frames: {meta.total_frames}")
        print(f"  robot_type: {meta.robot_type}")
        print(f"  features: {list(meta.features.keys())}")
        print(f"  shapes: {meta.shapes}")

        assert meta.fps == 30
        assert meta.total_episodes == 3
        assert meta.total_frames == 30
        assert meta.robot_type == "EVA_BIMANUAL"

        print("  ✓ ZarrDatasetMetadata tests passed!")


def test_zarr_dataset_basic():
    """Test basic ZarrDataset functionality."""
    print("\n=== Testing ZarrDataset Basic ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_mock_zarr_dataset(Path(tmpdir) / "test_dataset")

        dataset = ZarrDataset(repo_id="test", root=root)

        print(f"  len(dataset): {len(dataset)}")
        print(f"  num_episodes: {dataset.num_episodes}")
        print(f"  num_frames: {dataset.num_frames}")

        assert len(dataset) == 30
        assert dataset.num_episodes == 3
        assert dataset.num_frames == 30

        # Test __getitem__
        item = dataset[0]
        print(f"  item keys: {list(item.keys())}")
        print(f"  action shape: {item['action'].shape}")
        print(f"  observation.state shape: {item['observation.state'].shape}")
        print(f"  episode_index: {item['episode_index']}")

        assert "action" in item
        assert "observation.state" in item
        assert item["action"].shape == (7,)
        assert item["observation.state"].shape == (14,)
        assert item["episode_index"] == 0

        # Test middle frame
        item = dataset[15]
        assert item["episode_index"] == 1  # Second episode

        # Test last frame
        item = dataset[29]
        assert item["episode_index"] == 2  # Third episode

        print("  ✓ ZarrDataset basic tests passed!")


def test_zarr_dataset_episode_filter():
    """Test ZarrDataset episode filtering."""
    print("\n=== Testing ZarrDataset Episode Filtering ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_mock_zarr_dataset(Path(tmpdir) / "test_dataset")

        # Load only episode 1
        dataset = ZarrDataset(repo_id="test", root=root, episodes=[1])

        print(f"  len(dataset) with episodes=[1]: {len(dataset)}")
        assert len(dataset) == 10

        item = dataset[0]
        assert item["episode_index"] == 1

        # Load episodes 0 and 2
        dataset = ZarrDataset(repo_id="test", root=root, episodes=[0, 2])
        print(f"  len(dataset) with episodes=[0, 2]: {len(dataset)}")
        assert len(dataset) == 20

        print("  ✓ ZarrDataset episode filtering tests passed!")


def test_rldb_zarr_dataset():
    """Test RLDBZarrDataset with RLDB features."""
    print("\n=== Testing RLDBZarrDataset ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_mock_zarr_dataset(Path(tmpdir) / "test_dataset")

        # Test train mode
        dataset = RLDBZarrDataset(
            repo_id="test",
            root=root,
            mode="train",
            valid_ratio=0.2,
        )

        print(f"  Train mode len: {len(dataset)}")
        print(f"  Embodiment: {dataset.embodiment}")

        # Should have 2 episodes (train split)
        assert len(dataset) == 20
        assert dataset.embodiment == 8  # EVA_BIMANUAL

        # Test valid mode
        dataset = RLDBZarrDataset(
            repo_id="test",
            root=root,
            mode="valid",
            valid_ratio=0.2,
        )

        print(f"  Valid mode len: {len(dataset)}")
        assert len(dataset) == 10  # 1 episode

        # Test percent mode
        dataset = RLDBZarrDataset(
            repo_id="test",
            root=root,
            mode="percent",
            percent=0.5,
        )

        print(f"  Percent mode (50%) len: {len(dataset)}")
        assert len(dataset) == 10  # 50% of 20 train frames

        # Test task string injection
        dataset = RLDBZarrDataset(
            repo_id="test",
            root=root,
            mode="train",
            use_task_string=True,
            task_string="Pick up the cube",
        )

        item = dataset[0]
        print(f"  Task string: {item.get('high_level_language_prompt', 'NOT FOUND')}")
        assert item["high_level_language_prompt"] == "Pick up the cube"

        print("  ✓ RLDBZarrDataset tests passed!")


def test_hf_dataset_compatibility():
    """Test hf_dataset compatibility shim."""
    print("\n=== Testing hf_dataset Compatibility ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_mock_zarr_dataset(Path(tmpdir) / "test_dataset")

        dataset = ZarrDataset(repo_id="test", root=root)

        # Access hf_dataset property
        hf_ds = dataset.hf_dataset

        print(f"  len(hf_dataset): {len(hf_ds)}")
        assert len(hf_ds) == 30

        # Test __getitem__
        item = hf_ds[0]
        print(f"  hf_dataset[0] keys: {list(item.keys())}")
        assert "action" in item

        # Test select
        items = hf_ds.select([0, 5, 10])
        print(f"  hf_dataset.select([0, 5, 10]) len: {len(items)}")
        assert len(items) == 3

        print("  ✓ hf_dataset compatibility tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running ZarrDataset Tests")
    print("=" * 50)

    try:
        test_zarr_dataset_metadata()
        test_zarr_dataset_basic()
        test_zarr_dataset_episode_filter()
        test_rldb_zarr_dataset()
        test_hf_dataset_compatibility()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
