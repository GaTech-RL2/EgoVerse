#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of ZarrWriter for creating Zarr v3 episode stores.

This demonstrates the main use cases:
1. Basic writing with separate numeric and image data
2. One-shot convenience method with auto-detection
3. Incremental writing for step-by-step episode building
"""

import numpy as np
from pathlib import Path
from egomimic.rldb.zarr import ZarrWriter, ZarrEpisode


def example_basic_writing():
    """Example 1: Basic writing with explicit data separation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Writing")
    print("=" * 60)

    # Create test data
    numeric_data = {
        "observations.state": np.random.randn(100, 10),
        "actions_joints": np.random.randn(100, 7),
    }
    image_data = {
        "observations.images.cam1": np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8),
    }

    # Write episode
    writer = ZarrWriter(
        root_path="/tmp/zarr_examples",
        episode_index=0,
        total_frames=100,
        fps=30,
        robot_type="eva_bimanual",
        task="pick_and_place",
        jpeg_quality=95,
    )
    writer.write(numeric_data=numeric_data, image_data=image_data)

    print(f"Created episode at: {writer.episode_path}")

    # Read back with ZarrEpisode
    episode = ZarrEpisode(writer.episode_path)
    print(f"Episode has {len(episode)} frames")
    print(f"Available keys: {episode.keys}")


def example_convenience_method():
    """Example 2: One-shot convenience method with auto-detection."""
    print("\n" + "=" * 60)
    print("Example 2: Convenience Method")
    print("=" * 60)

    # Combined data - images will be auto-detected
    all_data = {
        "observations.state": np.random.randn(50, 10),
        "observations.images.cam1": np.random.randint(0, 255, (50, 480, 640, 3), dtype=np.uint8),
        "observations.images.cam2": np.random.randint(0, 255, (50, 240, 320, 3), dtype=np.uint8),
        "actions": np.random.randn(50, 7),
    }

    # One-shot write
    path = ZarrWriter.create_and_write(
        data=all_data,
        root_path="/tmp/zarr_examples",
        episode_index=1,
        auto_detect_images=True,  # Automatically detect 4D arrays with shape (..., H, W, 3)
        fps=30,
        robot_type="eva_bimanual",
        task="bimanual_assembly",
        jpeg_quality=90,
    )

    print(f"Created episode at: {path}")

    episode = ZarrEpisode(path)
    print(f"Episode has {len(episode)} frames")
    print(f"Available keys: {episode.keys}")


def example_incremental_writing():
    """Example 3: Incremental writing for step-by-step building."""
    print("\n" + "=" * 60)
    print("Example 3: Incremental Writing")
    print("=" * 60)

    # Create writer
    writer = ZarrWriter(
        episode_path="/tmp/zarr_examples/episode_000002.zarr",
        episode_index=2,
        total_frames=30,
        fps=30,
        robot_type="eva_bimanual",
        task="drawer_opening",
    )

    # Write arrays one at a time
    writer.write_incremental("observations.state", np.random.randn(30, 10), is_image=False)
    writer.write_incremental("actions", np.random.randn(30, 7), is_image=False)
    writer.write_incremental(
        "observations.images.cam1",
        np.random.randint(0, 255, (30, 480, 640, 3), dtype=np.uint8),
        is_image=True,
    )

    # Must call finalize after all incremental writes
    writer.finalize(metadata_override={"custom_field": "custom_value"})

    print(f"Created episode at: {writer.episode_path}")

    episode = ZarrEpisode(writer.episode_path)
    print(f"Episode has {len(episode)} frames")
    print(f"Available keys: {episode.keys}")
    print(f"Metadata: {episode.metadata}")


def example_custom_chunking():
    """Example 4: Custom chunking and sharding settings."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Chunking")
    print("=" * 60)

    numeric_data = {
        "observations.state": np.random.randn(100, 10),
    }

    # Explicit timestep-based chunking
    writer1 = ZarrWriter(
        root_path="/tmp/zarr_examples",
        episode_index=3,
        total_frames=100,
        chunk_timesteps=10,  # Exactly 10 timesteps per chunk
        enable_sharding=True,
    )
    writer1.write(numeric_data=numeric_data)
    print(f"Created episode with chunk_timesteps=10 at: {writer1.episode_path}")

    # Size-based chunking (default)
    writer2 = ZarrWriter(
        root_path="/tmp/zarr_examples",
        episode_index=4,
        total_frames=100,
        chunk_size_mb=1.0,  # Target ~1MB per chunk
        enable_sharding=True,
    )
    writer2.write(numeric_data=numeric_data)
    print(f"Created episode with chunk_size_mb=1.0 at: {writer2.episode_path}")

    # Without sharding (more chunks stored separately)
    writer3 = ZarrWriter(
        root_path="/tmp/zarr_examples",
        episode_index=5,
        total_frames=100,
        enable_sharding=False,  # Each chunk as separate file
    )
    writer3.write(numeric_data=numeric_data)
    print(f"Created episode without sharding at: {writer3.episode_path}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# ZarrWriter Usage Examples")
    print("#" * 60)

    example_basic_writing()
    example_convenience_method()
    example_incremental_writing()
    example_custom_chunking()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
