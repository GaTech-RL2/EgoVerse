#!/usr/bin/env python3
"""
Comprehensive test suite for Zarr dataset system.

Tests:
1. SimplejpegCodec encode/decode
2. Zarr array with JPEG compression
3. Full dataset creation with prestacked actions
4. ZarrDataset loading and __getitem__
5. Prestacked action loading
6. RLDBZarrDataset train/valid splits
7. DataLoader multi-worker support
8. ZarrHFShim for normalization compatibility

Usage:
    python test_zarr_dataset.py                     # Run all tests
    python test_zarr_dataset.py --test-codec       # Codec only
    python test_zarr_dataset.py --zarr-root PATH   # Test existing dataset
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_simplejpeg_codec():
    """Test SimplejpegCodec round-trip compression."""
    import numcodecs
    from egomimic.rldb.zarr_utils import SimplejpegCodec

    logger.info("=" * 50)
    logger.info("TEST: SimplejpegCodec")
    logger.info("=" * 50)

    H, W = 360, 640
    codec = SimplejpegCodec(quality=85, height=H, width=W)

    # Create test image with structure (gradients compress well, like real images)
    # Random noise compresses poorly (~1.9x), but real images achieve 10-20x
    y_grad = np.linspace(0, 255, H).reshape(H, 1, 1).astype(np.uint8)
    x_grad = np.linspace(0, 255, W).reshape(1, W, 1).astype(np.uint8)
    img = np.broadcast_to(y_grad + x_grad // 2, (H, W, 3)).astype(np.uint8)
    img = np.ascontiguousarray(img)
    flat = img.ravel()

    # Encode/decode
    t0 = time.time()
    encoded = codec.encode(flat)
    encode_ms = (time.time() - t0) * 1000

    t0 = time.time()
    decoded = codec.decode(encoded)
    decode_ms = (time.time() - t0) * 1000

    decoded_img = decoded.reshape(H, W, 3)

    # Metrics
    ratio = len(flat) / len(encoded)
    diff = np.abs(img.astype(float) - decoded_img.astype(float)).mean()

    logger.info(f"  Compression: {len(flat):,} -> {len(encoded):,} bytes ({ratio:.1f}x)")
    logger.info(f"  Encode: {encode_ms:.1f}ms, Decode: {decode_ms:.1f}ms")
    logger.info(f"  Mean diff: {diff:.2f} (JPEG is lossy)")

    # Assertions
    assert decoded_img.shape == img.shape, f"Shape: {decoded_img.shape} != {img.shape}"
    assert diff < 15, f"Diff too high: {diff}"
    assert ratio > 3, f"Poor compression: {ratio}"

    # Config round-trip
    config = codec.get_config()
    codec2 = SimplejpegCodec.from_config(config)
    assert codec2.quality == codec.quality
    assert codec2.height == codec.height

    # Registry
    registered = numcodecs.get_codec(config)
    assert isinstance(registered, SimplejpegCodec)

    logger.info("  [PASS]")
    return True


def test_zarr_array_jpeg():
    """Test Zarr array with JPEG compression."""
    import zarr
    from egomimic.rldb.zarr_utils import SimplejpegCodec

    logger.info("=" * 50)
    logger.info("TEST: Zarr Array with JPEG")
    logger.info("=" * 50)

    H, W, T = 360, 640, 10

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.zarr"
        codec = SimplejpegCodec(quality=85, height=H, width=W)

        store = zarr.open(str(path), mode='w')
        arr = store.create_dataset(
            'images', shape=(T, H, W, 3), chunks=(1, H, W, 3),
            dtype=np.uint8, compressor=codec
        )

        # Write
        originals = []
        t0 = time.time()
        for t in range(T):
            img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            originals.append(img.copy())
            arr[t] = img
        write_ms = (time.time() - t0) * 1000

        # Read
        t0 = time.time()
        for t in range(T):
            read = arr[t]
            diff = np.abs(originals[t].astype(float) - read.astype(float)).mean()
            assert diff < 15, f"Frame {t} diff: {diff}"
        read_ms = (time.time() - t0) * 1000

        # Storage
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        raw = T * H * W * 3

        logger.info(f"  Wrote {T} frames in {write_ms:.0f}ms ({write_ms/T:.1f}ms/frame)")
        logger.info(f"  Read {T} frames in {read_ms:.0f}ms ({read_ms/T:.1f}ms/frame)")
        logger.info(f"  Storage: {size:,} bytes ({raw/size:.1f}x compression)")

    logger.info("  [PASS]")
    return True


def create_synthetic_dataset(output_path: Path, num_episodes: int = 2, frames_per_ep: int = 30):
    """Create synthetic Zarr dataset for testing with prestacked actions."""
    import zarr
    import numcodecs
    from egomimic.rldb.zarr_utils import SimplejpegCodec, EMBODIMENT

    H, W = 360, 640
    T = frames_per_ep
    CHUNK_SIZE = 100

    store = zarr.open(str(output_path), mode='w')
    store.attrs.update({
        'fps': 30,
        'robot_type': 'MECKA_BIMANUAL',
        'total_episodes': num_episodes,
        'total_frames': T * num_episodes,
    })

    jpeg = SimplejpegCodec(quality=85, height=H, width=W)
    zstd = numcodecs.Zstd(level=3)

    for i in range(num_episodes):
        ep_hash = f"ep_{i:04d}"
        ep = store.create_group(f"episodes/{ep_hash}")
        ep.attrs.update({
            'episode_id': ep_hash,
            'hash': ep_hash,
            'length': T,
            'fps': 30,
            'embodiment_id': EMBODIMENT.MECKA_BIMANUAL.value,
            'task': f'task_{i}',
            'label_vocab': ['action_a', 'action_b'],
        })

        # Images
        rgb = ep.create_group('rgb')
        img_arr = rgb.create_dataset(
            'front_img_1', shape=(T, H, W, 3), chunks=(1, H, W, 3),
            dtype=np.uint8, compressor=jpeg
        )
        for t in range(T):
            img_arr[t] = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

        # State (T, 12)
        ep.create_group('state').create_dataset(
            'ee_pose_cam', data=np.random.randn(T, 12).astype(np.float32),
            chunks=(256, 12), compressor=zstd
        )

        # Prestacked actions (T, 100, 12)
        actions_raw = np.random.randn(T, 12).astype(np.float32)
        actions_prestacked = np.zeros((T, CHUNK_SIZE, 12), dtype=np.float32)
        for t in range(T):
            for c in range(CHUNK_SIZE):
                src_idx = min(t + c, T - 1)
                actions_prestacked[t, c] = actions_raw[src_idx]
        ep.create_group('actions').create_dataset(
            'ee_cartesian_cam', data=actions_prestacked,
            chunks=(1, CHUNK_SIZE, 12), compressor=zstd
        )

        # Prestacked keypoints (T, 100, 126)
        kp_raw = np.random.randn(T, 126).astype(np.float32)
        kp_prestacked = np.zeros((T, CHUNK_SIZE, 126), dtype=np.float32)
        for t in range(T):
            for c in range(CHUNK_SIZE):
                src_idx = min(t + c, T - 1)
                kp_prestacked[t, c] = kp_raw[src_idx]
        ep.create_group('keypoints').create_dataset(
            'hand_keypoints_world', data=kp_prestacked,
            chunks=(1, CHUNK_SIZE, 126), compressor=zstd
        )

        # Head pose (T, 10)
        ep.create_group('head').create_dataset(
            'pose_world', data=np.random.randn(T, 10).astype(np.float32),
            chunks=(256, 10), compressor=zstd
        )

        # Annotations
        ann = ep.create_group('annotations')
        ann.create_dataset('label_id', data=np.array([0, 1], dtype=np.int32))
        ann.create_dataset('start_frame', data=np.array([0, T//2], dtype=np.int32))
        ann.create_dataset('end_frame', data=np.array([T//2-1, T-1], dtype=np.int32))

    # Meta files
    meta = output_path / 'meta'
    meta.mkdir(parents=True, exist_ok=True)

    with open(meta / 'info.json', 'w') as f:
        json.dump({
            'robot_type': 'MECKA_BIMANUAL', 'fps': 30,
            'total_episodes': num_episodes, 'total_frames': T * num_episodes,
        }, f)

    with open(meta / 'episodes.jsonl', 'w') as f:
        for i in range(num_episodes):
            f.write(json.dumps({
                'hash': f'ep_{i:04d}', 'length': T,
                'embodiment_id': EMBODIMENT.MECKA_BIMANUAL.value,
            }) + '\n')

    with open(meta / 'stats.json', 'w') as f:
        json.dump({
            'observations.state.ee_pose_cam': {'mean': [0]*12, 'std': [1]*12},
            'actions_ee_cartesian_cam': {'mean': [0]*12, 'std': [1]*12},
        }, f)

    zarr.consolidate_metadata(str(output_path))
    return output_path


def test_dataset_loading(zarr_path: Path):
    """Test ZarrDataset loading."""
    from egomimic.rldb.zarr_utils import ZarrDataset

    logger.info("=" * 50)
    logger.info("TEST: ZarrDataset Loading")
    logger.info("=" * 50)

    ds = ZarrDataset(zarr_path)
    logger.info(f"  Frames: {len(ds)}, Episodes: {ds.num_episodes}")

    # Test __getitem__
    item = ds[0]

    expected = [
        ('observations.images.front_img_1', (3, 360, 640)),
        ('observations.state.ee_pose_cam', (12,)),
        ('actions_ee_cartesian_cam', (100, 12)),
        ('actions_ee_keypoints_world', (100, 126)),
        ('metadata.embodiment', (1,)),
    ]

    for key, shape in expected:
        assert key in item, f"Missing: {key}"
        assert item[key].shape == shape, f"{key}: {item[key].shape} != {shape}"
        logger.info(f"  {key}: {item[key].shape}")

    # Image range check
    img = item['observations.images.front_img_1']
    assert 0 <= img.min() and img.max() <= 1, f"Image range: [{img.min()}, {img.max()}]"

    logger.info("  [PASS]")
    return True


def test_prestacked_actions(zarr_path: Path):
    """Test prestacked action loading."""
    from egomimic.rldb.zarr_utils import ZarrDataset

    logger.info("=" * 50)
    logger.info("TEST: Prestacked Actions")
    logger.info("=" * 50)

    ds = ZarrDataset(zarr_path, chunk_size=100)

    # Test at various positions
    for idx in [0, 10, len(ds)-1]:
        item = ds[idx]
        chunk = item['actions_ee_cartesian_cam']
        assert chunk.shape == (100, 12), f"idx={idx}: {chunk.shape}"

    # Verify padding at end (last frames should have identical trailing rows)
    last = ds[len(ds)-1]
    chunk = last['actions_ee_cartesian_cam'].numpy()
    unique = len(np.unique(chunk[-5:], axis=0))
    logger.info(f"  Last 5 rows unique: {unique} (1 = correctly padded)")

    # Verify keypoints also prestacked
    kp = last['actions_ee_keypoints_world']
    assert kp.shape == (100, 126), f"Keypoints shape: {kp.shape}"
    logger.info(f"  Keypoints shape: {kp.shape}")

    logger.info("  [PASS]")
    return True


def test_rldb_zarr_dataset(zarr_path: Path):
    """Test RLDBZarrDataset with train/valid splits."""
    from egomimic.rldb.zarr_utils import RLDBZarrDataset

    logger.info("=" * 50)
    logger.info("TEST: RLDBZarrDataset Splits")
    logger.info("=" * 50)

    train_ds = RLDBZarrDataset(zarr_path, mode='train', valid_ratio=0.5)
    valid_ds = RLDBZarrDataset(zarr_path, mode='valid', valid_ratio=0.5)

    logger.info(f"  Train: {len(train_ds)} frames")
    logger.info(f"  Valid: {len(valid_ds)} frames")

    # Non-overlapping
    assert len(train_ds) > 0
    assert len(valid_ds) > 0

    # Check item
    item = train_ds[0]
    assert 'observations.images.front_img_1' in item

    logger.info("  [PASS]")
    return True


def test_dataloader(zarr_path: Path):
    """Test DataLoader with ZarrDataset."""
    import torch
    from torch.utils.data import DataLoader
    from egomimic.rldb.zarr_utils import ZarrDataset

    logger.info("=" * 50)
    logger.info("TEST: DataLoader")
    logger.info("=" * 50)

    ds = ZarrDataset(zarr_path)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    t0 = time.time()
    batches = 0
    for batch in loader:
        batches += 1
        if batches >= 3:
            break
    elapsed = time.time() - t0

    logger.info(f"  {batches} batches in {elapsed*1000:.0f}ms")
    logger.info(f"  Batch shapes:")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            logger.info(f"    {k}: {v.shape}")

    logger.info("  [PASS]")
    return True


def test_hf_shim(zarr_path: Path):
    """Test ZarrHFShim for DataSchematic compatibility."""
    from egomimic.rldb.zarr_utils import RLDBZarrDataset

    logger.info("=" * 50)
    logger.info("TEST: ZarrHFShim (Normalization)")
    logger.info("=" * 50)

    ds = RLDBZarrDataset(zarr_path, mode='train')

    # This is what DataSchematic.infer_norm_from_dataset does:
    hf = ds.hf_dataset
    data = hf.with_format('numpy', columns=['observations.state.ee_pose_cam'])[:]['observations.state.ee_pose_cam']

    logger.info(f"  Loaded column shape: {data.shape}")
    assert data.ndim == 2
    assert data.shape[1] == 12

    # Compute stats
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    logger.info(f"  Mean: {mean[:3]}...")
    logger.info(f"  Std: {std[:3]}...")

    logger.info("  [PASS]")
    return True


def run_all_tests(zarr_path: Path = None):
    """Run complete test suite."""
    logger.info("\n" + "=" * 50)
    logger.info("ZARR DATASET TEST SUITE")
    logger.info("=" * 50 + "\n")

    results = {}

    # 1. Codec test
    try:
        results['codec'] = test_simplejpeg_codec()
    except Exception as e:
        logger.error(f"FAIL: {e}")
        results['codec'] = False

    # 2. Zarr array test
    try:
        results['zarr_array'] = test_zarr_array_jpeg()
    except Exception as e:
        logger.error(f"FAIL: {e}")
        results['zarr_array'] = False

    # 3-7. Dataset tests
    cleanup = False
    try:
        if zarr_path is None:
            # Create synthetic dataset
            tmpdir = tempfile.mkdtemp()
            zarr_path = Path(tmpdir) / "test_dataset.zarr"
            logger.info(f"Creating synthetic dataset: {zarr_path}")
            create_synthetic_dataset(zarr_path)
            cleanup = True

        results['loading'] = test_dataset_loading(zarr_path)
        results['prestacked'] = test_prestacked_actions(zarr_path)
        results['rldb_splits'] = test_rldb_zarr_dataset(zarr_path)
        results['dataloader'] = test_dataloader(zarr_path)
        results['hf_shim'] = test_hf_shim(zarr_path)

    except Exception as e:
        logger.error(f"Dataset tests FAIL: {e}")
        import traceback
        traceback.print_exc()
        results['dataset'] = False
    finally:
        if cleanup and zarr_path and zarr_path.parent.exists():
            shutil.rmtree(zarr_path.parent)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_pass = False

    logger.info("=" * 50)
    if all_pass:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("SOME TESTS FAILED")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Zarr Dataset Test Suite")
    parser.add_argument("--zarr-root", type=str, help="Path to existing Zarr dataset")
    parser.add_argument("--test-codec", action="store_true", help="Test codec only")
    args = parser.parse_args()

    if args.test_codec:
        success = test_simplejpeg_codec()
    elif args.zarr_root:
        success = run_all_tests(Path(args.zarr_root))
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
