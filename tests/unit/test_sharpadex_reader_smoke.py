"""A converted SharpaDex episode stages and loads with the unchanged readers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
import zarr

from egomimic.rldb.embodiment.dexmate import Dexmate
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    _jpeg_probe_failed,
)
from egomimic.scripts.backfill_scripts import stage_processed_folder as staging
from egomimic.scripts.sharpa_process import sharpa_to_zarr
from egomimic.scripts.sharpa_process.sharpa_tracks import SharpaKinematics
from egomimic.scripts.sharpa_process.test_sharpa_to_zarr import (
    LENGTHS,
    OPTICAL,
    RIG,
    SIZE,
    _mount,
    _plan_manifest,
    build_sharpadex,
)
from egomimic.utils.aws import aws_data_utils


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """One calibrated SharpaDex episode converted under the provenance policy."""
    root = tmp_path_factory.mktemp("smoke")
    release = build_sharpadex(root / "release", SharpaKinematics())
    manifest = _plan_manifest(release, root)

    def camera(parent, mount):
        return {
            "K": [[70.0, 0.0, 47.5], [0.0, 70.0, 47.5], [0.0, 0.0, 1.0]],
            "model": "PINHOLE",
            "distortion": [],
            "resolution": [SIZE, SIZE],
            "rectified": True,
            "parent_link": parent,
            "parent_link_T_cam": mount.tolist(),
        }

    calibration = root / "rig.yaml"
    calibration.write_text(
        yaml.safe_dump(
            {
                RIG: {
                    "head_left": camera(
                        "head_base_link", _mount(OPTICAL, [0.05, 0, 0.03])
                    ),
                    "wrist_left": camera(
                        "left_hand_wrist", _mount(np.eye(3), [0, 0, 0.04])
                    ),
                    "wrist_right": camera(
                        "right_arm_Link_7", _mount(OPTICAL, [0, 0, 0.05])
                    ),
                }
            }
        )
    )
    out = root / "category"
    args = ["--profile", "sharpadex", "--dataset-root", str(release)]
    args += ["--manifest", str(manifest), "--output-dir", str(out), "--episodes", "1"]
    args += ["--calibration", str(calibration), "--status-policy", "provenance"]
    assert sharpa_to_zarr.main([*args, "--validate"]) == 0
    [path] = list(out.glob("*.zarr"))
    return path


def test_the_episode_stages_like_any_vendor_episode(converted, tmp_path, monkeypatch):
    from egomimic.rldb.zarr.test_upload_compatibility import LocalS3

    s3 = LocalS3(tmp_path / "s3")
    monkeypatch.setattr(aws_data_utils, "get_boto3_s3_client", lambda: s3)
    monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
    attrs = zarr.open_group(converted, mode="r").attrs
    assert attrs["data_status"] == "complete", attrs["conversion_report"]
    prefix = f"processed_v3/sharpa_dex/{converted.name}"
    aws_data_utils.upload_dir_to_s3(str(converted), "rldb", prefix)
    prefixes = list(staging.find_zarr_prefixes(s3, "processed_v3/sharpa_dex/"))
    rows = staging.read_batch(
        prefixes, "sharpa_dex", {"endpoint": None, "key": None, "secret": None}
    )
    [row] = rows
    assert row["episode_hash"] == converted.stem
    assert row["embodiment"] == "dexmate_bimanual" and row["task"] == "iron_garment"
    assert row["num_frames"] == LENGTHS[1] and row["has_annotations"]
    assert json.loads(row["raw_attrs"])["license"] == "CC-BY-NC-4.0"


def test_the_unchanged_dexmate_keymap_reads_every_frame(converted):
    keymap = Dexmate.get_keymap("keypoints", annotation_key="annotations")
    resolver = LocalEpisodeResolver(
        converted.parent,
        key_map=keymap,
        transform_list=Dexmate.get_transform_list("keypoints_headframe_quat"),
    )
    dataset = MultiDataset._from_resolver(resolver, mode="total")
    assert len(dataset) == LENGTHS[1]
    [inner] = dataset.datasets.values()
    assert _jpeg_probe_failed(inner) is None
    first = dataset[0]
    for index in (LENGTHS[1] // 2, LENGTHS[1] - 1):
        item = dataset[index]
        assert item.keys() == first.keys()
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                assert value.shape == first[key].shape, key
                assert np.isfinite(value).all(), key
    assert "actions_keypoints" in first
    images = [k for k in first if "images" in k and "front" in k]
    assert images and first[images[0]].shape == (3, SIZE, SIZE)
    assert inner._annotation_text_for_frame(0) == ["Grip the iron head."]
    assert inner._annotation_text_for_frame(LENGTHS[1] - 1) == ["Iron the garment."]


def test_tactile_images_decode_through_the_writer_s_image_keys(converted):
    import simplejpeg

    group = zarr.open_group(converted, mode="r")
    features = group.attrs["features"]
    assert features["left.obs_tactile_image"]["dtype"] == "jpeg"
    strip = simplejpeg.decode_jpeg(
        group["left.obs_tactile_image"][LENGTHS[1] - 1 : LENGTHS[1]][0],
        colorspace="RGB",
    )
    # The loader decodes jpeg-typed keys as RGB; a grayscale strip comes back
    # with three equal channels.
    assert strip.shape == (240, 1200, 3)
    assert np.abs(strip[..., 0].astype(int) - strip[..., 2]).max() == 0
