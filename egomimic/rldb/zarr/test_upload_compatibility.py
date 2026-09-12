"""Existing upload/conversion/staging boundaries, with no service credentials.

S3 stores real bytes on local disk. Only the hardware-specific Aria extraction
is substituted; upload, converters, Zarr I/O, staging and discovery are real.
"""

import asyncio
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.validate import validate_episode
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.backfill_scripts import stage_processed_folder as staging
from egomimic.scripts.data_upload import abstract_upload
from egomimic.utils.aws import aws_data_utils
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash


class LocalS3:
    def __init__(self, root):
        self.root = root

    def upload_file(self, filename, bucket, key, **kwargs):
        path = self.root / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(filename).read_bytes())

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO((self.root / Bucket / Key).read_bytes())}

    def list_objects_v2(self, *, Bucket, Prefix, **kwargs):
        root = self.root / Bucket / Prefix
        return {"CommonPrefixes": [
            {"Prefix": f"{Prefix}{p.name}/"} for p in sorted(root.iterdir()) if p.is_dir()
        ]}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    s3 = LocalS3(tmp_path / "s3")
    monkeypatch.setattr(abstract_upload, "get_boto3_s3_client", lambda: s3)
    monkeypatch.setattr(aws_data_utils, "get_boto3_s3_client", lambda: s3)
    monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
    return s3


def _stage_and_load(path, storage, *, status=None):
    if status is not None:
        zarr.open_group(path, mode="a").attrs["data_status"] = status
    prefix = f"processed_v3/test_vendor/category/{path.name}"
    aws_data_utils.upload_dir_to_s3(str(path), "rldb", prefix)
    prefixes = list(staging.find_zarr_prefixes(storage, "processed_v3/test_vendor/"))
    rows = staging.read_batch(prefixes, "test_vendor", {"endpoint": None, "key": None, "secret": None})
    folder = storage.root / "rldb/processed_v3/test_vendor/category"
    resolver = LocalEpisodeResolver(folder, key_map={"pose": {"zarr_key": "left.obs_ee_pose"}})
    dataset = MultiDataset._from_resolver(resolver, mode="total")
    assert len(rows) == 1
    row = rows[0]
    assert row["episode_hash"] == path.stem
    assert row["embodiment"] == zarr.open_group(path, mode="r").attrs["embodiment"]
    assert row["task"] == "packing" and row["num_frames"] == 3
    assert row["zarr_processed_path"] == f"s3://rldb/{prefix}"
    assert row["has_annotations"] and row["created_at"].endswith("+00:00")
    assert len(dataset) == 3
    assert dataset[2]["pose"].shape == (7,)
    return row


@pytest.mark.parametrize("kind", ["aria", "eva"])
def test_raw_upload_metadata_conversion_staging_and_discovery(tmp_path, monkeypatch, storage, kind):
    from egomimic.scripts.data_upload.aria_uploader import aria_uploader
    from egomimic.scripts.data_upload.eva_uploader import eva_uploader

    raw = tmp_path / "raw"
    raw.mkdir()
    stamp = 1780000000000
    if kind == "aria":
        source = raw / "recording.vrs"
        source.write_bytes(b"VRS extraction boundary fixture")
        source.with_suffix(".vrs.json").write_text('{}')
        uploader = aria_uploader()
    else:
        source = raw / "recording.hdf5"
        with h5py.File(source, "w") as h5:
            h5["observations/images/front_img_1"] = np.full((3, 32, 32, 3), 40, np.uint8)
            for key in ("observations/eepose", "actions/eepose", "observations/joints", "actions/joints"):
                h5[key] = np.full((3, 14), 0.1)
        uploader = eva_uploader()
    uploader.local_dir = raw
    uploader.directory_prompted = True
    uploader.use_batch_metadata = uploader.batch_metadata_asked = True
    uploader.batch_metadata = {"task": "packing", "task_description": "Pack the objects", "lab": "test_lab"}
    monkeypatch.setattr(uploader, "get_timestamp_name", lambda p: stamp)
    monkeypatch.setattr(uploader, "set_directory", lambda: None)
    monkeypatch.setattr(uploader, "delete_dir", lambda: None)
    asyncio.run(uploader.run())
    uploaded = storage.root / f"rldb/raw_v2/{kind}"
    metadata = json.loads((uploaded / f"{stamp}_metadata.json").read_text())
    assert metadata["episode_hash"] == stamp and metadata["embodiment"] == kind
    assert metadata["task"] == "packing"
    assert "created_at" not in uploader.metadata_keys
    assert source.exists()
    episode_hash = timestamp_ms_to_episode_hash(stamp)
    output = tmp_path / "converted"
    output.mkdir()
    if kind == "aria":
        from egomimic.scripts.aria_process.aria_to_zarr import (
            AriaVRSExtractor,
            DatasetConverter,
        )

        pose = np.tile([0, 0, 1, 1, 0, 0, 0], (3, 1)).astype(float)
        extracted = {f"{side}.obs_ee_pose": pose for side in ("left", "right")}
        extracted.update({"images.front_1": np.zeros((3, 240, 320, 3), np.uint8), "obs_head_pose": pose})
        monkeypatch.setattr(AriaVRSExtractor, "process_episode", lambda **kw: extracted)
        converter = DatasetConverter(uploaded, fps=30, arm="both", convert_mano=False, height=240, width=320)
        path, _ = converter.extract_episode(uploaded / f"{stamp}.vrs", output_dir=output,
                                            dataset_name=episode_hash, task_name=metadata["task"], chunk_timesteps=2)
    else:
        from egomimic.scripts.eva_process.eva_to_zarr import convert_episode

        path, _ = convert_episode(uploaded / f"{stamp}.hdf5", output, episode_hash,
                                  "both", 30, task_name=metadata["task"], chunk_timesteps=2)
    # Existing annotation writer, including retained-frame intervals.
    writer = ZarrWriter(path, embodiment="human_bimanual" if kind == "aria" else "eva_bimanual",
                        intrinsics=zarr.open_group(path, mode="r").attrs["intrinsics"])
    writer.append_annotations("annotations", [("Pack objects", 0, 3)], mode="w")
    group = zarr.open_group(path, mode="a")
    assert group.attrs["features"]["annotations"]["shape"] == [1]
    group.attrs.pop("data_status", None)  # unchanged legacy absence
    assert validate_episode(path).ok, validate_episode(path).text()
    _stage_and_load(path, storage)


@pytest.mark.parametrize("status", [None, "complete", "structural_sample", "in_review"])
def test_staging_and_resolver_agree_on_direct_vendor_status(tmp_path, storage, status):
    path = tmp_path / "2026-09-12-12-00-00-000000.zarr"
    ZarrWriter.create_and_write(path, numeric_data={
        "left.obs_ee_pose": np.tile([0, 0, 1, 1, 0, 0, 0], (3, 1)).astype(float),
    }, embodiment="dexmate_bimanual", task_name="packing", chunk_timesteps=2,
        annotations=[("Pack objects", 0, 3)], intrinsics={"front_1": np.eye(3, 4)})
    group = zarr.open_group(path, mode="a")
    if status is None:
        group.attrs.pop("data_status", None)
    else:
        group.attrs["data_status"] = status
    prefix = f"processed_v3/test_vendor/{path.name}"
    aws_data_utils.upload_dir_to_s3(str(path), "rldb", prefix)
    rows = staging.read_batch([prefix + "/"], "test_vendor", {"endpoint": None, "key": None, "secret": None})
    resolver = LocalEpisodeResolver(storage.root / "rldb/processed_v3/test_vendor", key_map={})
    datasets = resolver._load_zarr_datasets(
        search_path=storage.root / "rldb/processed_v3/test_vendor", valid_folder_names={path.stem})
    eligible = status in (None, "complete")
    assert bool(rows) == bool(datasets) == eligible
    if eligible:
        assert rows[0]["embodiment"] == "dexmate_bimanual"
        assert rows[0]["has_annotations"]
        assert rows[0]["zarr_processed_path"] == f"s3://rldb/{prefix}"


def test_staging_command_inserts_vendor_metadata_into_isolated_database(tmp_path, storage, monkeypatch):
    from sqlalchemy import create_engine, event, text

    path = tmp_path / "2026-09-12-12-00-00-000000.zarr"
    ZarrWriter.create_and_write(path, numeric_data={"left.obs_ee_pose": np.tile(
        [0, 0, 1, 1, 0, 0, 0], (3, 1)).astype(float)}, embodiment="dexmate_bimanual",
        intrinsics={"front_1": np.eye(3, 4)}, annotations=[("Pack objects", 0, 3)],
        task_name="packing", chunk_timesteps=2)
    aws_data_utils.upload_dir_to_s3(str(path), "rldb", f"processed_v3/test_vendor/{path.name}")
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")

    @event.listens_for(engine, "connect")
    def attach_schema(connection, record):
        connection.execute("ATTACH DATABASE ? AS app", (str(tmp_path / "app.sqlite"),))

    # Only PostgreSQL-specific type/default syntax is adapted for SQLite.
    monkeypatch.setattr(staging, "_CREATE_TABLE", staging._CREATE_TABLE.replace("now()", "CURRENT_TIMESTAMP"))
    monkeypatch.setattr(staging, "_INSERT", staging._INSERT.replace("CAST(:raw_attrs AS JSONB)", ":raw_attrs"))
    monkeypatch.setattr(staging, "create_default_engine", lambda: engine)
    monkeypatch.setattr("sys.argv", ["stage_processed_folder", "--folder", "test_vendor", "--no-ray", "--workers", "1"])
    assert staging.main() == 0
    with engine.connect() as connection:
        row = connection.execute(text("SELECT * FROM app.staging_test_vendor")).mappings().one()
    assert row["episode_hash"] == path.stem and row["embodiment"] == "dexmate_bimanual"
    assert row["num_frames"] == 3 and row["has_annotations"]
    assert json.loads(row["raw_attrs"])["data_status"] == "complete"
    assert row["zarr_processed_path"].endswith(path.name)
    engine.dispose()
