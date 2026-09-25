"""Small CPU fixtures for deterministic donor selection and byte integrity."""

import copy
import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from astra_reversal import image_donor_bank as module
from astra_reversal.interpolation_catalog import donor_for
from astra_reversal.records import digest, file_sha256


def source_pixels(frame, camera):
    image = np.empty((256, 256, 3), np.uint8)
    image[..., 0] = frame if camera == module.CAMERAS[0] else 200 - frame
    image[..., 1] = np.arange(256, dtype=np.uint8)[None]
    image[..., 2] = np.arange(256, dtype=np.uint8)[:, None]
    image[:20, :30] = (10, 20, 30)
    image[-20:, -30:] = (240, 230, 220)
    return image


def encoded_frame(frame, camera):
    output = io.BytesIO()
    Image.fromarray(source_pixels(frame, camera)).save(output, format="PNG")
    return {"bytes": output.getvalue(), "path": "embedded.png"}


def fixture_resize(image, height, width):
    assert image.shape == (256, 256, 3) and (height, width) == (224, 224)
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
    )


@pytest.fixture(scope="module")
def library_fixture(tmp_path_factory):
    """Actual parquet parsing; only upstream metadata and resize loading are mocked."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path_factory.mktemp("donor-parquet")
    dataset, output = root / "dataset", root / "library"
    donor = donor_for("10")
    relative = "data/chunk-000/episode_000379.parquet"
    parquet = dataset / relative
    parquet.parent.mkdir(parents=True)
    rows = [
        {
            "image": encoded_frame(frame, module.CAMERAS[0]),
            "wrist_image": encoded_frame(frame, module.CAMERAS[1]),
            "state": [0.0] * 8,
            "episode_index": 379,
            "frame_index": frame,
            "task_index": 10,
        }
        for frame in range(112)
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    (dataset / "file_index.json").write_text(
        json.dumps(
            {
                "id": module.DATASET_REPO,
                "sha": module.DATASET_REVISION,
                "siblings": [
                    {
                        "rfilename": relative,
                        "size": parquet.stat().st_size,
                        "lfs": {"sha256": file_sha256(parquet)},
                    }
                ],
            }
        )
    )
    source = {
        "source_id": "10",
        "prompt": donor.prompt,
        "dataset_task_index": donor.dataset_task_index,
        "episodes": [
            {"episode_index": 379, "frame_count": 112, "relative_path": relative}
        ],
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "demonstration_plan", lambda _: {"sources": [source]})
        patch.setattr(module, "_load_resizer", lambda _: fixture_resize)
        library = module.build_library(dataset, output, source_ids=["10"])
    return dataset, output, library


def reseal(directory, manifest):
    manifest["library_id"] = digest(
        {key: value for key, value in manifest.items() if key != "library_id"}
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))


def copy_library(tmp_path, fixture):
    directory = tmp_path / "library"
    shutil.copytree(fixture[1], directory)
    return directory, json.loads((directory / "manifest.json").read_text())


def test_phase_sampling_includes_endpoints_and_ties_round_up():
    assert module.phase_indices(112) == [0, 28, 56, 83, 111]
    assert module.phase_indices(7) == [0, 2, 3, 5, 6]
    assert module.phase_indices(5) == [0, 1, 2, 3, 4]
    for invalid in (4, True, 5.0):
        with pytest.raises(ValueError, match="five frames"):
            module.phase_indices(invalid)
    for invalid in ([], ["10", "10"], ["13", "10"], ["ood"], [10]):
        with pytest.raises(ValueError):
            module._source_ids(invalid)


def test_builder_preserves_camera_orientation_and_exact_selected_frames(
    library_fixture,
):
    _, output, library = library_fixture
    catalog = library.catalog()
    assert len(catalog) == 5
    assert len([p for p in output.rglob("*") if p.is_file()]) == 14
    for column, (row, frame) in enumerate(
        zip(catalog, (0, 28, 56, 83, 111), strict=True)
    ):
        assert row["donor_id"] == f"std10-e379-f{frame}"
        assert row["preview_position"] == {"row": 0, "column": column}
        assert row["phase"] == {"numerator": column, "denominator": 4}
        assert row["sample_sha256"] == digest(
            {k: v for k, v in row.items() if k not in ("sample_sha256", "library_id")}
        )
        for camera in module.CAMERAS:
            resolved = library.resolve(row["donor_id"], camera)
            expected = fixture_resize(source_pixels(frame, camera), 224, 224)
            np.testing.assert_array_equal(resolved.pixels, expected)
            assert resolved.provenance["source_pixels_sha256"] == digest(
                source_pixels(frame, camera)
            )
            assert resolved.provenance["pixels_sha256"] == digest(expected)
            assert resolved.provenance["library_id"] == library.library_id
    assert (
        library.metadata()["preprocessing"]["stored_demo_actions_state_or_object_poses"]
        is False
    )
    assert library.metadata()["plan"]["ood_demonstrations"] == 0


def test_loaded_pixels_and_returned_metadata_cannot_mutate_library(library_fixture):
    _, output, _ = library_fixture
    library = module.load_library(output)
    first = library.catalog()[0]
    image = library.resolve(first["donor_id"], module.CAMERAS[0])
    with pytest.raises(ValueError):
        image.pixels[0, 0] = 0
    with pytest.raises(ValueError):
        image.pixels.setflags(write=True)
    with pytest.raises(AttributeError):
        library.library_id = "other"
    image.provenance["phase"]["numerator"] = 99
    library.metadata()["samples"][0]["frame_index"] = 99
    first["cameras"].clear()
    assert image.provenance["phase"]["numerator"] == 0
    assert len(library.catalog()[0]["cameras"]) == 2
    for donor, camera in (("unknown", module.CAMERAS[0]), (image.donor_id, "other")):
        with pytest.raises(ValueError, match="Unknown donor"):
            library.resolve(donor, camera)


def test_contact_sheets_bind_each_camera_to_exact_labeled_grid(library_fixture):
    library = library_fixture[2]
    for sheet in library.contact_sheets():
        assert digest(sheet["image"]) == sheet["sha256"]
        assert sheet["image"].shape == (204, 600, 3)
        for sample in library.catalog():
            column = sample["preview_position"]["column"]
            actual = sheet["image"][60:172, 20 + 112 * column : 132 + 112 * column]
            expected = np.asarray(
                Image.fromarray(
                    library.resolve(sample["donor_id"], sheet["camera"]).pixels
                ).resize((112, 112), Image.Resampling.BILINEAR)
            )
            np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError):
            sheet["image"].setflags(write=True)
        sheet["layout"]["rows"] = 99
    assert library.contact_sheets()[0]["layout"]["rows"] == 1


@pytest.mark.parametrize("change", ["pixels", "camera", "donor_id", "library_id"])
def test_donor_object_rejects_mismatched_provenance(change):
    pixels = np.zeros(module.IMAGE_SHAPE, np.uint8)
    provenance = {
        "pixels_sha256": digest(pixels),
        "library_id": "a" * 64,
        "sample_sha256": "b" * 64,
        "donor_id": "test",
        "camera": module.CAMERAS[0],
    }
    if change == "pixels":
        pixels[0, 0] = 1
    else:
        provenance[change] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        module.DonorImage("test", module.CAMERAS[0], pixels, provenance)


@pytest.mark.parametrize("change", ["missing", "bytes", "unreferenced"])
def test_loader_rejects_missing_corrupt_or_unreferenced_bytes(
    library_fixture, tmp_path, change
):
    directory, manifest = copy_library(tmp_path, library_fixture)
    image = directory / manifest["image_files"][0]["cameras"][module.CAMERAS[0]]["file"]
    if change == "missing":
        image.unlink()
    elif change == "bytes":
        image.write_bytes(image.read_bytes() + b"tamper")
    else:
        (directory / "unreferenced.txt").write_text("unbound")
    with pytest.raises(ValueError, match="missing|checksum|unreferenced"):
        module.load_library(directory)


@pytest.mark.parametrize("change", ["phase", "camera", "path", "preview"])
def test_rehashed_manifest_cannot_hide_wrong_phase_camera_path_or_preview(
    library_fixture, tmp_path, change
):
    directory, manifest = copy_library(tmp_path, library_fixture)
    if change == "phase":
        sample = manifest["samples"][1]
        sample["frame_index"] += 1
        sample["sample_sha256"] = digest(
            {k: v for k, v in sample.items() if k != "sample_sha256"}
        )
    elif change == "camera":
        manifest["image_files"][0]["cameras"].pop(module.CAMERAS[1])
    elif change == "path":
        manifest["image_files"][0]["cameras"][module.CAMERAS[0]]["file"] = (
            "../outside.png"
        )
    else:
        sheet = manifest["contact_sheets"][0]
        path = directory / sheet["file"]
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        pixels[60, 20, 0] ^= 1
        Image.fromarray(pixels).save(path, format="PNG")
        sheet["file_sha256"] = file_sha256(path)
        sheet["pixels_sha256"] = digest(pixels)
    reseal(directory, manifest)
    with pytest.raises(ValueError, match="deterministic|camera|Unsafe|Preview tile"):
        module.load_library(directory)


def test_builder_requires_verified_local_sources_and_never_fetches_implicitly(
    library_fixture, tmp_path, monkeypatch
):
    plan = copy.deepcopy(library_fixture[2].metadata()["plan"])
    monkeypatch.setattr(module, "plan_library", lambda *a, **k: plan)
    monkeypatch.setattr(module, "_load_resizer", lambda _: fixture_resize)
    monkeypatch.setattr(module, "_download", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="Missing or corrupt"):
        module.build_library(tmp_path / "empty", tmp_path / "output")
    assert not (tmp_path / "output").exists()
    with pytest.raises(FileExistsError):
        module.build_library(library_fixture[0], library_fixture[1])


def test_native_resize_source_is_hash_checked_before_loading(tmp_path):
    source = tmp_path / "image_tools.py"
    source.write_text("raise AssertionError('unverified code executed')\n")
    with pytest.raises(ValueError, match="Unverified native RGB resize"):
        module._load_resizer(source)
