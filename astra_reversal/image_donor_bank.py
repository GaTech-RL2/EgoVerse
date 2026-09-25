"""Verified RGB pairs from sparse phases of pinned STANDARD LIBERO demonstrations.

This module never runs a policy or provider. Planning/building is offline;
retrieval is a separate explicit operation. Dataset PNGs already include the
RLDS 180-degree rotation. Policy pixels use the pinned native 224px resizer.
"""

import argparse
import copy
import importlib.util
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .interpolation_bank import _download, iter_episode_observations
from .interpolation_catalog import (
    DATASET_REPO,
    DATASET_REVISION,
    DONORS,
    METADATA_SHA256,
    demonstration_plan,
    donor_for,
)
from .records import digest, file_sha256

SCHEMA = "standard-libero-image-donors-1"
CAMERAS = ("observation/image", "observation/wrist_image")
IMAGE_SHAPE = (224, 224, 3)
PHASE_DENOMINATOR = 4
SELECTION = "first prescribed metadata episode per donor; phases 0,1/4,1/2,3/4,1; round nearest ties up"
IMAGE_TOOLS_SHA256 = "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
SAMPLE_FIELDS = {
    "donor_id",
    "source_id",
    "prompt",
    "episode_index",
    "frame_index",
    "phase",
    "preview_position",
    "cameras",
}
PREPROCESSING = {
    "source": "embedded 256x256 uint8 RGB PNG from the pinned standard training dataset",
    "rotation": "none; upstream RLDS conversion already rotated both cameras 180 degrees",
    "resize": "pinned openpi_client.image_tools.resize_with_pad(224,224), default PIL bilinear",
    "image_tools_sha256": IMAGE_TOOLS_SHA256,
    "output_shape": list(IMAGE_SHAPE),
    "output_dtype": "uint8",
    "policy_pixels_contain_labels_or_annotations": False,
    "stored_demo_actions_state_or_object_poses": False,
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def phase_indices(frame_count):
    """Five evenly spaced frames, with integer nearest rounding (ties up)."""
    _require(
        type(frame_count) is int and frame_count >= 5,
        "At least five frames are required",
    )
    return [(i * (frame_count - 1) + 2) // 4 for i in range(5)]


def _source_ids(values=None):
    result = [donor.source_id for donor in DONORS] if values is None else list(values)
    _require(
        result and len(result) == len(set(result)),
        "Source IDs must be nonempty and unique",
    )
    expected = [donor.source_id for donor in DONORS if donor.source_id in result]
    _require(
        result == expected, "Only canonical-order STANDARD donor IDs are supported"
    )
    return result


def plan_library(dataset_root, *, source_ids=None):
    """Validate cached metadata and select one fixed episode per training donor."""
    root = Path(dataset_root)
    sources = demonstration_plan(root / "meta")["sources"]
    index = json.loads((root / "file_index.json").read_text())
    _require(
        index["id"] == DATASET_REPO and index["sha"] == DATASET_REVISION,
        "Dataset identity differs from pinned STANDARD training data",
    )
    files = {item["rfilename"]: item for item in index["siblings"]}
    selected = []
    for source_id in _source_ids(source_ids):
        source = next(row for row in sources if row["source_id"] == source_id)
        episode = copy.deepcopy(source["episodes"][0])
        published = files[episode["relative_path"]]
        episode.update(sha256=published["lfs"]["sha256"], bytes=published["size"])
        _require(
            _sha(episode["sha256"])
            and type(episode["bytes"]) is int
            and episode["bytes"] > 0,
            "Missing published parquet hash or size",
        )
        selected.append(
            {
                "source_id": source_id,
                "prompt": source["prompt"],
                "dataset_task_index": source["dataset_task_index"],
                "episode": episode,
                "frame_indices": phase_indices(episode["frame_count"]),
            }
        )
    plan = {
        "schema_version": SCHEMA + "-plan",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "metadata_sha256": dict(METADATA_SHA256),
        "file_index_sha256": file_sha256(root / "file_index.json"),
        "selection": SELECTION,
        "sources": selected,
        "total_source_bytes": sum(row["episode"]["bytes"] for row in selected),
        "paired_samples": 5 * len(selected),
        "ood_demonstrations": 0,
    }
    plan["plan_id"] = digest(plan)
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    _require(
        plan["plan_id"] == digest({k: v for k, v in plan.items() if k != "plan_id"}),
        "Image plan identity mismatch",
    )
    _require(
        plan["schema_version"] == SCHEMA + "-plan"
        and plan["dataset_repo"] == DATASET_REPO
        and plan["dataset_revision"] == DATASET_REVISION
        and plan["metadata_sha256"] == METADATA_SHA256
        and _sha(plan["file_index_sha256"])
        and plan["selection"] == SELECTION
        and plan["ood_demonstrations"] == 0,
        "Image plan uses unapproved source data",
    )
    _source_ids(row["source_id"] for row in plan["sources"])
    for source in plan["sources"]:
        donor, episode = donor_for(source["source_id"]), source["episode"]
        _require(
            source["prompt"] == donor.prompt
            and source["dataset_task_index"] == donor.dataset_task_index
            and episode["episode_index"] == donor.episode_indices[0]
            and episode["frame_count"] == donor.episode_frame_counts[0],
            "Image plan changed the deterministic donor episode",
        )
        expected_path = f"data/chunk-{episode['episode_index'] // 1000:03d}/episode_{episode['episode_index']:06d}.parquet"
        _require(
            episode["relative_path"] == expected_path
            and _sha(episode["sha256"])
            and type(episode["bytes"]) is int
            and episode["bytes"] > 0,
            "Invalid pinned parquet identity",
        )
        _require(
            source["frame_indices"] == phase_indices(episode["frame_count"]),
            "Image plan changed deterministic phase samples",
        )
    _require(
        plan["paired_samples"] == 5 * len(plan["sources"])
        and plan["total_source_bytes"]
        == sum(row["episode"]["bytes"] for row in plan["sources"]),
        "Image plan counts disagree",
    )


def fetch_sources(plan, dataset_root):
    """Explicitly retrieve only the selected verified public parquet files."""
    _validate_plan(plan)
    root = Path(dataset_root)
    for source in plan["sources"]:
        episode = source["episode"]
        _download(
            f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{episode['relative_path']}",
            root / episode["relative_path"],
            expected_sha256=episode["sha256"],
            expected_size=episode["bytes"],
        )


def _load_resizer(source_path=None):
    if source_path is None:
        from openpi_client import image_tools

        source_path = Path(image_tools.__file__)
    else:
        source_path = Path(source_path)
        _require(
            file_sha256(source_path) == IMAGE_TOOLS_SHA256,
            "Unverified native RGB resize implementation",
        )
        spec = importlib.util.spec_from_file_location(
            "_pinned_donor_image_tools", source_path
        )
        image_tools = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(image_tools)
    _require(
        file_sha256(source_path) == IMAGE_TOOLS_SHA256,
        "Unverified native RGB resize implementation",
    )
    return image_tools.resize_with_pad


def _freeze_pixels(pixels, shape):
    value = np.asarray(pixels)
    _require(
        value.dtype == np.uint8 and value.shape == shape,
        "Expected exact uint8 RGB pixel geometry",
    )
    return np.frombuffer(value.tobytes(), dtype=np.uint8).reshape(shape)


@dataclass(frozen=True, init=False)
class DonorImage:
    donor_id: str
    camera: str
    pixels: np.ndarray
    _provenance_json: str

    def __init__(self, donor_id, camera, pixels, provenance):
        _require(
            isinstance(donor_id, str) and donor_id and camera in CAMERAS,
            "Invalid donor image identity",
        )
        pixels = _freeze_pixels(pixels, IMAGE_SHAPE)
        _require(
            provenance["pixels_sha256"] == digest(pixels)
            and _sha(provenance["library_id"])
            and _sha(provenance["sample_sha256"])
            and provenance.get("donor_id", donor_id) == donor_id
            and provenance.get("camera", camera) == camera,
            "Donor provenance does not bind its pixels/library",
        )
        object.__setattr__(self, "donor_id", donor_id)
        object.__setattr__(self, "camera", camera)
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(
            self,
            "_provenance_json",
            json.dumps(provenance, sort_keys=True, allow_nan=False),
        )

    @property
    def provenance(self):
        return json.loads(self._provenance_json)


def _save_png(directory, relative, pixels):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), "Refusing to replace an image artifact")
    Image.fromarray(pixels).save(path, format="PNG")
    return {
        "file": relative,
        "file_sha256": file_sha256(path),
        "pixels_sha256": digest(pixels),
        "shape": list(pixels.shape),
        "dtype": "uint8",
    }


def _sheet_layout(rows):
    return {
        "rows": rows,
        "columns": 5,
        "canvas_shape": [44 + 140 * rows + 20, 600, 3],
        "tile_shape": [112, 112, 3],
        "header_height": 44,
        "row_height": 140,
        "left": 20,
        "top_in_row": 16,
    }


def _contact_sheet(samples, camera, pixels):
    layout = _sheet_layout(len(samples) // 5)
    height, width, _ = layout["canvas_shape"]
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    heading, prompt_font, label = (
        ImageFont.load_default(size=16),
        ImageFont.load_default(size=11),
        ImageFont.load_default(size=10),
    )
    title = "external camera" if camera == CAMERAS[0] else "wrist camera"
    draw.text(
        (20, 5),
        "STANDARD LIBERO training frames / " + title,
        fill="black",
        font=heading,
    )
    for column in range(5):
        draw.text(
            (20 + 112 * column, 27), f"phase {column}/4", fill="black", font=label
        )
    for sample in samples:
        row = sample["preview_position"]["row"]
        column = sample["preview_position"]["column"]
        top, left = 44 + row * 140, 20 + column * 112
        if column == 0:
            text = sample["source_id"] + ": " + sample["prompt"]
            _require(
                draw.textbbox((0, 0), text, font=prompt_font)[2] <= 560,
                "Donor prompt does not fit the declared preview layout",
            )
            draw.text((20, top), text, fill="black", font=prompt_font)
        image = Image.fromarray(pixels[(sample["donor_id"], camera)])
        canvas.paste(
            image.resize((112, 112), Image.Resampling.BILINEAR), (left, top + 16)
        )
        _require(
            draw.textbbox((0, 0), sample["donor_id"], font=label)[2] <= 112,
            "Sample ID does not fit preview tile",
        )
        draw.text((left, top + 129), sample["donor_id"], fill="black", font=label)
    return np.asarray(canvas).copy(), layout


def build_library(dataset_root, output, *, source_ids=None, image_tools_source=None):
    """Build from local verified parquets only; existing output is never replaced."""
    root, output = Path(dataset_root), Path(output)
    plan = plan_library(root, source_ids=source_ids)
    resize = _load_resizer(image_tools_source)
    for source in plan["sources"]:
        episode = source["episode"]
        path = root / episode["relative_path"]
        _require(
            path.is_file()
            and path.stat().st_size == episode["bytes"]
            and file_sha256(path) == episode["sha256"],
            "Missing or corrupt pinned source parquet; use explicit fetch first",
        )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    samples, records, pixels = [], [], {}
    for row, source in enumerate(plan["sources"]):
        episode = source["episode"]
        path = root / episode["relative_path"]
        selected = {
            frame: ordinal for ordinal, frame in enumerate(source["frame_indices"])
        }
        count = 0
        for frame, observation in iter_episode_observations(
            path, episode, source["dataset_task_index"]
        ):
            count += 1
            if frame not in selected:
                continue
            column = selected[frame]
            donor_id = f"std{source['source_id']}-e{episode['episode_index']}-f{frame}"
            camera_records, cameras = {}, {}
            for camera, basename in zip(CAMERAS, ("external", "wrist"), strict=True):
                original = observation[camera]
                value = np.asarray(resize(original, 224, 224)).copy()
                _freeze_pixels(value, IMAGE_SHAPE)
                pixels[(donor_id, camera)] = value
                saved = _save_png(output, f"images/{donor_id}/{basename}.png", value)
                saved["source_pixels_sha256"] = digest(original)
                camera_records[camera] = saved
                cameras[camera] = {
                    key: saved[key] for key in ("shape", "dtype", "pixels_sha256")
                }
            sample = {
                "donor_id": donor_id,
                "source_id": source["source_id"],
                "prompt": source["prompt"],
                "episode_index": episode["episode_index"],
                "frame_index": frame,
                "phase": {"numerator": column, "denominator": 4},
                "preview_position": {"row": row, "column": column},
                "cameras": cameras,
            }
            sample["sample_sha256"] = digest(sample)
            samples.append(sample)
            records.append({"donor_id": donor_id, "cameras": camera_records})
        _require(
            count == episode["frame_count"] and file_sha256(path) == episode["sha256"],
            "Source frame coverage or bytes changed during extraction",
        )
    _require(len(samples) == plan["paired_samples"], "Missing selected donor frame")
    sheets = []
    for camera, name in zip(CAMERAS, ("external", "wrist"), strict=True):
        image, layout = _contact_sheet(samples, camera, pixels)
        sheets.append(
            {
                "camera": camera,
                "layout": layout,
                **_save_png(output, f"contact_sheets/{name}.png", image),
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "plan": plan,
        "preprocessing": PREPROCESSING,
        "samples": samples,
        "image_files": records,
        "contact_sheets": sheets,
        "software": {name: version(name) for name in ("numpy", "Pillow", "pyarrow")},
        "builder_source_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in (
                "image_donor_bank.py",
                "interpolation_bank.py",
                "interpolation_catalog.py",
                "records.py",
            )
        },
    }
    manifest["library_id"] = digest(manifest)
    _write_json(output / "manifest.json", manifest)
    return load_library(output)


def _read_png(root, record, expected_shape):
    relative = PurePosixPath(record["file"])
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "\\" not in record["file"]
        and relative.suffix == ".png",
        "Unsafe image artifact path",
    )
    path = root / relative
    _require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root.resolve()),
        "Image artifact is missing or escapes the library",
    )
    _require(file_sha256(path) == record["file_sha256"], "Image file checksum mismatch")
    with Image.open(path) as image:
        _require(
            image.format == "PNG" and image.mode == "RGB",
            "Expected lossless RGB PNG artifact",
        )
        pixels = np.asarray(image).copy()
    pixels = _freeze_pixels(pixels, expected_shape)
    _require(
        record["shape"] == list(expected_shape)
        and record["dtype"] == "uint8"
        and digest(pixels) == record["pixels_sha256"],
        "Image pixel hash/shape/dtype mismatch",
    )
    return pixels


class ImageDonorLibrary:
    def __init__(self, manifest, images, sheets):
        self._manifest_json = json.dumps(manifest, sort_keys=True, allow_nan=False)
        self._images, self._sheets = dict(images), tuple(sheets)
        self._library_id = manifest["library_id"]

    @property
    def library_id(self):
        return self._library_id

    def metadata(self):
        return json.loads(self._manifest_json)

    def catalog(self):
        return [
            {**sample, "library_id": self.library_id}
            for sample in self.metadata()["samples"]
        ]

    def resolve(self, donor_id, camera):
        _require(camera in CAMERAS, "Unknown donor camera")
        _require((donor_id, camera) in self._images, "Unknown donor sample ID")
        return self._images[(donor_id, camera)]

    donor_lookup = resolve

    def contact_sheets(self):
        return [
            {
                "camera": row["camera"],
                "image": image,
                "sha256": row["pixels_sha256"],
                "file_sha256": row["file_sha256"],
                "layout": copy.deepcopy(row["layout"]),
                "library_id": self.library_id,
            }
            for row, image in self._sheets
        ]


def load_library(directory):
    """CPU-only integrity verification; returns immutable pixels and copied metadata."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    _require(
        manifest["schema_version"] == SCHEMA
        and manifest["library_id"]
        == digest({k: v for k, v in manifest.items() if k != "library_id"}),
        "Image library manifest identity mismatch",
    )
    _validate_plan(manifest["plan"])
    _require(
        json.loads((root / "plan.json").read_text()) == manifest["plan"],
        "Image plan file differs from sealed manifest",
    )
    _require(
        manifest["preprocessing"] == PREPROCESSING,
        "Image preprocessing contract changed",
    )
    samples, records = manifest["samples"], manifest["image_files"]
    _require(
        len(samples) == len(records) == manifest["plan"]["paired_samples"],
        "Missing paired samples",
    )
    record_map = {record["donor_id"]: record for record in records}
    _require(len(record_map) == len(records), "Duplicate image record")
    images, expected_files = {}, {"manifest.json", "plan.json"}
    for index, sample in enumerate(samples):
        row, column = divmod(index, 5)
        source = manifest["plan"]["sources"][row]
        episode, frame = source["episode"], source["frame_indices"][column]
        donor_id = f"std{source['source_id']}-e{episode['episode_index']}-f{frame}"
        _require(
            set(sample) == SAMPLE_FIELDS | {"sample_sha256"}
            and sample["sample_sha256"]
            == digest({k: v for k, v in sample.items() if k != "sample_sha256"}),
            "Sample metadata checksum mismatch",
        )
        _require(
            sample["donor_id"] == donor_id
            and sample["source_id"] == source["source_id"]
            and sample["prompt"] == source["prompt"]
            and sample["episode_index"] == episode["episode_index"]
            and sample["frame_index"] == frame
            and sample["phase"] == {"numerator": column, "denominator": 4}
            and sample["preview_position"] == {"row": row, "column": column},
            "Sample is not the declared deterministic donor phase",
        )
        record = record_map[donor_id]
        _require(
            set(sample["cameras"]) == set(record["cameras"]) == set(CAMERAS),
            "Missing or unexpected paired camera",
        )
        for camera in CAMERAS:
            saved = record["cameras"][camera]
            _require(
                sample["cameras"][camera]
                == {key: saved[key] for key in ("shape", "dtype", "pixels_sha256")}
                and _sha(saved["source_pixels_sha256"]),
                "Sample camera hash does not bind image record",
            )
            pixels = _read_png(root, saved, IMAGE_SHAPE)
            _require(
                saved["file"] not in expected_files,
                "Image file is shared across donor/camera identities",
            )
            expected_files.add(saved["file"])
            provenance = {
                "library_id": manifest["library_id"],
                "sample_sha256": sample["sample_sha256"],
                "donor_id": donor_id,
                "camera": camera,
                "pixels_sha256": digest(pixels),
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "source_id": source["source_id"],
                "prompt": source["prompt"],
                "episode_index": episode["episode_index"],
                "frame_index": frame,
                "phase": copy.deepcopy(sample["phase"]),
                "parquet": copy.deepcopy(episode),
                "source_pixels_sha256": saved["source_pixels_sha256"],
                "png_file_sha256": saved["file_sha256"],
                "preprocessing": copy.deepcopy(PREPROCESSING),
            }
            images[(donor_id, camera)] = DonorImage(
                donor_id, camera, pixels, provenance
            )
    sheets = []
    _require(
        len(manifest["contact_sheets"]) == 2
        and {row["camera"] for row in manifest["contact_sheets"]} == set(CAMERAS),
        "Missing paired preview sheets",
    )
    for sheet in manifest["contact_sheets"]:
        _require(
            sheet["layout"] == _sheet_layout(len(manifest["plan"]["sources"])),
            "Preview grid contract changed",
        )
        pixels = _read_png(root, sheet, tuple(sheet["layout"]["canvas_shape"]))
        for sample in samples:
            row, column = (sample["preview_position"][key] for key in ("row", "column"))
            left, top = 20 + 112 * column, 44 + 140 * row + 16
            original = images[(sample["donor_id"], sheet["camera"])].pixels
            preview = np.asarray(
                Image.fromarray(original).resize((112, 112), Image.Resampling.BILINEAR)
            )
            _require(
                np.array_equal(pixels[top : top + 112, left : left + 112], preview),
                "Preview tile differs from its declared policy donor pixels",
            )
        _require(sheet["file"] not in expected_files, "Duplicate preview artifact")
        expected_files.add(sheet["file"])
        sheets.append((copy.deepcopy(sheet), pixels))
    _require(
        expected_files
        == {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()},
        "Missing or unreferenced library files",
    )
    return ImageDonorLibrary(manifest, images, sheets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fetch", "build"):
        item = sub.add_parser(name)
        item.add_argument("--dataset-root", type=Path, required=True)
        item.add_argument("--source-id", action="append")
        if name == "build":
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--image-tools-source", type=Path)
    item = sub.add_parser("verify")
    item.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        library = load_library(args.directory)
    elif args.command == "build":
        library = build_library(
            args.dataset_root,
            args.output,
            source_ids=args.source_id,
            image_tools_source=args.image_tools_source,
        )
    else:
        plan = plan_library(args.dataset_root, source_ids=args.source_id)
        if args.command == "fetch":
            fetch_sources(plan, args.dataset_root)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(
        json.dumps(
            {
                "library_id": library.library_id,
                "samples": len(library.catalog()),
                "cameras": list(CAMERAS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
