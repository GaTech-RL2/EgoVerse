"""Pinned STANDARD demonstrations with complete aligned action/observation streams.

Dataset fps is metadata, not a license to resample controller deltas. One selected
command occupies one control step; execution timing is recorded by the runner.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .image_donor_bank import _load_resizer
from .interpolation_bank import _decode_image, _download
from .interpolation_catalog import DATASET_REPO, DATASET_REVISION, METADATA_SHA256
from .records import digest, file_sha256

SCHEMA = "standard-libero-demo-sequences-1"
CAMERAS = ("observation/image", "observation/wrist_image")
INDEX_SHA256 = "4ac173b2f2b1c9cd0b64ae54939fa63fe4ed5289f6f06bb6b691d9703a5073e2"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def source_plan(root):
    """One lowest-index available metadata episode per all 40 standard tasks."""
    root = Path(root)
    if file_sha256(root / "file_index.json") != INDEX_SHA256:
        raise ValueError("Pinned source-file index checksum mismatch")
    for name, expected in METADATA_SHA256.items():
        if file_sha256(root / "meta" / name) != expected:
            raise ValueError("Demonstration metadata checksum mismatch")
    index = json.loads((root / "file_index.json").read_text())
    if index["id"] != DATASET_REPO or index["sha"] != DATASET_REVISION:
        raise ValueError("Wrong demonstration dataset/revision")
    files = {row["rfilename"]: row for row in index["siblings"]}
    tasks = [
        json.loads(line)
        for line in (root / "meta/tasks.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    plan = []
    for task in tasks:
        episode = min(
            (row for row in episodes if row["tasks"] == [task["task"]]),
            key=lambda row: row["episode_index"],
        )
        number = episode["episode_index"]
        relative = f"data/chunk-{number // 1000:03d}/episode_{number:06d}.parquet"
        published = files[relative]
        plan.append(
            {
                "source_id": str(task["task_index"]),
                "prompt": task["task"],
                "episode_index": number,
                "frame_count": episode["length"],
                "relative_path": relative,
                "sha256": published["lfs"]["sha256"],
                "bytes": published["size"],
            }
        )
    if len(plan) != 40 or {row["source_id"] for row in plan} != {
        str(i) for i in range(40)
    }:
        raise ValueError("Expected the pinned 40 standard LIBERO source tasks")
    return plan


def build_bank(dataset_root, output, *, image_tools_source=None, cached_only=False):
    """Build an immutable bank; cached-only is explicitly an incomplete pilot."""
    import pyarrow.parquet as pq

    root, output = Path(dataset_root), Path(output)
    plan = source_plan(root)
    output.mkdir(parents=True, exist_ok=False)
    resize = _load_resizer(image_tools_source)
    sources, missing = [], []
    for source in plan:
        path = root / source["relative_path"]
        if cached_only and not path.is_file():
            missing.append(source["source_id"])
            continue
        _download(
            f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{source['relative_path']}",
            path,
            expected_sha256=source["sha256"],
            expected_size=source["bytes"],
        )
        rows = pq.read_table(
            path,
            columns=[
                "image",
                "wrist_image",
                "state",
                "actions",
                "frame_index",
                "episode_index",
                "task_index",
            ],
        ).to_pylist()
        if len(rows) != source["frame_count"]:
            raise ValueError("Unexpected demonstration length")
        for step, row in enumerate(rows):
            if (row["episode_index"], row["frame_index"], row["task_index"]) != (
                source["episode_index"],
                step,
                int(source["source_id"]),
            ):
                raise ValueError("Misaligned demonstration observation/action row")
        arrays = {
            "state": np.asarray([r["state"] for r in rows], np.float32),
            "actions": np.asarray([r["actions"] for r in rows], np.float32),
        }
        if arrays["state"].shape != (len(rows), 8) or arrays["actions"].shape != (
            len(rows),
            7,
        ):
            raise ValueError("Unexpected source controller dimensions")
        if (
            not all(np.isfinite(a).all() for a in arrays.values())
            or np.max(np.abs(arrays["actions"])) > 1.000001
        ):
            raise ValueError("Non-finite or out-of-contract demonstration commands")
        for name in ("image", "wrist_image"):
            arrays[name] = np.stack(
                [resize(_decode_image(row[name]), 224, 224) for row in rows]
            )
        filename = f"episode_{source['episode_index']:06d}.npz"
        np.savez_compressed(output / filename, **arrays)
        frames = np.linspace(0, len(rows) - 1, 6).round().astype(int)
        preview = Image.new("RGB", (6 * 160, 2 * 160 + 28), "white")
        draw = ImageDraw.Draw(preview)
        for column, frame in enumerate(frames):
            draw.text(
                (column * 160 + 4, 5),
                f"source {source['source_id']} frame {frame}",
                fill="black",
            )
            for camera, name in enumerate(("image", "wrist_image")):
                preview.paste(
                    Image.fromarray(arrays[name][frame]).resize((160, 160)),
                    (column * 160, 28 + camera * 160),
                )
        preview_name = f"source_{source['source_id']}.png"
        preview.save(output / preview_name)
        cuts = np.flatnonzero(np.diff(arrays["actions"][:, 6]) != 0) + 1
        source = {
            **source,
            "file": filename,
            "file_sha256": file_sha256(output / filename),
            "preview": preview_name,
            "preview_sha256": file_sha256(output / preview_name),
            "preview_frames": frames.tolist(),
            "gripper_transition_frames": cuts.tolist(),
            "action_min": arrays["actions"].min(0).tolist(),
            "action_max": arrays["actions"].max(0).tolist(),
            "record_status": "observed_standard_demonstration",
            "transfer_success": None,
        }
        sources.append(source)
    if not sources:
        raise ValueError("No verified source sequences are available")
    overview = Image.new("RGB", (8 * 160, ((len(sources) + 7) // 8) * 190), "white")
    draw = ImageDraw.Draw(overview)
    for index, source in enumerate(sources):
        with np.load(output / source["file"], allow_pickle=False) as episode:
            tile = Image.fromarray(episode["image"][0]).resize((160, 160))
        x, y = (index % 8) * 160, (index // 8) * 190
        overview.paste(tile, (x, y + 25))
        draw.text((x + 4, y + 5), f"source {source['source_id']}", fill="black")
    overview.save(output / "overview.png")
    metadata = {
        "schema_version": SCHEMA,
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "dataset_metadata_fps": 10,
        "images": "224px pinned native resizer; upstream rotation retained",
        "action_semantics": "recorded unnormalized LIBERO OSC controller commands; no resampling",
        "gripper": "negative opens, positive closes",
        "sources": sources,
        "missing_source_ids": missing,
        "complete_standard_catalog": not missing,
        "planned_sources": plan,
        "ood_demonstrations": 0,
        "overview_sha256": file_sha256(output / "overview.png"),
    }
    metadata["bank_id"] = digest(metadata)
    write_json(output / "bank.json", metadata)
    return metadata


class DemoBank:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "bank.json").read_text())
        check = dict(self.metadata)
        bank_id = check.pop("bank_id")
        if check["schema_version"] != SCHEMA or bank_id != digest(check):
            raise ValueError("Demo-bank metadata identity mismatch")
        self.bank_id = bank_id
        self.sources = {row["source_id"]: row for row in check["sources"]}
        self._arrays = {}
        for row in self.sources.values():
            for field, sha in (("file", "file_sha256"), ("preview", "preview_sha256")):
                if (
                    Path(row[field]).name != row[field]
                    or file_sha256(self.directory / row[field]) != row[sha]
                ):
                    raise ValueError("Demo-bank asset checksum mismatch")
        if file_sha256(self.directory / "overview.png") != check["overview_sha256"]:
            raise ValueError("Demo-bank overview checksum mismatch")

    def catalog(self):
        return [
            {
                k: row[k]
                for k in (
                    "source_id",
                    "prompt",
                    "frame_count",
                    "preview_frames",
                    "gripper_transition_frames",
                    "record_status",
                )
            }
            for row in self.sources.values()
        ]

    def arrays(self, source_id):
        if source_id not in self.sources:
            raise ValueError("Unknown standard demonstration source")
        if source_id not in self._arrays:
            with np.load(
                self.directory / self.sources[source_id]["file"], allow_pickle=False
            ) as saved:
                self._arrays[source_id] = {k: saved[k].copy() for k in saved.files}
        return self._arrays[source_id]

    def observation(self, source_id, frame):
        arrays = self.arrays(source_id)
        if type(frame) is not int or not 0 <= frame < len(arrays["state"]):
            raise ValueError("Donor frame is outside the recorded demonstration")
        return {
            CAMERAS[0]: arrays["image"][frame].copy(),
            CAMERAS[1]: arrays["wrist_image"][frame].copy(),
            "observation/state": arrays["state"][frame].copy(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-tools-source", type=Path)
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_args()
    result = build_bank(
        args.dataset_root,
        args.output,
        image_tools_source=args.image_tools_source,
        cached_only=args.cached_only,
    )
    print(
        json.dumps(
            {
                "bank_id": result["bank_id"],
                "sources": len(result["sources"]),
                "frames": sum(s["frame_count"] for s in result["sources"]),
                "missing_sources": result["missing_source_ids"],
            }
        )
    )


if __name__ == "__main__":
    main()
