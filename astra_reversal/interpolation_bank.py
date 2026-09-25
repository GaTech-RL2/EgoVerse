"""Stream verified standard LIBERO observations into a frozen PI05 donor bank.

Images in the selected LeRobot parquet files already have the RLDS 180-degree
rotation. They are decoded as RGB and passed unchanged to the adapter. The
adapter owns resizing, normalization and tokenization. No simulator, action
sampler, optimizer or provider call is used by this module.
"""

import argparse
import contextlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .interpolation_catalog import (
    DATASET_REPO,
    DATASET_REVISION,
    METADATA_SHA256,
    demonstration_plan,
    donor_for,
)
from .interpolation_conditioning import CAPTURE_BOUNDARY, TextLatentBank
from .records import digest, file_sha256

ARRAY_KEYS = ("states", "token_ids", "token_mask", "instruction_mask")
SCHEMA = "astra-interpolation-demo-bank-1"
PREPROCESSING = {
    "images": "decode embedded RGB PNG; no extra rotation, resize or normalization",
    "camera_orientation": "already rotated 180 degrees in the upstream HDF5-to-RLDS conversion",
    "state": "raw float32 eef position(3), axis angle(3), gripper qpos(2)",
    "adapter": "native openpi_libero profile applies its own frozen preprocessing",
    "rlds_source": "https://github.com/moojink/rlds_dataset_builder/tree/6174b0b6bb69df6361f1117944952bf14afb0cc3/LIBERO_Goal",
    "lerobot_conversion_source": "https://github.com/QuanyiLi/pi0-text-latent/blob/587a6cbf64f16c7b87fa5805dc0ed934192239a4/examples/libero/convert_libero_data_to_lerobot.py",
}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _download(url, destination, *, expected_sha256=None, expected_size=None):
    """Fetch public immutable data; never print redirected signed URLs."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 and file_sha256(destination) != expected_sha256:
            raise ValueError(f"Cached dataset checksum mismatch: {destination.name}")
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError(f"Cached dataset size mismatch: {destination.name}")
        return destination
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise ValueError(
                    f"Downloaded dataset size mismatch: {destination.name}"
                )
            if expected_sha256 and file_sha256(temporary) != expected_sha256:
                raise ValueError(
                    f"Downloaded dataset checksum mismatch: {destination.name}"
                )
            temporary.replace(destination)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == 2:
                code = getattr(exc, "code", None)
                raise RuntimeError(
                    f"Public dataset download failed for {destination.name}; "
                    f"error type={type(exc).__name__}, HTTP status={code}"
                ) from None
            time.sleep(attempt + 1)
        finally:
            if temporary.exists():
                temporary.unlink()
    raise AssertionError("Unreachable download state")


def prepare_dataset(cache_dir):
    """Download only small pinned metadata; parquet files are fetched on demand."""
    root = Path(cache_dir) / f"libero-{DATASET_REVISION}"
    resolve = (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"
    )
    for name, sha in METADATA_SHA256.items():
        _download(f"{resolve}/meta/{name}", root / "meta" / name, expected_sha256=sha)
    plan = demonstration_plan(root / "meta")
    index_path = root / "file_index.json"
    _download(
        f"https://huggingface.co/api/datasets/{DATASET_REPO}/revision/{DATASET_REVISION}?blobs=true",
        index_path,
    )
    index = json.loads(index_path.read_text())
    if index.get("sha") != DATASET_REVISION or index.get("id") != DATASET_REPO:
        raise ValueError(
            "Dataset file index does not identify the pinned repository/revision"
        )
    files = {row["rfilename"]: row for row in index["siblings"]}
    for source in plan["sources"]:
        for episode in source["episodes"]:
            info = files.get(episode["relative_path"], {})
            sha = info.get("lfs", {}).get("sha256", "")
            size = info.get("size")
            if (
                len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)
                or type(size) is not int
                or size <= 0
            ):
                raise ValueError(
                    "Selected parquet is missing its published LFS digest/size"
                )
            episode.update({"sha256": sha, "bytes": size})
    plan["file_index_sha256"] = file_sha256(index_path)
    return root, plan


def _decode_image(value):
    from PIL import Image

    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
        raise ValueError(
            "Expected an embedded image; external image paths are unsupported"
        )
    with Image.open(io.BytesIO(value["bytes"])) as image:
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError("Frozen LIBERO images must be 256x256 RGB")
        pixels = np.asarray(image).copy()
    if pixels.dtype != np.uint8:
        raise ValueError("Frozen LIBERO images must be uint8")
    return pixels


def iter_episode_observations(path, episode, task_index):
    """Read bounded Arrow batches and prove exact episode/task/frame coverage."""
    import pyarrow.parquet as pq

    columns = [
        "image",
        "wrist_image",
        "state",
        "episode_index",
        "frame_index",
        "task_index",
    ]
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != episode["frame_count"]:
        raise ValueError("Parquet frame count disagrees with pinned metadata")
    count = 0
    for batch in parquet.iter_batches(batch_size=1, columns=columns, use_threads=False):
        row = batch.to_pylist()[0]
        if (row["episode_index"], row["frame_index"], row["task_index"]) != (
            episode["episode_index"],
            count,
            task_index,
        ):
            raise ValueError("Parquet episode/task/frame identity mismatch")
        state = np.asarray(row["state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Parquet proprioception must be a finite raw state8")
        yield (
            count,
            {
                "observation/image": _decode_image(row["image"]),
                "observation/wrist_image": _decode_image(row["wrist_image"]),
                "observation/state": state,
            },
        )
        count += 1
    if count != episode["frame_count"]:
        raise ValueError("Incomplete parquet episode")


class FrameMean:
    """Float64 accumulation with fixed source identity and one vote per frame."""

    def __init__(self):
        self.states_sum = None
        self.count = 0
        self.token_ids = self.token_mask = self.instruction_mask = None
        self.invariants = None

    def add(self, bank):
        provenance = bank.provenance
        invariants = {
            name: provenance[name]
            for name in ("source_prompt", "compatibility", "capture_boundary")
        }
        if self.states_sum is None:
            self.states_sum = np.zeros(bank.states.shape, np.float64)
            self.token_ids, self.token_mask, self.instruction_mask = (
                getattr(bank, name).copy() for name in ARRAY_KEYS[1:]
            )
            self.invariants = invariants
        elif (
            invariants != self.invariants
            or bank.states.shape != self.states_sum.shape
            or any(
                not np.array_equal(getattr(self, name), getattr(bank, name))
                for name in ARRAY_KEYS[1:]
            )
        ):
            raise ValueError(
                "Donor capture model/profile/prompt/token identity changed across frames"
            )
        # Out-of-scope states are never injected, and are explicitly zeroed.
        self.states_sum += np.where(
            bank.instruction_mask[None, :, :, None], bank.states, 0
        )
        self.count += 1

    def arrays(self):
        if self.count == 0:
            raise ValueError("Cannot checkpoint an empty donor mean")
        return {
            "states_sum": self.states_sum,
            "token_ids": self.token_ids,
            "token_mask": self.token_mask,
            "instruction_mask": self.instruction_mask,
        }

    def bank(self, provenance):
        if self.count == 0:
            raise ValueError("Cannot produce an empty donor bank")
        return TextLatentBank(
            (self.states_sum / self.count).astype(np.float32),
            self.token_ids,
            self.token_mask,
            self.instruction_mask,
            {**self.invariants, **provenance},
        )


def _implementation():
    root = Path(__file__).parent
    return {
        name: file_sha256(root / name)
        for name in (
            "interpolation_bank.py",
            "interpolation_catalog.py",
            "interpolation_conditioning.py",
            "lerobot_policy.py",
            "openpi_inputs.py",
            "records.py",
        )
    }


def _resume(directory, expected):
    accumulator = FrameMean()
    progress_path = directory / "progress.json"
    if not progress_path.exists():
        return accumulator, [], []
    progress = json.loads(progress_path.read_text())
    for key, value in expected.items():
        if progress.get(key) != value:
            raise ValueError(f"Extraction restart identity mismatch: {key}")
    checkpoint = directory / progress["resume_file"]
    if file_sha256(checkpoint) != progress["resume_sha256"]:
        raise ValueError("Extraction resume checkpoint checksum mismatch")
    with np.load(checkpoint, allow_pickle=False) as arrays:
        accumulator.states_sum = arrays["states_sum"].copy()
        for name in ARRAY_KEYS[1:]:
            setattr(accumulator, name, arrays[name].copy())
    accumulator.count = progress["frame_count"]
    accumulator.invariants = progress["invariants"]
    if (
        accumulator.states_sum.dtype != np.float64
        or not np.isfinite(accumulator.states_sum).all()
        or accumulator.count != len(progress["frames"])
        or accumulator.count != sum(row["frame_count"] for row in progress["episodes"])
    ):
        raise ValueError(
            "Extraction resume counts/dtype/finite values are inconsistent"
        )
    accumulator.bank({})  # Validate the stored token arrays and native identity.
    return accumulator, progress["episodes"], progress["frames"]


def load_bank(path):
    """Verify bank bytes, full array digests and source provenance before use."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Expected a completed demonstration bank")
    donor = donor_for(manifest["source_id"])
    source = manifest["source_plan"]
    episodes = source["episodes"]
    if (
        source["source_id"] != donor.source_id
        or source["prompt"] != donor.prompt
        or source["dataset_task_index"] != donor.dataset_task_index
        or tuple(row["episode_index"] for row in episodes) != donor.episode_indices
        or tuple(row["frame_count"] for row in episodes) != donor.episode_frame_counts
        or source["frame_count"] != donor.all_frame_count
        or manifest["frame_count"] != donor.all_frame_count
        or sum(row["frame_count"] for row in episodes) != donor.all_frame_count
        or manifest["episode_count"] != 20
        or manifest["dataset_revision"] != DATASET_REVISION
    ):
        raise ValueError(
            "Donor manifest does not cover the frozen source demonstrations"
        )
    bank_path = directory / manifest["bank_file"]
    if file_sha256(bank_path) != manifest["bank_sha256"]:
        raise ValueError("Donor bank file checksum mismatch")
    frames_path = directory / manifest["frame_ledger_file"]
    if file_sha256(frames_path) != manifest["frame_ledger_sha256"]:
        raise ValueError("Donor frame ledger checksum mismatch")
    expected_frames = (
        (episode["episode_index"], frame)
        for episode in episodes
        for frame in range(episode["frame_count"])
    )
    with frames_path.open() as stream:
        for expected in expected_frames:
            line = stream.readline()
            row = json.loads(line) if line else {}
            if (row.get("episode_index"), row.get("frame_index")) != expected:
                raise ValueError(
                    "Donor frame ledger has incomplete or duplicate coverage"
                )
            for name in ("observation_sha256", "capture_bank_id"):
                value = row.get(name, "")
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Donor frame ledger contains an invalid digest")
        if stream.read():
            raise ValueError("Donor frame ledger has extra frames")
    with np.load(bank_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(ARRAY_KEYS):
            raise ValueError("Unexpected donor bank arrays")
        bank = TextLatentBank(
            **{name: arrays[name] for name in ARRAY_KEYS},
            provenance=manifest["bank"]["provenance"],
        )
    if bank.metadata() != manifest["bank"]:
        raise ValueError(
            "Donor bank arrays/provenance do not match the saved bank identity"
        )
    provenance = bank.provenance
    expected_provenance = {
        "source_id": donor.source_id,
        "source_prompt": donor.prompt,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "metadata_sha256": METADATA_SHA256,
        "episodes": episodes,
        "frame_count": donor.all_frame_count,
        "frame_ledger_sha256": manifest["frame_ledger_sha256"],
        "capture_kind": "demonstration_mean",
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Donor bank provenance disagrees with its complete manifest")
    if np.any(bank.states * (~bank.instruction_mask)[None, :, :, None] != 0):
        raise ValueError("Demonstration bank contains nonzero out-of-scope states")
    return bank


def extract_donor(adapter, source_id, *, output_dir, cache_dir, progress_every=100):
    """Extract one source; restart reuses only verified completed episodes."""
    import fcntl

    donor = donor_for(source_id)
    if type(progress_every) is not int or progress_every <= 0:
        raise ValueError("Progress interval must be a positive integer")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".extract.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another extractor owns this donor output directory"
            ) from None
        return _extract_locked(
            adapter, donor, directory, Path(cache_dir), progress_every
        )


def _extract_locked(adapter, donor, directory, cache_dir, progress_every):
    started = time.perf_counter()
    cache_root, plan = prepare_dataset(cache_dir)
    source = next(row for row in plan["sources"] if row["source_id"] == donor.source_id)
    identity = {
        "schema": SCHEMA,
        "source_id": donor.source_id,
        "source_plan": source,
        "dataset_revision": DATASET_REVISION,
        "adapter_metadata_sha256": digest(adapter.metadata),
        "implementation_sha256": _implementation(),
    }
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "Completed donor bank belongs to a different extraction identity"
            )
        load_bank(directory)
        return manifest
    accumulator, completed, frames = _resume(directory, identity)
    if completed != source["episodes"][: len(completed)]:
        raise ValueError(
            "Restart episode coverage is not the prescribed ordered prefix"
        )
    initial_count = accumulator.count
    for episode in source["episodes"][len(completed) :]:
        relative = episode["relative_path"]
        path = _download(
            f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{relative}",
            cache_root / relative,
            expected_sha256=episode["sha256"],
            expected_size=episode["bytes"],
        )
        for frame_index, observation in iter_episode_observations(
            path, episode, donor.dataset_task_index
        ):
            observation_id = f"{DATASET_REPO}@{DATASET_REVISION}:episode{episode['episode_index']}:frame{frame_index}"
            bank = adapter.capture_text_latents(
                observation, donor.prompt, observation_id=observation_id
            )
            if (
                bank.provenance["source_prompt"] != donor.prompt
                or bank.provenance["capture_boundary"] != CAPTURE_BOUNDARY
            ):
                raise ValueError(
                    "Capture does not identify the prescribed source prompt/boundary"
                )
            accumulator.add(bank)
            frames.append(
                {
                    "episode_index": episode["episode_index"],
                    "frame_index": frame_index,
                    "observation_sha256": digest(observation),
                    "capture_bank_id": bank.bank_id,
                }
            )
            if accumulator.count % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "source_id": donor.source_id,
                            "frames": accumulator.count,
                            "expected_frames": source["frame_count"],
                            "new_frames_this_process": accumulator.count
                            - initial_count,
                            "wall_seconds_this_process": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        completed.append(episode)
        resume_name = f"resume_{len(completed):02d}.npz"
        _atomic_npz(directory / resume_name, **accumulator.arrays())
        _atomic_json(
            directory / "progress.json",
            {
                **identity,
                "episodes": completed,
                "frames": frames,
                "frame_count": accumulator.count,
                "invariants": accumulator.invariants,
                "resume_file": resume_name,
                "resume_sha256": file_sha256(directory / resume_name),
            },
        )
        if len(completed) > 1:
            old = directory / f"resume_{len(completed) - 1:02d}.npz"
            with contextlib.suppress(FileNotFoundError):
                old.unlink()
    if accumulator.count != source["frame_count"] or len(completed) != 20:
        raise ValueError(
            "Donor extraction did not cover all 20 prescribed episodes/frames"
        )
    ledger = directory / "frames.jsonl"
    temporary = ledger.with_suffix(".jsonl.tmp")
    with temporary.open("w") as out:
        for row in frames:
            out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(ledger)
    bank = accumulator.bank(
        {
            "capture_kind": "demonstration_mean",
            "source_id": donor.source_id,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "metadata_sha256": METADATA_SHA256,
            "episodes": source["episodes"],
            "frame_count": accumulator.count,
            "frame_ledger_sha256": file_sha256(ledger),
            "preprocessing": PREPROCESSING,
            "aggregation": "float64 sum of all equally weighted frames, divided once and cast float32; noninstruction slots zero",
            "implementation_sha256": _implementation(),
        }
    )
    _atomic_npz(
        directory / "bank.npz", **{name: getattr(bank, name) for name in ARRAY_KEYS}
    )
    manifest = {
        **identity,
        "status": "complete",
        "frame_count": accumulator.count,
        "episode_count": len(completed),
        "bank": bank.metadata(),
        "bank_file": "bank.npz",
        "bank_sha256": file_sha256(directory / "bank.npz"),
        "frame_ledger_file": "frames.jsonl",
        "frame_ledger_sha256": file_sha256(ledger),
        "dataset_file_index_sha256": plan["file_index_sha256"],
        "new_frames_this_process": accumulator.count - initial_count,
        "wall_seconds_this_process": time.perf_counter() - started,
        "resumed_frames": initial_count,
    }
    _atomic_json(directory / "manifest.json", manifest)
    load_bank(directory)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-assets", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    import torch

    from .policy_adapter import load_policy

    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    adapter = load_policy(
        args.checkpoint,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        input_profile="openpi_libero",
        reference_assets=args.reference_assets,
        provenance="frozen standard-demo task latent extraction",
        training_overlap="standard LIBERO training demonstrations; no OOD demonstrations",
    )
    manifest = extract_donor(
        adapter, args.source_id, output_dir=args.output, cache_dir=args.cache
    )
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "bank_id": manifest["bank"]["bank_id"],
                "frame_count": manifest["frame_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
