import io
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from astra_reversal import interpolation_bank as module
from astra_reversal.interpolation_catalog import donor_for
from astra_reversal.interpolation_conditioning import CAPTURE_BOUNDARY, TextLatentBank
from astra_reversal.records import file_sha256


def capture(
    value, *, prompt="put the bowl on the plate", compatibility=None, token_ids=None
):
    return TextLatentBank(
        np.full((3, 1, 6, 4), value, np.float32),
        np.array([[1, 10, 11, 12, 13, 0]], np.int64)
        if token_ids is None
        else token_ids,
        np.array([[1, 1, 1, 1, 1, 0]], bool),
        np.array([[0, 1, 1, 1, 0, 0]], bool),
        {
            "source_prompt": prompt,
            "compatibility": compatibility or {"weights": "frozen"},
            "capture_boundary": CAPTURE_BOUNDARY,
        },
    )


def test_frame_weighted_mean_preserves_tokens_and_zeros_noninstruction_slots():
    accumulator = module.FrameMean()
    for value in (0, 9, 9):
        accumulator.add(capture(value))
    bank = accumulator.bank({"capture_kind": "demonstration_mean"})
    assert accumulator.count == 3
    np.testing.assert_array_equal(bank.states[:, :, 1:4], 6)
    np.testing.assert_array_equal(bank.states[:, :, [0, 4, 5]], 0)
    np.testing.assert_array_equal(bank.token_ids, capture(0).token_ids)
    assert bank.states.dtype == np.float32


@pytest.mark.parametrize("change", ["weights", "tokens", "prompt"])
def test_mean_rejects_cross_frame_model_or_prompt_identity_drift(change):
    accumulator = module.FrameMean()
    accumulator.add(capture(1))
    kwargs = {
        "weights": {"compatibility": {"weights": "other"}},
        "tokens": {"token_ids": np.array([[1, 20, 11, 12, 13, 0]], np.int64)},
        "prompt": {"prompt": "other instruction"},
    }[change]
    with pytest.raises(ValueError, match="identity changed"):
        accumulator.add(capture(2, **kwargs))


def test_embedded_png_is_not_rotated_or_resized_again():
    pixels = np.zeros((256, 256, 3), np.uint8)
    pixels[0, 0] = [1, 2, 3]
    pixels[-1, -1] = [20, 30, 40]
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG")
    np.testing.assert_array_equal(
        module._decode_image({"bytes": out.getvalue(), "path": "unused.png"}), pixels
    )
    with pytest.raises(ValueError, match="embedded"):
        module._decode_image({"bytes": None, "path": "external.png"})


def test_cached_download_rejects_corrupt_file_without_network(tmp_path):
    path = tmp_path / "episode.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module._download("https://unused.invalid", path, expected_sha256="0" * 64)


@pytest.mark.parametrize("wrong_field", [None, "task_index", "frame_index", "state"])
def test_parquet_parser_proves_frame_identity_and_preserves_raw_inputs(
    tmp_path, wrong_field
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pixels = np.zeros((256, 256, 3), np.uint8)
    pixels[0, 0] = [10, 20, 30]
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG")
    encoded = {"bytes": out.getvalue(), "path": "frame.png"}
    rows = [
        {
            "image": encoded,
            "wrist_image": encoded,
            "state": list(range(8)),
            "episode_index": 379,
            "frame_index": frame,
            "task_index": 10,
        }
        for frame in range(2)
    ]
    if wrong_field == "state":
        rows[1][wrong_field] = [0] * 7
    elif wrong_field:
        rows[1][wrong_field] = -1
    path = tmp_path / "episode.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    parsed = module.iter_episode_observations(
        path, {"episode_index": 379, "frame_count": 2}, 10
    )
    if wrong_field:
        with pytest.raises(ValueError, match="identity mismatch|state8"):
            list(parsed)
    else:
        frames = list(parsed)
        assert [frame for frame, _ in frames] == [0, 1]
        np.testing.assert_array_equal(frames[0][1]["observation/image"], pixels)
        np.testing.assert_array_equal(
            frames[0][1]["observation/state"], np.arange(8, dtype=np.float32)
        )


@pytest.fixture
def extraction_fixture(tmp_path, monkeypatch):
    episodes = [
        {
            "episode_index": i,
            "frame_count": 2 if i == 1 else 1,
            "relative_path": f"data/{i}.parquet",
            "sha256": "a" * 64,
            "bytes": 1,
        }
        for i in range(20)
    ]
    donor = replace(
        donor_for("10"),
        episode_indices=tuple(range(20)),
        all_frame_count=21,
        release_frame_count=21,
        episode_frame_counts=tuple(row["frame_count"] for row in episodes),
    )
    source = {
        "source_id": donor.source_id,
        "prompt": donor.prompt,
        "dataset_task_index": donor.dataset_task_index,
        "frame_count": 21,
        "episodes": episodes,
    }
    monkeypatch.setattr(module, "donor_for", lambda _: donor)
    monkeypatch.setattr(
        module,
        "prepare_dataset",
        lambda _: (
            tmp_path / "cache",
            {"sources": [source], "file_index_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        module, "_download", lambda url, destination, **kwargs: destination
    )

    def observations(path, episode, task_index):
        assert task_index == 10
        for frame in range(episode["frame_count"]):
            yield frame, {"value": episode["episode_index"] * 10 + frame}

    monkeypatch.setattr(module, "iter_episode_observations", observations)

    class Adapter:
        metadata = {"model": "frozen", "input_profile": "openpi_libero"}

        def __init__(self, fail_on=None):
            self.calls = 0
            self.fail_on = fail_on

        def capture_text_latents(self, observation, prompt, *, observation_id):
            self.calls += 1
            if self.calls == self.fail_on:
                raise RuntimeError("simulated interruption")
            return capture(observation["value"], prompt=prompt)

    return tmp_path, source, Adapter


def test_extraction_restart_replays_only_uncommitted_episode_and_verifies_complete_bank(
    extraction_fixture,
):
    root, source, Adapter = extraction_fixture
    output = root / "donor10"
    failing = Adapter(fail_on=3)
    with pytest.raises(RuntimeError, match="interruption"):
        module.extract_donor(failing, "10", output_dir=output, cache_dir=root / "cache")
    progress = json.loads((output / "progress.json").read_text())
    assert progress["frame_count"] == 1
    assert [row["episode_index"] for row in progress["episodes"]] == [0]
    adapter = Adapter()
    result = module.extract_donor(
        adapter, "10", output_dir=output, cache_dir=root / "cache"
    )
    assert adapter.calls == 20
    assert result["resumed_frames"] == 1
    assert result["episode_count"] == 20
    assert result["frame_count"] == 21
    bank = module.load_bank(output)
    expected = np.float32(
        sum(i * 10 + j for i in range(20) for j in range(2 if i == 1 else 1)) / 21
    )
    np.testing.assert_array_equal(bank.states[:, :, 1:4], expected)
    assert len(list(output.glob("resume_*.npz"))) == 1
    before = (output / "manifest.json").read_bytes()
    module.extract_donor(adapter, "10", output_dir=output, cache_dir=root / "cache")
    assert adapter.calls == 20
    assert (output / "manifest.json").read_bytes() == before
    assert bank.bank_id == module.load_bank(output / "bank.npz").bank_id


def test_resume_rejects_changed_adapter_and_corrupted_snapshot(extraction_fixture):
    root, _, Adapter = extraction_fixture
    output = root / "donor10"
    with pytest.raises(RuntimeError):
        module.extract_donor(
            Adapter(fail_on=2), "10", output_dir=output, cache_dir=root / "cache"
        )
    changed = Adapter()
    changed.metadata = {"model": "other"}
    with pytest.raises(ValueError, match="restart identity"):
        module.extract_donor(changed, "10", output_dir=output, cache_dir=root / "cache")
    next(output.glob("resume_*.npz")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        module.extract_donor(
            Adapter(), "10", output_dir=output, cache_dir=root / "cache"
        )


def test_loader_detects_array_change_even_when_outer_file_digest_is_updated(
    extraction_fixture,
):
    root, _, Adapter = extraction_fixture
    output = root / "donor10"
    module.extract_donor(Adapter(), "10", output_dir=output, cache_dir=root / "cache")
    with np.load(output / "bank.npz", allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    arrays["states"][0, 0, 1, 0] += 1
    module._atomic_npz(output / "bank.npz", **arrays)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["bank_sha256"] = file_sha256(output / "bank.npz")
    module._atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="arrays/provenance"):
        module.load_bank(output)


def test_loader_rejects_duplicate_frame_ledger_after_outer_hash_update(
    extraction_fixture,
):
    root, _, Adapter = extraction_fixture
    output = root / "donor10"
    module.extract_donor(Adapter(), "10", output_dir=output, cache_dir=root / "cache")
    lines = (output / "frames.jsonl").read_text().splitlines()
    lines[1] = lines[0]
    (output / "frames.jsonl").write_text("\n".join(lines) + "\n")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["frame_ledger_sha256"] = file_sha256(output / "frames.jsonl")
    module._atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="duplicate coverage"):
        module.load_bank(output)
