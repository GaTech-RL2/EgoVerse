"""Integrity checks for the shareable recording export; no GPU or model calls."""

import base64
import json

import pytest

from experiments.libero_hardware_interface.tools.build_process_dashboard import (
    canonical,
    export_value,
    sha,
    validate_archive,
    validate_chain,
)


def chain():
    rows = []
    previous = "0" * 64
    for i, kind in enumerate(["trial_start", "tool_response", "trial_end"]):
        row = {
            "sequence": i,
            "event": kind,
            "previous_sha256": previous,
            "response": {"accepted": i != 1},
        }
        row["sha256"] = sha(canonical(row))
        previous = row["sha256"]
        rows.append(row)
    return rows


def test_recorded_chain_rejects_changed_result_and_missing_event():
    rows = chain()
    assert validate_chain(rows) == rows[-1]["sha256"]
    rows[1]["response"]["accepted"] = True
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_chain(rows)
    with pytest.raises(ValueError, match="Broken event chain"):
        validate_chain(chain()[1:])
    with pytest.raises(ValueError, match="complete, terminated"):
        validate_chain(chain()[:-1])


def test_archive_receipt_authenticates_original_bytes(tmp_path):
    raw = b'{"event":"original"}\n'
    (tmp_path / "events.jsonl").write_bytes(raw)
    receipt = {
        "prefix": "experiments/test/trials/example/",
        "files": [{"path": "events.jsonl", "sha256": sha(raw)}],
    }
    (tmp_path / "archive-receipt.json").write_text(json.dumps(receipt))
    assert validate_archive(tmp_path)["verified_files"] == ["events.jsonl"]
    (tmp_path / "events.jsonl").write_bytes(b"edited")
    with pytest.raises(ValueError, match="Archive digest mismatch"):
        validate_archive(tmp_path)
    receipt["files"][0]["path"] = "../events.jsonl"
    (tmp_path / "archive-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        validate_archive(tmp_path)


def test_png_externalization_preserves_pixels_sample_step_and_other_fields(tmp_path):
    from io import BytesIO

    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (3, 2), (27, 63, 118)).save(stream, "PNG")
    raw = stream.getvalue()
    (tmp_path / "media").mkdir()
    value = {
        "channel": "camera.wrist",
        "metadata": {"simulator_step": 17},
        "value": {
            "encoding": "image/png",
            "shape": [2, 3, 3],
            "base64": base64.b64encode(raw).decode(),
        },
    }
    images = []
    exported = export_value(value, tmp_path, images, step=19)
    assert images[0]["camera"] == "Wrist"
    assert images[0]["step"] == 17
    assert (tmp_path / images[0]["path"]).read_bytes() == raw
    assert exported["metadata"] == value["metadata"]
    assert exported["value"]["png_sha256"] == sha(raw)
    assert "base64" in value["value"]  # Original input is never modified.


def test_opaque_reasoning_is_not_presented_as_readable_reasoning(tmp_path):
    value = {"encrypted_content": "opaque-provider-bytes", "output": "visible message"}
    exported = export_value(value, tmp_path, [])
    assert exported["encrypted_content"] == {
        "opaque_payload_omitted": True,
        "sha256": sha(b"opaque-provider-bytes"),
        "bytes": 21,
    }
    assert exported["output"] == "visible message"
