"""Exercise standalone upload contents without downloading model assets."""

import hashlib
import json
import sys
import tarfile

import pytest

from astra_reversal.osmo import package_payload


@pytest.fixture
def upload_tree(tmp_path, monkeypatch):
    files = {
        "astra_reversal/__init__.py": "",
        "astra_reversal/.deps/tokenizers/paligemma-3b-pt-224/tokenizer.json": "{}",
        "astra_reversal/checkpoints/paligemma_tokenizer.json": json.dumps(
            {"files": [{"path": "tokenizer.json"}]}
        ),
        "astra_reversal/.deps/reference/cpu_libero_probe.npz": "fixture",
        "astra_reversal/.deps/reference/pi05_libero/norm_stats.json": "{}",
        "astra_reversal/.deps/reference/pi05_libero/paligemma_tokenizer.model": "fixture",
        "astra_reversal/.deps/private.key": "synthetic-secret-must-stay-local",
        "astra_reversal/.deps/checkpoints/model.safetensors": "excluded-weights",
        "astra_reversal/artifacts/old.json": "{}",
        "tests/unit/astra/test_smoke.py": "def test_smoke(): pass\n",
        "tests/integration/test_astra_lerobot_policy.py": "# native smoke\n",
        "tests/fixtures/astra/lerobot_pi05/config.json": "{}",
        "tests/conftest.py": "raise AssertionError('unrelated project hooks')\n",
        "tests/unit/test_unrelated.py": "# unrelated\n",
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "astra_reversal/.deps/new-upload/payload.tar.gz"
    monkeypatch.setattr(sys, "argv", ["package_payload", "--output", str(output)])
    return output


def test_upload_includes_relocated_tests_and_only_allowed_assets(upload_tree):
    package_payload.main()

    with tarfile.open(upload_tree) as archive:
        names = {member.name for member in archive if member.isfile()}
    assert {
        "tests/unit/astra/test_smoke.py",
        "tests/integration/test_astra_lerobot_policy.py",
        "tests/fixtures/astra/lerobot_pi05/config.json",
        "astra_reversal/.deps/tokenizers/paligemma-3b-pt-224/tokenizer.json",
        "astra_reversal/.deps/reference/pi05_libero/norm_stats.json",
    } <= names
    assert (
        not {
            "tests/conftest.py",
            "tests/unit/test_unrelated.py",
            "astra_reversal/.deps/private.key",
            "astra_reversal/.deps/checkpoints/model.safetensors",
            "astra_reversal/artifacts/old.json",
        }
        & names
    )


def test_explicit_payload_cannot_overwrite_a_frozen_upload(upload_tree):
    package_payload.main()
    before = upload_tree.read_bytes()
    checksum = upload_tree.parent / "payload.sha256"
    assert checksum.read_text().strip() == hashlib.sha256(before).hexdigest()

    with pytest.raises(FileExistsError, match="immutable"):
        package_payload.main()

    assert upload_tree.read_bytes() == before
    assert checksum.read_text().strip() == hashlib.sha256(before).hexdigest()
