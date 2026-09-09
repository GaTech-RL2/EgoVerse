import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "ice" / "capture_runtime_lock.py"
)
SPEC = importlib.util.spec_from_file_location("capture_runtime_lock", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_canonical_bytes_are_stable_and_newline_terminated():
    first = MODULE.canonical_bytes({"z": 1, "a": [2, 3]})
    second = MODULE.canonical_bytes({"a": [2, 3], "z": 1})

    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first) == {"a": [2, 3], "z": 1}


def test_atomic_create_refuses_overwrite(tmp_path):
    output = tmp_path / "locks" / "runtime.json"
    MODULE.atomic_create(output, b"first\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.atomic_create(output, b"second\n")

    assert output.read_bytes() == b"first\n"


def test_runtime_lock_source_has_no_deployment_specific_paths():
    text = MODULE_PATH.read_text()

    assert "/coc/" not in text
    assert "/storage/" not in text
    assert "paphiwetsa3" not in text
