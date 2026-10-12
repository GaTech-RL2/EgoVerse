"""Completed trial archives must coexist with the final archive without overwrite."""

import sys
from types import SimpleNamespace

import pytest

from astra_reversal.hardware_interface.osmo_worker import archive


def test_per_trial_and_final_archives_never_replace_prior_work(tmp_path, monkeypatch):
    objects = {}

    def upload(path, bucket, key):
        assert key not in objects
        objects[key] = path

    def put(**kwargs):
        assert kwargs["Key"] not in objects
        objects[kwargs["Key"]] = kwargs["Body"]

    client = SimpleNamespace(
        list_objects_v2=lambda **kwargs: {
            "KeyCount": int(any(key.startswith(kwargs["Prefix"]) for key in objects))
        },
        upload_file=upload,
        put_object=put,
    )
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *a, **k: client)
    )
    for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "unit-test")
    (tmp_path / "receipt.json").write_text('{"passed":true}')
    workflow = "libero-hardware-interface-20261008-unit-test"
    archive(tmp_path, workflow, "trials/pilot-F")
    archive(tmp_path, workflow, "final")
    snapshot = dict(objects)
    with pytest.raises(FileExistsError, match="archive_prefix_already_exists"):
        archive(tmp_path, workflow, "trials/pilot-F")
    assert objects == snapshot and len(objects) == 4
