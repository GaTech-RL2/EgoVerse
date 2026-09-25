"""The worker payload must contain donor bytes even when its source is linked."""

import json
import sys
import tarfile
from types import SimpleNamespace

from astra_reversal import image_donor_bank
from astra_reversal.osmo import package_payload


def test_linked_donor_root_is_portable_regular_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "astra_reversal"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints/paligemma_tokenizer.json").write_text(
        json.dumps({"files": []})
    )
    for name in (
        "reference/cpu_libero_probe.npz",
        "reference/pi05_libero/norm_stats.json",
        "reference/pi05_libero/paligemma_tokenizer.model",
    ):
        path = root / ".deps" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unrelated authorized asset")
    cache = tmp_path / "external_donor_cache"
    (cache / "images").mkdir(parents=True)
    (cache / "manifest.json").write_bytes(b"verified manifest")
    (cache / "images/camera.png").write_bytes(b"verified image bytes")
    linked = root / ".deps/image-perturbations/donors"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(cache, target_is_directory=True)
    # Pixel/manifest validation is covered by the donor loader's tests. This
    # regression isolates tar handling after that validation has succeeded.
    seen = []

    def verified_library(path):
        seen.append(path.resolve())
        return SimpleNamespace(catalog=lambda: [{}] * 45)

    monkeypatch.setattr(image_donor_bank, "load_library", verified_library)
    monkeypatch.setattr(package_payload, "TEST_PATHS", ())
    output = tmp_path / "payload.tar.gz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["package_payload", "--include-image-donors", "--output", str(output)],
    )
    package_payload.main()
    assert seen == [cache.resolve()]
    with tarfile.open(output) as archive:
        assert all(not member.issym() and not member.islnk() for member in archive)
        archive.extractall(tmp_path / "worker", filter="data")
    received = tmp_path / "worker/astra_reversal/.deps/image-perturbations/donors"
    assert (received / "manifest.json").read_bytes() == b"verified manifest"
    assert (received / "images/camera.png").read_bytes() == b"verified image bytes"
