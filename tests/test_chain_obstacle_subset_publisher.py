import hashlib
import json
import os
from pathlib import Path

import scripts.data.publish_chain_obstacle_3000_subset as publisher


def test_balanced_3000_selection_contract() -> None:
    manual, generated, selected = publisher._selected_indices(
        source_per_level=128,
        manual_per_level=16,
        target_per_level=100,
    )

    assert manual == list(range(16))
    assert len(generated) == len(set(generated)) == 84
    assert generated[0] == 16
    assert generated[-1] == 127
    assert len(selected) == len(set(selected)) == 100
    assert selected == sorted(selected)


def test_publication_is_atomic_audited_and_does_not_chmod_sources(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(publisher, "LEVELS", (1, 2))
    source = tmp_path / "output_8"
    for level in publisher.LEVELS:
        level_root = source / f"level_{level:02d}" / "chain_gripper" / "T"
        for index in range(8):
            episode = level_root / f"episode_obs{level}_{index:06d}.zarr"
            episode.mkdir(parents=True)
            (episode / "zarr.json").write_text(
                json.dumps({"attributes": {"total_frames": 100 + index}})
            )

    source_audit = {
        "generated_per_level": 6,
        "levels": list(publisher.LEVELS),
        "manual_per_level": 2,
        "status": "PASS",
        "target_total_per_level": 8,
    }
    source_audit_path = source / "audit_report.json"
    source_audit_path.write_text(json.dumps(source_audit))
    source_audit_sha = hashlib.sha256(source_audit_path.read_bytes()).hexdigest()

    manifest, inventory_bytes, audit, audit_bytes = publisher._build_publication(
        source_root=source,
        source_audit_sha256=source_audit_sha,
        source_per_level=8,
        manual_per_level=2,
        target_per_level=5,
    )
    output = tmp_path / "output_10_balanced"
    publisher._publish(
        output_root=output,
        source_root=source,
        manifest=manifest,
        inventory_bytes=inventory_bytes,
        audit_bytes=audit_bytes,
    )
    publisher._verify_tree(
        output_root=output,
        source_root=source,
        manifest=manifest,
        inventory_bytes=inventory_bytes,
        audit_bytes=audit_bytes,
    )

    assert audit["status"] == "PASS"
    assert audit["total_episodes"] == 10
    assert len(manifest["episodes"]) == 10
    assert (output.stat().st_mode & 0o777) == 0o555
    assert ((output / "subset_manifest.json").stat().st_mode & 0o777) == 0o444
    for episode in source.glob("level_*/chain_gripper/T/*.zarr"):
        assert (episode.stat().st_mode & 0o777) == 0o755

    # Restore only the test's real directories so tmp_path cleanup can remove it.
    for root, _dirs, _files in os.walk(output, topdown=False, followlinks=False):
        Path(root).chmod(0o755)
