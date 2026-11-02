"""An eval-only change may reuse unchanged stats through one reviewed edge."""

import hashlib
import json
import subprocess

import pytest

from experiments.fold_clothes_moving_opscale import (
    normalization_provenance as provenance,
)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def compatibility(tmp_path):
    repo, root = tmp_path / "repo", tmp_path / "runs"
    repo.mkdir()

    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Unit Test")
    module = repo / provenance.DATASET_MODULE
    module.parent.mkdir(parents=True)
    module.write_text(
        "class ZarrDataset:\n    def __getitem__(self, i):\n        return i\n\nclass EvenStrideDataset:\n    def _read_exact(self, i):\n        return i\n"
    )
    git("add", ".")
    git("commit", "-qm", "original")
    old = git("rev-parse", "HEAD")
    module.write_text(
        module.read_text().replace(
            "class EvenStrideDataset:",
            "    def evaluation_tracking_validity(self, i):\n        return True\n\nclass EvenStrideDataset:",
        )
    )
    git("add", ".")
    git("commit", "-qm", "eval metadata only")
    new = git("rev-parse", "HEAD")
    design = root / "private/moving_opscale_design.json"
    canonical = root / "private/canonical_zarr/_episode_index.json"
    save(design, {"synthetic": True})
    save(canonical, {"synthetic": True})
    entries = {}
    for budget in (24, 32, 48, 64, 72, 96):
        for operators in (4, 8, 16, 32, 64, 128):
            cell = f"b{budget}_op{operators}"
            folder = root / "norm" / cell
            stats = folder / "norm_stats.json"
            save(stats, {"mean": [1], "std": [2]})
            report = {
                "status": "passed",
                "commit": old,
                "budget_hours": budget,
                "operator_count": operators,
                "design_sha256": sha(design),
                "canonical_index_sha256": sha(canonical),
                "config_sha256": "c" * 64,
                "split_sha256": "d" * 64,
                "stats_path": str(stats),
                "stats_sha256": sha(stats),
            }
            path = folder / "norm_complete.json"
            save(path, report)
            entries[cell] = {
                key: report[key]
                for key in ("stats_sha256", "config_sha256", "split_sha256")
            }
            entries[cell]["report_sha256"] = sha(path)
    report = json.loads((root / "norm/b24_op4/norm_complete.json").read_text())
    comparison = provenance.source_equivalence(repo, old, new)
    review_path = root / "operations/review.json"
    review = {
        "decision": "approved",
        "normalization_reuse": {
            **comparison,
            "checks": dict.fromkeys(provenance.REUSE_CHECKS, True),
        },
    }
    save(review_path, review)
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scope": "evaluation_only_source_change",
        "source_commit": old,
        "target_commit": new,
        **comparison,
        "review_path": str(review_path),
        "review_sha256": sha(review_path),
        "design_sha256": sha(design),
        "canonical_index_sha256": sha(canonical),
        "reports": entries,
    }
    review["normalization_reuse"]["artifacts_sha256"] = (
        provenance.reuse_artifacts_digest(receipt)
    )
    save(review_path, review)
    receipt["review_sha256"] = sha(review_path)
    receipt_path = root / "normalization_reuse.json"
    save(receipt_path, receipt)
    return root, repo, old, new, report, receipt, receipt_path, git, module


def test_reviewed_edge_accepts_original_report_without_rewriting_it(compatibility):
    root, repo, old, new, report, *_ = compatibility
    path = root / "norm/b24_op4/norm_complete.json"
    original = path.read_bytes()
    assert provenance.normalization_source_matches(root, report, new, repo)
    assert report["commit"] == old
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        "missing_receipt",
        "other_target",
        "altered_report",
        "altered_stats",
        "missing_cell",
        "unreviewed_diff",
        "changed_data",
    ],
)
def test_reuse_fails_closed_on_changed_provenance(compatibility, change):
    root, repo, _, new, report, receipt, receipt_path, *_ = compatibility
    if change == "missing_receipt":
        receipt_path.unlink()
    elif change == "other_target":
        receipt["target_commit"] = "f" * 40
        save(receipt_path, receipt)
    elif change == "altered_report":
        save(root / "norm/b24_op4/norm_complete.json", {**report, "status": "running"})
    elif change == "altered_stats":
        save(root / "norm/b24_op4/norm_stats.json", {"mean": [99]})
    elif change == "missing_cell":
        receipt["reports"].pop("b96_op128")
        save(receipt_path, receipt)
    elif change == "unreviewed_diff":
        receipt["source_diff_sha256"] = "0" * 64
        save(receipt_path, receipt)
    else:
        save(root / "private/canonical_zarr/_episode_index.json", {"changed": True})
    with pytest.raises(RuntimeError):
        provenance.normalization_source_matches(root, report, new, repo)


def test_changed_training_read_kernel_cannot_be_approved_as_eval_only(compatibility):
    _, repo, old, _, _, _, _, git, module = compatibility
    module.write_text(module.read_text().replace("return i", "return i + 1", 1))
    git("add", ".")
    git("commit", "-qm", "changed training read")
    with pytest.raises(RuntimeError, match="kernel changed"):
        provenance.source_equivalence(repo, old, git("rev-parse", "HEAD"))


def test_rewriting_stats_report_and_outer_receipt_cannot_reuse_old_review(
    compatibility,
):
    root, repo, _, new, report, receipt, receipt_path, *_ = compatibility
    stats = root / "norm/b24_op4/norm_stats.json"
    save(stats, {"mean": [999]})
    report["stats_sha256"] = sha(stats)
    original = root / "norm/b24_op4/norm_complete.json"
    save(original, report)
    receipt["reports"]["b24_op4"].update(
        stats_sha256=sha(stats), report_sha256=sha(original)
    )
    save(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="Independent review"):
        provenance.normalization_source_matches(root, report, new, repo)


def test_model_or_config_changes_cannot_reuse_old_normalization(compatibility):
    _, repo, old, _, _, _, _, git, _ = compatibility
    config = repo / "egomimic/hydra_configs/model/recipe.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("batch_size: 99\n")
    git("add", ".")
    git("commit", "-qm", "changed recipe")
    with pytest.raises(RuntimeError, match="training/configuration"):
        provenance.source_equivalence(repo, old, git("rev-parse", "HEAD"))
