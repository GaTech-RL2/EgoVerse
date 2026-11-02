"""Accept an original normalization proof only across one reviewed eval-only edge.

The original report and statistics are never rewritten. An external receipt must
bind the exact source diff, independent review, data, configuration and all 36
original report/stat hashes. Ordinary same-commit validation remains unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

DATASET_MODULE = "egomimic/rldb/zarr/zarr_dataset_multi.py"
OPERATIONAL_FILES = {
    "launch_pace.py",
    "run_preflight.py",
    "pipeline.py",
    "validate_setup.py",
    "normalization_provenance.py",
}
REUSE_CHECKS = {
    "normalization_kernel_unchanged",
    "training_recipe_unchanged",
    "default_training_reads_unchanged",
    "original_norm_reports_preserved",
    "data_splits_and_stats_unchanged",
    "full_unit_suite_passed",
}


def _need(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _allowed_path(name):
    path = Path(name)
    if name == DATASET_MODULE:
        return True
    if name.startswith(("egomimic/eval/", "tests/", "docs/")):
        return True
    prefix = "experiments/fold_clothes_moving_opscale/"
    if name.startswith(prefix):
        tail = name[len(prefix) :]
        return (
            tail.startswith("operations/")
            or tail in OPERATIONAL_FILES
            or path.suffix == ".md"
        )
    return False


def _normalization_kernel(text):
    """Existing non-validation functions and class attributes are the kernel.

    Additive evaluation readers may be introduced, but every existing training,
    transform/window, normalization, sampling and bounds method must be identical.
    """
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef) and node.name != "EvenStrideDataset":
            result[f"{node.name}:bases"] = repr(
                (
                    [ast.dump(base) for base in node.bases],
                    [ast.dump(item) for item in node.keywords],
                    [ast.dump(item) for item in node.decorator_list],
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{item.name}"] = ast.dump(
                        item, include_attributes=False
                    )
                elif not (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    result[
                        f"{node.name}:attribute:{ast.dump(item, include_attributes=False)}"
                    ] = ast.dump(item, include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
            value = ast.dump(node, include_attributes=False)
            result[f"module:{value}"] = value
    return result


def source_equivalence(repo, source_commit, target_commit):
    for value in (source_commit, target_commit):
        _need(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            "Normalization reuse requires full source commits",
        )
    changed = sorted(
        _git(repo, "diff", "--name-only", source_commit, target_commit)
        .decode()
        .splitlines()
    )
    _need(
        changed and all(_allowed_path(name) for name in changed),
        "Normalization reuse includes unreviewable training/configuration changes",
    )
    before = _normalization_kernel(
        _git(repo, "show", f"{source_commit}:{DATASET_MODULE}").decode()
    )
    after = _normalization_kernel(
        _git(repo, "show", f"{target_commit}:{DATASET_MODULE}").decode()
    )
    _need(
        all(after.get(name) == value for name, value in before.items()),
        "Normalization/training data kernel changed",
    )
    diff = _git(repo, "diff", "--binary", "--full-index", source_commit, target_commit)
    return {
        "reviewed_changed_paths": changed,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def reuse_artifacts_digest(receipt):
    """Bind the complete original artifact inventory to independent approval."""
    identity = {
        key: receipt[key]
        for key in (
            "source_commit",
            "target_commit",
            "design_sha256",
            "canonical_index_sha256",
            "reports",
        )
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalization_source_matches(run_root, report, target_commit, repo):
    """Validate one preserved original report; callers retain all other gates."""
    source_commit = report.get("commit")
    if source_commit == target_commit:
        return True
    root, repo = Path(run_root).resolve(), Path(repo).resolve()
    receipt_path = root / "normalization_reuse.json"
    _need(receipt_path.is_file(), "No reviewed normalization compatibility receipt")
    receipt = json.loads(receipt_path.read_text())
    _need(
        receipt.get("schema_version") == 1
        and receipt.get("status") == "passed"
        and receipt.get("scope") == "evaluation_only_source_change"
        and receipt.get("source_commit") == source_commit
        and receipt.get("target_commit") == target_commit,
        "Normalization compatibility edge does not match original/current source",
    )
    _need(
        _git(repo, "rev-parse", "HEAD").decode().strip() == target_commit,
        "Current normalization consumer source changed",
    )
    _need(
        not _git(repo, "status", "--porcelain").strip(),
        "Normalization consumer source is dirty",
    )
    comparison = source_equivalence(repo, source_commit, target_commit)
    _need(
        all(receipt.get(key) == value for key, value in comparison.items()),
        "Reviewed normalization source diff changed",
    )
    review_path = Path(receipt["review_path"]).resolve()
    _need(
        review_path.is_relative_to(root / "operations")
        and _sha(review_path) == receipt.get("review_sha256"),
        "Normalization reuse review changed",
    )
    review = json.loads(review_path.read_text())
    _need(
        review.get("decision") == "approved"
        and review.get("normalization_reuse", {}).get("artifacts_sha256")
        == reuse_artifacts_digest(receipt)
        and review.get("normalization_reuse", {}).get("source_diff_sha256")
        == comparison["source_diff_sha256"]
        and all(
            review.get("normalization_reuse", {}).get("checks", {}).get(key) is True
            for key in REUSE_CHECKS
        ),
        "Independent review did not approve this normalization compatibility edge",
    )
    for key, path in (
        ("design_sha256", root / "private/moving_opscale_design.json"),
        ("canonical_index_sha256", root / "private/canonical_zarr/_episode_index.json"),
    ):
        _need(
            report.get(key) == receipt.get(key) == _sha(path),
            "Normalization compatibility data identity changed",
        )
    expected_cells = {
        f"b{b}_op{n}" for b in (24, 32, 48, 64, 72, 96) for n in (4, 8, 16, 32, 64, 128)
    }
    entries = receipt.get("reports", {})
    _need(
        set(entries) == expected_cells,
        "Normalization compatibility must bind all 36 original reports",
    )
    cell = f"b{report['budget_hours']}_op{report['operator_count']}"
    entry = entries[cell]
    original_path = root / "norm" / cell / "norm_complete.json"
    _need(
        json.loads(original_path.read_text()) == report,
        "Normalization reuse received a modified report",
    )
    _need(
        entry.get("report_sha256") == _sha(original_path),
        "Original normalization report changed",
    )
    _need(
        entry.get("stats_sha256")
        == report.get("stats_sha256")
        == _sha(report["stats_path"]),
        "Original normalization values changed",
    )
    for field in ("config_sha256", "split_sha256"):
        _need(
            entry.get(field) == report.get(field),
            "Normalization configuration/split receipt changed",
        )
    return True


def normalization_reuse_identity(run_root, target_commit):
    """Record the compatibility proof alongside its original producing source."""
    path = Path(run_root).resolve() / "normalization_reuse.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    _need(
        receipt.get("target_commit") == target_commit,
        "Stale normalization reuse identity",
    )
    return {
        "path": str(path),
        "sha256": _sha(path),
        "producer_commit": receipt["source_commit"],
        "consumer_commit": target_commit,
    }
