#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ICE_DIR = Path(__file__).parents[1] / "scripts" / "ice"
INVENTORY_PATH = ICE_DIR / "ice_checkpoint_archive_inventory.py"
sys.path.insert(0, str(ICE_DIR))
SPEC = importlib.util.spec_from_file_location("ice_checkpoint_archive_inventory", INVENTORY_PATH)
INVENTORY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = INVENTORY
SPEC.loader.exec_module(INVENTORY)


def make_validator(root: Path) -> Path:
    validator = root / "validate.py"
    validator.write_text("#!/bin/sh\nexit 0\n")
    validator.chmod(0o700)
    return validator


def write_complete(run: Path, checkpoint: Path, step: int) -> None:
    (run / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "checkpoint": {
                    "global_step": step,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            }
        )
    )


class ArchiveInventoryTest(unittest.TestCase):
    def test_distinguishes_completed_archived_pending_and_unregistered(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "scratch" / "campaign"
            state_dir = root / "service" / "state"
            state_dir.mkdir(parents=True)
            validator = make_validator(Path(raw))

            archived = root / "archived"
            archived_ckpt = archived / "checkpoints" / "step-10.ckpt"
            archived_ckpt.parent.mkdir(parents=True)
            archived_ckpt.write_text("archived")
            write_complete(archived, archived_ckpt, 10)

            pending = root / "pending"
            pending_ckpt = pending / "checkpoints" / "step-20.ckpt"
            pending_ckpt.parent.mkdir(parents=True)
            pending_ckpt.write_text("pending")
            write_complete(pending, pending_ckpt, 20)

            unknown = root / "unknown"
            unknown_ckpt = unknown / "checkpoints" / "step-30.ckpt"
            unknown_ckpt.parent.mkdir(parents=True)
            unknown_ckpt.write_text("unknown")

            manifest = Path(raw) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "runs": [
                            {
                                "id": "archived",
                                "local_root": str(archived),
                                "checkpoint_glob": "checkpoints/*.ckpt",
                                "validator": str(validator),
                                "remote_dir": "/remote/archive/archived",
                            },
                            {
                                "id": "pending",
                                "local_root": str(pending),
                                "checkpoint_glob": "checkpoints/*.ckpt",
                                "validator": str(validator),
                                "remote_dir": "/remote/archive/pending",
                            },
                        ],
                    }
                )
            )
            (state_dir / "mirror-state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "files": {
                            str(archived_ckpt.resolve()): {
                                "run_id": "archived",
                                "remote_verified": True,
                                "global_step": 10,
                                "sha256": hashlib.sha256(archived_ckpt.read_bytes()).hexdigest(),
                            }
                        },
                    }
                )
            )
            output = state_dir / "archive-inventory.json"
            code = INVENTORY.main(
                [
                    "--search-root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--state-dir",
                    str(state_dir),
                    "--output",
                    str(output),
                    "--require-clear",
                ]
            )
            self.assertEqual(code, 1)
            report = json.loads(output.read_text())
            statuses = {Path(row["run_root"]).name: row["status"] for row in report["runs"]}
            self.assertEqual(statuses["archived"], "complete_archived")
            self.assertEqual(statuses["pending"], "complete_needs_transfer")
            self.assertEqual(statuses["unknown"], "unregistered")

    def test_slurm_like_metadata_does_not_imply_completion(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "scratch" / "campaign"
            run = root / "run"
            checkpoint = run / "checkpoints" / "step.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("checkpoint")
            (run / "slurm-COMPLETED.txt").write_text("COMPLETED")
            report = INVENTORY.build_inventory(
                root,
                {"schema_version": 1, "runs": []},
                {"schema_version": 2, "files": {}},
                4,
            )
            self.assertFalse(report["runs"][0]["complete"])

    def test_completed_run_with_missing_checkpoint_is_not_invisible(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "scratch" / "campaign"
            run = root / "run"
            run.mkdir(parents=True)
            (run / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "COMPLETE",
                        "checkpoint": {"global_step": 10, "sha256": "a" * 64},
                    }
                )
            )
            report = INVENTORY.build_inventory(
                root,
                {"schema_version": 1, "runs": []},
                {"schema_version": 2, "files": {}},
                4,
            )
            self.assertEqual(report["runs"][0]["status"], "unregistered")
            self.assertTrue(report["runs"][0]["complete"])
            self.assertEqual(report["runs"][0]["checkpoint_count"], 0)

    def test_registered_complete_run_with_missing_checkpoint_requires_attention(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "scratch" / "campaign"
            run = root / "run"
            run.mkdir(parents=True)
            sentinel = run / "COMPLETE.json"
            sentinel.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "COMPLETE",
                        "checkpoint": {"global_step": 10, "sha256": "a" * 64},
                    }
                )
            )
            report = INVENTORY.build_inventory(
                root,
                {
                    "schema_version": 1,
                    "runs": [
                        {
                            "id": "run",
                            "local_root": str(run),
                            "checkpoint_glob": "checkpoints/*.ckpt",
                            "validator": "/unused/by_inventory",
                            "remote_dir": "/remote/archive/run",
                            "completion_sentinel": str(sentinel),
                        }
                    ],
                },
                {"schema_version": 2, "files": {}},
                4,
            )
            self.assertEqual(report["runs"][0]["status"], "complete_missing_checkpoints")
            self.assertEqual(report["attention_required"], [str(run.resolve())])


if __name__ == "__main__":
    unittest.main()
