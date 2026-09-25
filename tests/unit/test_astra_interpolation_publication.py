"""Publication integrity tests with explicitly synthetic temporary artifacts."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "astra_reversal/reports/phase_interpolation/report_publication/publish_report.py"
)
SPEC = importlib.util.spec_from_file_location("report_publication", SOURCE)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private = self.root / "private/report"
        self.private.mkdir(parents=True)
        self.output = self.root / "public/results"
        self.private_receipt = self.root / "private/publication.json"
        report = {
            "schema_version": "astra-phase-interpolation-report-1",
            "status": "complete",
            "phase": "development",
            "expected_cases": 3,
            "seed": 19,
            "synthetic_test_only": True,
            "cases": [{"episode_id": str(index)} for index in range(3)],
            "postprocessor": {"path": str(self.root / "tool.py"), "sha256": "a" * 64},
            "sources": [
                {"path": str(self.root / 'inputs/café".json'), "sha256": "b" * 64}
            ],
            "metrics": {"synthetic_number": 1.0, "synthetic_integer": 3},
            "container_path": "/workspace/astra/model",
        }
        for name in publication.FILES:
            (self.private / name).write_bytes(
                publication.encoded(report)
                if name == "report.json"
                else b"SYNTHETIC TEST ONLY\n"
            )
        self.rebuild_manifest()

    def rebuild_manifest(self):
        report = publication.decode((self.private / "report.json").read_bytes())
        manifest = {
            "schema": "astra-phase-interpolation-report-files-1",
            "phase": "development",
            "postprocessor": report["postprocessor"],
            "files": {
                name: publication.receipt((self.private / name).read_bytes())
                for name in publication.FILES
            },
        }
        (self.private / "manifest.json").write_bytes(publication.encoded(manifest))

    def publish(self, **options):
        return publication.publish(
            self.private,
            self.output,
            self.private_receipt,
            [self.root],
            "development",
            **options,
        )

    def test_paths_only_verbatim_files_private_mapping_and_rebuilt_manifest(self):
        before = {path.name: path.read_bytes() for path in self.private.iterdir()}
        result = self.publish(
            expected_report_sha256=publication.sha(before["report.json"])
        )
        self.assertEqual(result["path_values_changed"], 3)
        self.assertEqual(result["byte_identical_original_files"], 8)
        self.assertEqual(
            before, {path.name: path.read_bytes() for path in self.private.iterdir()}
        )
        report = publication.decode((self.output / "report.json").read_bytes())
        self.assertEqual(report["sources"][0]["path"], 'inputs/café".json')
        self.assertEqual(
            report["metrics"], {"synthetic_number": 1.0, "synthetic_integer": 3}
        )
        self.assertEqual(report["container_path"], "/workspace/astra/model")
        for name in publication.FILES - {"report.json"}:
            self.assertEqual((self.output / name).read_bytes(), before[name])
        manifest = publication.decode((self.output / "manifest.json").read_bytes())
        for name in publication.FILES:
            self.assertEqual(
                manifest["files"][name],
                publication.receipt((self.output / name).read_bytes()),
            )
        exact = publication.decode(self.private_receipt.read_bytes())
        self.assertEqual(
            exact["exact_path_prefix_map"], [{"from": str(self.root) + "/", "to": ""}]
        )
        self.assertEqual(self.private_receipt.stat().st_mode & 0o777, 0o600)
        public = publication.decode((self.output / "publication.json").read_bytes())
        self.assertEqual(
            public["private_mapping_receipt_sha256"],
            publication.sha(self.private_receipt.read_bytes()),
        )
        self.assertEqual(
            (self.output / "source/publish_report.py.txt").read_bytes(),
            SOURCE.read_bytes(),
        )
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn(str(self.root).encode(), path.read_bytes())

    def test_input_checksum_failure_creates_no_publication(self):
        (self.private / "curves.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "completed manifest"):
            self.publish()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.private_receipt.exists())

    def test_incomplete_report_and_unmapped_path_fail_closed(self):
        report = publication.decode((self.private / "report.json").read_bytes())
        report["status"] = "running"
        (self.private / "report.json").write_bytes(publication.encoded(report))
        self.rebuild_manifest()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.publish()
        report["status"] = "complete"
        report["sources"][0]["path"] = "/Users/unmapped/repository/file"
        (self.private / "report.json").write_bytes(publication.encoded(report))
        self.rebuild_manifest()
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            self.publish()
        self.assertFalse(self.output.exists())

    def test_existing_publication_and_wrong_expected_hash_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "expected frozen checksum"):
            self.publish(expected_report_sha256="0" * 64)
        self.publish()
        frozen = (self.output / "publication.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.publish()
        self.assertEqual((self.output / "publication.json").read_bytes(), frozen)

    def test_embedded_prefix_does_not_silently_edit_prose(self):
        with self.assertRaisesRegex(ValueError, "non-path string"):
            publication.normalize(
                json.dumps({"text": "See " + str(self.root) + "/file"}).encode(),
                [str(self.root) + "/"],
            )


if __name__ == "__main__":
    unittest.main()
