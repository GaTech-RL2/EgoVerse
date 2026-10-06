"""Immutable candidate archive, prompt-only control, and gated final bundles."""

import ast
import json
from pathlib import Path

from astra_reversal.demo_segments import write_json
from astra_reversal.records import digest, file_sha256

from .harness import Harness


def prompt_only_change(baseline, candidate):
    def masked(source):
        tree = ast.parse(source)
        changed = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for index, key in enumerate(node.keys):
                    if isinstance(key, ast.Constant) and key.value == "instruction":
                        if not isinstance(
                            node.values[index], ast.Constant
                        ) or not isinstance(node.values[index].value, str):
                            raise ValueError(
                                "Prompt-only search may edit a literal instruction only"
                            )
                        node.values[index] = ast.Constant(value="PROMPT_ONLY_SLOT")
                        changed += 1
        if changed != 1:
            raise ValueError("Expected one prompt-only slot")
        return ast.dump(tree, include_attributes=False)

    return masked(baseline) == masked(candidate)


class Archive:
    def __init__(self, root, manifest, *, create=True):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            write_json(self.root / "manifest.json", manifest)
            (self.root / "candidates").mkdir()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest != manifest:
            raise ValueError("Cannot change a study manifest during search")
        self.manifest_sha256 = file_sha256(self.root / "manifest.json")

    def register(self, source, hypothesis, *, kind, baseline):
        if (self.root / "selected_bundle").exists():
            raise ValueError("The selected system is frozen")
        if (
            kind not in ("initial", "prompt_only", "meta_harness")
            or not isinstance(hypothesis, str)
            or not hypothesis.strip()
        ):
            raise ValueError("A candidate needs a kind and a falsifiable hypothesis")
        identity = f"candidate_{len(list((self.root / 'candidates').iterdir())):03d}"
        directory = self.root / "candidates" / identity
        directory.mkdir()
        (directory / "harness.py").write_text(source)
        row = {
            "candidate_id": identity,
            "hypothesis": hypothesis,
            "kind": kind,
            "source_sha256": file_sha256(directory / "harness.py"),
            "status": "admitted",
            "manifest_sha256": self.manifest_sha256,
        }
        try:
            Harness(source)
            if kind == "prompt_only" and not prompt_only_change(baseline, source):
                raise ValueError("Prompt-only candidate changed executable logic")
        except (ValueError, SyntaxError) as error:
            row.update(status="rejected", reason=str(error))
        write_json(directory / "candidate.json", row)
        return row

    def record(self, candidate_id, results, aggregate):
        directory = self.root / "candidates" / candidate_id
        row = json.loads((directory / "candidate.json").read_text())
        if row["status"] != "admitted" or (directory / "search_metrics.json").exists():
            raise ValueError("Rejected or already evaluated candidate")
        if file_sha256(directory / "harness.py") != row["source_sha256"]:
            raise ValueError("Candidate changed during evaluation")
        expected = set(self.manifest["search_episode_ids"])
        ids = [r["episode_id"] for r in results]
        valid = bool(results) and set(ids) == expected and len(ids) == len(expected)
        violations = []
        for result in results:
            c = result["controller"]
            if (
                c["mode"] != "async"
                or c["runtime_requests"] > self.manifest["limits"]["calls"]
                or result["actions_executed"] > 300
                or c["contract_errors"]
            ):
                violations.append("execution_contract")
            for record in c["provider_records"]:
                if (
                    record.get("hard_token_cap_enforced") is False
                    or record.get("error")
                    or not record.get("usage")
                ):
                    violations.append("provider_or_token_contract")
            if result["card_manifest_sha256"] != self.manifest["card_manifest_sha256"]:
                violations.append("card_manifest_changed")
            if result["checkpoint_identity"] != self.manifest["checkpoint_identity"]:
                violations.append("checkpoint_identity_changed")
        valid = valid and not violations
        write_json(
            directory / "search_metrics.json",
            {
                "eligible": valid,
                "violations": sorted(set(violations)),
                "aggregate": aggregate,
                "episode_ids": ids,
                "results_sha256": digest(results),
                "manifest_sha256": self.manifest_sha256,
            },
        )

    def freeze(self, candidate_id):
        directory = self.root / "candidates" / candidate_id
        row = json.loads((directory / "candidate.json").read_text())
        metrics = json.loads((directory / "search_metrics.json").read_text())
        if (
            not metrics["eligible"]
            or file_sha256(directory / "harness.py") != row["source_sha256"]
        ):
            raise ValueError("Only a valid evaluated candidate can be frozen")
        output = self.root / "selected_bundle"
        output.mkdir(exist_ok=False)
        (output / "harness.py").write_bytes((directory / "harness.py").read_bytes())
        write_json(
            output / "manifest.json",
            {
                "study": self.manifest,
                "candidate": row,
                "search_metrics_sha256": file_sha256(directory / "search_metrics.json"),
            },
        )
        return output


class SearchView:
    """A capability limited to this search's candidates, never final evaluation."""

    def __init__(self, archive):
        self.root = (archive.root / "candidates").resolve()

    def files(self):
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and p.suffix in (".json", ".jsonl", ".py", ".png")
        )

    def read(self, relative, *, offset=0, count=16000):
        path = self.root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(self.root)
            or relative not in self.files()
            or type(offset) is not int
            or offset < 0
            or type(count) is not int
            or not 1 <= count <= 16000
        ):
            raise ValueError("File is outside the search evidence capability")
        if path.suffix == ".png":
            raise ValueError("Use the separate image reader")
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(count)
        return {
            "path": relative,
            "offset": offset,
            "text": raw.decode(errors="replace"),
            "file_sha256": file_sha256(path),
            "size": path.stat().st_size,
        }


def rank_candidates(archive):
    rows = []
    for directory in (archive.root / "candidates").iterdir():
        path = directory / "search_metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text())
            if metrics["eligible"]:
                rows.append({"candidate_id": directory.name, **metrics["aggregate"]})
    # Selection cost is secondary only on exact success ties. Close, nonidentical
    # results require matched replication before claiming a reliable advantage.
    return sorted(
        rows,
        key=lambda r: (
            -r["task_macro_success"],
            r["input_tokens_known"] + r["output_tokens_known"],
            r["candidate_id"],
        ),
    )


def split_plan():
    suites = ("libero_goal_ood", "libero_spatial_ood")
    return {
        "development_tasks": [[s, i] for s in suites for i in range(2, 8)],
        "search_tasks": [[s, i] for s in suites for i in (0, 1)],
        "final_tasks": [[s, i] for s in suites for i in (8, 9)],
        "search_resets": [14, 15, 16, 17, 18],
        "final_resets": list(range(19, 44)),
        "seed": 137,
        "generalization_claim": "fresh resets of previously inspected/adapted compositions, NOT unseen-task transfer",
    }
