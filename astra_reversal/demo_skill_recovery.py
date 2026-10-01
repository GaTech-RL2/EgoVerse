"""Continue an interrupted search by verifying and replaying recorded evidence.

Replaying control flow does not execute old robot actions or call Astra again.
Unknown provider attempts remain in the ledger; fresh calls get a new identity.
"""

import base64
import copy
import io
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .astra_client import _append_record
from .demo_skill_agent import PROMPT_TEMPLATE_VERSION, SYSTEM_PROMPT, parse_proposal
from .records import digest, file_sha256


def confined(root, relative):
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("Recovery artifact escapes its recorded directory")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Recovery artifact escapes its recorded directory")
    return path


class RecoveryArchive:
    def __init__(self, results, *, arm, bank_id, protocol, namespace):
        self.results = Path(results)
        self.directory = self.results / "experiment"
        self.report = json.loads((self.directory / "summary.json").read_text())
        runtime = json.loads((self.results / "runtime.json").read_text())
        if (
            self.report["arm"] != arm
            or self.report["bank_id"] != bank_id
            or self.report["protocol"] != protocol
            or self.report["status"] == "complete"
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,180}", namespace)
            or namespace == runtime["workflow"]
        ):
            raise ValueError(
                "Recovery requires the same incomplete protocol and a new run"
            )
        self.namespace = namespace
        self.astra = protocol["astra"]
        self.rollouts = {r["attempt_id"]: r for r in self.report["physical_rollouts"]}
        if len(self.rollouts) != len(self.report["physical_rollouts"]):
            raise ValueError("Recovery contains duplicate physical attempts")
        self.decisions, self.requests = {}, {}
        for path in sorted((self.directory / "decisions").glob("*.json")):
            index = int(path.stem)
            self.decisions[index] = json.loads(path.read_text())
            self.requests[index] = json.loads(
                (self.directory / "requests" / path.name).read_text()
            )
        if sorted(self.decisions) != list(range(1, len(self.decisions) + 1)):
            raise ValueError(
                "Accepted recovery decisions must form an uninterrupted prefix"
            )
        self.records = self.report["provider_records"]
        self.cursor = 0
        self.used_rollouts, self.used_decisions = set(), set()
        self.receipt = {
            "source_workflow": runtime["workflow"],
            "source_payload_sha256": runtime["payload_sha256"],
            "source_summary_sha256": file_sha256(self.directory / "summary.json"),
            "source_rollouts": len(self.rollouts),
            "source_provider_records": len(self.records),
            "source_accepted_decisions": len(self.decisions),
            "fresh_request_namespace": namespace,
            "replayed_evidence_is_not_new_execution": True,
        }

    def attempt_name(self, index, default):
        if index in self.requests:
            return self.requests[index]["attempt_id"]
        return f"{default}:continuation:{self.namespace}"

    def bind_request(self, request):
        """Keep original wire bytes after checking every field and decoded pixel.

        PNG compression can differ between macOS and Linux Pillow builds. That
        must neither invalidate identical observations nor allow changed pixels.
        """
        original = self.requests.get(request["request_index"])
        if original is None:
            return request

        def content(value):
            if isinstance(value, dict):
                if value.get("encoding") == "base64_png":
                    with Image.open(
                        io.BytesIO(base64.b64decode(value["data"], validate=True))
                    ) as picture:
                        if picture.mode != "RGB" or picture.size != (
                            value["width"],
                            value["height"],
                        ):
                            raise ValueError("Recovered request image format differs")
                        return {**value, "data": digest(np.asarray(picture))}
                return {k: content(v) for k, v in value.items()}
            if isinstance(value, list):
                return [content(v) for v in value]
            return value

        a, b = dict(request), dict(original)
        for value in (a, b):
            fingerprint = value.pop("request_fingerprint")
            if digest(value) != fingerprint:
                raise ValueError("Recovered request fingerprint differs")
        if content(a) != content(b):
            raise ValueError(
                "Recovered Astra request differs from its complete recorded inputs"
            )
        return copy.deepcopy(original)

    @property
    def replaying(self):
        return self.used_rollouts != set(self.rollouts) or self.used_decisions != set(
            self.decisions
        )

    def rollout(self, entry, program, attempt_id, split, destination):
        if attempt_id not in self.rollouts:
            if self.replaying:
                raise ValueError(
                    "Continuation skipped recorded evidence before a new rollout"
                )
            return None
        if attempt_id in self.used_rollouts:
            raise ValueError("Recovered physical attempt was requested twice")
        root = confined(self.directory / "rollouts", attempt_id)
        summary = json.loads((root / "summary.json").read_text())
        trace = json.loads((root / "trace_manifest.json").read_text())
        if (
            summary != self.rollouts[attempt_id]
            or summary["program_sha256"] != digest(program)
            or trace["sha256"] != summary["trace_sha256"]
            or digest(trace["files"]) != trace["sha256"]
            or "events.jsonl" not in trace["files"]
        ):
            raise ValueError("Recovered rollout identity differs")
        for name, expected in trace["files"].items():
            if file_sha256(confined(root, name)) != expected:
                raise ValueError("Recovered trace bytes differ")
        events = [
            json.loads(line)
            for line in (root / "events.jsonl").read_text().splitlines()
        ]
        starts = [e for e in events if e["kind"] == "program_start"]
        ends = [e for e in events if e["kind"] == "rollout_result"]
        if (
            len(starts) != 1
            or len(ends) != 1
            or starts[0]["entry"] != entry
            or starts[0]["program"] != program
            or starts[0]["split"] != split
            or starts[0]["arm"] != self.report["arm"]
        ):
            raise ValueError("Recovery cannot change an episode, program or split")

        def restore(value):
            if isinstance(value, dict):
                if set(value) == {"array", "shape", "dtype", "sha256"}:
                    if value["array"] not in trace["files"]:
                        raise ValueError(
                            "Snapshot array is missing from the trace manifest"
                        )
                    array = np.load(confined(root, value["array"]), allow_pickle=False)
                    if (
                        list(array.shape) != value["shape"]
                        or str(array.dtype) != value["dtype"]
                        or digest(array) != value["sha256"]
                    ):
                        raise ValueError("Recovered snapshot differs")
                    return array
                return {k: restore(v) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v) for v in value]
            return value

        result = restore(ends[0]["result"])
        if {k: v for k, v in result.items() if k != "snapshots"} != {
            k: v for k, v in summary.items() if k != "trace_sha256"
        }:
            raise ValueError("Recorded result differs from its summary")
        shutil.copytree(root, destination)
        self.used_rollouts.add(attempt_id)
        return {**result, "trace_sha256": summary["trace_sha256"]}

    def exhausted(self):
        if (
            self.used_rollouts != set(self.rollouts)
            or self.used_decisions != set(self.decisions)
            or self.cursor != len(self.records)
        ):
            raise ValueError("Continuation did not consume all recorded evidence")


class RecoveryClient:
    def __init__(self, client, archive):
        self.client, self.archive = client, archive
        self.records = client.records

    def _retain(self, record):
        self.records.append(copy.deepcopy(record))
        _append_record(self.client.response_log, record)

    def propose(self, request):
        import hashlib

        archive = self.archive
        index = request["request_index"]
        if index in archive.decisions:
            if index in archive.used_decisions or request != archive.requests[index]:
                raise ValueError(
                    "Recovered Astra request differs from its complete recorded inputs"
                )
            while archive.cursor < len(archive.records):
                row = archive.records[archive.cursor]
                archive.cursor += 1
                if row.get("accepted"):
                    if (
                        row["request_fingerprint"] != request["request_fingerprint"]
                        or row["prompt_template_version"] != PROMPT_TEMPLATE_VERSION
                        or row["system_prompt_sha256"]
                        != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
                        or row.get("configured_model") != archive.astra["model"]
                        or row.get("reasoning_effort")
                        != archive.astra["reasoning_effort"]
                    ):
                        raise ValueError(
                            "Recovered provider receipt has a different prompt or identity"
                        )
                    self._retain(row)
                    break
                self._retain(row)
            else:
                raise ValueError(
                    "Recovered decision lacks an accepted provider receipt"
                )
            decision = archive.decisions[index]
            parsed = parse_proposal(
                {
                    k: decision[k]
                    for k in (
                        "selected_sources",
                        "program",
                        "failure_hypothesis",
                        "expected_effect",
                    )
                },
                request,
            )
            if parsed != decision:
                raise ValueError("Recovered decision failed its original contract")
            archive.used_decisions.add(index)
            return parsed
        # Preserve interrupted attempts, including unknown usage, before a fresh
        # request with the continuation identity. Never retry their invocation IDs.
        for row in archive.records[archive.cursor :]:
            if row.get("accepted"):
                raise ValueError("Continuation skipped an accepted provider decision")
            self._retain(row)
            archive.cursor += 1
        archive.exhausted()
        return self.client.propose(request)
