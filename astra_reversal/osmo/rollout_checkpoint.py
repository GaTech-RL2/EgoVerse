"""Immutable evidence checkpoints for completed Codex physical rollouts.

These are cumulative evidence snapshots, not task-completion seals or resumable
simulator state. A remote commit receipt is published only after its archive.
"""

import hashlib
import io
import json
import re
import secrets
import tarfile
from pathlib import Path

from astra_reversal.astra_client import _strict_json
from astra_reversal.records import digest, file_sha256

_FAMILIES = {
    "frs-codex-frozen-evaluation-1.0": "frs",
    "vision-codex-representation-screen-1.0": "representation",
}
_METADATA = (
    "runtime.json",
    "checkpoint.json",
    "protocol.json",
    "frozen_plan.json",
    "reset_manifest.json",
    "prompts.json",
    "frozen_weights_before.json",
)
_EVALUATION_FIELDS = (
    "method",
    "round_index",
    "physical_run_id",
    "attempt_id",
    "episode_id",
    "success",
)


def _encoded(value):
    return json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"


def checkpoint_progress(archive, results, task_id, protocol, before, weight_receipt):
    """Leave the established HTTP workers' callback exactly unchanged."""
    if protocol.get("schema_version") not in _FAMILIES:
        return archive.sync
    if protocol.get("astra", {}).get("backend") != "codex_relay":
        raise ValueError("Codex rollout checkpoint requires the Codex backend")
    return RolloutCheckpoint(
        archive, results, task_id, protocol, before, weight_receipt
    )


class RolloutCheckpoint:
    def __init__(self, archive, results, task_id, protocol, before, weight_receipt):
        self.archive, self.results = archive, Path(results)
        self.directory = self.results / f"task_{task_id}"
        self.task_id, self.family = task_id, _FAMILIES[protocol["schema_version"]]
        self.protocol_sha256 = digest(protocol)
        self.before = _strict_json(_encoded(before))
        if not before.get("tensors") or before.get("sha256") != digest(
            before["tensors"]
        ):
            raise ValueError("Initial frozen-weight receipt is invalid")
        self.weight_receipt = weight_receipt
        self.closed_ids = ()
        self.failed = False

    def __call__(self):
        if self.failed:
            raise RuntimeError("A failed rollout checkpoint cannot advance")
        try:
            self.archive.sync()
            return self._checkpoint()
        except Exception:
            self.failed = True
            raise

    def _completed(self, summary):
        rows = summary.get("physical_rollouts")
        if not isinstance(rows, list) or summary.get("task_id") != self.task_id:
            raise ValueError(
                "Checkpoint summary has the wrong task or rollout inventory"
            )
        if summary.get("protocol_sha256") != self.protocol_sha256:
            raise ValueError("Checkpoint summary differs from the frozen protocol")
        ids = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("episode_id"), str):
                raise ValueError("Completed rollout lacks episode identity")
            attempt = row.get("attempt_id")
            if (
                not isinstance(attempt, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]+", attempt) is None
            ):
                raise ValueError("Completed rollout has an invalid attempt identity")
            identity = f"{row['episode_id']}:{attempt}"
            if self.family == "frs" and row.get("physical_run_id") != identity:
                raise ValueError("Completed FRS physical identity differs")
            ids.append(identity)
        ids = tuple(ids)
        if len(set(ids)) != len(ids) or ids[: len(self.closed_ids)] != self.closed_ids:
            raise ValueError("Completed rollout identities changed or repeat")
        if len(ids) < len(self.closed_ids):
            raise ValueError("Completed rollout inventory moved backward")
        if ids == self.closed_ids:
            return None
        if len(ids) != len(self.closed_ids) + 1:
            raise ValueError(
                "Each physical rollout requires its own evidence checkpoint"
            )
        if self.family == "frs":
            expected = [{key: row[key] for key in _EVALUATION_FIELDS} for row in rows]
            evaluations = summary.get("evaluation")
            # rollout() saves its result just before evaluate() appends this row.
            if evaluations == expected[:-1]:
                return None
            if evaluations != expected:
                raise ValueError(
                    "Checkpoint evaluation rows differ from completed rollouts"
                )
        return rows, ids

    def _closed_events(self, rows):
        opened, closed = set(), []
        with (self.directory / "events.jsonl").open() as stream:
            for line in stream:
                event = _strict_json(line)
                kind = event.get("kind")
                if kind in ("rollout_start", "representation_rollout_start"):
                    attempt = event["attempt_id"]
                    if attempt in opened:
                        raise ValueError("Rollout start is repeated")
                    opened.add(attempt)
                elif kind in ("rollout_end", "representation_rollout_complete"):
                    result = event["result" if kind == "rollout_end" else "attempt"]
                    attempt = result["attempt_id"]
                    if attempt not in opened:
                        raise ValueError("Completed rollout lacks its start event")
                    opened.remove(attempt)
                    closed.append((result["episode_id"], attempt))
        if opened or closed != [(row["episode_id"], row["attempt_id"]) for row in rows]:
            raise ValueError("Checkpoint includes an open or mismatched rollout")
        for row in rows:
            video = self.directory / f"{row['attempt_id']}.mp4"
            if not video.is_file() or video.is_symlink():
                raise ValueError("Completed rollout lacks its closed video")

    def _checkpoint(self):
        summary = _strict_json((self.directory / "summary.json").read_bytes())
        completed = self._completed(summary)
        if completed is None:
            return None
        rows, ids = completed
        self._closed_events(rows)
        names = _METADATA + (
            ("bank_inventory.json", "image_library.json")
            if self.family == "representation"
            else ()
        )
        metadata, metadata_bytes = {}, {}
        for name in names:
            path = self.results / name
            if not path.is_file() or path.is_symlink():
                raise ValueError("Checkpoint requires regular worker metadata files")
            metadata_bytes[name] = path.read_bytes()
            metadata[name] = _strict_json(metadata_bytes[name])
        if (
            metadata["frozen_weights_before.json"] != self.before
            or digest(metadata["protocol.json"]) != self.protocol_sha256
        ):
            raise ValueError(
                "Checkpoint worker metadata differs from its initial binding"
            )
        assigned = set(metadata["frozen_plan.json"]["assigned_episodes"])
        resets = {
            entry["episode_id"]
            for entry in metadata["reset_manifest.json"]["episodes"]
            if entry["task_id"] == self.task_id
        }
        if any(row["episode_id"] not in assigned & resets for row in rows):
            raise ValueError(
                "Checkpoint rollout is outside the assigned reset inventory"
            )
        after = self.weight_receipt()
        if self.before != after:
            raise RuntimeError(
                "Native policy tensor bytes changed before rollout checkpoint"
            )
        runtime = metadata["runtime.json"]
        checkpoint_id = f"closed_{len(ids):04d}_{secrets.token_hex(8)}"
        key = f"{self.archive.prefix}/rollout_checkpoints/task_{self.task_id}/{checkpoint_id}"
        staging = (
            self.results.parent
            / "rollout_checkpoints"
            / f"task_{self.task_id}"
            / checkpoint_id
        )
        staging.mkdir(parents=True, exist_ok=False)
        bundle = staging / "evidence.tar.gz"
        files = {}
        with tarfile.open(bundle, "x:gz", compresslevel=1) as stream:

            def add(name, raw):
                item = tarfile.TarInfo(name)
                item.size, item.mode = len(raw), 0o600
                stream.addfile(item, io.BytesIO(raw))
                files[name] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }

            for path in sorted(self.directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError(
                        "Checkpoint task evidence may not contain symlinks"
                    )
                if path.is_file():
                    add(
                        f"results/task_{self.task_id}/{path.relative_to(self.directory).as_posix()}",
                        path.read_bytes(),
                    )
            for name in names:
                add(f"results/{name}", metadata_bytes[name])
            add("checkpoint/frozen_weights_after.json", _encoded(after))
            manifest = {
                "schema_version": "codex-rollout-evidence-1.0",
                "scope": "Completed physical rollouts; not task completion or simulator recovery",
                "family": self.family,
                "task_id": self.task_id,
                "workflow": runtime["workflow"],
                "worker": runtime["worker"],
                "protocol_sha256": self.protocol_sha256,
                "native_tensor_sha256": after["sha256"],
                "closed_rollout_ids": list(ids),
                "evaluation": summary.get("evaluation")
                if self.family == "frs"
                else None,
                "files": dict(files),
            }
            add("checkpoint/manifest.json", _encoded(manifest))
        receipt = {
            "schema_version": "codex-rollout-checkpoint-commit-1.0",
            "checkpoint_id": checkpoint_id,
            "scope": manifest["scope"],
            "family": self.family,
            "workflow": runtime["workflow"],
            "worker": runtime["worker"],
            "task_id": self.task_id,
            "closed_rollout_ids": list(ids),
            "protocol_sha256": self.protocol_sha256,
            "native_tensor_sha256": after["sha256"],
            "archive": {
                "key": key + ".tar.gz",
                "sha256": file_sha256(bundle),
                "bytes": bundle.stat().st_size,
            },
            "manifest": files["checkpoint/manifest.json"],
        }
        # The receipt's remote existence is the commit point. A failed upload
        # leaves an uncommitted archive and prevents further physical rollouts.
        self.archive.client.upload_file(str(bundle), "rldb", receipt["archive"]["key"])
        receipt_path = staging / "commit.json"
        with receipt_path.open("xb") as stream:
            stream.write(_encoded(receipt))
        self.archive.client.upload_file(str(receipt_path), "rldb", key + ".json")
        self.closed_ids = ids
        bundle.unlink()
        return receipt
