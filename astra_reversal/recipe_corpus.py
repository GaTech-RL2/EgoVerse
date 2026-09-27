"""Extract audited successful rollouts without labeling unexecuted predictions.

Archive inputs may be local files or explicitly supplied HTTPS read URLs. URLs
and local absolute paths are never written into the portable corpus metadata.
This module loads no policy, contacts no inference service, and runs no simulator.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .interpolation_catalog import donor_catalog
from .records import digest, file_sha256

SCHEMA_VERSION = "recipe-corpus-1.0"
ARRAY_KEYS = (
    "observation_image",
    "observation_wrist_image",
    "observation_state",
    "controller_actions",
    "executed_mask",
)
CAMERAS = ("observation/image", "observation/wrist_image")
NATIVE_MODES = {"recovered_noise", "native"}
ASTRA_MODES = {"astra_tei", "astra_tli"}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _events(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line in stream:
            yield json.loads(line)


def _uncompressed_sha(path):
    result = hashlib.sha256()
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def _safe_member(name):
    path = PurePosixPath(name)
    _need(
        bool(name) and not path.is_absolute() and ".." not in path.parts,
        "Unsafe archive member",
    )
    return str(path)


def _hash_value(value):
    return value["sha256"] if isinstance(value, dict) else value


@dataclass(frozen=True, repr=False)
class CorpusSource:
    """One immutable case and the receipts that authorize its completed rows."""

    source_id: str
    family: str
    case_dir: Path
    archive: str | Path
    archive_prefix: str
    array_audit: Path
    feedback_audit: Path
    allowed_modes: tuple[str, ...]


def _verified_inputs(source, summary, events_path):
    arrays, feedback = _json(source.array_audit), _json(source.feedback_audit)
    hashes = {
        "summary.json": file_sha256(Path(source.case_dir) / "summary.json"),
        "events.jsonl": _uncompressed_sha(events_path),
    }
    episode = summary["episode_id"]
    if source.family == "phase_interpolation":
        _need(arrays["status"] == "passed" and arrays["complete"], "Array audit failed")
        matched = [case for case in arrays["cases"] if case["episode_id"] == episode]
        _need(
            len(matched) == 1 and matched[0]["status"] == "passed", "Case audit missing"
        )
        case = matched[0]
        _need(
            case["summary_sha256"] == hashes["summary.json"]
            and case["events_sha256"] == hashes["events.jsonl"],
            "Array receipt does not bind these case bytes",
        )
        _need(feedback["status"] == "passed", "Feedback audit failed")
        feedback_hashes = feedback["input_file_sha256"]
    else:
        _need(source.family == "representation", "Unsupported source family")
        _need(
            arrays["status"] in ("passed", "verified_partial_archive"),
            "Representation array audit failed",
        )
        if arrays["status"] == "verified_partial_archive":
            _need(
                arrays["checks"]["full_archive_stream_verified"]
                and arrays["checks"]["all_available_npy_references_verified"]
                and arrays["checks"]["published_small_files_match_archive"],
                "Partial recording lacks available-array verification",
            )
        else:
            _need(
                arrays["checks"]["complete_sealed_case"]
                and arrays["checks"]["all_npy_references_verified"],
                "Complete representation recording lacks verification",
            )
        _need(
            feedback["status"] in ("complete", "preserved_prefix")
            and feedback["archive_audit"]["receipt_sha256"]
            == file_sha256(source.array_audit),
            "Provider audit does not bind the array receipt",
        )
        case = arrays
        feedback_hashes = feedback["source_file_sha256"]
        _need(
            all(
                arrays["input_file_sha256"][name] == value
                for name, value in hashes.items()
            ),
            "Representation archive does not bind these case bytes",
        )
    _need(
        case["episode_id"] == feedback["episode_id"] == episode
        and case.get(
            "protocol_sha256", case.get("identities", {}).get("protocol_sha256")
        )
        == feedback["protocol_sha256"]
        == summary["protocol_sha256"],
        "Case/protocol identities differ",
    )
    _need(
        all(
            _hash_value(feedback_hashes[name]) == value
            for name, value in hashes.items()
        ),
        "Feedback receipt does not bind these case bytes",
    )
    return arrays, {
        "source_id": source.source_id,
        "family": source.family,
        "episode_id": episode,
        "protocol_sha256": summary["protocol_sha256"],
        "archive_sha256": arrays["archive"]["sha256"],
        "archive_bytes": arrays["archive"]["bytes"],
        "array_audit_sha256": file_sha256(source.array_audit),
        "array_audit_status": arrays["status"],
        "feedback_audit_sha256": file_sha256(source.feedback_audit),
        "feedback_audit_status": feedback["status"],
        "input_file_sha256": hashes,
        "allowed_modes": list(source.allowed_modes),
        "checkpoint_identity": {
            key: summary["checkpoint"][key]
            for key in (
                "artifact_sha256",
                "tokenizer_sha256",
                "normalization",
                "input_profile",
            )
            if key in summary.get("checkpoint", {})
        },
    }


def _phase_attempts(summary):
    candidates = [summary["baseline"], *summary["controls"].values()]
    for arm in summary["arms"].values():
        candidates.extend(arm["attempts"])
    unique = {}
    for row in candidates:
        key = row["attempt_id"]
        _need(
            key not in unique or unique[key] == row,
            "Inconsistent shared-baseline duplicate",
        )
        unique[key] = row
    return unique


def _choice(generation, mode, family, accepted, instruction):
    native = mode in NATIVE_MODES
    active_key = (
        "active_interpolation"
        if family == "phase_interpolation"
        else "active_intervention"
    )
    active = generation[active_key]
    provenance = generation.get("conditioning", generation.get("provenance"))
    if native:
        _need(
            active is None and provenance is None, "Native anchor has an intervention"
        )
        return {"operator": "native"}, None
    _need(
        mode in ASTRA_MODES and active is not None,
        "Text teacher has no accepted choice",
    )
    decision_id = (
        generation["applied_accepted_decision_id"]
        if family == "phase_interpolation"
        else active["decision_id"]
    )
    _need(decision_id in accepted, "Applied choice has no accepted origin")
    origin = accepted[decision_id]
    expected_active = (
        {key: origin[key] for key in ("source_a_id", "source_b_id", "alpha")}
        if family == "phase_interpolation"
        else origin
    )
    _need(active == expected_active, "Applied choice differs from accepted origin")
    preceding = [
        row
        for row in accepted.values()
        if row["attempt_id"] == generation["attempt_id"]
        and row["observation_step"] <= generation["observation_step"]
    ]
    _need(
        preceding
        and decision_id
        == max(preceding, key=lambda row: row["decision_index"])["decision_id"],
        "Applied choice is not the latest accepted decision for this attempt",
    )
    language = active if family == "phase_interpolation" else active["language"]
    operator = mode.removeprefix("astra_")
    choice = {
        "operator": operator,
        **{key: language[key] for key in ("source_a_id", "source_b_id", "alpha")},
    }
    _need(
        provenance is not None
        and provenance["operator"] == operator
        and provenance["alpha"] == choice["alpha"]
        and provenance["target_prompt"] == instruction,
        "Generation provenance does not match the selected text operator",
    )
    _need(
        type(choice["alpha"]) in (int, float) and 0 <= choice["alpha"] <= 1,
        "Invalid interpolation weight",
    )
    prompts = {row["source_id"]: row["prompt"] for row in donor_catalog()}
    _need(
        choice["source_a_id"] in prompts and choice["source_b_id"] in prompts,
        "Unknown text donor ID",
    )
    _need(
        provenance["source_prompts"]
        == [prompts[choice[key]] for key in ("source_a_id", "source_b_id")],
        "Applied source prompts differ from selected donor IDs",
    )
    return choice, decision_id


def _plan_source(source):
    _need(
        source.allowed_modes
        and set(source.allowed_modes) <= NATIVE_MODES | ASTRA_MODES,
        "Only native anchors and pure Astra TEI/TLI successes are admissible",
    )
    case_dir = Path(source.case_dir)
    summary = _json(case_dir / "summary.json")
    events_path = case_dir / "events.jsonl"
    if not events_path.exists():
        events_path = case_dir / "events.jsonl.gz"
    arrays, binding = _verified_inputs(source, summary, events_path)
    all_attempts = (
        _phase_attempts(summary)
        if source.family == "phase_interpolation"
        else {row["attempt_id"]: row for row in summary["physical_rollouts"]}
    )
    selected = {
        key: row
        for key, row in all_attempts.items()
        if row["success"] and row.get("mode", row.get("arm")) in source.allowed_modes
    }
    generations, completed, accepted, entry, action_spec = {}, {}, {}, None, None
    specs, entries = {}, {}
    for event in _events(events_path):
        kind = event["kind"]
        if kind == "case":
            entry = event["entry"]
        elif kind == "inversion_initialization":
            action_spec = event["action_spec"]
        elif kind == "representation_rollout_start":
            entries[event["attempt_id"]] = event["entry"]
            specs[event["attempt_id"]] = event["action_spec"]
        elif kind in ("interpolation_decision", "representation_decision"):
            decision = event["decision"]
            if decision["accepted"]:
                proposal = decision["proposal"]
                accepted[proposal["decision_id"]] = proposal
        elif kind in ("phase_generation", "representation_generation"):
            if event["attempt_id"] in selected:
                generations.setdefault(event["attempt_id"], []).append(event)
        elif kind in ("phase_attempt", "representation_rollout_complete"):
            row = event["attempt"]
            if row["attempt_id"] in selected:
                _need(
                    row == selected[row["attempt_id"]],
                    "Completed event differs from summary",
                )
                completed[row["attempt_id"]] = event
    _need(set(completed) == set(selected), "A selected success has no completed event")
    windows, trajectories = [], []
    for attempt_id, attempt in selected.items():
        mode = attempt.get("mode", attempt.get("arm"))
        current_entry = (
            entry if source.family == "phase_interpolation" else entries[attempt_id]
        )
        spec = (
            action_spec if source.family == "phase_interpolation" else specs[attempt_id]
        )
        _need(
            spec is not None
            and spec["horizon"] == 10
            and spec["model_action_dim"] == 32,
            "Unsupported recorded action specification",
        )
        _need(
            attempt["execute_steps"] == 5 and not attempt["initial_success"],
            "Unsupported execution chunk or initial-success label",
        )
        count = attempt["actions_executed"]
        _need(type(count) is int and 0 < count <= 300, "Invalid completed action count")
        rows = generations.get(attempt_id, [])
        _need(
            [row["observation_step"] for row in rows] == list(range(0, count, 5)),
            "Missing, repeated or noncontiguous executed windows",
        )
        if arrays["status"] == "verified_partial_archive":
            verified = {
                row["attempt_id"]: row for row in arrays["verified_completed_attempts"]
            }
            _need(
                attempt_id in verified
                and verified[attempt_id]["attempt_sha256"] == digest(attempt)
                and verified[attempt_id]["completed_event_digest"]
                == digest(completed[attempt_id]),
                "Partial archive does not verify this completed success",
            )
        trajectory_id = digest(
            {
                "source": source.source_id,
                "episode": summary["episode_id"],
                "attempt": attempt_id,
            }
        )
        trajectory = {
            "trajectory_id": trajectory_id,
            "source_id": source.source_id,
            "episode_id": summary["episode_id"],
            "attempt_id": attempt_id,
            "source_mode": mode,
            "revision": (
                attempt["iteration"] - 1
                if source.family == "phase_interpolation"
                else attempt["revision"]
            ),
            "sample_kind": "native_anchor" if mode in NATIVE_MODES else "astra_success",
            "instruction": current_entry["instruction"],
            "trajectory_length": count,
            "success": True,
            "reset_audit": attempt["reset_audit"],
            "entry_sha256": digest(current_entry),
            "action_spec": spec,
            "completion_event_sha256": digest(completed[attempt_id]),
        }
        trajectories.append(trajectory)
        for event in rows:
            start = event["observation_step"]
            _need(
                not event.get("vision") and not event.get("vision_has_effect", False),
                "Image-modified trajectory is outside the corpus",
            )
            choice, decision_id = _choice(
                event, mode, source.family, accepted, trajectory["instruction"]
            )
            refs = {
                "observation_image": event["observation"][CAMERAS[0]],
                "observation_wrist_image": event["observation"][CAMERAS[1]],
                "observation_state": event["observation"]["observation/state"],
                "controller_actions": event["controller_actions"],
            }
            for ref in refs.values():
                _safe_member(ref["array"])
            window = {
                "sample_id": digest(
                    {"trajectory": trajectory_id, "action_start": start}
                ),
                "trajectory_id": trajectory_id,
                "source_id": source.source_id,
                "episode_id": summary["episode_id"],
                "attempt_id": attempt_id,
                "source_study": "phase"
                if source.family == "phase_interpolation"
                else "representation",
                "arm": mode,
                "revision": trajectory["revision"],
                "instruction": trajectory["instruction"],
                "sample_kind": trajectory["sample_kind"],
                "action_start": start,
                "executed_count": min(5, count - start),
                "trajectory_length": count,
                "choice": choice,
                "accepted_decision_id": decision_id,
                "held_after_provider_error": bool(
                    event.get("held_text_after_failed_call", False)
                ),
                "condition_id": event["condition_id"],
                "generation_event_sha256": digest(event),
                "array_refs": refs,
                "action_spec_id": spec["action_spec_id"],
                "source_model_actions_sha256": event["generated_actions"]["sha256"],
                "source_noise_sha256": event.get("latent", event.get("noise"))[
                    "sha256"
                ],
                "source_prediction_clipping": event["clipping"],
                "clipping_scope": "Full recorded 10-row prediction; labels are the actually executed clipped prefix.",
                "lossy_label_processing": False,
            }
            windows.append(window)
    return binding, trajectories, windows


def plan_corpus(sources):
    """Verify existing receipts and select complete successes, without archive I/O."""
    _need(
        sources and len({s.source_id for s in sources}) == len(sources),
        "Sources must have unique IDs",
    )
    bindings, trajectories, windows = [], [], []
    for source in sorted(sources, key=lambda row: row.source_id):
        binding, selected, samples = _plan_source(source)
        bindings.append(binding)
        trajectories.extend(selected)
        windows.extend(samples)
    windows.sort(
        key=lambda row: (row["source_id"], row["attempt_id"], row["action_start"])
    )
    _need(
        len({row["sample_id"] for row in windows}) == len(windows),
        "Duplicate corpus samples",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": bindings,
        "trajectories": trajectories,
        "windows": windows,
        "trajectory_count": len(trajectories),
        "window_count": len(windows),
        "executed_action_count": sum(row["executed_count"] for row in windows),
        "label_contract": "Only first K<=5 actually executed controller actions; no generated suffix is a label.",
        "anchor_target": "Frozen original-condition native velocity (zero residual); executed commands are retained as provenance.",
        "feature_contract": "Only raw paired cameras, robot state and original instruction are inputs; outcome/reset/teacher metadata is not an input feature.",
        "split_contract": "Historical source resets are training data. Evaluate on predeclared disjoint reset states; never split adjacent windows across train and evaluation.",
    }


class _HashedReader:
    def __init__(self, stream):
        self.stream, self.sha, self.size = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.sha.update(value)
        self.size += len(value)
        return value


def _extract_archive(location, expected, references):
    address = str(location)
    remote = address.startswith("https://")
    _need(remote or "://" not in address, "Unsupported archive transport")
    try:
        stream = (
            urllib.request.urlopen(address, timeout=120)
            if remote
            else Path(location).open("rb")
        )
        with stream:
            if remote:
                _need(stream.status == 200, "Archive HTTP response was not 200")
                _need(
                    int(stream.headers["Content-Length"]) == expected["archive_bytes"],
                    "Archive HTTP size differs",
                )
            hashed = _HashedReader(stream)
            found = {}
            with gzip.GzipFile(fileobj=hashed) as decoded:
                with tarfile.open(fileobj=decoded, mode="r|") as archive:
                    for member in archive:
                        name = _safe_member(member.name)
                        if name not in references:
                            continue
                        _need(
                            name not in found
                            and member.isfile()
                            and not member.issym()
                            and member.size < 2**20,
                            "Invalid selected array member",
                        )
                        data = archive.extractfile(member).read()
                        value = np.load(io.BytesIO(data), allow_pickle=False)
                        ref = references[name]
                        _need(
                            list(value.shape) == ref["shape"]
                            and str(value.dtype) == ref["dtype"]
                            and digest(value) == ref["sha256"],
                            "Selected NPY differs from recorded descriptor",
                        )
                        found[name] = value
                while decoded.read(2**20):
                    pass
            while hashed.read(2**20):
                pass
            _need(
                hashed.size == expected["archive_bytes"]
                and hashed.sha.hexdigest() == expected["archive_sha256"],
                "Compressed archive hash/size differs",
            )
            _need(
                set(found) == set(references),
                "Selected array is absent from the archive",
            )
            return found
    except urllib.error.HTTPError as exc:
        raise ValueError(f"archive_http_status_{exc.code}") from None
    except urllib.error.URLError:
        raise ValueError("archive_transport_error") from None


def _save_npz(path, arrays):
    """Deterministic ZIP headers; arrays never use pickle/object dtype."""
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for key in ARRAY_KEYS:
            data = io.BytesIO()
            np.save(data, arrays[key], allow_pickle=False)
            entry = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data.getvalue(), compresslevel=6)


def build_corpus(sources, output_dir):
    """Stream selected arrays once per archive and publish a portable corpus."""
    plan = plan_corpus(sources)
    _need(plan["window_count"] > 0, "No admissible successful windows")
    # Two image stacks plus their extraction copies remain under 500 MB here.
    _need(
        plan["window_count"] <= 700, "Corpus exceeds the bounded extraction memory plan"
    )
    output_dir = Path(output_dir)
    _need(
        not output_dir.exists() or not any(output_dir.iterdir()),
        "Corpus output must be empty",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_id = {source.source_id: source for source in sources}
    bindings = {row["source_id"]: row for row in plan["sources"]}
    grouped = {}
    for window in plan["windows"]:
        source = source_by_id[window["source_id"]]
        checksum = bindings[source.source_id]["archive_sha256"]
        group = grouped.setdefault(checksum, {"source": source, "refs": {}})
        for ref in window["array_refs"].values():
            member = _safe_member(
                str(PurePosixPath(source.archive_prefix) / ref["array"])
            )
            _need(
                member not in group["refs"] or group["refs"][member] == ref,
                "Conflicting array references",
            )
            group["refs"][member] = ref
    extracted = {}
    for checksum, group in grouped.items():
        source = group["source"]
        found = _extract_archive(
            source.archive, bindings[source.source_id], group["refs"]
        )
        extracted.update({(checksum, name): value for name, value in found.items()})
    count = plan["window_count"]
    arrays = {
        "observation_image": np.empty((count, 224, 224, 3), dtype=np.uint8),
        "observation_wrist_image": np.empty((count, 224, 224, 3), dtype=np.uint8),
        "observation_state": np.empty((count, 8), dtype=np.float32),
        "controller_actions": np.zeros((count, 5, 7), dtype=np.float32),
        "executed_mask": np.zeros((count, 5), dtype=np.bool_),
    }
    trajectories = {row["trajectory_id"]: row for row in plan["trajectories"]}
    for index, window in enumerate(plan["windows"]):
        source = source_by_id[window["source_id"]]
        checksum = bindings[source.source_id]["archive_sha256"]
        for key, ref in window["array_refs"].items():
            name = str(PurePosixPath(source.archive_prefix) / ref["array"])
            value = extracted[checksum, name]
            if key == "controller_actions":
                _need(
                    value.shape == (10, 7)
                    and value.dtype == np.float32
                    and np.isfinite(value).all(),
                    "Invalid recorded controller chunk",
                )
                spec = trajectories[window["trajectory_id"]]["action_spec"]
                length = window["executed_count"]
                prefix = value[:length]
                _need(
                    np.all(prefix >= spec["lower"]) and np.all(prefix <= spec["upper"]),
                    "Executed controller command is outside recorded bounds",
                )
                arrays[key][index, :length] = prefix
                arrays["executed_mask"][index, :length] = True
            else:
                _need(
                    value.shape == arrays[key].shape[1:]
                    and value.dtype == arrays[key].dtype
                    and np.isfinite(value).all(),
                    "Invalid original observation array",
                )
                arrays[key][index] = value
    del extracted
    temporary = output_dir / "arrays.pending.npz"
    _save_npz(temporary, arrays)
    temporary.replace(output_dir / "arrays.npz")
    plan["arrays"] = {
        "file": "arrays.npz",
        "sha256": file_sha256(output_dir / "arrays.npz"),
        "contents": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": digest(value),
            }
            for key, value in arrays.items()
        },
    }
    plan["status"] = "complete"
    plan["builder_source_sha256"] = file_sha256(__file__)
    plan["corpus_id"] = digest(plan)
    (output_dir / "metadata.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    return plan


def load_corpus(directory):
    """Validate immutable data and masks before handing it to feature capture."""
    directory = Path(directory)
    metadata = _json(directory / "metadata.json")
    identity = metadata.copy()
    checksum = identity.pop("corpus_id")
    _need(
        metadata["schema_version"] == SCHEMA_VERSION
        and metadata["status"] == "complete"
        and digest(identity) == checksum,
        "Corpus metadata identity differs",
    )
    _need(
        file_sha256(directory / "arrays.npz") == metadata["arrays"]["sha256"],
        "Corpus NPZ hash differs",
    )
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        _need(set(archive.files) == set(ARRAY_KEYS), "Unexpected corpus arrays")
        arrays = {key: archive[key] for key in ARRAY_KEYS}
    for key, value in arrays.items():
        expected = metadata["arrays"]["contents"][key]
        _need(
            list(value.shape) == expected["shape"]
            and str(value.dtype) == expected["dtype"]
            and digest(value) == expected["sha256"],
            "Corpus array identity differs",
        )
    for index, row in enumerate(metadata["windows"]):
        mask = arrays["executed_mask"][index]
        length = row["executed_count"]
        _need(
            type(length) is int
            and 1 <= length <= 5
            and np.array_equal(mask, np.arange(5) < length)
            and not np.any(arrays["controller_actions"][index, ~mask]),
            "Unexecuted suffix or noncontiguous mask entered training labels",
        )
    return metadata, arrays


def iter_samples(directory):
    """Yield the runner's canonical sample view with no padded action rows."""
    metadata, arrays = load_corpus(directory)
    sources = {row["source_id"]: row for row in metadata["sources"]}
    for index, row in enumerate(metadata["windows"]):
        length = row["executed_count"]
        yield {
            "sample_id": row["sample_id"],
            "trajectory_id": row["trajectory_id"],
            "episode_id": row["episode_id"],
            "original_prompt": row["instruction"],
            "observation_step": row["action_start"],
            "executed_actions": arrays["controller_actions"][index, :length].copy(),
            "observation": {
                CAMERAS[0]: arrays["observation_image"][index].copy(),
                CAMERAS[1]: arrays["observation_wrist_image"][index].copy(),
                "observation/state": arrays["observation_state"][index].copy(),
            },
            "choice": row["choice"].copy(),
            "kind": "anchor" if row["sample_kind"] == "native_anchor" else "correction",
            "source_receipt_sha256": sources[row["source_id"]]["feedback_audit_sha256"],
            "source_array_audit_sha256": sources[row["source_id"]][
                "array_audit_sha256"
            ],
            "source_study": row["source_study"],
            "arm": row["arm"],
            "revision": row["revision"],
        }
