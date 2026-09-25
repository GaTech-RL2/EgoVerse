"""Offline integrity and pixel replay for recorded image-perturbation runs.

This module makes no model, provider or simulator calls. Pixel operations are
replayed against the pinned donor library; recorded physics and hidden model
states are not independently re-executed.
"""

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .flow import error_metrics
from .image_perturbations import (
    CAMERAS,
    DEFAULT_LIMITS,
    ImagePerturbationLimits,
    apply_image_perturbations,
)
from .interventions import noise_basis, perturb_noise, random_noise_proposal
from .records import digest, file_sha256

SCHEMA_VERSION = "image-perturbation-audit-1.0"
ARRAY_FIELDS = {"array", "shape", "dtype", "sha256"}
OBSERVATION_FIELDS = {*CAMERAS, "observation/state"}
FLOW_SHAPE = (1, 10, 32)
ARMS = (
    "random_noise",
    "random_occlusion",
    "random_demo_blend",
    "astra_occlusion",
    "astra_demo_blend",
)
LIMITATIONS = [
    "No policy/model/provider/simulator calls occur. Recorded task outcomes and hidden model states are not independently re-executed.",
    "Every referenced NPY is verified, and both edited camera arrays are reconstructed bitwise from raw observations and the pinned donor library. Condition IDs bind the edited observations and unchanged task, not independently recomputed hidden embeddings.",
    "The same-condition inversion endpoints and image probe endpoints are recomputed for numerical errors, but the solves themselves are not rerun. The inherited zero text-hook parity endpoint is not separately stored and is checked as worker-reported provenance only.",
    "The donor library loader verifies stored PNGs, metadata and contact-sheet tile bindings. This audit does not independently rebuild the library from the original training parquet files.",
    "Fresh/held noise replay is exact. Random-noise float32 reconstruction allows max absolute 1e-7, and seeded normalized coefficients allow four float64 ULPs for the already measured cross-platform reduction difference.",
    "Controller decoding uses the pinned normalization file and permits max absolute 1e-7 CPU arithmetic drift. Pixel replay allows no tolerance.",
]


def _float32(actual, expected, label, atol=0):
    require(
        actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape,
        f"{label}: invalid shape/dtype",
    )
    maximum = float(
        np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    )
    require(maximum <= atol, f"{label}: numerical reconstruction differs")
    return maximum


def _metrics(recorded, a, b, label):
    expected = error_metrics(a, b)
    require(set(recorded) == set(expected), f"{label}: malformed metrics")
    require(
        all(
            type(recorded[key]) in (int, float)
            and math.isclose(recorded[key], value, rel_tol=1e-12, abs_tol=1e-15)
            for key, value in expected.items()
        ),
        f"{label}: endpoint errors differ",
    )
    return expected


def _physical(summary):
    require(
        set(summary["arms"]) == set(ARMS)
        and set(summary["controls"]) == {"known_noise", "policy_fresh"},
        "Missing or extra physical arms/controls",
    )
    baseline = summary["baseline"]
    for mode, row in (
        ("recovered_noise", baseline),
        *summary["controls"].items(),
    ):
        require(
            row["mode"] == mode
            and row["iteration"] == 1
            and row["attempt_id"] == f"{mode}_1",
            "Native baseline/control role differs from its physical attempt",
        )
    rows = [
        baseline,
        summary["controls"]["known_noise"],
        summary["controls"]["policy_fresh"],
    ]
    for arm in ARMS:
        attempts = summary["arms"][arm]["attempts"]
        require(
            attempts and attempts[0] == baseline,
            "Arms must share the exact physical baseline",
        )
        rows.extend(attempts[1:])
    require(
        len({row["attempt_id"] for row in rows}) == len(rows),
        "Physical attempts are duplicated",
    )
    return rows


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate recorded JSON object key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite recorded JSON number: {value}")

    return json.loads(
        text, object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )


class ArtifactStore:
    """Read exact immutable bytes and require complete array-reference coverage."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.files = {}
        self.references = {}
        self.inventory = {}
        self.reference_count = 0

    def path(self, relative):
        require(type(relative) is str, "Recorded artifact path must be a string")
        name = PurePosixPath(relative)
        require(
            not name.is_absolute() and ".." not in name.parts and "\\" not in relative,
            "Unsafe recorded artifact path",
        )
        path = self.directory / relative
        require(
            not path.is_symlink() and path.resolve().is_relative_to(self.directory),
            "Recorded artifact escapes its case directory",
        )
        return path

    def remember(self, relative):
        path = self.path(relative)
        checksum = file_sha256(path)
        require(
            relative not in self.files or checksum == self.files[relative],
            "Artifact bytes changed during audit",
        )
        self.files[relative] = checksum
        return path

    def read_json(self, relative):
        return _json(self.remember(relative).read_text())

    def lines(self, relative):
        return [
            _json(line)
            for line in self.remember(relative).read_text().splitlines()
            if line.strip()
        ]

    def register(self, value):
        if isinstance(value, dict):
            if "array" in value:
                require(set(value) == ARRAY_FIELDS, "Malformed NPY array descriptor")
                name = value["array"]
                require(type(name) is str, "NPY array path must be a string")
                parts = PurePosixPath(name).parts
                require(
                    len(parts) == 2 and parts[0] == "arrays" and name.endswith(".npy"),
                    "NPY references must be direct children of arrays/",
                )
                self.path(name)
                require(_sha(value["sha256"]), "Malformed recorded array digest")
                require(
                    name not in self.references or self.references[name] == value,
                    "Conflicting descriptors for the same array file",
                )
                self.references[name] = copy.deepcopy(value)
                self.reference_count += 1
            else:
                for item in value.values():
                    self.register(item)
        elif isinstance(value, list):
            for item in value:
                self.register(item)

    def array(self, descriptor):
        require(
            type(descriptor) is dict
            and set(descriptor) == ARRAY_FIELDS
            and self.references.get(descriptor["array"]) == descriptor,
            "Array was not registered with its exact descriptor",
        )
        name = descriptor["array"]
        path = self.remember(name)
        array = np.load(path, allow_pickle=False)
        require(
            isinstance(array, np.ndarray)
            and array.dtype.kind in "buif"
            and np.isfinite(array).all()
            and list(array.shape) == descriptor["shape"]
            and str(array.dtype) == descriptor["dtype"]
            and digest(array) == descriptor["sha256"],
            "Recorded array hash/shape/dtype/finiteness mismatch",
        )
        self.inventory[name] = {
            "file_sha256": self.files[name],
            "array_sha256": descriptor["sha256"],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": path.stat().st_size,
        }
        return array

    def resolve(self, value):
        if isinstance(value, dict):
            if "array" in value:
                return self.array(value)
            return {key: self.resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        return value

    def observation(self, descriptors):
        require(
            type(descriptors) is dict and set(descriptors) == OBSERVATION_FIELDS,
            "Unexpected raw observation fields",
        )
        result = {name: self.array(value) for name, value in descriptors.items()}
        for camera in CAMERAS:
            require(
                result[camera].dtype == np.uint8
                and result[camera].shape == (224, 224, 3),
                "Expected raw uint8 camera pixels at the frozen 224px policy resolution",
            )
        require(
            result["observation/state"].shape == (8,)
            and result["observation/state"].dtype == np.float32,
            "Expected unchanged native float32 proprioception8",
        )
        return result

    def finish(self):
        actual = {
            str(path.relative_to(self.directory))
            for path in (self.directory / "arrays").rglob("*")
            if path.is_file()
        }
        require(
            actual == set(self.references), "Missing or unreferenced NPY array files"
        )
        for name, descriptor in self.references.items():
            if name not in self.inventory:
                self.array(descriptor)
        for name, checksum in self.files.items():
            require(
                file_sha256(self.path(name)) == checksum,
                "Artifact changed before audit completion",
            )
        return {
            "array_files": len(self.inventory),
            "array_references": self.reference_count,
            "array_bytes": sum(value["bytes"] for value in self.inventory.values()),
            "array_inventory_sha256": digest(self.inventory),
        }


def replay_image_application(
    raw, modified, operations, library, recorded, *, limits=DEFAULT_LIMITS
):
    """Require byte-identical replay of both cameras and all untouched fields."""
    expected, audit = apply_image_perturbations(
        raw, operations, library.resolve, limits=limits
    )
    require(
        digest(modified) == digest(expected),
        "Modified observation differs from exact pinned pixel replay",
    )
    require(
        recorded == audit, "Image-application audit differs from reconstructed pixels"
    )
    return audit


class _CaseAudit:
    def __init__(self, directory, *, library, metadata, assets):
        self.store, self.library, self.metadata = (
            ArtifactStore(directory),
            library,
            metadata,
        )
        self.summary = self.store.read_json("summary.json")
        self.events = self.store.lines("events.jsonl")
        require(
            [row["sequence"] for row in self.events] == list(range(len(self.events))),
            "Missing or duplicate event sequence",
        )
        allowed = {
            "case",
            "inversion_initialization",
            "image_gate_reference",
            "image_gate_check",
            "image_gate",
            "image_request",
            "image_decision",
            "random_image_decision",
            "image_generation",
            "image_attempt",
            "image_development_validation",
        }
        require(
            all(row["kind"] in allowed for row in self.events),
            "Unknown event kind in image experiment",
        )
        self.store.register(self.events)
        self.groups = {
            kind: [row for row in self.events if row["kind"] == kind]
            for kind in allowed
        }
        self.case = self.one("case")
        require(self.case["sequence"] == 0, "Case declaration is not the first event")
        self.entry, self.protocol = self.case["entry"], self.case["protocol"]
        self.episode_id = self.entry["episode_id"]
        self.checkpoint = metadata["checkpoint.json"]
        require(
            self.protocol == metadata["protocol.json"],
            "Case protocol differs from worker",
        )
        require(
            self.protocol["arms"] == list(ARMS)
            and self.protocol["attempt_budget"] == 3
            and self.protocol["action_budget"] == 300
            and self.protocol["execute_steps"] == 5
            and self.protocol["astra"]["call_interval"]
            == self.protocol["images"]["valid_for_actions"]
            == 25,
            "Unexpected image study schedule",
        )
        require(
            self.protocol["execution_solver"]
            == {"solver": "euler", "steps": 10, "time_power": 1.0}
            and self.protocol["inversion_solver"]
            == {"solver": "rk4", "steps": 100, "time_power": 3.0},
            "Unexpected image study sampler",
        )
        entries = {
            row["episode_id"]: row
            for row in metadata["reset_manifest.json"]["episodes"]
        }
        require(
            self.entry == entries.get(self.episode_id)
            and self.episode_id in metadata["frozen_plan.json"]["assigned_episodes"],
            "Case reset/assignment differs from frozen worker",
        )
        require(
            self.summary["schema_version"] == "image-perturbations-1.0"
            and self.summary["status"] == "complete"
            and self.summary["episode_id"] == self.episode_id
            and self.summary["protocol_sha256"] == digest(self.protocol)
            and self.summary["reset_entry_sha256"] == digest(self.entry)
            and self.summary["checkpoint"] == self.checkpoint,
            "Incomplete or mismatched image case",
        )
        require(
            self.summary["image_library"]
            == metadata["image_library.json"]
            == library.metadata(),
            "Image library differs from the frozen pixel library",
        )
        require(
            self.checkpoint["input_profile"] == "openpi_libero"
            and self.checkpoint["frozen"] is True
            and self.checkpoint["horizon"] == 10
            and self.checkpoint["model_action_dim"] == 32,
            "Unexpected frozen native checkpoint profile",
        )
        path = Path(assets) / "norm_stats.json"
        self.asset_sha256 = file_sha256(path)
        require(
            self.asset_sha256
            == self.checkpoint["input_profile_assets"]["norm_stats"]["sha256"],
            "Action normalization asset differs from frozen checkpoint",
        )
        stats = _json(path.read_text())["norm_stats"]["actions"]
        self.q01, self.q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        require(
            self.q01.shape == self.q99.shape == (7,)
            and np.isfinite(self.q01).all()
            and np.isfinite(self.q99).all()
            and np.all(self.q99 > self.q01),
            "Invalid action normalization arrays",
        )
        self.physical = _physical(self.summary)
        self.attempts = {row["attempt_id"]: row for row in self.physical}
        self.attempt_events = {
            row["attempt"]["attempt_id"]: row for row in self.groups["image_attempt"]
        }
        require(
            len(self.attempt_events) == len(self.groups["image_attempt"])
            and set(self.attempt_events) == set(self.attempts),
            "Attempt-event coverage differs from physical summaries",
        )
        require(
            all(
                self.attempt_events[key]["attempt"] == value
                for key, value in self.attempts.items()
            ),
            "Attempt summary differs from its physical event",
        )
        require(
            [row["attempt"]["attempt_id"] for row in self.groups["image_attempt"]]
            == [row["attempt_id"] for row in self.physical],
            "Physical attempt order differs from the protocol",
        )
        self.seed = [self.protocol["seed"], int(digest(self.episode_id)[:8], 16)]
        self.basis = self.store.array(self.case["basis"])
        expected_basis, basis_id = noise_basis(
            FLOW_SHAPE, np.random.SeedSequence(self.seed + [1])
        )
        require(
            self.case["basis_id"] == basis_id == digest(self.basis)
            and np.array_equal(self.basis, expected_basis),
            "Noise basis differs from prescribed seed",
        )
        require(
            type(self.summary["development"]) is bool
            and self.summary["development"]
            == (metadata["runtime.json"]["phase"] == "development")
            and self.summary["seed"] == self.entry["seed"] == self.protocol["seed"]
            and self.summary["suite"] == self.entry["suite"]
            and self.summary["task_id"] == self.entry["task_id"],
            "Case phase/seed/suite/task differs from the frozen assignment",
        )
        self.counts = Counter(
            {
                key: 0
                for key in (
                    "requests",
                    "physical_provider_calls",
                    "preflight_failures",
                    "accepted_provider_proposals",
                    "raw_snapshot_bindings",
                    "decoded_camera_bindings",
                )
            }
        )
        self.bindings = []
        self.maxima = {
            "noise_reconstruction_max_abs": 0.0,
            "controller_decode_max_abs": 0.0,
        }

    def one(self, kind):
        require(len(self.groups[kind]) == 1, f"Expected exactly one {kind} event")
        return self.groups[kind][0]

    def condition(self, observation, condition_id):
        require(
            condition_id
            == digest({**observation, "prompt": self.entry["instruction"]}),
            "Native condition ID does not bind edited RGB, state and unchanged task",
        )

    def decode(self, actions):
        require(
            actions.shape == FLOW_SHAPE and actions.dtype == np.float32,
            "Full model endpoint must be float32 [1,10,32]",
        )
        return (actions[0, :, :7] + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01

    def initialization(self):
        event = self.one("inversion_initialization")
        reported = self.summary["initialization"]
        require(
            all(event[key] == value for key, value in reported.items()),
            "Initialization summary differs from event",
        )
        observation = self.store.observation(event["observation"])
        self.initial_observation_id = digest(observation)
        self.condition(observation, event["condition_id"])
        self.known = self.store.array(event["known_noise"])
        self.recovered = self.store.array(event["recovered_noise"])
        expected = (
            np.random.default_rng(np.random.SeedSequence(self.seed))
            .standard_normal(FLOW_SHAPE)
            .astype(np.float32)
        )
        _float32(self.known, expected, "Seeded reference noise")
        require(
            event["known_noise_sha256"] == digest(self.known)
            and event["recovered_noise_sha256"] == digest(self.recovered),
            "Initial noise digest mismatch",
        )
        reference, roundtrip = (
            self.store.array(event[key]) for key in ("reference", "roundtrip")
        )
        native, adapter = (
            self.store.array(event[key])
            for key in ("native_actions", "adapter_actions")
        )
        errors = event["errors"]
        _metrics(errors["noise"], self.known, self.recovered, "Initial noise roundtrip")
        _metrics(
            errors["actions"], reference, roundtrip, "Initial policy endpoint roundtrip"
        )
        _metrics(errors["native_parity"], native, adapter, "Native sampler parity")
        require(
            event["passed"] is True
            and errors["noise"]["max_abs"]
            <= self.protocol["numerical_gate"]["noise_max_abs"]
            and errors["actions"]["max_abs"]
            <= self.protocol["numerical_gate"]["action_max_abs"]
            and errors["native_parity"]["max_abs"] <= 1e-5
            and errors["zero_embedding_hook_parity"] == {"rmse": 0.0, "max_abs": 0.0},
            "Initialization numerical gate did not pass",
        )
        require(
            event["development_embedding_probe"] is None
            and event["velocity_evaluations"] == 1230,
            "Unexpected initialization compute or unrelated embedding probe",
        )
        zero = event["zero_embedding_hook"]
        require(
            zero["condition_id"] == event["condition_id"]
            and zero["original_prompt"] == self.entry["instruction"]
            and zero["has_effect"] is False,
            "Zero-hook provenance is inconsistent",
        )
        self.spec = event["action_spec"]
        payload = {
            key: self.spec[key]
            for key in (
                "horizon",
                "model_action_dim",
                "timestep_seconds",
                "lower",
                "upper",
                "semantics",
            )
        }
        require(
            self.spec["action_spec_id"] == digest(payload)
            and self.spec["horizon"] == 10
            and self.spec["model_action_dim"] == 32,
            "Invalid frozen controller specification",
        )
        self.counts["initialization_velocity_evaluations"] = 1230
        return observation

    def gate(self, initial_observation):
        event, gate = self.one("image_gate"), self.summary["image_gate"]
        require(
            all(event[key] == value for key, value in gate.items()),
            "Image gate summary/event mismatch",
        )
        require(
            gate["passed"] is True
            and gate["complete"] is True
            and gate["status"] == "passed"
            and gate["velocity_evaluations_complete"] is True
            and gate["velocity_evaluations"] == 60
            and gate["solver"] == self.protocol["execution_solver"],
            "Image gate is incomplete, failed or has wrong compute",
        )
        require(
            gate["observation_id"] == self.initial_observation_id
            and gate["known_noise_sha256"] == digest(self.known),
            "Image gate uses another observation/noise",
        )
        reference = self.one("image_gate_reference")
        raw = self.store.observation(reference["observation"])
        require(
            digest(raw) == digest(initial_observation),
            "Image gate reference observation changed",
        )
        _float32(
            self.store.array(reference["latent"]), self.known, "Image gate fixed noise"
        )
        self.condition(raw, reference["condition_id"])
        require(
            reference["condition_id"] == gate["native_condition_id"]
            and reference["velocity_evaluations"] == 10,
            "Image gate reference identity/compute mismatch",
        )
        generated = self.store.array(reference["generated_actions"])
        decoded = self.store.array(reference["decoded_actions"])
        require(
            np.max(np.abs(decoded - self.decode(generated))) <= 1e-7,
            "Image gate reference decoding mismatch",
        )
        height, width = raw[CAMERAS[0]].shape[:2]
        blend = {
            "kind": "demo_blend",
            "camera": CAMERAS[0],
            "donor_id": self.library.catalog()[0]["donor_id"],
            "alpha": 0.5,
        }
        mask = {
            "kind": "occlusion",
            "camera": CAMERAS[0],
            "box_xyxy": [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            "fill_rgb": [127, 127, 127],
            "strength": 1.0,
        }
        specifications = {
            "empty_identity": [],
            "blend_zero_identity": [{**blend, "alpha": 0.0}],
            "occlusion_zero_identity": [{**mask, "strength": 0.0}],
            "blend_nonzero": [blend],
            "occlusion_nonzero": [mask],
        }
        checks = {row["label"]: row for row in self.groups["image_gate_check"]}
        require(
            set(checks) == set(specifications) == set(gate["checks"])
            and len(checks) == len(self.groups["image_gate_check"]),
            "Image gate probe coverage differs",
        )
        for label, operations in specifications.items():
            row = checks[label]
            require(
                reference["sequence"] < row["sequence"] < event["sequence"]
                and all(
                    row[key] == value for key, value in gate["checks"][label].items()
                ),
                "Image gate probe ordering/summary mismatch",
            )
            raw = self.store.observation(row["observation"])
            modified = self.store.observation(row["modified_observation"])
            require(
                digest(raw) == self.initial_observation_id
                and row["operations"] == operations,
                "Image gate probe input/spec changed",
            )
            replay_image_application(
                raw, modified, operations, self.library, row["image_audit"]
            )
            self.condition(modified, row["condition_id"])
            value = self.store.array(row["generated_actions"])
            actions = self.store.array(row["decoded_actions"])
            require(
                np.max(np.abs(actions - self.decode(value))) <= 1e-7
                and row["velocity_evaluations"] == 10,
                "Image probe decoding/compute mismatch",
            )
            _metrics(row["errors"], generated, value, "Image probe full endpoint")
            _metrics(
                row["controlled_channel_errors"],
                generated[:, :, :7],
                value[:, :, :7],
                "Image probe controlled endpoint",
            )
            _metrics(
                row["decoded_action_errors"],
                decoded,
                actions,
                "Image probe decoded endpoint",
            )
            nonzero = label.endswith("nonzero")
            changed = digest(raw) != digest(modified)
            require(
                row["expected_effect"] == ("nonzero" if nonzero else "identity")
                and row["passed"] is True
                and row["pixels_changed"] == changed,
                "Image probe effect flags mismatch",
            )
            require(
                (
                    changed
                    and row["controlled_channel_errors"]["max_abs"] > 0
                    and row["decoded_action_errors"]["max_abs"] > 0
                )
                if nonzero
                else (
                    not changed
                    and np.array_equal(generated, value)
                    and np.array_equal(decoded, actions)
                    and row["condition_id"] == gate["native_condition_id"]
                ),
                "Image probe does not establish its required identity/nonzero effect",
            )
        self.counts["image_gate_velocity_evaluations"] = 60

    def snapshot(self, wire, recorded, *, request, scope):
        require(
            (wire["label"], wire["step"]) == (recorded["label"], recorded["step"]),
            "Feedback labels/steps differ from recorded raw observations",
        )
        raw = self.store.observation(recorded["observation"])
        for camera in CAMERAS:
            encoded = wire["observation"][camera]
            require(
                set(encoded) == {"encoding", "data"}
                and encoded["encoding"] == "base64_png",
                "Feedback must be lossless raw PNG",
            )
            content = base64.b64decode(encoded["data"], validate=True)
            with Image.open(io.BytesIO(content)) as image:
                require(
                    image.format == "PNG" and image.mode == "RGB",
                    "Feedback is not RGB PNG",
                )
                pixels = np.array(image, copy=True)
            require(
                np.array_equal(pixels, raw[camera]),
                "Astra feedback pixels differ from the recorded unmodified camera",
            )
            self.counts["decoded_camera_bindings"] += 1
        require(
            wire["observation"]["observation/state"]
            == raw["observation/state"].tolist(),
            "Feedback state differs from raw robot proprioception",
        )
        self.counts["raw_snapshot_bindings"] += 1
        self.bindings.append(
            {
                "kind": "feedback",
                "request_fingerprint": request["request_fingerprint"],
                "scope": scope,
                "step": recorded["step"],
                "raw_observation_sha256": digest(raw),
            }
        )

    def provider(self, request, record, decision):
        from .astra_client import ClientError
        from .image_perturbation_agent import (
            PROMPT_TEMPLATE_VERSION,
            build_payload,
            parse_proposal,
        )
        from .image_perturbation_agent import SCHEMA_VERSION as CLIENT_SCHEMA
        from .intervention_agent import normalize_usage

        settings = self.protocol["astra"]
        payload = build_payload(
            request,
            settings["model"],
            sampling={
                key: settings[key]
                for key in ("reasoning_effort", "max_completion_tokens")
            },
        )
        require(
            record["client_schema_version"] == CLIENT_SCHEMA
            and record["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
            and record["requested_model"] == settings["model"]
            and type(record["provider_call"]) is bool
            and type(record["accepted"]) is bool,
            "Provider identity/physical-call flags differ",
        )
        identity = (
            "schema_version",
            "episode_id",
            "attempt_id",
            "decision_index",
            "observation_step",
            "image_mode",
            "request_id",
            "request_fingerprint",
        )
        require(
            all(record[key] == request[key] for key in identity),
            "Provider ledger is bound to another request",
        )
        require(
            record["payload_sha256"]
            == hashlib.sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest()
            and record["cache"] == {"no-cache": True}
            and record["response_format"] == {"type": "json_object"}
            and record["sampling_settings"]
            == {
                key: settings[key]
                for key in ("reasoning_effort", "max_completion_tokens")
            },
            "Provider payload/model/cache/sampling differs from frozen request",
        )
        require(
            record["donor_catalog_sha256"] == digest(self.library.catalog())
            and record["library_id"] == self.library.metadata()["library_id"]
            and record["contact_sheet_sha256"]
            == {row["camera"]: row["sha256"] for row in self.library.contact_sheets()},
            "Provider donor/contact-sheet binding differs",
        )
        response = record.get("response") or {}
        require(
            record["token_usage"] == normalize_usage(response.get("usage")),
            "Provider normalized usage differs from reported response",
        )
        parsed = None
        if response:
            try:
                require(
                    response.get("model") == settings["model"]
                    and type(record.get("http_status")) is int
                    and 200 <= record["http_status"] < 300,
                    "Wrong response model/status",
                )
                choices = response["choices"]
                require(
                    type(choices) is list
                    and len(choices) == 1
                    and choices[0].get("finish_reason") == "stop",
                    "Incomplete response choice",
                )
                message = choices[0]["message"]
                require(
                    message.get("role") == "assistant" and not message.get("refusal"),
                    "Refused/nonassistant response",
                )
                parsed = parse_proposal(message["content"], request)
            except (KeyError, IndexError, TypeError, ValueError, ClientError):
                parsed = None
        require(
            record["accepted"] == decision["accepted"] == (parsed is not None)
            and decision["proposal"] == parsed,
            "Accepted provider body does not match the executed locally bound proposal",
        )
        if parsed is not None:
            require(
                record["provider_call"]
                and record["decision_id"] == parsed["decision_id"]
                and decision["error"] is None,
                "Accepted proposal has no physical bound call",
            )
        else:
            require(
                type(decision["error"]) is str
                and decision["error"]
                and type(record.get("error")) is str
                and record["error"],
                "Rejected call lacks error evidence",
            )
        self.counts["requests"] += 1
        self.counts["physical_provider_calls"] += record["provider_call"]
        self.counts["preflight_failures"] += not record["provider_call"]
        self.counts["accepted_provider_proposals"] += record["accepted"]
        self.bindings.append(
            {
                "kind": "provider",
                "request_fingerprint": request["request_fingerprint"],
                "request_id": request["request_id"],
                "payload_sha256": record["payload_sha256"],
                "provider_call": record["provider_call"],
                "accepted": record["accepted"],
                "decision_id": parsed["decision_id"] if parsed else None,
            }
        )

    def online_requests(self, attempt, generations):
        from .image_perturbation_agent import _validate_request
        from .interpolation_search import outcome_feedback

        name = attempt["attempt_id"]
        requests = [
            row
            for row in self.groups["image_request"]
            if row["request"]["attempt_id"] == name
        ]
        decisions = [
            row for row in self.groups["image_decision"] if row["attempt_id"] == name
        ]
        require(
            len(requests)
            == len(decisions)
            == len(attempt["decisions"])
            == len(attempt["provider_records"]),
            "Scheduled request/decision/provider counts differ",
        )
        arm_rows = self.summary["arms"][attempt["mode"]]["attempts"]
        history = arm_rows[: attempt["iteration"] - 1]
        previous = history[-1]
        prior_event = self.attempt_events[previous["attempt_id"]]
        for index, (request_event, decision_event, decision, record) in enumerate(
            zip(
                requests,
                decisions,
                attempt["decisions"],
                attempt["provider_records"],
                strict=True,
            )
        ):
            request = request_event["request"]
            _validate_request(request)
            step = index * 25
            current = next(
                row for row in generations if row["observation_step"] == step
            )
            require(
                prior_event["sequence"]
                < request_event["sequence"]
                < decision_event["sequence"]
                < current["sequence"]
                and decision_event["decision"] == decision
                and decision["decision_index"] == index + 1
                and decision["observation_step"] == step,
                "Online request/decision ordering differs from scheduled application",
            )
            require(
                request["episode_id"] == self.episode_id
                and request["attempt_id"] == name
                and request["decision_index"] == index + 1
                and request["observation_step"] == step
                and request["image_mode"] == attempt["mode"].removeprefix("astra_")
                and request["target_task"] == self.entry["instruction"]
                and request["action_spec"] == self.spec,
                "Astra request task/action/slot identity mismatch",
            )
            require(
                request["limits"]["call_interval"] == 25
                and request["limits"]["max_calls"] == 12
                and request["limits"]["action_budget"] == 300,
                "Astra request changed its frozen decision/action budget",
            )
            require(
                request["donor_catalog"] == self.library.catalog(),
                "Astra request does not contain the pinned donor catalog",
            )
            for observed, expected in zip(
                request["contact_sheets"], self.library.contact_sheets(), strict=True
            ):
                require(
                    {key: value for key, value in observed.items() if key != "image"}
                    == {
                        key: value for key, value in expected.items() if key != "image"
                    },
                    "Astra donor preview metadata differs from the pinned library",
                )
                with Image.open(
                    io.BytesIO(
                        base64.b64decode(observed["image"]["data"], validate=True)
                    )
                ) as image:
                    require(
                        image.mode == "RGB"
                        and image.format == "PNG"
                        and np.array_equal(np.asarray(image), expected["image"]),
                        "Astra donor preview pixels differ from the pinned library",
                    )
            require(
                request["previous_decisions"] == attempt["decisions"][:index]
                and request["completed_rollout_feedback"]
                == [outcome_feedback(row) for row in history],
                "Astra request leaks another arm or changes its decision/rollout history",
            )
            previous_request = request["previous_attempt"]
            require(
                previous_request["feedback"] == outcome_feedback(previous)
                and previous_request["decisions"] == previous["decisions"],
                "Previous-attempt feedback differs from the same-arm completed rollout",
            )
            recent = [row for row in generations if row["observation_step"] <= step][
                -4:
            ]
            require(
                len(request["observations"]) == len(recent),
                "Current raw history window differs",
            )
            for wire, generation in zip(request["observations"], recent, strict=True):
                self.snapshot(
                    wire,
                    {
                        "label": f"step_{generation['observation_step']}",
                        "step": generation["observation_step"],
                        "observation": generation["observation"],
                    },
                    request=request,
                    scope="current",
                )
            require(
                len(previous_request["snapshots"]) == len(prior_event["snapshots"]),
                "Previous rollout snapshot coverage differs",
            )
            for wire, snapshot in zip(
                previous_request["snapshots"], prior_event["snapshots"], strict=True
            ):
                self.snapshot(
                    wire, snapshot, request=request, scope="previous_completed"
                )
            require(
                decision_event["valid_until_step"] == step + 25
                and decision_event["raw_condition_fallback"]
                is (not decision["accepted"]),
                "Decision expiry/failure-clear declaration differs",
            )
            self.provider(request, record, decision)

    def attempt(self, attempt, expected_noise_proposal):
        from .image_perturbation_search import random_image_operations
        from .interpolation_audit import verify_reset

        name, mode = attempt["attempt_id"], attempt["mode"]
        require(
            name == f"{mode}_{attempt['iteration']}"
            and attempt["episode_id"] == self.episode_id,
            "Physical attempt identity mismatch",
        )
        event = self.attempt_events[name]
        generations = [
            row for row in self.groups["image_generation"] if row["attempt_id"] == name
        ]
        count = attempt["actions_executed"]
        require(
            type(count) is int
            and 0 < count <= 300
            and [row["observation_step"] for row in generations]
            == list(range(0, count, 5))
            and attempt["policy_replans"] == len(generations),
            "Physical action/replan coverage differs",
        )
        index = self.physical.index(attempt)
        preceding = (
            self.attempt_events[self.physical[index - 1]["attempt_id"]]["sequence"]
            if index
            else self.one("image_gate")["sequence"]
        )
        require(
            all(preceding < row["sequence"] < event["sequence"] for row in generations),
            "Generation occurred outside its physical attempt interval",
        )
        raw_initial = self.store.observation(generations[0]["observation"])
        verify_reset(
            attempt,
            self.entry,
            digest(raw_initial),
            self.summary["baseline"]["reset_audit"],
        )
        require(
            attempt["reset_audit"]["post_stabilization_observation_sha256"]
            == self.initial_observation_id,
            "Paired reset initial pixels differ",
        )
        require(
            attempt["noise_proposal"] == expected_noise_proposal,
            "Noise proposal differs from its assigned arm",
        )
        random_image = mode in ("random_occlusion", "random_demo_blend")
        astra = mode.startswith("astra_")
        expected_steps = list(range(0, count, 25)) if random_image or astra else []
        require(
            [row["observation_step"] for row in attempt["decisions"]] == expected_steps
            and [row["decision_index"] for row in attempt["decisions"]]
            == list(range(1, len(expected_steps) + 1)),
            "Scheduled decision slots are missing or duplicated",
        )
        random_events = [
            row
            for row in self.groups["random_image_decision"]
            if row["attempt_id"] == name
        ]
        if astra:
            require(not random_events, "Astra attempt contains random decisions")
            self.online_requests(attempt, generations)
        elif random_image:
            require(
                len(random_events) == len(attempt["decisions"])
                and all(
                    row["decision"] == decision
                    for row, decision in zip(
                        random_events, attempt["decisions"], strict=True
                    )
                ),
                "Random decision event/summary mismatch",
            )
        else:
            require(
                not random_events and not attempt["decisions"],
                "Native/noise control acquired image decisions",
            )
        if not astra:
            require(not attempt["provider_records"], "Control acquired provider calls")
        path = f"{name}_provider.jsonl"
        if attempt["provider_records"]:
            require(
                self.store.lines(path) == attempt["provider_records"],
                "Provider sidecar differs from physical summary",
            )
        else:
            require(not self.store.path(path).exists(), "Unreported provider sidecar")
        fresh = np.random.default_rng(np.random.SeedSequence(self.seed + [2]))
        random_rng = np.random.default_rng(
            np.random.SeedSequence(
                self.seed + [int(digest(mode)[:8], 16), attempt["iteration"]]
            )
        )
        totals = Counter()
        applied_ids = []
        image_mode = mode.removeprefix("random_").removeprefix("astra_")
        limits = (
            ImagePerturbationLimits(allowed_kinds=(image_mode,))
            if random_image or astra
            else DEFAULT_LIMITS
        )
        current_operations, current_id, current_decision = [], None, None
        for generation in generations:
            step = generation["observation_step"]
            require(
                generation["mode"] == mode
                and generation["iteration"] == attempt["iteration"]
                and generation["instruction"] == self.entry["instruction"],
                "Generation arm/iteration/task changed",
            )
            raw = self.store.observation(generation["observation"])
            modified = self.store.observation(generation["modified_observation"])
            if (random_image or astra) and step % 25 == 0:
                current_decision = attempt["decisions"][step // 25]
                current_operations, current_id = [], None
                proposal = current_decision["proposal"]
                if current_decision["accepted"]:
                    current_operations = proposal["image_perturbations"]
                    current_id = proposal["decision_id"]
                if random_image:
                    expected = random_image_operations(
                        image_mode,
                        raw,
                        [row["donor_id"] for row in self.library.catalog()],
                        random_rng,
                    )
                    require(
                        current_decision
                        == {
                            "decision_index": step // 25 + 1,
                            "observation_step": step,
                            "accepted": True,
                            "error": None,
                            "proposal": {
                                "decision_id": f"{name}_decision_{step // 25 + 1}",
                                "image_perturbations": expected,
                            },
                        },
                        "Random image control differs from its deterministic RNG",
                    )
                    require(
                        random_events[step // 25]["sequence"] < generation["sequence"],
                        "Random decision occurred after application",
                    )
            require(
                generation["image_perturbations"] == current_operations
                and generation["applied_accepted_decision_id"] == current_id,
                "Image edits were held beyond expiry, inherited across attempts, or survived a failed refresh",
            )
            if current_decision is not None:
                require(
                    current_decision["observation_step"]
                    <= step
                    < current_decision["observation_step"] + 25,
                    "Image decision outlived its 25-action interval",
                )
            replay_image_application(
                raw,
                modified,
                current_operations,
                self.library,
                generation["image_audit"],
                limits=limits,
            )
            require(
                generation["observation_sha256"] == digest(raw)
                and generation["modified_observation_sha256"] == digest(modified),
                "Generation raw/modified observation hashes differ",
            )
            self.condition(modified, generation["condition_id"])
            changed = digest(raw) != digest(modified)
            fallback = astra and current_id is None
            require(
                generation["image_has_effect"] is changed
                and generation["native_condition_fallback"] is fallback,
                "Generation effect/fallback flags differ from actual pixels/decisions",
            )
            noise = self.store.array(generation["latent"])
            expected_noise = (
                (
                    self.known
                    if step == 0
                    else fresh.standard_normal(FLOW_SHAPE).astype(np.float32)
                )
                if mode == "policy_fresh"
                else self.known
                if mode == "known_noise"
                else perturb_noise(self.recovered, self.basis, expected_noise_proposal)
            )
            self.maxima["noise_reconstruction_max_abs"] = max(
                self.maxima["noise_reconstruction_max_abs"],
                _float32(
                    noise,
                    expected_noise,
                    "Generation fixed/fresh/random noise",
                    1e-7 if expected_noise_proposal else 0,
                ),
            )
            endpoint = self.store.array(generation["generated_actions"])
            decoded = self.decode(endpoint).astype(np.float32)
            expected_actions = np.clip(
                decoded, self.spec["lower"], self.spec["upper"]
            ).astype(np.float32)
            actions = self.store.array(generation["controller_actions"])
            self.maxima["controller_decode_max_abs"] = max(
                self.maxima["controller_decode_max_abs"],
                _float32(actions, expected_actions, "Controller decoding", 1e-7),
            )
            clipping = {
                "count": int(np.count_nonzero(decoded != expected_actions)),
                "max_abs": float(np.max(np.abs(decoded - expected_actions))),
            }
            require(
                generation["clipping"] == clipping
                and generation["velocity_evaluations"] == 10,
                "Generation clipping/velocity evaluation count mismatch",
            )
            totals["clipped_values"] += clipping["count"]
            totals["velocity_evaluations"] += 10
            totals["image_active_policy_calls"] += bool(current_operations)
            totals["image_changed_policy_calls"] += changed
            totals["accepted_decision_policy_calls"] += current_id is not None
            totals["native_condition_fallback_policy_calls"] += fallback
            executed = min(5, count - step)
            totals["actions_with_accepted_decision"] += executed * (
                current_id is not None
            )
            totals["actions_with_changed_image"] += executed * changed
            totals["native_condition_fallback_actions"] += executed * fallback
            if current_id is not None and current_id not in applied_ids:
                applied_ids.append(current_id)
        require(
            all(attempt[key] == value for key, value in totals.items())
            and attempt["applied_accepted_decision_ids"] == applied_ids
            and attempt["accepted_decisions_executed"] == len(applied_ids),
            "Physical attempt application/action/cost counters differ",
        )
        require(
            type(attempt["success"]) is bool
            and type(attempt["terminated"]) is bool
            and attempt["status"]
            == (
                "success"
                if attempt["success"]
                else "terminated"
                if attempt["terminated"]
                else "budget_exhausted"
            )
            and (attempt["success"] or attempt["terminated"] or count == 300),
            "Attempt status disagrees with binary outcome or action cap",
        )
        snapshots = event["snapshots"]
        require(
            snapshots
            and snapshots[0]["step"] == 0
            and snapshots[-1]["step"] == count
            and len(snapshots) <= 4,
            "Rollout snapshots lack initial/final coverage",
        )
        require(
            [row["step"] for row in snapshots]
            == sorted({row["step"] for row in snapshots}),
            "Rollout snapshots are not distinct and ordered",
        )
        raw_by_step = {
            row["observation_step"]: row["observation_sha256"] for row in generations
        }
        for snapshot in snapshots:
            raw = self.store.observation(snapshot["observation"])
            if snapshot["step"] in raw_by_step:
                require(
                    digest(raw) == raw_by_step[snapshot["step"]],
                    "Rollout snapshot differs from the same-step raw policy observation",
                )
        require(
            digest(self.store.observation(snapshots[0]["observation"]))
            == self.initial_observation_id,
            "Initial rollout snapshot differs from paired reset",
        )
        self.counts.update(
            physical_rollouts=1,
            physical_actions=count,
            generations=len(generations),
            rollout_velocity_evaluations=totals["velocity_evaluations"],
            accepted_decisions_executed=len(applied_ids),
        )
        self.counts.update(
            {
                key: totals[key]
                for key in (
                    "image_active_policy_calls",
                    "image_changed_policy_calls",
                    "actions_with_accepted_decision",
                    "actions_with_changed_image",
                    "native_condition_fallback_actions",
                )
            }
        )

    def summaries(self):
        from .image_perturbation_agent import summarize_calls

        setup = (
            self.counts["initialization_velocity_evaluations"]
            + self.counts["image_gate_velocity_evaluations"]
        )
        for arm in ARMS:
            rows, value = (
                self.summary["arms"][arm]["attempts"],
                self.summary["arms"][arm]["summary"],
            )
            require(
                [row["iteration"] for row in rows] == list(range(1, len(rows) + 1))
                and 1 <= len(rows) <= 3,
                "Arm attempt iterations/cap differ",
            )
            require(
                all(row["mode"] == arm for row in rows[1:]),
                "Arm contains another intervention method",
            )
            first = next((row for row in rows if row["success"]), None)
            prefix = [
                row
                for row in rows
                if first is None or row["iteration"] <= first["iteration"]
            ]
            forced = (
                self.summary["development"]
                and self.protocol["development_force_one_intervention"]
            )
            require(
                len(rows)
                == (max(2 if forced else 1, first["iteration"]) if first else 3),
                "Arm stopping/censoring does not match its complete budget",
            )
            calls = [call for row in rows for call in row["provider_records"]]
            prefix_calls = [call for row in prefix for call in row["provider_records"]]
            provider = summarize_calls(calls)
            prefix_provider = summarize_calls(prefix_calls)
            expected = {
                "success": first is not None,
                "first_success_attempt": first["iteration"] if first else None,
                "full_rollout_revisions_to_success": first["iteration"] - 1
                if first
                else None,
                "within_successful_rollout_decisions": len(first["decisions"])
                if first
                else None,
                "decisions_through_success_or_cap": sum(
                    len(row["decisions"]) for row in prefix
                ),
                "physical_decisions": sum(len(row["decisions"]) for row in rows),
                "rollouts_executed": len(rows),
                "attempt_budget": 3,
                "censored_without_success": first is None,
                "success_by_attempt": [
                    bool(first and first["iteration"] <= budget) for budget in (1, 2, 3)
                ],
                "actions_through_success_or_cap": sum(
                    row["actions_executed"] for row in prefix
                ),
                "velocity_evaluations_through_success_or_cap": sum(
                    row["velocity_evaluations"] for row in prefix
                ),
                "standalone_velocity_evaluations_through_success_or_cap": setup
                + sum(row["velocity_evaluations"] for row in prefix),
                "rollout_seconds_through_success_or_cap": sum(
                    row["wall_seconds"] for row in prefix
                ),
                "physical_actions": sum(row["actions_executed"] for row in rows),
                "physical_velocity_evaluations": sum(
                    row["velocity_evaluations"] for row in rows
                ),
                "provider": provider,
                "provider_through_success_or_cap": prefix_provider,
                "tokens_to_first_success": prefix_provider if first else None,
                "development_extra_rollouts_after_success": len(rows) - len(prefix),
            }
            require(
                value == expected,
                "Arm success/censor/prefix/physical accounting differs from recorded attempts",
            )
        cost = self.summary["physical_cost"]
        all_calls = [call for row in self.physical for call in row["provider_records"]]
        expected = {
            "rollouts": len(self.physical),
            "simulated_actions": sum(row["actions_executed"] for row in self.physical),
            "rollout_velocity_evaluations": sum(
                row["velocity_evaluations"] for row in self.physical
            ),
            "initialization_velocity_evaluations": self.counts[
                "initialization_velocity_evaluations"
            ],
            "image_gate_velocity_evaluations": self.counts[
                "image_gate_velocity_evaluations"
            ],
            "velocity_evaluations": setup
            + sum(row["velocity_evaluations"] for row in self.physical),
            "rollout_wall_seconds": sum(row["wall_seconds"] for row in self.physical),
            "token_usage": summarize_calls(all_calls),
        }
        require(
            all(cost[key] == value for key, value in expected.items()),
            "Physical cost does not count each actual attempt/setup exactly once",
        )
        self.provider_summary = expected["token_usage"]
        self.counts["velocity_evaluations"] = expected["velocity_evaluations"]
        require(
            self.counts["physical_rollouts"] == cost["rollouts"]
            and self.counts["physical_actions"] == cost["simulated_actions"]
            and self.counts["physical_provider_calls"]
            == self.provider_summary["provider_calls"]
            and self.counts["accepted_provider_proposals"]
            == self.provider_summary["accepted_proposals"],
            "Independent execution counters disagree with summary",
        )

    def run(self):
        initial = self.initialization()
        self.gate(initial)
        rng = np.random.default_rng(np.random.SeedSequence(self.seed + [3]))
        for attempt in self.physical:
            proposal = attempt["noise_proposal"]
            if attempt["mode"] == "random_noise":
                expected = random_noise_proposal(self.basis, rng)
                require(
                    type(proposal) is dict
                    and set(proposal) == set(expected)
                    and proposal["basis_id"] == expected["basis_id"]
                    and proposal["kind"] == expected["kind"]
                    and proposal["perturbation_scale"]
                    == expected["perturbation_scale"],
                    "Random noise basis/kind/scale differs from seeded control",
                )
                coefficients = proposal["coefficients"]
                require(
                    type(coefficients) is list
                    and len(coefficients) == 8
                    and all(
                        type(value) in (int, float) and math.isfinite(value)
                        for value in coefficients
                    ),
                    "Invalid random noise coefficient schema",
                )
                observed, wanted = (
                    np.asarray(coefficients, np.float64),
                    np.asarray(expected["coefficients"], np.float64),
                )
                require(
                    np.all(np.abs(observed - wanted) <= 4 * np.abs(np.spacing(wanted))),
                    "Random coefficient differs by more than four float64 ULPs",
                )
            else:
                require(
                    proposal is None, "Image/native arm changed the recovered noise"
                )
            self.attempt(attempt, proposal)
        require(
            len(self.groups["image_generation"]) == self.counts["generations"]
            and len(self.groups["image_request"]) == self.counts["requests"]
            and len(self.groups["image_decision"]) == self.counts["requests"],
            "Unassigned generation/provider event",
        )
        expected_sidecars = {
            f"{row['attempt_id']}_provider.jsonl"
            for row in self.physical
            if row["provider_records"]
        }
        require(
            {path.name for path in self.store.directory.glob("*_provider.jsonl")}
            == expected_sidecars,
            "Unassigned provider sidecar",
        )
        random_count = sum(
            len(row["decisions"])
            for row in self.physical
            if row["mode"] in ("random_occlusion", "random_demo_blend")
        )
        require(
            len(self.groups["random_image_decision"]) == random_count,
            "Unassigned random decision event",
        )
        self.summaries()
        if self.summary["development"]:
            validation = self.one("image_development_validation")
            require(
                all(
                    validation[key] == value
                    for key, value in self.summary["development_validation"].items()
                )
                and validation["passed"] is True,
                "Development validation failed or differs from event",
            )
            require(
                set(validation["arms"]) == {"astra_occlusion", "astra_demo_blend"},
                "Missing development Astra validation arm",
            )
            for arm, check in validation["arms"].items():
                rows = self.summary["arms"][arm]["attempts"][1:]
                require(
                    check["passed"] is True
                    and check["provider_bindings_verified"] is True
                    and check["accepted_decisions_executed"]
                    == sum(row["accepted_decisions_executed"] for row in rows)
                    > 0
                    and check["image_changed_policy_calls"]
                    == sum(row["image_changed_policy_calls"] for row in rows),
                    "Development lacks a genuine accepted-and-executed decision",
                )
        else:
            require(
                not self.groups["image_development_validation"],
                "Evaluation includes development validation/extras",
            )
        arrays = self.store.finish()
        self.counts.update(
            {
                key: value
                for key, value in arrays.items()
                if key != "array_inventory_sha256"
            }
        )
        counts = dict(self.counts)
        counts.update(
            actions=counts["physical_actions"],
            provider_calls=self.provider_summary["provider_calls"],
            accepted_proposals=self.provider_summary["accepted_proposals"],
            executed_decisions=counts["accepted_decisions_executed"],
        )
        root = Path(__file__).parent
        sources = [
            "image_perturbation_audit.py",
            "image_perturbations.py",
            "image_donor_bank.py",
            "image_perturbation_agent.py",
            "image_perturbation_search.py",
            "interventions.py",
            "interpolation_audit.py",
            "intervention_rollout.py",
            "intervention_agent.py",
            "action_adapter.py",
            "openpi_inputs.py",
            "flow.py",
            "records.py",
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "complete": True,
            "episode_id": self.episode_id,
            "protocol_sha256": digest(self.protocol),
            "library_id": self.library.metadata()["library_id"],
            "summary_sha256": self.store.files["summary.json"],
            "events_sha256": self.store.files["events.jsonl"],
            "reset_entry_sha256": digest(self.entry),
            "all_arrays_verified": True,
            "provider_bindings_verified": True,
            "reset_pairing_verified": True,
            "decision_lifetimes_verified": True,
            "array_inventory_sha256": arrays["array_inventory_sha256"],
            "counts": counts,
            "provider": self.provider_summary,
            "numerical_maxima": self.maxima,
            "input_file_sha256": self.store.files,
            "asset_sha256": {"norm_stats.json": self.asset_sha256},
            "bindings": self.bindings,
            "source_sha256": {name: file_sha256(root / name) for name in sources},
            "limitations": LIMITATIONS,
        }


def audit_case(case_dir, *, library, metadata, assets):
    """Audit one completed case against a loaded, verified donor library."""
    return _CaseAudit(case_dir, library=library, metadata=metadata, assets=assets).run()


def _audit_worker(directory, *, library, assets):
    from .config import BenchmarkConfig
    from .image_perturbation_search import load_protocol
    from .intervention_rollout import load_reset_manifest

    store = ArtifactStore(directory)
    files = (
        "runtime.json",
        "protocol.json",
        "checkpoint.json",
        "reset_manifest.json",
        "image_library.json",
        "frozen_plan.json",
        "progress.json",
    )
    metadata = {name: store.read_json(name) for name in files}
    runtime, plan = metadata["runtime.json"], metadata["frozen_plan.json"]
    phase = runtime["phase"]
    require(
        phase in ("development", "evaluation")
        and metadata["progress.json"]["status"] == "complete",
        "Worker is not a completed image experiment",
    )
    protocol = load_protocol()
    if phase == "development":
        protocol["seed"] = protocol["development_seed"]
    require(
        metadata["protocol.json"] == protocol
        and plan["protocol_sha256"] == digest(protocol)
        and plan["runtime"] == runtime,
        "Worker protocol/runtime differs from its frozen declaration",
    )
    require(
        metadata["image_library.json"] == library.metadata()
        and plan["image_library_sha256"] == digest(library.metadata())
        and len(library.catalog()) == 45,
        "Worker donor library differs from complete pinned library",
    )
    manifest = load_reset_manifest(
        Path(directory) / "reset_manifest.json",
        BenchmarkConfig.preset(runtime["assignment"]["suite"]),
    )
    require(
        plan["manifest_sha256"] == manifest["sha256"],
        "Worker full reset manifest digest differs",
    )
    assignment = runtime["assignment"]
    expected = (
        [row for row in manifest["episodes"] if row["task_id"] == assignment["task_id"]]
        if phase == "development"
        else manifest["episodes"][assignment["case_shard"] :: assignment["case_shards"]]
    )
    require(
        plan["assigned_episodes"] == [row["episode_id"] for row in expected],
        "Worker episode partition differs from the frozen reset manifest",
    )
    expected_dirs = {
        f"case_{row['task_id']}_{row['initial_state_id']}" for row in expected
    }
    require(
        {path.name for path in Path(directory).glob("case_*")} == expected_dirs,
        "Worker has missing/extra case directories",
    )
    cases = [
        audit_case(
            Path(directory) / name, library=library, metadata=metadata, assets=assets
        )
        for name in sorted(expected_dirs)
    ]
    require(
        {row["episode_id"] for row in cases} == set(plan["assigned_episodes"]),
        "Audited case identities differ from assignments",
    )
    store.finish()
    counts = Counter()
    for case in cases:
        counts.update(case["counts"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "complete": True,
        "phase": phase,
        "worker": runtime["worker"],
        "workflow": runtime["workflow"],
        "payload_sha256": runtime["payload_sha256"],
        "protocol_sha256": digest(protocol),
        "library_id": library.metadata()["library_id"],
        "reset_manifest_sha256": manifest["sha256"],
        "source_metadata_sha256": store.files,
        "counts": dict(counts),
        "cases": cases,
        "limitations": LIMITATIONS,
    }


def audit_worker(results_root, *, library, assets, output=None):
    """Read an extracted worker or full tar.gz; optionally write a new receipt."""
    from .image_donor_bank import load_library
    from .interpolation_audit import _extract_archive

    library = load_library(library) if isinstance(library, (str, Path)) else library
    path = Path(results_root)
    if path.is_dir():
        receipt = _audit_worker(path, library=library, assets=assets)
    else:
        before = file_sha256(path)
        with tempfile.TemporaryDirectory(prefix="image-audit-") as temporary:
            uncompressed_bytes = _extract_archive(path, Path(temporary))
            receipt = _audit_worker(
                Path(temporary) / "results", library=library, assets=assets
            )
        require(file_sha256(path) == before, "Archive changed during audit")
        receipt["archive"] = {
            "sha256": before,
            "bytes": path.stat().st_size,
            "uncompressed_bytes": uncompressed_bytes,
            "gzip_crc_verified": True,
        }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            stream.write(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = audit_worker(
        args.results, library=args.library, assets=args.assets, output=args.output
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cases": len(receipt["cases"]),
                "receipt_sha256": file_sha256(args.output),
                "counts": receipt["counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
