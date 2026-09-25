"""Read-only tensor audit for complete phase-interpolation worker artifacts.

No model, provider, or simulator calls occur. Stored tensors, tokenization,
noise operations, image rendering, endpoint metrics and decoding are checked on
CPU. Hidden states/embedding hashes are provenance checks, not a GPU replay.
"""

import argparse
import copy
import gzip
import json
import math
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from .config import BenchmarkConfig
from .flow import error_metrics
from .interpolation_catalog import donor_catalog, oracle_for, paper_alpha
from .interpolation_conditioning import ALIGNMENT, CAPTURE_BOUNDARY
from .interpolation_search import ARMS, load_protocol
from .intervention_rollout import load_reset_manifest
from .interventions import (
    apply_vision,
    noise_basis,
    perturb_noise,
    random_noise_proposal,
)
from .records import digest, file_sha256

SCHEMA = "phase-interpolation-array-audit-1"
ARRAY_KEYS = {"array", "shape", "dtype", "sha256"}
NOISE_ATOL = 1e-7
RANDOM_COEFFICIENT_ULPS = 4
RENDERER = "Pillow-Linux-x86_64-separate-float32-blend"
FLOW_SHAPE = (1, 10, 32)
OBSERVATION_KEYS = {"observation/image", "observation/wrist_image", "observation/state"}
GATE_CHECKS = {
    "tei_identical_sources": ("tei", 0.37, False),
    "tli_zero_residual": ("tli", 0.5, False),
    "tei_nonzero_oracle_sources": ("tei", 0.0, True),
    "tli_nonzero_oracle_banks": ("tli", 0.0, True),
}
LIMITATIONS = [
    "No GPU solve, hidden-state extraction, model inference, provider call or simulator replay is performed.",
    "Text mask/token IDs, bank identities, condition hashes and recorded boundary/norm flags are checked. Unstored prefix embeddings and per-layer residuals are not independently recomputed.",
    "Initial zero pooled-embedding parity is worker-reported: that separate endpoint is not stored. The four new interpolation probe endpoints are stored and independently compared.",
    "Bank metadata is bound to the frozen bank inventory. Bank NPZ bytes and demonstration extraction correctness require the separate donor-bank audit.",
    "Provider request/response binding, visual feedback scope and online hold/expiry replay require the separate feedback audit.",
    "Recorded task outcomes are not independently re-executed. Numerical consistency does not imply intervention efficacy or generalization.",
]


class AuditError(ValueError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    def nonfinite(value):
        raise AuditError(f"Nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=_object, parse_constant=nonfinite)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _metrics(recorded, expected, label):
    require(set(recorded) == {"max_abs", "rmse"}, f"{label}: malformed error metrics")
    for name, value in expected.items():
        require(
            type(recorded[name]) in (int, float)
            and math.isfinite(recorded[name])
            and math.isclose(recorded[name], value, rel_tol=1e-12, abs_tol=1e-15),
            f"{label}: recomputed {name} differs",
        )


def _same_float32(actual, expected, label, *, atol=0):
    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32,
        f"{label}: shape/dtype mismatch",
    )
    maximum = float(
        np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    )
    require(maximum <= atol, f"{label}: maximum difference {maximum} exceeds {atol}")
    return maximum


def render_linux(observation, annotations):
    """Reproduce the measured Linux blend exactly; never accept pixel tolerance."""
    from PIL import Image, ImageDraw

    validated = apply_vision(observation, annotations)
    if not annotations:
        return validated
    result = copy.deepcopy(observation)
    for annotation in annotations:
        gain = annotation["gain"]
        if gain == 0:
            continue
        camera = annotation["camera"]
        image = Image.fromarray(result[camera])
        layer = image.copy()
        draw = ImageDraw.Draw(layer)
        coordinates = np.asarray(annotation["coordinates"], dtype=np.float64).tolist()
        if annotation["kind"] == "box":
            draw.rectangle(coordinates, outline=(255, 0, 255), width=3)
        else:
            x, y = coordinates
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 255))
        original = np.asarray(image, dtype=np.float32)
        delta = np.asarray(layer, dtype=np.float32) - original
        scaled = np.multiply(delta, np.float32(gain), dtype=np.float32)
        result[camera] = np.clip(
            np.add(original, scaled, dtype=np.float32), 0, 255
        ).astype(np.uint8)
    return result


class ArrayStore:
    """Require exact one-to-one coverage of array files and recorded references."""

    def __init__(self, directory, rows):
        self.directory = Path(directory).resolve()
        self.references, self.inventory = {}, {}
        self.reference_count = 0
        for row in rows:
            self._scan(row)
        files = {
            str(path.relative_to(self.directory))
            for path in (self.directory / "arrays").rglob("*")
            if path.is_file()
        }
        require(files == set(self.references), "Missing or unreferenced array files")

    def _scan(self, value):
        if isinstance(value, dict):
            if "array" in value:
                require(set(value) == ARRAY_KEYS, "Malformed array reference")
                name = value["array"]
                path = PurePosixPath(name)
                require(
                    not path.is_absolute()
                    and path.parts[:1] == ("arrays",)
                    and len(path.parts) == 2
                    and path.suffix == ".npy"
                    and ".." not in path.parts
                    and "\\" not in name,
                    "Unsafe array reference",
                )
                require(
                    name not in self.references or self.references[name] == value,
                    "Conflicting array references",
                )
                require(_sha(value["sha256"]), "Malformed array digest")
                self.references[name] = value
                self.reference_count += 1
            else:
                for item in value.values():
                    self._scan(item)
        elif isinstance(value, list):
            for item in value:
                self._scan(item)

    def read(self, reference):
        name = reference["array"]
        require(
            name in self.references and self.references[name] == reference,
            "Unregistered array reference",
        )
        path = self.directory / name
        require(
            not path.is_symlink() and path.resolve().is_relative_to(self.directory),
            "Array path escapes its case",
        )
        array = np.load(path, allow_pickle=False)
        require(
            isinstance(array, np.ndarray)
            and array.dtype.kind in "buif"
            and np.isfinite(array).all(),
            f"Invalid/nonfinite saved array: {name}",
        )
        require(
            list(array.shape) == reference["shape"]
            and str(array.dtype) == reference["dtype"]
            and digest(array) == reference["sha256"],
            f"Array hash/shape/dtype mismatch: {name}",
        )
        if name not in self.inventory:
            self.inventory[name] = {
                "file_sha256": file_sha256(path),
                "record_sha256": reference["sha256"],
                "bytes": path.stat().st_size,
            }
        else:
            require(
                self.inventory[name]["file_sha256"] == file_sha256(path),
                "Array changed during audit",
            )
        return array

    def resolve(self, value):
        if isinstance(value, dict):
            if "array" in value:
                return self.read(value)
            return {key: self.resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        return value

    def finish(self):
        for name, reference in self.references.items():
            if name not in self.inventory:
                self.read(reference)
            else:
                require(
                    file_sha256(self.directory / name)
                    == self.inventory[name]["file_sha256"],
                    "Array bytes changed before audit completion",
                )
        return {
            "array_files": len(self.inventory),
            "array_references": self.reference_count,
            "array_bytes": sum(row["bytes"] for row in self.inventory.values()),
            "array_inventory_sha256": digest(self.inventory),
        }


def validate_observation(observation):
    require(set(observation) == OBSERVATION_KEYS, "Unexpected observation channels")
    for key in OBSERVATION_KEYS - {"observation/state"}:
        require(
            observation[key].dtype == np.uint8
            and observation[key].shape == (224, 224, 3),
            "Raw camera shape/dtype differs from frozen policy view",
        )
    state = observation["observation/state"]
    require(
        state.shape == (8,) and state.dtype == np.float32 and np.isfinite(state).all(),
        "Invalid raw state8",
    )


class Context:
    def __init__(self, metadata, assets):
        import sentencepiece

        self.metadata, self.checkpoint = metadata, metadata["checkpoint.json"]
        assets = Path(assets)
        profile = self.checkpoint["input_profile_assets"]
        files = {
            "norm_stats": assets / "norm_stats.json",
            "tokenizer": assets / "paligemma_tokenizer.model",
        }
        for key, path in files.items():
            require(
                file_sha256(path) == profile[key]["sha256"],
                f"{key} asset checksum mismatch",
            )
        self.asset_hashes = {key: file_sha256(path) for key, path in files.items()}
        stats = _json(files["norm_stats"].read_text())["norm_stats"]["actions"]
        self.q01, self.q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=files["tokenizer"].read_bytes()
        )
        self.token_cache = {}
        self.bank_inventory = metadata["bank_inventory.json"]["donors"]
        require(
            set(self.bank_inventory) == {row["source_id"] for row in donor_catalog()},
            "Missing donor bank inventory entries",
        )
        self.banks = {}
        for donor in donor_catalog():
            entry = self.bank_inventory[donor["source_id"]]
            require(
                "bank" in entry,
                "Frozen bank inventory must contain full donor bank metadata",
            )
            bank = entry["bank"]
            require(
                bank["bank_id"]
                == digest(
                    {key: value for key, value in bank.items() if key != "bank_id"}
                ),
                "Invalid donor bank metadata identity",
            )
            require(
                bank["provenance"]["source_prompt"] == donor["prompt"],
                "Donor prompt differs from catalog",
            )
            self.banks[donor["source_id"]] = bank
        root = Path(__file__).parent
        self.operator_sha = file_sha256(root / "interpolation_conditioning.py")
        require(
            self.checkpoint["adapter_source_sha256"]
            == file_sha256(root / "lerobot_policy.py"),
            "Adapter source differs from frozen checkpoint metadata",
        )
        require(
            self.checkpoint["input_profile"] == "openpi_libero"
            and self.checkpoint["input_profile_source_sha256"]
            == file_sha256(root / "openpi_inputs.py")
            and self.checkpoint["frozen"] is True
            and self.checkpoint["horizon"] == 10
            and self.checkpoint["model_action_dim"] == 32,
            "Unexpected checkpoint/profile",
        )
        self.compatibility = {
            key: copy.deepcopy(self.checkpoint.get(key))
            for key in (
                "artifact_sha256",
                "model_source_sha256",
                "adapter_source_sha256",
                "processor_source_sha256",
                "transformers_source_sha256",
                "input_profile",
                "input_profile_source_sha256",
                "input_profile_assets",
            )
        }
        self.compatibility.update(
            operator_source_sha256=self.operator_sha,
            layer_count=18,
            width=2048,
            text_slots=200,
            native_parameter_dtypes=["torch.float32"],
        )
        inventory = _json(
            (root / "checkpoints/lerobot_pi05_libero_base.json").read_text()
        )
        published = {row["path"]: row["sha256"] for row in inventory["files"]}
        for name in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            require(
                self.checkpoint["artifact_sha256"][name] == published[name],
                "Checkpoint metadata does not name the pinned published weights/processors",
            )
        for source_id, bank in self.banks.items():
            require(
                bank["provenance"]["compatibility"] == self.compatibility
                and bank["provenance"]["capture_boundary"] == CAPTURE_BOUNDARY
                and bank["provenance"]["capture_kind"] == "demonstration_mean",
                f"Donor bank model/capture mismatch: {source_id}",
            )
            for name, values in zip(
                ("token_ids", "token_mask", "instruction_mask"),
                self.tokens(bank["provenance"]["source_prompt"]),
                strict=True,
            ):
                require(
                    bank["arrays"][name]
                    == {
                        "shape": list(values.shape),
                        "dtype": str(values.dtype),
                        "sha256": digest(values),
                    },
                    "Donor bank token/mask array digest mismatch",
                )

    def tokens(self, prompt):
        if prompt not in self.token_cache:
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            text = self.tokenizer.encode(cleaned, add_bos=False)
            ids = self.tokenizer.encode(cleaned, add_bos=True) + self.tokenizer.encode(
                "\n"
            )
            require(
                ids[: len(text) + 1] == [self.tokenizer.bos_id(), *text]
                and len(ids) <= 200
                and text,
                "Unsupported instruction token layout",
            )
            tokens = np.array([ids + [0] * (200 - len(ids))], np.int64)
            valid = np.arange(200)[None] < len(ids)
            instruction = np.zeros_like(valid)
            instruction[:, 1 : len(text) + 1] = True
            self.token_cache[prompt] = (tokens, valid, instruction)
        return self.token_cache[prompt]

    def decode(self, value):
        require(
            value.shape == FLOW_SHAPE and value.dtype == np.float32,
            "Full native endpoint must be float32 [1,10,32]",
        )
        return (value[0, :, :7] + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01

    def mapping(self, source, target):
        a = np.flatnonzero(self.tokens(source)[2][0]).tolist()
        b = np.flatnonzero(self.tokens(target)[2][0]).tolist()
        count = min(len(a), len(b))
        return {
            "rule": ALIGNMENT,
            "source_positions": a,
            "target_positions": b,
            "mapped_positions": [
                list(pair) for pair in zip(a[:count], b[:count], strict=True)
            ],
            "source_instruction_tokens": len(a),
            "target_instruction_tokens": len(b),
            "zero_padded_tokens": len(b) - count,
            "truncated_tokens": len(a) - count,
        }

    def text_metrics(self, row, target, sources=None, *, factor=None):
        for name in ("text_before_sha256", "text_after_sha256"):
            require(_sha(row[name]), "Malformed hidden-state hash")
        require(
            type(row["has_effect"]) is bool and row["protected_text_unchanged"] is True,
            "Protected text slots changed",
        )
        require(
            row["has_effect"]
            == (row["text_before_sha256"] != row["text_after_sha256"]),
            "Hidden-state effect/hash mismatch",
        )
        for key in ("delta_frobenius", "delta_rms", "base_frobenius"):
            require(
                type(row[key]) in (int, float)
                and math.isfinite(row[key])
                and row[key] >= 0,
                "Invalid text norm provenance",
            )
        require(
            (row["delta_frobenius"] > 0) == row["has_effect"],
            "Text effect/norm inconsistency",
        )
        count = int(self.tokens(target)[2].sum()) * 2048
        require(
            math.isclose(
                row["delta_rms"],
                row["delta_frobenius"] / math.sqrt(count),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ),
            "Text RMS/Frobenius inconsistency",
        )
        relative = (
            row["delta_frobenius"] / row["base_frobenius"]
            if row["base_frobenius"]
            else (0.0 if row["delta_frobenius"] == 0 else None)
        )
        require(row["relative_rms"] == relative, "Relative text RMS inconsistency")
        if sources is not None:
            require(
                row["mapping_a"] == self.mapping(sources[0], target)
                and row["mapping_b"] == self.mapping(sources[1], target)
                and row["factor"] == factor,
                "Instruction alignment/factor mismatch",
            )

    def conditioning(
        self, value, observation, target, sources, operator, alpha, bank_ids=None
    ):
        observation_id = digest(observation)
        raw_id = digest({**observation, "prompt": target})
        require(
            value["raw_condition_id"] == raw_id
            and value["target_prompt"] == target
            and value["source_prompts"] == list(sources),
            "Interpolation prompt/observation mismatch",
        )
        # Vision callers bind observation_id to the original raw frame separately.
        require(_sha(value["observation_id"]), "Invalid original observation ID")
        require(
            value["operator"] == operator
            and value["operator_version"] == 1
            and value["alpha"] == alpha
            and value["operator_source_sha256"] == self.operator_sha
            and value["compatibility"] == self.compatibility,
            "Interpolation operator/model/alpha mismatch",
        )
        require(
            value["alignment"] == ALIGNMENT
            and value["tli_boundary"] == CAPTURE_BOUNDARY
            and value["tei_orientation"] == "A_at_0_B_at_1"
            and value["tei_formula"] == "(1-alpha)*E_A+alpha*E_B"
            and value["tli_formula"] == "(1-2*alpha)*(T_A-T_B)",
            "Interpolation formula/boundary changed",
        )
        for key in (
            "text_mask_fixed",
            "direct_writes_instruction_only",
            "later_vision_hidden_states_may_change",
            "cache_rebuilt",
            "vision_prefix_unchanged",
            "protected_embedding_slots_unchanged",
            "hooks_removed",
        ):
            require(value[key] is True, f"Interpolation boundary flag failed: {key}")
        for name in (
            "prefix_before_sha256",
            "prefix_after_sha256",
            "prefix_padding_sha256",
            "prefix_attention_sha256",
            "prefix_positions_sha256",
        ):
            require(_sha(value[name]), "Malformed prefix digest")
        tokens, valid, instruction = self.tokens(target)
        require(
            value["target_token_sha256"] == digest(tokens)
            and value["target_token_mask_sha256"] == digest(valid)
            and value["target_instruction_mask_sha256"] == digest(instruction)
            and value["target_instruction_positions"]
            == np.flatnonzero(instruction[0]).tolist(),
            "Target text token/mask mismatch",
        )
        expected_sources = [
            {
                "token_sha256": digest(parts[0]),
                "token_mask_sha256": digest(parts[1]),
                "instruction_mask_sha256": digest(parts[2]),
            }
            for parts in map(self.tokens, sources)
        ]
        require(
            value["source_tokens"] == expected_sources, "Source token/mask mismatch"
        )
        # The pinned export has two real 16x16 patch grids plus one masked camera.
        padding = np.concatenate(
            (np.ones((1, 512), bool), np.zeros((1, 256), bool), valid), axis=1
        )
        require(
            value["text_start"] == 768
            and value["text_slots"] == 200
            and value["embedding_dtype"] == "torch.float32"
            and value["prefix_padding_sha256"] == digest(padding)
            and value["prefix_attention_sha256"] == digest(np.zeros_like(padding))
            and value["prefix_positions_sha256"]
            == digest(np.cumsum(padding, axis=1) - 1),
            "Native prefix masks/positions changed",
        )
        tei = operator in ("tei", "tei_tli")
        if tei:
            require(
                len(value["source_scaled_embedding_sha256"]) == 2
                and all(_sha(item) for item in value["source_scaled_embedding_sha256"]),
                "Missing TEI native source embedding hashes",
            )
        self.text_metrics(value["tei"], target, sources if tei else None)
        require(
            value["tei"]["has_effect"]
            == (value["prefix_before_sha256"] != value["prefix_after_sha256"]),
            "TEI prefix hash/effect mismatch",
        )
        if not tei:
            require(
                not value["tei"]["has_effect"],
                "TLI-only prefill changed input embeddings",
            )
        active_tli = operator in ("tli", "tei_tli") and alpha != 0.5
        require(
            value["tli_active"] == active_tli
            and value["tli_layer_indices"] == (list(range(17)) if active_tli else []),
            "Wrong effective TLI layers",
        )
        if active_tli:
            require(
                bank_ids is not None and set(value["banks"]) == {"a", "b"},
                "Missing TLI bank binding",
            )
            for label, source_id, prompt in zip(
                ("a", "b"), bank_ids, sources, strict=True
            ):
                bank = self.banks[source_id]
                require(
                    value["banks"][label] == bank
                    and bank["provenance"]["source_prompt"] == prompt
                    and bank["provenance"]["compatibility"] == self.compatibility,
                    "TLI bank differs from frozen donor inventory",
                )
                require(
                    bank["arrays"]["states"]["shape"] == [18, 1, 200, 2048]
                    and bank["arrays"]["states"]["dtype"] == "float32"
                    and bank["captured_layer_indices"] == list(range(18))
                    and bank["effective_layer_indices"] == list(range(17)),
                    "Donor bank layer/shape mismatch",
                )
            require(
                [row["layer_index"] for row in value["tli_layers"]] == list(range(17)),
                "Missing or duplicate TLI residual layer",
            )
            for row in value["tli_layers"]:
                require(
                    row["hidden_dtype"] == "torch.float32"
                    and row["direct_vision_write_unchanged"] is True,
                    "TLI changed native dtype or direct image slots",
                )
                self.text_metrics(row, target, sources, factor=1 - 2 * alpha)
        else:
            require(
                value["banks"] is None and value["tli_layers"] == [],
                "Disabled TLI injected residuals",
            )
        effect = value["tei"]["has_effect"] or any(
            row["has_effect"] for row in value["tli_layers"]
        )
        require(
            value["has_effect"] == effect
            and value["native_target_equivalent"] == (not effect),
            "Overall interpolation effect is inconsistent",
        )
        identity = {key: item for key, item in value.items() if key != "condition_id"}
        expected = (
            digest({"raw_condition_id": raw_id, "interpolation": identity})
            if effect
            else raw_id
        )
        require(
            value["condition_id"] == expected,
            "Interpolated condition ID does not bind its full provenance",
        )
        return expected, observation_id


def verify_reset(attempt, entry, observation_id, expected=None):
    reset = attempt["reset_audit"]
    require(
        reset["sha256"]
        == digest({key: value for key, value in reset.items() if key != "sha256"}),
        "Paired reset audit checksum mismatch",
    )
    if expected is not None:
        require(reset == expected, "Attempt reset differs from paired baseline")
    for key in (
        "episode_id",
        "seed",
        "reset_state_sha256",
        "reset_model_sha256",
        "bddl_sha256",
    ):
        require(reset[key] == entry[key], f"Paired reset {key} differs from manifest")
    require(
        reset["post_stabilization_observation_sha256"] == observation_id
        and reset["initial_success"] is False
        and reset["initial_terminated"] is False
        and attempt["initial_success"] is False
        and attempt["zero_action_success"] is False,
        "Invalid initial reset observation or zero-action outcome",
    )
    return reset


def verify_noise(
    actual, *, mode, step, known, recovered, basis, proposal=None, fresh_rng=None
):
    if mode == "policy_fresh":
        require(fresh_rng is not None, "Missing fresh-noise seed stream")
        expected = (
            known
            if step == 0
            else fresh_rng.standard_normal(FLOW_SHAPE).astype(np.float32)
        )
    elif mode == "known_noise":
        expected = known
    else:
        require(
            (proposal is not None) == (mode == "random_noise"),
            "Noise proposal leaked into a held-noise arm",
        )
        expected = perturb_noise(recovered, basis, proposal)
    return _same_float32(
        actual,
        expected,
        "Generation noise construction",
        atol=NOISE_ATOL if proposal is not None else 0,
    )


def _physical(summary):
    require(
        set(summary["controls"]) == {"known_noise", "policy_fresh"}
        and set(summary["arms"]) == set(ARMS),
        "Missing or extra reported arms/controls",
    )
    baseline = summary["baseline"]
    physical = [baseline, *summary["controls"].values()]
    for arm in ARMS:
        attempts = summary["arms"][arm]["attempts"]
        require(
            attempts and attempts[0] == baseline,
            "Arm does not reuse the exact shared baseline record",
        )
        physical.extend(attempts[1:])
    result = {row["attempt_id"]: row for row in physical}
    require(len(result) == len(physical), "Duplicate physical attempt")
    return result


class CaseAudit:
    def __init__(self, directory, context):
        self.directory, self.context = Path(directory), context
        self.source_hashes = {
            name: file_sha256(self.directory / name)
            for name in ("summary.json", "events.jsonl")
        }
        self.summary = _json((self.directory / "summary.json").read_text())
        self.rows = [
            _json(line)
            for line in (self.directory / "events.jsonl").read_text().splitlines()
        ]
        require(
            [row["sequence"] for row in self.rows] == list(range(len(self.rows))),
            "Missing/duplicate event sequence",
        )
        self.store = ArrayStore(self.directory, self.rows)
        self.groups = {}
        for row in self.rows:
            self.groups.setdefault(row["kind"], []).append(row)
        allowed = {
            "case",
            "inversion_initialization",
            "interpolation_gate_reference",
            "interpolation_gate_check",
            "interpolation_gate",
            "phase_generation",
            "phase_attempt",
            "interpolation_request",
            "interpolation_decision",
            "phase_development_validation",
        }
        require(
            set(self.groups) <= allowed, "Unknown event kind in frozen phase recording"
        )
        self.case = self.one("case")
        self.entry, self.protocol = self.case["entry"], self.case["protocol"]
        self.episode_id = self.entry["episode_id"]
        metadata = context.metadata
        require(
            self.protocol == metadata["protocol.json"],
            "Case protocol differs from worker protocol",
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
            self.summary["status"] == "complete"
            and self.summary["episode_id"] == self.episode_id
            and self.summary["protocol_sha256"] == digest(self.protocol)
            and self.summary["reset_entry_sha256"] == digest(self.entry)
            and self.summary["checkpoint"] == context.checkpoint,
            "Incomplete or mismatched case summary",
        )
        self.oracle = oracle_for(self.entry["suite"], self.entry["task_id"]).metadata()
        require(
            self.summary["oracle"] == self.oracle
            and self.summary["source_catalog"] == donor_catalog(),
            "Case oracle/catalog differs from frozen source mapping",
        )
        self.source_prompts = {
            row["source_id"]: row["prompt"] for row in donor_catalog()
        }
        self.attempts = _physical(self.summary)
        self.seed_words = [self.protocol["seed"], int(digest(self.episode_id)[:8], 16)]
        self.basis = self.store.read(self.case["basis"])
        seeded_basis, basis_id = noise_basis(
            FLOW_SHAPE, np.random.SeedSequence(self.seed_words + [1])
        )
        require(
            self.basis.shape == (8, *FLOW_SHAPE)
            and self.basis.dtype == np.float32
            and digest(self.basis) == self.case["basis_id"] == basis_id
            and np.array_equal(self.basis, seeded_basis),
            "Seeded noise basis mismatch",
        )
        self.counts = Counter()
        self.maxima = {
            "noise_reconstruction_max_abs": 0.0,
            "controller_decode_max_abs": 0.0,
        }

    def one(self, kind):
        rows = self.groups.get(kind, [])
        require(len(rows) == 1, f"Expected exactly one {kind} event")
        return rows[0]

    def initialization(self):
        row = self.store.resolve(self.one("inversion_initialization"))
        summary = self.summary["initialization"]
        require(
            all(row[key] == value for key, value in summary.items()),
            "Initialization event differs from summary",
        )
        observation = row["observation"]
        validate_observation(observation)
        self.initial_observation_id = digest(observation)
        self.initial_raw_id = digest(
            {**observation, "prompt": self.entry["instruction"]}
        )
        self.known, self.recovered = row["known_noise"], row["recovered_noise"]
        expected_known = (
            np.random.default_rng(np.random.SeedSequence(self.seed_words))
            .standard_normal(FLOW_SHAPE)
            .astype(np.float32)
        )
        _same_float32(self.known, expected_known, "Initial seeded noise")
        require(
            self.recovered.shape == FLOW_SHAPE and self.recovered.dtype == np.float32,
            "Recovered noise shape/dtype mismatch",
        )
        for name in ("reference", "roundtrip"):
            require(
                row[name].shape == FLOW_SHAPE and row[name].dtype == np.float32,
                f"Initial {name} must be full float32 [1,10,32]",
            )
        for name in ("native_actions", "adapter_actions"):
            require(
                row[name].shape == (10, 7) and row[name].dtype == np.float64,
                f"Initial {name} must be decoded float64 [10,7]",
            )
        require(
            row["condition_id"] == self.initial_raw_id
            and row["known_noise_sha256"] == digest(self.known)
            and row["recovered_noise_sha256"] == digest(self.recovered),
            "Initial condition/latent binding mismatch",
        )
        errors = {
            "noise": error_metrics(self.known, self.recovered),
            "actions": error_metrics(row["reference"], row["roundtrip"]),
            "native_parity": error_metrics(
                row["native_actions"], row["adapter_actions"]
            ),
        }
        for label, metrics in errors.items():
            _metrics(row["errors"][label], metrics, "Initial " + label)
        gate = self.protocol["numerical_gate"]
        require(
            row["passed"] is True
            and errors["noise"]["max_abs"] <= gate["noise_max_abs"] == 0.1
            and errors["actions"]["max_abs"] <= gate["action_max_abs"] == 0.02
            and errors["native_parity"]["max_abs"] <= 1e-5,
            "Initial fixed numerical gate failed",
        )
        require(
            row["velocity_evaluations"] == 1230
            and row["development_embedding_probe"] is None
            and row["errors"]["zero_embedding_hook_parity"]
            == {"max_abs": 0.0, "rmse": 0.0},
            "Unexpected initial probe/VF/parity record",
        )
        zero = row["zero_embedding_hook"]
        require(
            zero["condition_id"] == zero["raw_condition_id"] == self.initial_raw_id
            and zero["observation_id"] == self.initial_observation_id
            and zero["enabled"] is False
            and zero["has_effect"] is False
            and zero["delta_frobenius"] == 0
            and zero["prefix_before_sha256"] == zero["prefix_after_sha256"],
            "Disabled original embedding hook changed its condition",
        )
        for key in (
            "text_mask_fixed",
            "padding_unchanged",
            "vision_prefix_unchanged",
            "attention_unchanged",
            "positions_unchanged",
        ):
            require(zero[key] is True, "Initial disabled embedding boundary violation")
        self.action_spec = row["action_spec"]
        require(
            self.action_spec["horizon"] == 10
            and self.action_spec["model_action_dim"] == 32,
            "Action spec changed native dimensions",
        )
        self.initial_adapter_actions = row["adapter_actions"]
        self.counts["initialization_velocity_evaluations"] = row["velocity_evaluations"]
        return errors

    def interpolation_gate(self):
        gate = self.one("interpolation_gate")
        declared = self.summary["interpolation_gate"]
        require(
            {
                key: value
                for key, value in gate.items()
                if key not in ("kind", "sequence", "timestamp")
            }
            == declared,
            "Interpolation gate event differs from summary",
        )
        require(
            gate["passed"] is True
            and gate["complete"] is True
            and gate["status"] == "passed"
            and gate["velocity_evaluations_complete"] is True
            and gate["velocity_evaluations"] == 50
            and gate["reference_velocity_evaluations"] == 10
            and set(gate["checks"]) == set(GATE_CHECKS),
            "Incomplete or failed weighted interpolation gate",
        )
        require(
            gate["solver"] == self.protocol["execution_solver"]
            and gate["known_noise_sha256"] == digest(self.known)
            and gate["observation_id"] == self.initial_observation_id
            and gate["native_condition_id"] == self.initial_raw_id,
            "Interpolation gate reference binding changed",
        )
        reference = self.store.resolve(self.one("interpolation_gate_reference"))
        observation = reference["observation"]
        validate_observation(observation)
        require(
            digest(observation)
            == reference["observation_id"]
            == self.initial_observation_id
            and reference["condition_id"] == self.initial_raw_id
            and reference["solver"] == gate["solver"]
            and reference["velocity_evaluations"] == 10,
            "Wrong weighted gate reference",
        )
        _same_float32(reference["latent"], self.known, "Weighted gate initial latent")
        _metrics(
            error_metrics(
                reference["decoded_actions"],
                self.context.decode(reference["generated_actions"]),
            ),
            {"max_abs": 0.0, "rmse": 0.0},
            "Reference endpoint decoding",
        )
        require(
            np.array_equal(reference["decoded_actions"], self.initial_adapter_actions),
            "Weighted reference differs from initial native Euler endpoint",
        )
        probes = self.groups.get("interpolation_gate_check", [])
        require(
            len(probes) == 4 and {row["label"] for row in probes} == set(GATE_CHECKS),
            "Missing/duplicate weighted probe",
        )
        oracle_ids = (self.oracle["source_a_id"], self.oracle["source_b_id"])
        for recorded in probes:
            row = self.store.resolve(recorded)
            label = row["label"]
            expected_row = {
                key: value
                for key, value in recorded.items()
                if key
                not in (
                    "kind",
                    "sequence",
                    "timestamp",
                    "label",
                    "observation_id",
                    "latent_sha256",
                    "generated_actions",
                    "decoded_actions",
                )
            }
            require(
                expected_row == gate["checks"][label],
                "Probe event differs from declared gate row",
            )
            operator, alpha, nonzero = GATE_CHECKS[label]
            require(
                row["operator"] == operator
                and row["alpha"] == alpha
                and row["expected_effect"] == ("nonzero" if nonzero else "identity")
                and row["source_ids"] == (list(oracle_ids) if nonzero else None),
                "Weighted probe specification changed",
            )
            require(
                row["observation_id"] == self.initial_observation_id
                and row["latent_sha256"] == digest(self.known)
                and row["velocity_evaluations"] == 10,
                "Weighted probe condition/latent/solver mismatch",
            )
            sources = (
                tuple(self.source_prompts[source_id] for source_id in oracle_ids)
                if nonzero
                else (self.entry["instruction"],) * 2
            )
            condition_id, _ = self.context.conditioning(
                row["provenance"],
                observation,
                self.entry["instruction"],
                sources,
                operator,
                alpha,
                oracle_ids if nonzero and operator == "tli" else None,
            )
            require(
                condition_id == row["condition_id"]
                and row["provenance"]["observation_id"] == self.initial_observation_id,
                "Weighted probe condition ID mismatch",
            )
            metrics = {
                "errors": error_metrics(
                    reference["generated_actions"], row["generated_actions"]
                ),
                "controlled_channel_errors": error_metrics(
                    reference["generated_actions"][:, :, :7],
                    row["generated_actions"][:, :, :7],
                ),
                "decoded_action_errors": error_metrics(
                    reference["decoded_actions"], row["decoded_actions"]
                ),
            }
            for name, measured in metrics.items():
                _metrics(row[name], measured, label + " " + name)
            require(
                np.array_equal(
                    row["decoded_actions"],
                    self.context.decode(row["generated_actions"]),
                ),
                "Weighted endpoint action decoding mismatch",
            )
            passed = (
                (
                    row["provenance"]["has_effect"]
                    and metrics["controlled_channel_errors"]["max_abs"] > 0
                    and metrics["decoded_action_errors"]["max_abs"] > 0
                )
                if nonzero
                else (
                    not row["provenance"]["has_effect"]
                    and condition_id == self.initial_raw_id
                    and metrics["errors"]["max_abs"] == 0
                    and metrics["decoded_action_errors"]["max_abs"] == 0
                )
            )
            require(
                passed and row["passed"] is True and row["status"] == "passed",
                "Recomputed weighted probe gate failed",
            )
        self.counts["interpolation_gate_velocity_evaluations"] = 50

    def attempts_and_generations(self):
        recorded_attempts = self.groups.get("phase_attempt", [])
        require(
            len(recorded_attempts) == len(self.attempts),
            "Physical attempt event count mismatch",
        )
        event_attempts = {
            row["attempt"]["attempt_id"]: row for row in recorded_attempts
        }
        require(
            set(event_attempts) == set(self.attempts),
            "Missing/duplicate physical attempt event",
        )
        generation_groups = {key: [] for key in self.attempts}
        for row in self.groups.get("phase_generation", []):
            require(
                row["attempt_id"] in generation_groups,
                "Generation belongs to unrecorded physical attempt",
            )
            generation_groups[row["attempt_id"]].append(row)
        random_rng = np.random.default_rng(
            np.random.SeedSequence(self.seed_words + [3])
        )
        baseline_reset = None
        for attempt_id, attempt in self.attempts.items():
            event = event_attempts[attempt_id]
            require(event["attempt"] == attempt, "Attempt event differs from summary")
            require(
                attempt_id == f"{attempt['mode']}_{attempt['iteration']}"
                and type(attempt["actions_executed"]) is int
                and 0 < attempt["actions_executed"] <= self.protocol["action_budget"]
                and attempt["execute_steps"] == 5
                and attempt["action_budget"] == self.protocol["action_budget"],
                "Invalid attempt identity/budget",
            )
            baseline_reset = verify_reset(
                attempt, self.entry, self.initial_observation_id, baseline_reset
            )
            snapshots = event["snapshots"]
            snapshot_steps = [row["step"] for row in snapshots]
            require(
                snapshots
                and snapshot_steps
                == np.rint(
                    np.linspace(
                        0,
                        attempt["actions_executed"],
                        min(4, attempt["actions_executed"] + 1),
                    )
                )
                .astype(int)
                .tolist(),
                "Missing initial/final physical rollout snapshots",
            )
            snapshot_hashes = {}
            for snapshot in snapshots:
                value = self.store.resolve(snapshot["observation"])
                validate_observation(value)
                snapshot_hashes[snapshot["step"]] = digest(value)
            first = self.store.resolve(snapshots[0]["observation"])
            validate_observation(first)
            require(
                digest(first) == self.initial_observation_id,
                "Rollout initial pixels/state differ from paired reset",
            )
            rows = generation_groups[attempt_id]
            require(
                [row["observation_step"] for row in rows]
                == list(range(0, attempt["actions_executed"], 5))
                and len(rows) == attempt["policy_replans"],
                "Generation steps do not cover executed actions exactly",
            )
            proposal = attempt["noise_proposal"]
            if attempt["mode"] == "random_noise":
                expected = random_noise_proposal(self.basis, random_rng)
                require(
                    {
                        key: value
                        for key, value in proposal.items()
                        if key != "coefficients"
                    }
                    == {
                        key: value
                        for key, value in expected.items()
                        if key != "coefficients"
                    },
                    "Seeded random proposal basis/scale changed",
                )
                actual, wanted = (
                    np.asarray(proposal["coefficients"], np.float64),
                    np.asarray(expected["coefficients"], np.float64),
                )
                require(
                    actual.shape == wanted.shape
                    and np.all(
                        np.abs(actual - wanted)
                        <= RANDOM_COEFFICIENT_ULPS * np.spacing(np.abs(wanted))
                    ),
                    "Seeded random coefficient differs beyond four float64 ULPs",
                )
            else:
                require(
                    proposal is None,
                    "Non-random arm received an undeclared noise perturbation",
                )
            fresh_rng = np.random.default_rng(
                np.random.SeedSequence(self.seed_words + [2])
            )
            held_latent_hash, counters = None, Counter()
            for recorded in rows:
                row = self.store.resolve(recorded)
                require(
                    row["mode"] == attempt["mode"]
                    and row["iteration"] == attempt["iteration"]
                    and row["velocity_evaluations"] == 10,
                    "Generation mode/iteration/VF mismatch",
                )
                noise_error = verify_noise(
                    row["latent"],
                    mode=row["mode"],
                    step=row["observation_step"],
                    known=self.known,
                    recovered=self.recovered,
                    basis=self.basis,
                    proposal=proposal,
                    fresh_rng=fresh_rng,
                )
                self.maxima["noise_reconstruction_max_abs"] = max(
                    self.maxima["noise_reconstruction_max_abs"], noise_error
                )
                if row["mode"] != "policy_fresh":
                    latent_hash = digest(row["latent"])
                    require(
                        held_latent_hash is None or latent_hash == held_latent_hash,
                        "Held latent changed across fresh observations",
                    )
                    held_latent_hash = latent_hash
                observation = row["observation"]
                validate_observation(observation)
                observation_id = digest(observation)
                if row["observation_step"] in snapshot_hashes:
                    require(
                        observation_id == snapshot_hashes[row["observation_step"]],
                        "Generation raw frame differs from same-step rollout snapshot",
                    )
                if row["observation_step"] == 0:
                    require(
                        observation_id == self.initial_observation_id,
                        "First generation does not use paired reset observation",
                    )
                marks, active = row["vision"], row["active_interpolation"]
                require(
                    not marks or row["mode"] == "astra_tli_vision",
                    "Vision marks leaked into a nonvision arm",
                )
                modified = render_linux(observation, marks)
                require(
                    digest(modified) == row["modified_observation_sha256"],
                    "Strict Linux vision redraw hash mismatch",
                )
                require(
                    row["vision_has_effect"] == (digest(modified) != observation_id),
                    "Recorded vision effect is inconsistent",
                )
                if row["mode"].startswith("oracle_"):
                    expected = {
                        "source_a_id": self.oracle["source_a_id"],
                        "source_b_id": self.oracle["source_b_id"],
                        "alpha": paper_alpha(
                            row["observation_step"] // 5, self.oracle["lambda_calls"]
                        ),
                    }
                    require(
                        active == expected,
                        "Oracle source pair or phase schedule changed",
                    )
                if active is None:
                    require(
                        row["conditioning"] is None
                        and row["text_has_effect"] is False
                        and not marks
                        and row["condition_id"]
                        == digest({**observation, "prompt": self.entry["instruction"]}),
                        "Native condition fallback changed its inputs",
                    )
                else:
                    require(
                        row["mode"].startswith(("astra_", "oracle_"))
                        and set(active) == {"source_a_id", "source_b_id", "alpha"}
                        and type(active["alpha"]) in (int, float)
                        and 0 <= active["alpha"] <= 1,
                        "Unexpected active interpolation",
                    )
                    source_ids = (active["source_a_id"], active["source_b_id"])
                    require(
                        all(source in self.source_prompts for source in source_ids),
                        "Unknown interpolation source",
                    )
                    sources = tuple(
                        self.source_prompts[source] for source in source_ids
                    )
                    operator = (
                        row["mode"]
                        .removeprefix("astra_")
                        .removeprefix("oracle_")
                        .removesuffix("_vision")
                    )
                    value = row["conditioning"]
                    require(
                        value["observation_id"] == observation_id,
                        "Condition is not bound to its fresh raw observation",
                    )
                    condition_id, _ = self.context.conditioning(
                        value,
                        modified,
                        self.entry["instruction"],
                        sources,
                        operator,
                        active["alpha"],
                        source_ids,
                    )
                    require(
                        condition_id == row["condition_id"]
                        and row["text_has_effect"] == value["has_effect"],
                        "Generation condition/effect binding mismatch",
                    )
                    self.counts["interpolated_generations"] += 1
                    self.counts["tli_layer_records"] += len(value["tli_layers"])
                decoded = self.context.decode(row["generated_actions"]).astype(
                    np.float32
                )
                clipped = np.clip(
                    decoded, self.action_spec["lower"], self.action_spec["upper"]
                ).astype(np.float32)
                error = _same_float32(
                    row["controller_actions"],
                    clipped,
                    "Controller action decoding",
                    atol=NOISE_ATOL,
                )
                self.maxima["controller_decode_max_abs"] = max(
                    self.maxima["controller_decode_max_abs"], error
                )
                expected_clipping = {
                    "count": int(np.count_nonzero(decoded != clipped)),
                    "max_abs": float(np.abs(decoded - clipped).max()),
                }
                require(
                    row["clipping"] == expected_clipping,
                    "Controller clipping record mismatch",
                )
                count = min(5, attempt["actions_executed"] - row["observation_step"])
                counters["velocity_evaluations"] += 10
                counters["clipped_values"] += expected_clipping["count"]
                counters["vision_active_policy_calls"] += bool(marks)
                counters["vision_changed_policy_calls"] += row["vision_has_effect"]
                counters["text_nonzero_policy_calls"] += row["text_has_effect"]
                counters["accepted_decision_policy_calls"] += (
                    row["applied_accepted_decision_id"] is not None
                )
                counters["native_condition_fallback_policy_calls"] += row[
                    "native_condition_fallback"
                ]
                for flag, name in (
                    ("text_has_effect", "actions_with_nonzero_text"),
                    ("vision_has_effect", "actions_with_changed_vision"),
                    (
                        "held_text_after_failed_call",
                        "actions_with_held_text_after_failed_call",
                    ),
                    ("native_condition_fallback", "native_condition_fallback_actions"),
                ):
                    counters[name] += count * row[flag]
                counters["actions_with_accepted_decision"] += count * (
                    row["applied_accepted_decision_id"] is not None
                )
                self.counts["generations"] += 1
            for key, count in counters.items():
                require(
                    attempt[key] == count,
                    f"Attempt {key} disagrees with actual generation records",
                )
            self.counts["physical_attempts"] += 1
            self.counts["physical_actions"] += attempt["actions_executed"]
            self.counts["rollout_velocity_evaluations"] += counters[
                "velocity_evaluations"
            ]

    def run(self):
        initial_errors = self.initialization()
        self.interpolation_gate()
        self.attempts_and_generations()
        array_receipt = self.store.finish()
        self.counts.update(
            {key: value for key, value in array_receipt.items() if type(value) is int}
        )
        total = sum(
            self.counts[name]
            for name in (
                "initialization_velocity_evaluations",
                "interpolation_gate_velocity_evaluations",
                "rollout_velocity_evaluations",
            )
        )
        costs = self.summary["physical_cost"]
        for name, value in (
            ("rollouts", self.counts["physical_attempts"]),
            ("simulated_actions", self.counts["physical_actions"]),
            ("velocity_evaluations", total),
            (
                "rollout_velocity_evaluations",
                self.counts["rollout_velocity_evaluations"],
            ),
            ("initialization_velocity_evaluations", 1230),
            ("interpolation_gate_velocity_evaluations", 50),
        ):
            require(
                costs[name] == value,
                f"Physical {name} differs from unique recorded work",
            )
        self.counts["velocity_evaluations"] = total
        require(
            self.source_hashes
            == {
                name: file_sha256(self.directory / name) for name in self.source_hashes
            },
            "Case summary/events changed during audit",
        )
        return {
            "status": "passed",
            "episode_id": self.episode_id,
            "protocol_sha256": digest(self.protocol),
            "reset_entry_sha256": digest(self.entry),
            "summary_sha256": self.source_hashes["summary.json"],
            "events_sha256": self.source_hashes["events.jsonl"],
            "counts": dict(self.counts),
            "maximum_errors": self.maxima,
            "array_inventory_sha256": array_receipt["array_inventory_sha256"],
            "initialization_errors_recomputed": initial_errors,
            "gates": {
                "all_arrays_verified": True,
                "paired_resets_verified": True,
                "noise_construction_verified": True,
                "vision_hashes_verified": True,
                "weighted_endpoint_gates_verified": True,
                "recorded_text_boundaries_and_banks_verified": True,
            },
            "physical_attempt_ids": list(self.attempts),
        }


def audit_case(case_dir, *, metadata, assets):
    """Audit one extracted case against already verified worker metadata."""
    context = metadata if isinstance(metadata, Context) else Context(metadata, assets)
    return CaseAudit(case_dir, context).run()


def _audit_results(root, assets):
    root = Path(root).resolve()
    if (root / "results").is_dir() and not (root / "runtime.json").is_file():
        root = root / "results"
    names = (
        "runtime.json",
        "protocol.json",
        "checkpoint.json",
        "bank_inventory.json",
        "frozen_plan.json",
        "reset_manifest.json",
        "progress.json",
    )
    require(
        all(
            (root / name).is_file() and not (root / name).is_symlink() for name in names
        ),
        "Full worker metadata is missing",
    )
    hashes = {name: file_sha256(root / name) for name in names}
    metadata = {name: _json((root / name).read_text()) for name in names}
    runtime, plan = metadata["runtime.json"], metadata["frozen_plan.json"]
    require(
        runtime["phase"] in ("development", "evaluation")
        and "L40S" in runtime["gpu"]
        and runtime["tf32"] is False
        and _sha(runtime["payload_sha256"]),
        "Unexpected runtime/phase/GPU",
    )
    require(
        metadata["progress.json"]["status"] == "complete"
        and not (root / "failure.json").exists(),
        "Worker did not complete cleanly",
    )
    protocol = load_protocol()
    if runtime["phase"] == "development":
        protocol["seed"] = protocol["development_seed"]
    require(
        metadata["protocol.json"] == protocol
        and plan["runtime"] == runtime
        and plan["protocol_sha256"] == digest(protocol)
        and plan["bank_inventory_sha256"] == hashes["bank_inventory.json"],
        "Worker frozen plan/input hash mismatch",
    )
    benchmark = BenchmarkConfig.preset(runtime["assignment"]["suite"])
    manifest = load_reset_manifest(root / "reset_manifest.json", benchmark)
    require(
        manifest == metadata["reset_manifest.json"]
        and manifest["sha256"] == plan["manifest_sha256"]
        and manifest["seed"] == protocol["seed"],
        "Worker reset manifest/seed mismatch",
    )
    assigned = plan["assigned_episodes"]
    require(
        assigned and len(set(assigned)) == len(assigned),
        "Missing/duplicate assigned episodes",
    )
    context = Context(metadata, assets)
    directories = sorted(path for path in root.glob("case_*") if path.is_dir())
    require(
        directories and len(directories) == len(assigned),
        "Missing/extra full case directories",
    )
    cases = []
    covered_arrays = set()
    for directory in directories:
        require(not directory.is_symlink(), "Case directory must not be a symlink")
        case = audit_case(directory, metadata=context, assets=assets)
        cases.append(case)
        covered_arrays.update(
            path.resolve() for path in (directory / "arrays").glob("*.npy")
        )
    require(
        {case["episode_id"] for case in cases} == set(assigned)
        and len({case["episode_id"] for case in cases}) == len(cases),
        "Full array audit has wrong assigned episode coverage",
    )
    require(
        covered_arrays == {path.resolve() for path in root.rglob("*.npy")},
        "Unaudited NPY tensor outside declared case arrays",
    )
    require(
        hashes == {name: file_sha256(root / name) for name in names},
        "Worker metadata changed during audit",
    )
    counts = Counter()
    for case in cases:
        counts.update(case["counts"])
    source_root = Path(__file__).parent
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "complete": True,
        "workflow": runtime["workflow"],
        "worker": runtime["worker"],
        "phase": runtime["phase"],
        "payload_sha256": runtime["payload_sha256"],
        "reset_manifest_sha256": manifest["sha256"],
        "protocol_sha256": digest(protocol),
        "bank_inventory_sha256": hashes["bank_inventory.json"],
        "source_metadata_sha256": hashes,
        "asset_sha256": context.asset_hashes,
        "counts": dict(counts),
        "cases": cases,
        "renderer": RENDERER,
        "random_coefficient_tolerance_ulps": RANDOM_COEFFICIENT_ULPS,
        "noise_delta_and_decoding_cpu_tolerance": NOISE_ATOL,
        "held_and_known_noise_tolerance": 0,
        "postprocessor_source_sha256": {
            name: file_sha256(source_root / name)
            for name in (
                "interpolation_audit.py",
                "configs/phase_interpolation_v1.json",
                "interpolation_conditioning.py",
                "interpolation_catalog.py",
                "interpolation_search.py",
                "intervention_rollout.py",
                "interventions.py",
                "config.py",
                "libero_runner.py",
                "lerobot_policy.py",
                "openpi_inputs.py",
                "checkpoints/lerobot_pi05_libero_base.json",
                "records.py",
                "flow.py",
            )
        },
        "limitations": LIMITATIONS,
    }


def _extract_archive(path, destination):
    """Extract only regular result files; force gzip CRC validation to EOF."""
    seen, total = set(), 0
    with gzip.open(path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                require(
                    relative.parts[:1] == ("results",)
                    and not relative.is_absolute()
                    and ".." not in relative.parts
                    and "\\" not in member.name
                    and member.name not in seen,
                    "Unsafe or duplicate archive member",
                )
                seen.add(member.name)
                target = Path(destination).joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                require(member.isfile(), "Archive links/devices are not allowed")
                total += member.size
                require(
                    total <= 8 * 1024**3,
                    "Worker extraction exceeds the declared 8 GiB bound",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                require(source is not None, "Missing regular archive member data")
                with target.open("xb") as out:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        out.write(chunk)
                require(
                    target.stat().st_size == member.size, "Truncated archive member"
                )
        while stream.read(1024 * 1024):
            pass
    return total


def audit_worker(results_root, *, assets, output=None):
    """Audit an extracted results directory or a local complete .tar.gz archive.

    Raises AuditError on an integrity failure. When output is supplied, failure
    is also written as an explicit failed receipt; no partial run is marked passed.
    """
    source = Path(results_root)
    try:
        if source.is_file():
            archive_sha = file_sha256(source)
            with tempfile.TemporaryDirectory(
                prefix="astra-phase-array-audit-"
            ) as temporary:
                extracted_bytes = _extract_archive(source, temporary)
                result = _audit_results(Path(temporary) / "results", assets)
            require(file_sha256(source) == archive_sha, "Archive changed during audit")
            result["archive"] = {
                "sha256": archive_sha,
                "bytes": source.stat().st_size,
                "extracted_bytes": extracted_bytes,
                "gzip_crc_verified": True,
            }
        else:
            result = _audit_results(source, assets)
            result["archive"] = None
        if output is not None:
            _write(output, result)
        return result
    except Exception as exc:
        if output is not None:
            _write(
                output,
                {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "complete": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "limitations": LIMITATIONS,
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Full extracted worker results or local artifacts.tar.gz",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Pinned pi05_libero tokenizer and norm_stats directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_worker(args.results, assets=args.assets, output=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
