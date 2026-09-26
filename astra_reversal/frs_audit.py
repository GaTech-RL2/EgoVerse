"""Strict offline audit of complete FRS task artifacts; no learned-model solves.

CPU code replays action transforms, seeded noise, calibration projections and
request construction. It verifies stored training labels/optimizer counters and
checkpoint identities, but does not rerun CNN inference, optimization, pi05 or
physics. Recorded outcomes and hidden learned computations remain observations.
"""

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

from .action_adapter import ActionAdapter, ActionSpec
from .flow import error_metrics, noise_statistics
from .frs_agent import (
    SYSTEM_PROMPTS,
    action_edit_request,
    build_payload,
    critique_request,
    direction_request,
    judge_request,
    parse_proposal,
    prompt_manifest,
    summarize_calls,
)
from .frs_experiment import compact_arrays, first_success, visible_rollout
from .frs_guide import project_policy_pixels
from .frs_operators import (
    directional_reference,
    edit_native,
    repeated_gaussian_noise,
    resample_padding_noise,
)
from .image_perturbation_audit import ArtifactStore, _json, require
from .records import digest, file_sha256

SCHEMA_VERSION = "frs-artifact-audit-1.0"
CAMERAS = ("observation/image", "observation/wrist_image")
FLOW_SHAPE = (1, 10, 32)
SOLVER = {"solver": "euler", "steps": 10, "time_power": 1.0}
LIMITATIONS = [
    "No pi05/CNN inference, optimization, provider, simulator or GPU work is rerun.",
    "Full recorded arrays, transforms, random streams, provider payloads, labels and checkpoints are verified. Hidden model computation and physics are not independently reproduced.",
    "Finite Euler FRS deliberately changes inverse padding and may repeat physical noise; reported action deviation is not known-noise recovery or an exact-roundtrip gate.",
    "Controller float32 arithmetic permits at most 1e-7 replay drift; stored hashes and seeded noise remain exact.",
    "Camera projection is replayed from recorded calibration, not a reconstructed MuJoCo scene.",
    "Training receipts and Adam counters are checked against accepted observed labels; gradients and recorded prediction values are not recomputed.",
]


def _same(a, b, message):
    require(digest(a) == digest(b), message)


def _array(actual, expected, message, *, atol=0.0):
    require(
        isinstance(actual, np.ndarray)
        and actual.shape == expected.shape
        and actual.dtype == expected.dtype,
        f"{message}: shape/dtype",
    )
    maximum = float(
        np.max(
            np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0
        )
    )
    require(np.isfinite(actual).all() and maximum <= atol, message)
    return maximum


def _flow(value, label):
    require(
        isinstance(value, np.ndarray)
        and value.shape == FLOW_SHAPE
        and value.dtype == np.float32
        and np.isfinite(value).all(),
        f"{label}: expected finite float32[1,10,32]",
    )
    return value


def _metrics(recorded, a, b, label):
    expected = error_metrics(a, b)
    require(
        set(recorded) == set(expected)
        and all(
            math.isclose(recorded[k], v, rel_tol=1e-12, abs_tol=1e-15)
            for k, v in expected.items()
        ),
        f"{label}: incorrect recorded metrics",
    )
    return expected


def _rng(entry, step, stream):
    return np.random.default_rng(
        np.random.SeedSequence(
            [entry["seed"], int(digest(entry["episode_id"])[:8], 16), step, stream]
        )
    )


def verify_reset_entry(entry, suite):
    require(
        entry["suite"] == suite
        and type(entry["seed"]) is int
        and 0 <= entry["seed"] < 2**32
        and type(entry["task_id"]) is int
        and 0 <= entry["task_id"] < 10
        and type(entry["initial_state_id"]) is int
        and entry["initial_state_id"] >= 0,
        "Invalid reset identity",
    )
    require(
        entry["episode_id"]
        == f"{suite}:seed{entry['seed']}:task{entry['task_id']}:state{entry['initial_state_id']}",
        "Reset episode ID differs from its actual seed/task/index",
    )
    state = np.asarray(entry["reset_state"], np.float64)
    model = {
        name: np.asarray(value, np.float64)
        for name, value in entry["reset_model"].items()
    }
    require(
        state.ndim == 1
        and state.size
        and np.isfinite(state).all()
        and digest(state) == entry["reset_state_sha256"],
        "Actual reset state bytes/hash differ",
    )
    require(
        set(model) == {"body_pos", "body_quat"}
        and model["body_pos"].ndim == model["body_quat"].ndim == 2
        and model["body_pos"].shape[1] == 3
        and model["body_quat"].shape == (model["body_pos"].shape[0], 4)
        and all(np.isfinite(value).all() for value in model.values())
        and digest(model) == entry["reset_model_sha256"],
        "Actual reset model bytes/hash differ",
    )
    return state, model


def normalization_adapter(spec, q01, q99):
    """Pinned OpenPI quantile formula, without instantiating either policy."""
    q01, q99 = np.asarray(q01), np.asarray(q99)
    require(
        q01.shape == q99.shape == (7,)
        and np.isfinite(q01).all()
        and np.isfinite(q99).all()
        and np.all(q99 > q01),
        "Invalid action normalization statistics",
    )

    def encode(data):
        values = (data["actions"] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return {"actions": np.pad(values.astype(np.float32), ((0, 0), (0, 25)))}

    def decode(data):
        values = data["actions"][..., :7]
        return {"actions": (values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01}

    return ActionAdapter(spec, encode, decode)


def replay_guide(observation, guide, receipt):
    """Require exact raster reconstruction and independently derive axis signs."""
    require(
        receipt["camera"] == "agentview"
        and receipt["image_convention"] == "opengl_then_flip_both_axes"
        and receipt["policy_receives_guide"] is False,
        "Guide convention changed",
    )
    eef = observation["observation/state"][:3].astype(np.float64)
    table = eef.copy()
    table[2] = receipt["table_height"]
    points = np.stack((eef, table))
    calibration = np.asarray(receipt["camera_transform"], np.float64)
    xy = project_policy_pixels(points, calibration)
    probes = np.stack(
        (table, table + [0.01, 0, 0], table + [0, 0.01, 0], table + [0, 0, 0.01])
    )
    pixels = project_policy_pixels(probes, calibration)
    deltas = pixels[1:] - pixels[0]
    signs = [int(np.sign(deltas[0, 1])), int(np.sign(deltas[1, 0])), 1]
    require(
        0 not in signs and deltas[2, 1] < 0, "Invalid camera-to-controller geometry"
    )
    for name, expected in (
        ("world_points", points),
        ("policy_pixels_xy", xy),
        ("axis_probe_delta_pixels", deltas),
    ):
        require(
            np.asarray(receipt[name]).shape == expected.shape
            and np.allclose(receipt[name], expected, rtol=0, atol=1e-10),
            f"Guide {name} disagrees with calibration",
        )
    require(
        receipt["camera_to_controller_signs"] == signs,
        "Guide direction signs are not calibrated",
    )
    raw = observation[CAMERAS[0]]
    base = Image.fromarray(raw).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)
    drawing.line([tuple(p) for p in xy], fill=(35, 100, 255, 155), width=2)
    x, y = xy[0]
    drawing.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(35, 100, 255, 200))
    expected = np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))
    _array(guide, expected, "Guide pixels differ from calibrated drawing")
    require(
        receipt["raw_sha256"] == digest(raw)
        and receipt["guide_sha256"] == digest(guide),
        "Guide/raw pixel hash mismatch",
    )
    return signs


def replay_generation(event, *, entry, adapter):
    """Replay the recorded action/noise boundary, never either velocity solve."""
    method, step, raw = event["method"], event["step"], event["observation"]
    require(
        type(step) is int and 0 <= step < 300 and step % 10 == 0,
        "Generation step does not follow H10 execution",
    )
    require(
        set(raw) == {*CAMERAS, "observation/state"}, "Extra hidden observation fields"
    )
    for camera in CAMERAS:
        require(
            raw[camera].shape == (224, 224, 3) and raw[camera].dtype == np.uint8,
            "Wrong raw RGB shape/dtype",
        )
    require(
        raw["observation/state"].shape == (8,)
        and raw["observation/state"].dtype == np.float32,
        "Wrong proprioception shape/dtype",
    )
    require(
        event["observation_sha256"] == digest(raw)
        and event["condition_id"] == digest({**raw, "prompt": entry["instruction"]}),
        "Condition does not bind raw cameras/state/task",
    )
    is_paper = method in ("astra_direction_direct", "astra_frs")
    is_loop = method.startswith("critique_frs_")
    repeated = is_loop or method in ("native_repeated_noise", "learned_noise")
    base_rng = _rng(entry, step, 0)
    expected_base = (
        repeated_gaussian_noise(base_rng)["noise"]
        if repeated
        else base_rng.standard_normal(FLOW_SHAPE).astype(np.float32)
    )
    _array(event["base_noise"], expected_base, "Base noise stream differs")
    predicted = _flow(event["predicted_noise"], "Predicted noise")
    actor = event["actor_receipt"]
    if actor is None or actor["source"] == "exact_untrained_base_fallback":
        _array(predicted, expected_base, "Untrained/native fallback changed base noise")
        if actor is not None:
            require(
                actor["trained_rounds"] == 0
                and actor["auxiliary_forward_performed"] is False
                and actor["rng_state_before_sha256"]
                == actor["rng_state_after_sha256"]
                == digest(_rng(entry, step, 1).bit_generator.state),
                "Untrained actor consumed RNG or inferred",
            )
    else:
        require(
            actor["source"] == "learned_deterministic_mean_fresh_gaussian_padding"
            and actor["auxiliary_forward_performed"] is True
            and actor["learned_variance"] is False,
            "Unknown actor prediction source",
        )
        mean = predicted[0, 0, :7]
        require(
            np.array_equal(predicted[0, :, :7], np.broadcast_to(mean, (10, 7)))
            and np.max(np.abs(mean)) <= 5,
            "Learned noise mean is not bounded/repeated",
        )
        padding_rng = _rng(entry, step, 1)
        require(
            actor["rng_state_before_sha256"] == digest(padding_rng.bit_generator.state),
            "Learned padding initial RNG mismatch",
        )
        _array(
            predicted[..., 7:],
            padding_rng.standard_normal((1, 10, 25)).astype(np.float32),
            "Learned Gaussian padding differs",
        )
        require(
            actor["rng_state_after_sha256"] == digest(padding_rng.bit_generator.state),
            "Learned padding final RNG mismatch",
        )
        from .frs_noise_policy import preprocess_observation

        images, state = preprocess_observation(raw)
        require(
            actor["observation_sha256"] == digest(raw)
            and actor["images84_sha256"] == digest(images)
            and actor["state_sha256"] == digest(state)
            and actor["mean_first7_sha256"] == digest(mean),
            "Learned observation/mean binding differs",
        )
    if actor is not None:
        require(
            actor["base_noise_sha256"] == digest(expected_base)
            and actor["generation_noise_sha256"] == digest(predicted),
            "Actor noise hashes differ",
        )
    native = _flow(event["native_model_actions"], "Native endpoint")
    decoded, clipping = adapter.decode(native, np.zeros((1, 32), np.float32))
    _array(
        event["native_actions"],
        decoded,
        "Native controller decoding differs",
        atol=1e-7,
    )
    _same(event["native_clipping"], clipping, "Native clipping receipt differs")
    proposal = event["proposal"]
    steer = proposal is not None and (
        (is_paper and not proposal["fine"]) or (is_loop and proposal["mode"] == "edit")
    )
    reference = None
    if is_paper:
        signs = replay_guide(raw, event["guide"], event["guide_receipt"])
    else:
        require(
            event["guide"] is None and event["guide_receipt"] is None,
            "Non-paper generation added a policy-image guide",
        )
    if steer:
        reference = (
            directional_reference(
                adapter,
                raw,
                coords=[c * s for c, s in zip(proposal["coords"], signs, strict=True)],
                motion_amount=proposal["motion_amount"],
            )
            if is_paper
            else edit_native(
                adapter,
                raw,
                event["native_actions"],
                delta_xyz=proposal["delta_xyz"],
                apply_steps=proposal["apply_steps"],
                gripper=proposal["gripper"],
            )
        )
        _same(
            event["reference"],
            reference,
            "Action reference/receipt differs from replay",
        )
        if method == "astra_direction_direct":
            expected_model = reference["target_model_actions"]
            require(
                event["executed_noise"] is None
                and event["generation_kind"] == "direct_reference"
                and event["reversed_noise"] is None
                and event["noise_transform"] is None
                and event["reconstruction"] is None,
                "Direct direction incorrectly recorded an inverse/noise",
            )
            _array(
                event["generated_model_actions"],
                expected_model,
                "Direct target changed before decode",
            )
        else:
            inverse = _flow(event["reversed_noise"], "Reversed endpoint")
            transformed = resample_padding_noise(
                inverse, _rng(entry, step, 2), repeat_mean=is_loop
            )
            _same(
                event["noise_transform"],
                transformed,
                "Post-inversion mean/padding transform differs",
            )
            _array(
                event["executed_noise"],
                transformed["noise"],
                "Executed FRS noise is not the recorded transform",
            )
            require(
                event["generation_kind"] == "frs_edit",
                "FRS edit lost its generation role",
            )
            _metrics(
                event["reconstruction"],
                reference["target_model_actions"],
                event["generated_model_actions"],
                "Post-transform action deviation",
            )
    else:
        require(
            all(
                event[key] is None
                for key in (
                    "reference",
                    "reversed_noise",
                    "noise_transform",
                    "reconstruction",
                )
            )
            and event["generation_kind"] == "native_defer",
            "Deferred native generation acquired an FRS transform",
        )
        _array(event["executed_noise"], predicted, "Deferred noise changed")
        _array(
            event["generated_model_actions"], native, "Deferred native endpoint changed"
        )
    generated = _flow(event["generated_model_actions"], "Executed generation endpoint")
    actions, clipping = adapter.decode(generated, np.zeros((1, 32), np.float32))
    _array(event["actions"], actions, "Executed controller decoding differs", atol=1e-7)
    _same(event["clipping"], clipping, "Executed clipping receipt differs")
    if event["executed_noise"] is not None:
        _flow(event["executed_noise"], "Executed noise")
        for name, value in noise_statistics(event["executed_noise"]).items():
            require(
                math.isclose(
                    event["noise_statistics"][name], value, rel_tol=1e-12, abs_tol=1e-15
                ),
                "Noise statistics differ",
            )
        if repeated:
            noise = event["executed_noise"]
            require(
                np.array_equal(
                    noise[0, :, :7], np.broadcast_to(noise[0, 0, :7], (10, 7))
                ),
                "Loop noise is not exactly repeated before execution",
            )
    else:
        require(
            event["noise_statistics"] is None,
            "Direct action has fabricated noise statistics",
        )
    return {
        "velocity_evaluations": 10
        + (20 if steer and method != "astra_direction_direct" else 0),
        "interventions": int(steer),
        "deferred": int((is_paper or is_loop) and not steer),
        "rejected": int((is_paper or is_loop) and proposal is None),
        "clipped_actions": clipping["count"],
    }


def verify_provider_binding(request, decision, row, settings):
    """Join actual provider bytes to the exact observable-only reconstructed input."""
    require(
        request["request_fingerprint"]
        == decision["request_fingerprint"]
        == row["request_fingerprint"],
        "Provider/request fingerprint mismatch",
    )
    for key in (
        "role",
        "episode_id",
        "attempt_id",
        "request_index",
        "observation_step",
        "request_id",
        "schema_version",
    ):
        require(row[key] == request[key], f"Provider identity mismatch: {key}")
    model = settings["model"]
    require(
        row["requested_model"] == model
        and row["cache"] == {"no-cache": True}
        and row["response_format"] == {"type": "json_object"},
        "Provider model/cache/format changed",
    )
    sampling = {
        "reasoning_effort": settings["reasoning_effort"],
        "max_completion_tokens": settings["max_completion_tokens"],
    }
    require(row["sampling_settings"] == sampling, "Provider sampling settings changed")
    payload = build_payload(request, model, sampling=sampling)
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    require(
        row["payload_sha256"] == hashlib.sha256(body).hexdigest()
        and row["system_prompt_sha256"]
        == hashlib.sha256(SYSTEM_PROMPTS[request["role"]].encode()).hexdigest(),
        "Provider payload/prompt bytes differ",
    )
    require(
        row["accepted"] == (decision["response"] is not None),
        "Provider/controller acceptance mismatch",
    )
    if row["accepted"]:
        envelope = row["response"]
        require(
            row["provider_call"] is True
            and row["http_status"] == 200
            and envelope["model"] == model
            and len(envelope["choices"]) == 1,
            "Accepted provider response lacks exact successful model identity",
        )
        choice = envelope["choices"][0]
        require(
            choice["finish_reason"] == "stop"
            and choice["message"]["role"] == "assistant"
            and not choice["message"].get("refusal"),
            "Accepted response was incomplete/refused",
        )
        actual = parse_proposal(choice["message"]["content"], request)
        _same(
            actual,
            decision["response"],
            "Recorded accepted decision differs from raw provider response",
        )
        require(
            actual["response_id"] == row["response_id"] and decision["error"] is None,
            "Accepted response identity/error differs",
        )
    else:
        require(
            decision["error"] and row.get("error_kind"),
            "Rejected provider attempt lacks failure provenance",
        )
    return bool(row["accepted"])


def read_auxiliary_checkpoint(store, directory):
    """Verify and load inert CPU tensors with PyTorch's weights-only reader."""
    import torch

    from .frs_noise_policy import _tree

    manifest = store.read_json(f"{directory}/manifest.json")
    require(
        manifest["schema_version"] == "frs-noise-checkpoint-1.0"
        and set(manifest["files"]) == {"state.pt"},
        "Checkpoint schema/files changed",
    )
    path = store.remember(f"{directory}/state.pt")
    require(
        manifest["files"]["state.pt"]
        == {"sha256": file_sha256(path), "bytes": path.stat().st_size},
        "Checkpoint state bytes differ",
    )
    payload = _tree(torch.load(path, map_location="cpu", weights_only=True))
    require(
        set(payload)
        == {"model", "optimizer", "replay", "history", "accepted_rollouts"},
        "Checkpoint payload keys changed",
    )
    metadata = manifest["policy"]
    for value in payload["model"].values():
        require(
            isinstance(value, np.ndarray)
            and value.dtype == np.float32
            and np.isfinite(value).all(),
            "Invalid auxiliary parameter tensor",
        )
    for name, key in (
        ("parameter_sha256", "model"),
        ("optimizer_sha256", "optimizer"),
        ("replay_sha256", "replay"),
        ("history_sha256", "history"),
    ):
        require(
            metadata[name] == digest(payload[key]),
            f"Checkpoint {name} differs from actual saved tensors",
        )
    require(
        metadata["state_id"]
        == digest({k: v for k, v in metadata.items() if k != "state_id"}),
        "Auxiliary state identity is invalid",
    )
    receipt = {**manifest, "manifest_sha256": store.files[f"{directory}/manifest.json"]}
    return receipt, payload


class _TaskAudit:
    def __init__(self, directory, norm_stats_path, expected_protocol):
        self.store = ArtifactStore(directory)
        self.summary = self.store.read_json("summary.json")
        self.events = self.store.lines("events.jsonl")
        self.providers = self.store.lines("provider.jsonl")
        require(
            self.summary["status"] == "complete"
            and self.summary["schema_version"] == "frs-task-1.0",
            "Task is not a complete FRS artifact",
        )
        require(
            [row["sequence"] for row in self.events] == list(range(len(self.events))),
            "Event sequence is incomplete/duplicated",
        )
        allowed = {
            "task",
            "noise_policy_initial",
            "rollout_start",
            "native_parity",
            "guide_probe",
            "astra_decision",
            "generation",
            "rollout_end",
            "adaptation_round",
        }
        require(
            all(row["kind"] in allowed for row in self.events), "Unknown FRS event kind"
        )
        self.store.register(self.events)
        self.groups = {
            kind: [row for row in self.events if row["kind"] == kind]
            for kind in allowed
        }
        require(
            len(self.groups["task"]) == 1 and self.groups["task"][0]["sequence"] == 0,
            "Missing initial task declaration",
        )
        task = self.groups["task"][0]
        self.protocol, self.checkpoint = task["protocol"], task["checkpoint"]
        if expected_protocol is None:
            expected_protocol = _json(
                (
                    Path(__file__).parent / "configs/frs_policy_improvement_v1.json"
                ).read_text()
            )
            if self.summary["development"]:
                expected_protocol["seed"] = expected_protocol["development_seed"]
        if expected_protocol is not None:
            _same(
                self.protocol,
                expected_protocol,
                "Protocol differs from expected immutable input",
            )
        require(
            self.protocol["solver"] == SOLVER
            and self.protocol["execute_steps"] == 10
            and self.protocol["action_budget"] == 300
            and self.protocol["rounds"] == 3,
            "FRS solver/execution/training protocol changed",
        )
        require(
            self.summary["protocol_sha256"] == digest(self.protocol)
            and self.summary["entries_sha256"] == digest(task["entries"]),
            "Task input fingerprints differ",
        )
        _same(
            self.summary["checkpoint"],
            self.checkpoint,
            "Summary checkpoint metadata differs",
        )
        require(
            self.checkpoint["input_profile"] == "openpi_libero"
            and self.checkpoint["horizon"] == 10
            and self.checkpoint["model_action_dim"] == 32,
            "Wrong native checkpoint/input profile",
        )
        weight_inventory = _json(
            (
                Path(__file__).parent / "checkpoints/lerobot_pi05_libero_base.json"
            ).read_text()
        )
        expected_artifacts = {
            Path(row["path"]).name: row["sha256"]
            for row in weight_inventory["files"]
            if row["path"] in ("config.json", "model.safetensors")
            or (
                Path(row["path"]).name.startswith("policy_")
                and "processor" in Path(row["path"]).name
            )
        }
        require(
            self.checkpoint["artifact_sha256"] == expected_artifacts
            and self.checkpoint["provenance"]
            == weight_inventory["repo_id"] + "@" + weight_inventory["revision"]
            and self.checkpoint["frozen"] is True
            and self.checkpoint["format"] == "lerobot_pi05"
            and self.checkpoint["dtype"] == "float32",
            "Native weights/export identity differs from pinned inventory",
        )
        require(
            self.checkpoint["adapter_source_sha256"]
            == file_sha256(Path(__file__).with_name("lerobot_policy.py"))
            and self.checkpoint["input_profile_source_sha256"]
            == file_sha256(Path(__file__).with_name("openpi_inputs.py")),
            "Native adapter/profile source differs",
        )
        inventory = _json(
            (
                Path(__file__).parent / "checkpoints/openpi_libero_input_assets.json"
            ).read_text()
        )
        self.norm_hash = file_sha256(norm_stats_path)
        require(
            self.norm_hash
            == inventory["norm_stats"]["sha256"]
            == self.checkpoint["input_profile_assets"]["norm_stats"]["sha256"],
            "Normalization bytes differ from the pinned checkpoint",
        )
        stats = _json(Path(norm_stats_path).read_text())["norm_stats"]["actions"]
        self.stats = (np.asarray(stats["q01"]), np.asarray(stats["q99"]))
        self.entries = {row["episode_id"]: row for row in task["entries"]}
        require(
            len(self.entries) == 11
            and {e["initial_state_id"] for e in self.entries.values()}
            == set(range(11)),
            "Task needs distinct adaptation0 and evaluation1..10 reset entries",
        )
        require(
            all(
                e["task_id"] == self.summary["task_id"]
                and e["instruction"] == self.summary["instruction"]
                and e["seed"] == self.protocol["seed"]
                for e in self.entries.values()
            ),
            "Task/reset/seed identities differ",
        )
        require(
            len(
                {
                    (e["reset_state_sha256"], e["reset_model_sha256"])
                    for e in self.entries.values()
                }
            )
            == 11,
            "Adaptation/evaluation reset states are not distinct",
        )
        self.entry0 = next(
            e for e in self.entries.values() if e["initial_state_id"] == 0
        )
        for entry in self.entries.values():
            verify_reset_entry(entry, self.summary["suite"])
        self.starts = {row["attempt_id"]: row for row in self.groups["rollout_start"]}
        self.ends = {
            row["result"]["attempt_id"]: row for row in self.groups["rollout_end"]
        }
        require(
            len(self.starts) == len(self.groups["rollout_start"])
            and len(self.ends) == len(self.groups["rollout_end"])
            and self.starts.keys() == self.ends.keys(),
            "Duplicate or unpaired physical rollouts",
        )
        self.generation_groups = defaultdict(list)
        for row in self.groups["generation"]:
            require(
                row["attempt_id"] in self.starts,
                "Generation belongs to no physical attempt",
            )
            self.generation_groups[row["attempt_id"]].append(row)
        self.decisions = {
            row["request_index"]: row for row in self.groups["astra_decision"]
        }
        require(
            sorted(self.decisions) == list(range(1, len(self.providers) + 1))
            and len(self.decisions) == len(self.groups["astra_decision"]),
            "Provider invocation ledger coverage differs",
        )
        require(
            [row["request_index"] for row in self.groups["astra_decision"]]
            == list(range(1, len(self.providers) + 1)),
            "Provider invocation sequence changed",
        )
        self.provider_used = set()
        self.result_cache, self.samples, self.actor_uses = {}, {}, []
        self.counts = Counter()
        self.deviations = []

    def result(self, attempt_id):
        if attempt_id not in self.result_cache:
            self.result_cache[attempt_id] = self.store.resolve(
                self.ends[attempt_id]["result"]
            )
        return self.result_cache[attempt_id]

    def physical(self):
        expected = {("native_repeated_noise", 0, 0)}
        for method in self.protocol["adaptation_methods"]:
            expected.update((method, 0, round_index) for round_index in range(1, 4))
        states = (
            self.protocol["evaluation_states"][:1]
            if self.summary["development"]
            else self.protocol["evaluation_states"]
        )
        require(
            self.protocol["evaluation_states"] == list(range(1, 11)),
            "Held-out reset indexes changed",
        )
        for method in self.protocol["evaluation_methods"]:
            expected.update(
                (method, state, round_index)
                for state in states
                for round_index in (range(1, 4) if method == "learned_noise" else [3])
            )
        observed, summary_rows, reset_pairs = set(), {}, {}
        for row in self.summary["physical_rollouts"]:
            require(
                row["attempt_id"] not in summary_rows, "Summary repeats a physical run"
            )
            summary_rows[row["attempt_id"]] = row
        require(
            summary_rows.keys() == self.starts.keys(),
            "Summary/event physical coverage differs",
        )
        for attempt_id, start in self.starts.items():
            end, result = self.ends[attempt_id], self.result(attempt_id)
            require(
                start["sequence"] < end["sequence"], "Attempt ends before it starts"
            )
            entry = self.entries[start["episode_id"]]
            key = (start["method"], entry["initial_state_id"], start["round_index"])
            require(key not in observed, "Duplicate method/reset/round")
            observed.add(key)
            require(
                attempt_id == f"{key[0]}_state{key[1]}_round{key[2]}"
                and start["reset_entry_sha256"] == digest(entry),
                "Physical attempt/reset ID changed",
            )
            for name in (
                "attempt_id",
                "method",
                "episode_id",
                "round_index",
                "evaluation",
            ):
                require(result[name] == start[name], f"Start/end {name} differs")
            require(
                start["evaluation"] == (key[1] != 0),
                "Adaptation/held-out role mismatch",
            )
            compact = {
                k: v
                for k, v in result.items()
                if k not in ("snapshots", "executed_actions", "video_path")
            }
            compact["physical_run_id"] = f"{entry['episode_id']}:{attempt_id}"
            _same(
                compact,
                summary_rows[attempt_id],
                "Summary physical row differs from archived result",
            )
            reset = result["reset_audit"]
            require(
                reset["sha256"]
                == digest({k: v for k, v in reset.items() if k != "sha256"}),
                "Reset audit digest mismatch",
            )
            for name in (
                "episode_id",
                "seed",
                "reset_state_sha256",
                "reset_model_sha256",
                "bddl_sha256",
            ):
                require(
                    reset[name] == entry[name],
                    f"Reset {name} differs from frozen entry",
                )
            if entry["episode_id"] in reset_pairs:
                _same(
                    reset,
                    reset_pairs[entry["episode_id"]],
                    "Paired attempt reset changed",
                )
            reset_pairs[entry["episode_id"]] = reset
            actions = result["actions_executed"]
            require(
                type(actions) is int
                and 0 <= actions <= 300
                and result["execute_steps"] == 10
                and result["action_budget"] == 300,
                "Physical execution budget differs",
            )
            generations = self.generation_groups[attempt_id]
            require(
                [g["step"] for g in generations] == list(range(0, actions, 10))
                and len(generations) == result["policy_replans"],
                "Generation/executed-step coverage differs",
            )
            spec = ActionSpec(**start["action_spec"])
            require(
                spec.action_spec_id
                == digest(
                    {
                        name: start["action_spec"][name]
                        for name in (
                            "horizon",
                            "model_action_dim",
                            "timestep_seconds",
                            "lower",
                            "upper",
                            "semantics",
                        )
                    }
                ),
                "Environment action-spec self-digest differs",
            )
            adapter = normalization_adapter(spec, *self.stats)
            assembled, counts, labels = [], Counter(), []
            for descriptor in generations:
                require(
                    start["sequence"] < descriptor["sequence"] < end["sequence"]
                    and descriptor["method"] == start["method"],
                    "Generation crossed physical attempt boundary",
                )
                event = self.store.resolve(descriptor)
                counts.update(replay_generation(event, entry=entry, adapter=adapter))
                assembled.extend(event["actions"])
                if event["actor_receipt"] is not None:
                    self.actor_uses.append((start, event["actor_receipt"]))
                    counts["auxiliary_inferences"] += int(
                        event["actor_receipt"]["auxiliary_forward_performed"]
                    )
                if start["method"] == "critique_frs_learning":
                    from .frs_noise_policy import make_training_sample

                    labels.append(
                        make_training_sample(
                            event["observation"],
                            event["executed_noise"],
                            kind=event["generation_kind"],
                            observation_step=event["step"],
                            source_id=f"{attempt_id}:step{event['step']}",
                        )
                    )
                if event["reconstruction"] is not None:
                    self.deviations.append(
                        {
                            "attempt_id": attempt_id,
                            "step": event["step"],
                            "method": start["method"],
                            **event["reconstruction"],
                        }
                    )
            self.samples[attempt_id] = labels
            expected_actions = np.asarray(assembled, np.float32).reshape(-1, 7)[
                :actions
            ]
            _array(
                result["executed_actions"],
                expected_actions,
                "Recorded execution is not the actual generated prefix",
            )
            parity = [
                row
                for row in self.groups["native_parity"]
                if row["attempt_id"] == attempt_id
            ]
            counts["velocity_evaluations"] += 10 * len(parity)
            for name in (
                "velocity_evaluations",
                "interventions",
                "deferred",
                "rejected",
                "clipped_actions",
                "auxiliary_inferences",
            ):
                require(
                    result["counters"][name] == counts[name],
                    f"Physical counter differs: {name}",
                )
            self.counts.update(counts)
            self.counts["actions"] += actions
            self.counts["generations"] += len(generations)
            indexes = result["provider_record_indexes"]
            require(
                indexes == sorted(set(indexes))
                and all(
                    self.providers[i]["attempt_id"] == attempt_id
                    and self.providers[i]["role"] in ("paper_direction", "action_edit")
                    for i in indexes
                ),
                "Rollout provider scope differs",
            )
            _same(
                result["provider_usage"],
                summarize_calls([self.providers[i] for i in indexes]),
                "Rollout provider cost differs",
            )
            require(
                len(indexes) == len(generations)
                if start["method"].startswith("critique_frs_")
                or start["method"] in ("astra_direction_direct", "astra_frs")
                else len(indexes) == 0,
                "Rollout provider call cadence changed",
            )
            snapshots = result["snapshots"]
            require(
                [s["step"] for s in snapshots]
                == np.rint(np.linspace(0, actions, min(4, actions + 1)))
                .astype(int)
                .tolist(),
                "Rollout snapshot selection changed",
            )
            if generations:
                first = self.store.resolve(generations[0]["observation"])
                _same(
                    snapshots[0]["observation"],
                    first,
                    "First raw snapshot differs from first policy observation",
                )
            by_step = {row["step"]: row for row in generations}
            for snapshot in snapshots:
                if snapshot["step"] in by_step:
                    _same(
                        snapshot["observation"],
                        self.store.resolve(by_step[snapshot["step"]]["observation"]),
                        "Rollout feedback snapshot differs from its actual policy observation",
                    )
            require(
                reset["post_stabilization_observation_sha256"]
                == digest(snapshots[0]["observation"]),
                "Reset observation differs from first snapshot",
            )
        require(
            observed == expected,
            "Missing/extra physical methods, rounds or held-out resets",
        )
        evaluations = [row for row in self.summary["evaluation"]]
        require(
            len(evaluations)
            == len([s for s in self.starts.values() if s["evaluation"]]),
            "Evaluation index is incomplete/duplicated",
        )
        require(
            len({row["physical_run_id"] for row in evaluations}) == len(evaluations),
            "Evaluation index duplicates physical run",
        )
        for row in evaluations:
            physical = summary_rows[row["attempt_id"]]
            require(
                all(physical[k] == value for k, value in row.items()),
                "Evaluation reference differs from its physical run",
            )
        self.counts["physical_rollouts"] = len(observed)

    def gates(self):
        require(
            len(self.groups["native_parity"]) == len(self.groups["guide_probe"]) == 1,
            "Missing/duplicate native parity or calibration gate",
        )
        event = self.store.resolve(self.groups["native_parity"][0])
        first = self.store.resolve(self.generation_groups[event["attempt_id"]][0])
        require(
            event["step"] == first["step"] == 0,
            "Native gate did not use initial current condition",
        )
        _array(
            event["noise"], first["predicted_noise"], "Native gate used another noise"
        )
        spec = ActionSpec(**self.starts[event["attempt_id"]]["action_spec"])
        adapter = normalization_adapter(spec, *self.stats)
        decoded = adapter.output_transform(
            {"actions": first["native_model_actions"][0]}
        )["actions"]
        _array(
            event["decoded"],
            decoded,
            "Native parity decoded endpoint differs",
            atol=1e-7,
        )
        errors = _metrics(
            event["errors"], event["native"], event["decoded"], "Native Euler10 parity"
        )
        require(errors["max_abs"] <= 1e-5, "Native Euler10 parity gate failed")
        guide = self.store.resolve(self.groups["guide_probe"][0])
        _array(
            guide["raw"],
            first["observation"][CAMERAS[0]],
            "Calibration gate raw image differs",
        )
        replay_guide(first["observation"], guide["guide"], guide["receipt"])
        self.guide_signs = guide["receipt"]["camera_to_controller_signs"]
        return errors

    def bind_request(self, request, decision):
        index = decision["provider_record_index"]
        require(
            index not in self.provider_used and decision["request_index"] == index + 1,
            "Provider record reused/out of order",
        )
        require(decision["role"] == request["role"], "Decision role mismatch")
        accepted = verify_provider_binding(
            request, decision, self.providers[index], self.protocol["astra"]
        )
        self.provider_used.add(index)
        self.counts["provider_bindings"] += 1
        self.counts["accepted_provider_bindings"] += int(accepted)

    def identity(self, decision):
        row = self.providers[decision["provider_record_index"]]
        return {
            name: row[name]
            for name in (
                "episode_id",
                "attempt_id",
                "request_index",
                "observation_step",
            )
        } | {"target_task": self.summary["instruction"]}

    def online_requests(self):
        for attempt_id, generations in self.generation_groups.items():
            start, previous = self.starts[attempt_id], []
            roles = (
                "paper_direction"
                if start["method"] in ("astra_direction_direct", "astra_frs")
                else "action_edit"
                if start["method"].startswith("critique_frs_")
                else None
            )
            if roles is None:
                continue
            for descriptor in generations:
                candidates = [
                    row
                    for row in self.groups["astra_decision"]
                    if self.providers[row["provider_record_index"]]["attempt_id"]
                    == attempt_id
                    and self.providers[row["provider_record_index"]]["observation_step"]
                    == descriptor["step"]
                    and row["role"] == roles
                ]
                require(len(candidates) == 1, "Missing/duplicate online decision")
                decision, event = candidates[0], self.store.resolve(descriptor)
                require(
                    self.providers[decision["provider_record_index"]]["episode_id"]
                    == start["episode_id"],
                    "Online provider request belongs to another reset",
                )
                require(
                    start["sequence"] < decision["sequence"] < event["sequence"],
                    "Online decision is not current for the generation",
                )
                if roles == "action_edit":
                    reasoner_spec = event["reasoner_action_spec"]
                    _same(
                        {
                            k: v
                            for k, v in reasoner_spec.items()
                            if k != "camera_direction_to_world_signs"
                        },
                        start["action_spec"],
                        "Reasoner controller action spec changed",
                    )
                    mapping = reasoner_spec["camera_direction_to_world_signs"]
                    require(
                        [
                            mapping["toward_camera_world_x"],
                            mapping["image_right_world_y"],
                            mapping["up_world_z"],
                        ]
                        == self.guide_signs
                        and isinstance(mapping["source"], str),
                        "Action editor calibration context differs",
                    )
                request = (
                    direction_request(
                        **self.identity(decision),
                        external_image=event["observation"][CAMERAS[0]],
                        guide_image=event["guide"],
                    )
                    if roles == "paper_direction"
                    else action_edit_request(
                        **self.identity(decision),
                        observation=event["observation"],
                        native_actions=event["native_actions"],
                        rules=start["rules"],
                        previous_decisions=previous[-2:],
                        action_spec=reasoner_spec,
                    )
                )
                self.bind_request(request, decision)
                _same(
                    event["proposal"],
                    decision["response"],
                    "Executed proposal differs from accepted current decision",
                )
                previous.append(
                    {
                        "request_index": decision["request_index"],
                        "observation_step": event["step"],
                        "proposal": event["proposal"],
                        "accepted": event["proposal"] is not None,
                        "error": None
                        if event["proposal"] is not None
                        else "call_rejected",
                    }
                )

    def adaptation(self):
        baseline_id = "native_repeated_noise_state0_round0"
        self.learning_rounds = []
        for method in self.protocol["adaptation_methods"]:
            reported = self.summary["adaptation"][method]
            require(
                reported["baseline_attempt_id"] == baseline_id,
                "Adaptation arms do not share one exact physical baseline",
            )
            events = [
                row
                for row in self.groups["adaptation_round"]
                if row["method"] == method
            ]
            require(
                [row["round_index"] for row in events] == [1, 2, 3]
                and len(reported["rounds"]) == 3,
                "Adaptation did not execute the fixed three rounds",
            )
            incumbent = latest = baseline_id
            best_rules, attempts = [], [self.result(baseline_id)]
            for descriptor, compact in zip(events, reported["rounds"], strict=True):
                event = self.store.resolve(descriptor)
                require(
                    compact_arrays(
                        {
                            k: v
                            for k, v in event.items()
                            if k not in ("kind", "sequence", "timestamp", "method")
                        }
                    )
                    == compact,
                    "Adaptation event/summary differs",
                )
                candidate = f"{method}_state0_round{event['round_index']}"
                require(
                    event["candidate_attempt_id"] == candidate
                    and event["incumbent_attempt_id"] == incumbent,
                    "Judge incumbent is not the last Astra-promoted same-arm rollout",
                )
                start, end = self.starts[candidate], self.ends[candidate]
                critique = [
                    row
                    for row in self.groups["astra_decision"]
                    if row["role"] == "critique"
                    and self.ends[latest]["sequence"]
                    < row["sequence"]
                    < start["sequence"]
                    and row["provider_record_index"] not in self.provider_used
                ]
                judge = [
                    row
                    for row in self.groups["astra_decision"]
                    if row["role"] == "judge"
                    and end["sequence"] < row["sequence"] < event["sequence"]
                ]
                require(
                    len(critique) == len(judge) == 1,
                    "Missing/extra critique or judge for round",
                )
                critique, judge = critique[0], judge[0]
                require(
                    all(
                        self.providers[row["provider_record_index"]]["episode_id"]
                        == self.entry0["episode_id"]
                        for row in (critique, judge)
                    ),
                    "Critique/judge consumed a held-out reset identity",
                )
                self.bind_request(
                    critique_request(
                        **self.identity(critique),
                        rollout=visible_rollout(self.result(latest)),
                    ),
                    critique,
                )
                self.bind_request(
                    judge_request(
                        **self.identity(judge),
                        incumbent=visible_rollout(self.result(incumbent)),
                        candidate=visible_rollout(self.result(candidate)),
                    ),
                    judge,
                )
                _same(
                    event["critique"],
                    critique["response"],
                    "Critique response binding differs",
                )
                _same(
                    event["judge"], judge["response"], "Judge response binding differs"
                )
                rules = event["critique"]["rules"] if event["critique"] else []
                _same(
                    event["rules"],
                    rules,
                    "Round rules did not come from latest critique",
                )
                _same(start["rules"], rules, "Candidate used another rule set")
                promoted = (
                    event["critique"] is not None
                    and event["judge"] is not None
                    and event["judge"]["verdict"] == "better"
                )
                require(
                    event["promoted"] == promoted,
                    "Promotion used a condition other than validated Astra better",
                )
                require(
                    event["incumbent_success_evaluation_only"]
                    == self.result(incumbent)["success"]
                    and event["candidate_success_evaluation_only"]
                    == self.result(candidate)["success"],
                    "Evaluation-only outcomes differ from physical records",
                )
                if method == "critique_frs_learning":
                    self.learning_rounds.append(event)
                else:
                    require(
                        event["update"] is None and event["checkpoint"] is None,
                        "No-learning arm trained or saved an auxiliary policy",
                    )
                if promoted:
                    incumbent, best_rules = candidate, rules
                latest = candidate
                attempts.append(self.result(candidate))
            _same(
                reported["first_success"],
                first_success(attempts),
                "First-success summary differs from physical rounds",
            )
            _same(
                reported["final_rules"],
                best_rules,
                "Final rules are not the best Astra-promoted same-arm rules",
            )
            for start in self.starts.values():
                if start["method"] == method and start["evaluation"]:
                    _same(
                        start["rules"],
                        best_rules,
                        "Held-out no-learning rollout used another rule set",
                    )
        require(
            self.provider_used == set(range(len(self.providers))),
            "Unbound or duplicated provider invocations",
        )

    def checkpoints(self):
        from .frs_noise_policy import (
            _validate_sample,
            training_config,
        )

        initial_receipt, initial_payload = read_auxiliary_checkpoint(
            self.store, "noise_policy_initial"
        )
        initial = initial_receipt["policy"]
        require(
            len(self.groups["noise_policy_initial"]) == 1,
            "Missing/duplicate initial auxiliary checkpoint event",
        )
        initial_event = self.groups["noise_policy_initial"][0]
        require(
            initial_event["sequence"]
            < min(row["sequence"] for row in self.starts.values()),
            "Initial actor checkpoint was saved after rollout feedback",
        )
        _same(
            initial_event["checkpoint"],
            initial_receipt,
            "Initial actor event/checkpoint differs",
        )
        _same(
            self.summary["initial_noise_policy_checkpoint"],
            initial_receipt,
            "Initial actor summary/checkpoint differs",
        )
        task_id = f"{self.summary['suite']}:{self.summary['task_id']}"
        seed = int(digest(self.entry0["episode_id"])[:8], 16)
        require(
            initial["task_id"] == task_id
            and initial["seed"] == seed
            and initial["effective_seed"]
            == int(digest({"task": task_id, "seed": seed})[:16], 16) % (2**63),
            "Initial actor task/seed differs",
        )
        expected_sources = {
            key: file_sha256(Path(__file__).with_name(name))
            for key, name in (
                ("noise_policy_sha256", "frs_noise_policy.py"),
                ("operators_sha256", "frs_operators.py"),
                ("records_sha256", "records.py"),
            )
        }
        require(
            initial["sources"] == expected_sources
            and initial["config"] == training_config()
            and initial["synthetic_cpu_test_only"] is False
            and initial["test_only_cpu_steps"] is None,
            "Initial actor source/configuration differs",
        )
        require(
            initial["trained"] is False
            and initial["rounds"] == initial["replay_samples"] == 0
            and initial["accepted_rollouts"]
            == initial_payload["accepted_rollouts"]
            == initial_payload["replay"]
            == initial_payload["history"]
            == []
            and not initial_payload["optimizer"]["state"],
            "Initial actor is not an untrained exact-fallback policy",
        )
        self.initial_checkpoint_receipt = {
            "manifest_sha256": initial_receipt["manifest_sha256"],
            "state_file_sha256": initial_receipt["files"]["state.pt"]["sha256"],
            "state_id": initial["state_id"],
            "parameter_sha256": initial["parameter_sha256"],
        }
        current = initial
        expected_samples, expected_history, accepted = [], [], []
        checkpoints, states_by_round = [], {0: initial}
        for event in self.learning_rounds:
            round_index = event["round_index"]
            directory = f"noise_policy_round{round_index}"
            recorded, payload = read_auxiliary_checkpoint(self.store, directory)
            manifest = {k: v for k, v in recorded.items() if k != "manifest_sha256"}
            _same(
                event["checkpoint"],
                recorded,
                "Recorded checkpoint manifest binding differs",
            )
            model, optimizer, replay, history = (
                payload[name] for name in ("model", "optimizer", "replay", "history")
            )
            for value in model.values():
                require(
                    isinstance(value, np.ndarray)
                    and value.dtype == np.float32
                    and np.isfinite(value).all(),
                    "Invalid auxiliary parameter tensor",
                )
            metadata = manifest["policy"]
            for name in (
                "schema_version",
                "task_id",
                "seed",
                "effective_seed",
                "config",
                "sources",
                "preprocessing_pillow_version",
                "synthetic_cpu_test_only",
                "test_only_cpu_steps",
            ):
                require(
                    metadata[name] == initial[name],
                    f"Auxiliary checkpoint compatibility changed: {name}",
                )
            require(
                metadata["config"] == training_config()
                and metadata["synthetic_cpu_test_only"] is False,
                "Test/training recipe cannot be used as production evidence",
            )
            update = event["update"]
            samples = self.samples[event["candidate_attempt_id"]]
            if event["promoted"] and samples:
                require(
                    update is not None
                    and update["status"] == "updated"
                    and update["optimizer_updates"] == 1000
                    and update["synthetic_cpu_test_only"] is False
                    and str(update["device"]).startswith("cuda"),
                    "Accepted rollout lacks actual fixed CUDA training",
                )
                require(
                    update["state_before"] == current["state_id"]
                    and update["judge_sha256"] == digest(event["judge"])
                    and update["judge_verdict"] == "better"
                    and update["rollout_id"] == event["candidate_attempt_id"],
                    "Training was not bound to the Astra-better adaptation candidate",
                )
                expected_samples.extend(samples)
                accepted.append(event["candidate_attempt_id"])
                expected_history.append(
                    {k: v for k, v in update.items() if k != "state_after"}
                )
                require(
                    update["round"] == len(accepted)
                    and update["sample_ids"]
                    == [s["metadata"]["sample_id"] for s in expected_samples],
                    "Training sample order/round differs",
                )
                targets = np.stack([s["target"] for s in expected_samples])
                _array(
                    update["targets_raw"],
                    targets,
                    "Training target differs from accepted executed noise",
                )
                _array(
                    update["targets_clipped_for_diagnostics_only"],
                    np.clip(targets, -5, 5),
                    "Training clipping diagnostic differs",
                )
                outside = np.abs(targets) > 5
                _same(
                    update["target_out_of_range_mask"],
                    outside,
                    "Training clipping mask differs",
                )
                require(
                    update["loss_targets_clipped"] is False
                    and update["target_out_of_range_fraction"] == float(outside.mean())
                    and update["mean_regularization"] == 0.001
                    and update["batch_size"] == 128
                    and update["learning_rate"] == 1e-4,
                    "Training loss/batch configuration changed",
                )
                predictions = update["predictions_after"]
                require(
                    predictions.shape == targets.shape
                    and predictions.dtype == np.float32
                    and np.isfinite(predictions).all()
                    and np.max(np.abs(predictions)) <= 5,
                    "Invalid recorded auxiliary output",
                )
                difference = predictions.astype(np.float64) - targets
                _array(
                    update["mean_target_error"], difference, "Mean-target error differs"
                )
                require(
                    math.isclose(
                        update["mean_target_rmse"],
                        float(np.sqrt(np.mean(difference**2))),
                        rel_tol=1e-12,
                    )
                    and update["mean_target_max_abs"]
                    == float(np.max(np.abs(difference))),
                    "Auxiliary fitting metrics differ",
                )
                trace = update["loss_trace"]
                require(
                    trace.shape == (1000, 3)
                    and trace.dtype == np.float64
                    and np.isfinite(trace).all()
                    and np.all(trace >= 0)
                    and np.allclose(
                        trace[:, 0],
                        trace[:, 1] + 0.001 * trace[:, 2],
                        rtol=2e-6,
                        atol=1e-7,
                    ),
                    "Training loss trace does not match fixed MSE objective",
                )
                require(
                    metadata["parameter_sha256"] != current["parameter_sha256"]
                    and update["state_after"] == metadata["state_id"],
                    "Training did not change the auxiliary parameters or state binding",
                )
                self.counts["accepted_policy_updates"] += 1
                self.counts["optimizer_steps"] += 1000
            elif event["promoted"]:
                require(
                    update is not None
                    and update["status"] == "not_updated"
                    and update["reason"] == "no_executed_samples"
                    and update["optimizer_updates"] == 0,
                    "Empty accepted rollout unexpectedly trained",
                )
                _same(metadata, current, "Empty training rollout changed actor state")
            else:
                require(update is None, "Unpromoted rollout entered training")
                _same(
                    metadata,
                    current,
                    "Rejected judgment changed weights/optimizer/replay",
                )
            for sample in replay:
                _validate_sample(sample)
            _same(
                replay,
                expected_samples,
                "Checkpoint replay includes another task, held-out reset or rejected rollout",
            )
            _same(
                history,
                expected_history,
                "Checkpoint training history differs from recorded accepted updates",
            )
            require(
                payload["accepted_rollouts"] == accepted
                and metadata["accepted_rollouts"] == accepted
                and metadata["rounds"] == len(accepted)
                and metadata["trained"] == bool(accepted)
                and metadata["replay_samples"] == len(expected_samples),
                "Checkpoint replay/update counts differ",
            )
            for name, value in (
                ("parameter_sha256", model),
                ("optimizer_sha256", optimizer),
                ("replay_sha256", replay),
                ("history_sha256", history),
            ):
                require(
                    metadata[name] == digest(value),
                    f"Checkpoint {name} differs from actual saved tensors",
                )
            require(
                metadata["state_id"]
                == digest({k: v for k, v in metadata.items() if k != "state_id"}),
                "Auxiliary state identity is invalid",
            )
            require(
                len(optimizer["param_groups"]) == 1
                and optimizer["param_groups"][0]["lr"] == 1e-4,
                "Checkpoint Adam groups/learning rate changed",
            )
            if accepted:
                require(
                    len(optimizer["state"]) == len(model),
                    "Adam state does not cover all trainable tensors",
                )
                for state in optimizer["state"].values():
                    require(
                        float(state["step"]) == 1000 * len(accepted)
                        and np.isfinite(state["exp_avg"]).all()
                        and np.isfinite(state["exp_avg_sq"]).all(),
                        "Adam step/moment state differs from accepted fixed updates",
                    )
            else:
                require(
                    not optimizer["state"], "Untrained actor acquired optimizer state"
                )
            current = metadata
            states_by_round[round_index] = metadata
            checkpoints.append(
                {
                    "round_index": round_index,
                    "manifest_sha256": recorded["manifest_sha256"],
                    "state_file_sha256": manifest["files"]["state.pt"]["sha256"],
                    "state_id": metadata["state_id"],
                    "parameter_sha256": metadata["parameter_sha256"],
                    "accepted_updates": len(accepted),
                    "replay_samples": len(replay),
                    "promoted": event["promoted"],
                }
            )
        require(len(checkpoints) == 3, "Missing saved policy round")
        for start, actor in self.actor_uses:
            index = start["round_index"] - (
                1 if start["method"] == "critique_frs_learning" else 0
            )
            require(
                start["method"] in ("critique_frs_learning", "learned_noise")
                and actor["state_id"] == states_by_round[index]["state_id"]
                and actor["trained_rounds"] == states_by_round[index]["rounds"],
                "Rollout used another task/round's auxiliary checkpoint",
            )
        for start in self.starts.values():
            count = sum(
                s["attempt_id"] == start["attempt_id"] for s, _ in self.actor_uses
            )
            required = (
                len(self.generation_groups[start["attempt_id"]])
                if start["method"] in ("critique_frs_learning", "learned_noise")
                else 0
            )
            require(count == required, "Missing/extra auxiliary actor invocations")
        return checkpoints

    def costs(self):
        _same(
            self.summary["provider_usage"],
            summarize_calls(self.providers),
            "All-call usage differs from immutable provider ledger",
        )
        costs = self.summary["physical_cost"]
        expected = {
            "rollouts": self.counts["physical_rollouts"],
            "actions": self.counts["actions"],
            "velocity_evaluations": self.counts["velocity_evaluations"],
            "provider_calls": self.summary["provider_usage"]["provider_calls"],
            "client_attempts": len(self.providers),
            "accepted_policy_updates": self.counts["accepted_policy_updates"],
            "optimizer_steps": self.counts["optimizer_steps"],
            "auxiliary_inferences": self.counts["auxiliary_inferences"],
        }
        for name, value in expected.items():
            require(costs[name] == value, f"Total physical cost differs: {name}")
        for name, value in (
            (
                "rollout_wall_seconds",
                sum(self.result(key)["wall_seconds"] for key in self.starts),
            ),
            (
                "training_seconds",
                sum(
                    e["update"]["wall_seconds"]
                    for e in self.learning_rounds
                    if e["update"] and e["update"]["status"] == "updated"
                ),
            ),
            (
                "auxiliary_request_seconds",
                sum(
                    self.result(key)["counters"]["auxiliary_request_seconds"]
                    for key in self.starts
                ),
            ),
        ):
            require(
                math.isclose(costs[name], value, rel_tol=1e-12, abs_tol=1e-9),
                f"Total recorded timing differs: {name}",
            )
        return costs


def _worker_inputs(directory, audit):
    root = Path(directory).resolve()
    names = (
        "runtime.json",
        "protocol.json",
        "frozen_plan.json",
        "reset_manifest.json",
        "checkpoint.json",
        "prompts.json",
        "frozen_weights_before.json",
    )
    values, hashes = {}, {}
    for name in names:
        path = root / name
        require(not path.is_symlink(), "Worker metadata may not be symlinked")
        hashes[name], values[name] = file_sha256(path), _json(path.read_text())
    runtime, plan, manifest = (
        values["runtime.json"],
        values["frozen_plan.json"],
        values["reset_manifest.json"],
    )
    _same(values["protocol.json"], audit.protocol, "Worker/task protocols differ")
    _same(values["checkpoint.json"], audit.checkpoint, "Worker/task checkpoints differ")
    _same(
        values["prompts.json"], prompt_manifest(), "Worker/auditor prompt text differs"
    )
    _same(plan["runtime"], runtime, "Worker runtime/frozen plan differs")
    require(
        plan["protocol_sha256"] == digest(audit.protocol)
        and plan["manifest_sha256"] == manifest["sha256"]
        and runtime["tf32"] is False
        and "L40S" in runtime["gpu"],
        "Frozen worker runtime differs",
    )
    require(
        runtime["phase"]
        == ("development" if audit.summary["development"] else "evaluation")
        and audit.summary["task_id"] in plan["assigned_tasks"]
        and plan["assigned_episodes"]
        == [row["episode_id"] for row in manifest["episodes"]],
        "Worker phase/task/reset assignment differs",
    )
    require(
        manifest["sha256"]
        == digest({k: v for k, v in manifest.items() if k != "sha256"}),
        "Reset manifest self-digest differs",
    )
    entries = [
        row
        for row in manifest["episodes"]
        if row["task_id"] == audit.summary["task_id"]
    ]
    _same(
        entries,
        audit.groups["task"][0]["entries"],
        "Task reset entries differ from worker manifest",
    )
    before = values["frozen_weights_before.json"]
    require(
        before["sha256"] == digest(before["tensors"]),
        "Frozen native tensor receipt self-digest differs",
    )
    require(bool(before["tensors"]), "Empty native weight receipt")
    seal_path = audit.store.path("completion_receipt.json")
    task_after_path = audit.store.path("frozen_weights_after.json")
    seal_sha, task_after_sha = None, None
    if seal_path.exists():
        seal = audit.store.read_json("completion_receipt.json")
        require(
            set(seal)
            == {
                "schema_version",
                "task_id",
                "workflow",
                "worker",
                "native_tensor_sha256",
                "task_files_sha256",
                "worker_metadata_sha256",
            }
            and seal["schema_version"] == "frs-completed-task-1.0"
            and audit.summary["status"] == "complete"
            and type(seal["task_id"]) is int
            and seal["task_id"] == audit.summary["task_id"]
            and type(seal["worker"]) is int
            and seal["worker"] == runtime["worker"]
            and seal["workflow"] == runtime["workflow"]
            and seal["native_tensor_sha256"] == before["sha256"],
            "Task completion seal identity differs",
        )
        task_after = audit.store.read_json("frozen_weights_after.json")
        _same(
            before, task_after, "Native pi05 tensor bytes changed during completed task"
        )
        task_names = {
            "summary.json",
            "events.jsonl",
            "provider.jsonl",
            "frozen_weights_after.json",
        }
        require(
            set(seal["task_files_sha256"]) == task_names,
            "Task completion seal omits or adds a recording file",
        )
        for name in task_names:
            audit.store.remember(name)
            require(
                seal["task_files_sha256"][name] == audit.store.files[name],
                "Task completion seal recording bytes differ",
            )
        require(
            seal["worker_metadata_sha256"] == hashes,
            "Task completion seal worker metadata bytes differ",
        )
        seal_sha = audit.store.files["completion_receipt.json"]
        task_after_sha = audit.store.files["frozen_weights_after.json"]
    else:
        require(
            not task_after_path.exists(),
            "Task-level frozen weights lack a completion seal",
        )

    final_after_path = root / "frozen_weights_after.json"
    require(not final_after_path.is_symlink(), "Worker metadata may not be symlinked")
    if final_after_path.exists():
        hashes["frozen_weights_after.json"] = file_sha256(final_after_path)
        after = _json(final_after_path.read_text())
        _same(before, after, "Native pi05 tensor bytes changed during worker execution")
        weight_check_scope = "worker_complete"
    else:
        require(
            seal_sha is not None,
            "Missing final worker weight receipt and verified task completion seal",
        )
        weight_check_scope = "completed_task"
    for name, checksum in hashes.items():
        require(
            file_sha256(root / name) == checksum, "Worker metadata changed during audit"
        )
    return {
        "input_file_sha256": hashes,
        "runtime": runtime,
        "native_tensor_sha256": before["sha256"],
        "weight_check_scope": weight_check_scope,
        "completion_receipt_sha256": seal_sha,
        "task_frozen_weights_after_sha256": task_after_sha,
        "reset_manifest_sha256": manifest["sha256"],
        "checkpoint_metadata_sha256": digest(audit.checkpoint),
    }


def audit_task(
    task_dir, *, norm_stats_path, expected_protocol=None, worker_metadata_dir=None
):
    """Require a complete task and sealed or worker-final native weight proof."""
    audit = _TaskAudit(task_dir, norm_stats_path, expected_protocol)
    audit.physical()
    parity = audit.gates()
    audit.online_requests()
    audit.adaptation()
    checkpoints = audit.checkpoints()
    costs = audit.costs()
    worker = _worker_inputs(worker_metadata_dir or Path(task_dir).parent, audit)
    arrays = audit.store.finish()
    sources = {
        path.name: file_sha256(path)
        for path in (
            Path(__file__),
            *(
                Path(__file__).with_name(name)
                for name in (
                    "frs_operators.py",
                    "frs_noise_policy.py",
                    "frs_experiment.py",
                    "frs_guide.py",
                    "frs_agent.py",
                    "action_adapter.py",
                    "records.py",
                    "flow.py",
                    "image_perturbation_audit.py",
                    "astra_client.py",
                    "intervention_agent.py",
                    "interpolation_agent.py",
                    "image_perturbations.py",
                    "interventions.py",
                    "lerobot_policy.py",
                    "openpi_inputs.py",
                )
            ),
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "suite": audit.summary["suite"],
        "task_id": audit.summary["task_id"],
        "development": audit.summary["development"],
        "summary_sha256": audit.store.files["summary.json"],
        "events_sha256": audit.store.files["events.jsonl"],
        "provider_sha256": audit.store.files["provider.jsonl"],
        "input_file_sha256": {
            name: checksum
            for name, checksum in audit.store.files.items()
            if not name.startswith("arrays/")
        },
        "norm_stats_sha256": audit.norm_hash,
        "postprocessor_sources": sources,
        "postprocessor_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "platform": platform.platform(),
        },
        "worker": worker,
        "native_parity": parity,
        "arrays": arrays,
        "counts": dict(audit.counts),
        "zero_action_outcomes": [
            {
                "physical_run_id": row["physical_run_id"],
                "initial_success": row["initial_success"],
                "success": row["success"],
            }
            for row in audit.summary["physical_rollouts"]
            if row["actions_executed"] == 0
        ],
        "physical_cost": costs,
        "provider_usage": audit.summary["provider_usage"],
        "checkpoints": checkpoints,
        "initial_checkpoint": audit.initial_checkpoint_receipt,
        "physical_runs": [
            {
                "episode_id": row["episode_id"],
                "method": row["method"],
                "round_index": row["round_index"],
                "physical_run_id": row["physical_run_id"],
                "attempt_id": row["attempt_id"],
                "evaluation": row["evaluation"],
                "success": row["success"],
                "actions": row["actions_executed"],
                "velocity_evaluations": row["counters"]["velocity_evaluations"],
                "reset_sha256": row["reset_audit"]["sha256"],
            }
            for row in audit.summary["physical_rollouts"]
        ],
        "post_transform_action_deviation": {
            "pairs": len(audit.deviations),
            "max_abs": max((row["max_abs"] for row in audit.deviations), default=None),
            "per_generation": audit.deviations,
            "interpretation": "recorded_action_deviation_after_deliberate_padding_and_optional_mean_projection_not_an_exactness_gate",
        },
        "validations": {
            "complete_physical_coverage": True,
            "all_array_hashes": True,
            "native_weight_bytes_unchanged": True,
            "separated_reset_streams": True,
            "exact_request_provider_bindings": True,
            "calibrated_direction_and_action_transforms": True,
            "executed_loop_noise_repeated": True,
            "training_labels_only_Astra_better_adaptation": True,
            "no_heldout_feedback_or_labels": True,
            "checkpoint_optimizer_replay_bindings": True,
        },
        "known_generating_noise_for_Astra_reference": None,
        "limitations": LIMITATIONS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--worker-metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an audit receipt")
    receipt = audit_task(
        args.task_dir,
        norm_stats_path=args.norm_stats,
        worker_metadata_dir=args.worker_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "task_id": receipt["task_id"],
                "counts": receipt["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
