"""Stream one immutable intervention archive; never call a model or simulator."""

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sentencepiece
from frozen_intervention.intervention_agent import normalize_usage, parse_proposal
from frozen_intervention.interventions import apply_vision, perturb_noise, success_curve
from frozen_intervention.records import digest

PAYLOAD = "fde89be9786f85f6ed5e889d7f736784a500dd6fc2f81e9db65480cadb606fb4"
OPERATOR = "c6b0bd6a84108386af2e85627584f5740398ff1e70885d06d95e2cb758a3c67d"
BOUNDARIES = (
    "text_mask_fixed",
    "padding_unchanged",
    "vision_prefix_unchanged",
    "attention_unchanged",
    "positions_unchanged",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def error(reference, actual):
    require(reference.shape == actual.shape, "Endpoint shape mismatch")
    delta = actual.astype(np.float64) - reference
    return {
        "max_abs": float(np.abs(delta).max()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


class Catalog:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())

    def url(self, key):
        item = self.data.get("objects", self.data)[key]
        return item["url"] if isinstance(item, dict) else item

    def read(self, key):
        with urllib.request.urlopen(self.url(key), timeout=60) as response:
            return response.read()

    def receipt(self, key):
        request = urllib.request.Request(self.url(key), headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read(1)
            return {
                "status": response.status,
                "etag": response.headers.get("ETag"),
                "content_range": response.headers.get("Content-Range"),
            }


class HashReader:
    def __init__(self, source):
        self.source, self.hash, self.bytes = source, hashlib.sha256(), 0

    def read(self, size=-1):
        value = self.source.read(size)
        self.hash.update(value)
        self.bytes += len(value)
        return value


class CaseAudit:
    def __init__(self, name, metadata, data, assets):
        self.name, self.metadata, self.data = name, metadata, data
        self.summary = json.loads(data["summary.json"])
        self.rows = [
            json.loads(line) for line in data["events.jsonl"].splitlines() if line
        ]
        require(self.summary["status"] == "complete", "Case summary is incomplete")
        require(
            [row["sequence"] for row in self.rows] == list(range(len(self.rows))),
            "Noncontiguous event sequence",
        )
        self.case = next(row for row in self.rows if row["kind"] == "case")
        self.entry, self.protocol = self.case["entry"], self.case["protocol"]
        require(
            self.protocol == metadata["protocol.json"],
            "Case protocol differs from frozen worker",
        )
        entries = {
            row["episode_id"]: row
            for row in metadata["reset_manifest.json"]["episodes"]
        }
        require(
            self.entry == entries[self.entry["episode_id"]],
            "Case reset entry differs from manifest",
        )
        require(
            self.entry["episode_id"]
            in metadata["frozen_plan.json"]["assigned_episodes"],
            "Unassigned case",
        )
        self.initial = next(
            row for row in self.rows if row["kind"] == "inversion_initialization"
        )
        self.generations = [
            row for row in self.rows if row["kind"] == "candidate_generation"
        ]
        self.proposals = {
            (row["arm"], row["proposal"]["candidate_id"]): row["proposal"]
            for row in self.rows
            if row["kind"] == "intervention_proposal"
        }
        require(
            len(self.proposals)
            == sum(row["kind"] == "intervention_proposal" for row in self.rows),
            "Duplicate arm-scoped proposal ID",
        )
        self.refs, self.values, self.inventory = {}, {}, {}
        self.groups, self.consumers = {}, defaultdict(list)
        self.conditioning_counts, self.max_text_relative = Counter(), 0.0
        self.vision_effects, self.verified_conditions = 0, 0

        def walk(value):
            if isinstance(value, dict):
                if {"array", "shape", "dtype", "sha256"} <= value.keys():
                    name = value["array"]
                    require(
                        name.startswith("arrays/") and ".." not in Path(name).parts,
                        "Unsafe array path",
                    )
                    require(
                        name not in self.refs or self.refs[name] == value,
                        "Conflicting array reference",
                    )
                    self.refs[name] = value
                else:
                    for entry in value.values():
                        walk(entry)
            elif isinstance(value, list):
                for entry in value:
                    walk(entry)

        for row in self.rows:
            walk(row)
        observations = [("initial", self.initial)] + [
            (str(row["sequence"]), row) for row in self.generations
        ]
        for key, row in observations:
            self.groups[key] = {"row": row, "values": {}}
            for field, ref in row["observation"].items():
                self.consumers[ref["array"]].append((key, field))
        checkpoint = metadata["checkpoint.json"]
        profile = checkpoint["input_profile_assets"]
        norm = (assets / "norm_stats.json").read_bytes()
        tokens = (assets / "paligemma_tokenizer.model").read_bytes()
        require(
            sha(norm) == profile["norm_stats"]["sha256"], "Normalization asset mismatch"
        )
        require(
            sha(tokens) == profile["tokenizer"]["sha256"], "Tokenizer asset mismatch"
        )
        stats = json.loads(norm)["norm_stats"]["actions"]
        self.q01, self.q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        self.sp = sentencepiece.SentencePieceProcessor(model_proto=tokens)
        self.tokens = {}

    def token_identity(self, prompt):
        if prompt not in self.tokens:
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            ids = self.sp.encode(cleaned, add_bos=True) + self.sp.encode("\n")
            require(len(ids) <= 200, "Recorded text exceeds frozen token budget")
            padded = np.asarray([ids + [0] * (200 - len(ids))], dtype=np.int64)
            mask = np.asarray([[True] * len(ids) + [False] * (200 - len(ids))])
            self.tokens[prompt] = (digest(padded), digest(mask), len(ids))
        return self.tokens[prompt]

    def conditioning(self, value, raw_id, observation_id):
        require(
            value["raw_condition_id"] == raw_id
            and value["observation_id"] == observation_id,
            "Condition observation binding mismatch",
        )
        require(
            value["operator_source_sha256"] == OPERATOR,
            "Unexpected text operator source",
        )
        require(
            all(value[field] for field in BOUNDARIES),
            "Recorded prefix boundary changed",
        )
        require(
            value["vision_prefix_before_sha256"] == value["vision_prefix_after_sha256"],
            "Vision prefix hash changed",
        )
        require(
            0 <= value["relative_rms"] <= 0.25
            and 0 <= value["delta_frobenius"] <= value["bound_frobenius"],
            "Embedding bound violated",
        )
        require(
            value["has_effect"]
            == (value["prefix_before_sha256"] != value["prefix_after_sha256"]),
            "Embedding effect/hash inconsistency",
        )
        require(value["embedding_dtype"] == "torch.float32", "Embedding dtype changed")
        token_hash, mask_hash, count = self.token_identity(self.entry["instruction"])
        require(
            (
                value["original_token_sha256"],
                value["original_token_mask_sha256"],
                value["original_valid_text_tokens"],
            )
            == (token_hash, mask_hash, count),
            "Original token identity mismatch",
        )
        requested = value["requested"]
        enabled = (
            requested is not None
            and requested["alpha"] > 0
            and requested["max_relative_norm"] > 0
        )
        require(
            value["enabled"] == enabled == value["guidance_evaluated"],
            "Text enabled state mismatch",
        )
        if enabled:
            ids, masks, count = self.token_identity(requested["guidance_prompt"])
            require(
                (
                    value["guidance_token_sha256"],
                    value["guidance_token_mask_sha256"],
                    value["guidance_valid_text_tokens"],
                )
                == (ids, masks, count),
                "Guidance token identity mismatch",
            )
        else:
            require(
                value["delta_frobenius"] == 0 and not value["has_effect"],
                "Disabled embedding edit changed prefix",
            )
        identity = {key: item for key, item in value.items() if key != "condition_id"}
        expected = (
            digest({"raw_condition_id": raw_id, "text_intervention": identity})
            if enabled
            else raw_id
        )
        require(value["condition_id"] == expected, "Intervened condition ID mismatch")
        return expected

    def observation(self, key, row, observation):
        instruction = self.entry["instruction"]
        observation_id = digest(observation)
        if key == "initial":
            raw_id = digest({**observation, "prompt": instruction})
            require(row["condition_id"] == raw_id, "Initial raw condition mismatch")
            self.conditioning(row["zero_embedding_hook"], raw_id, observation_id)
            probe = row.get("development_embedding_probe")
            if probe:
                self.conditioning(probe["conditioning"], raw_id, observation_id)
            return
        proposal = self.proposals.get((row["mode"], row["candidate_id"]))
        modified = apply_vision(observation, proposal["vision"] if proposal else [])
        require(
            digest(modified) == row["modified_observation_sha256"],
            "Vision redraw differs from recorded condition",
        )
        self.vision_effects += digest(modified) != observation_id
        raw_id = digest({**modified, "prompt": instruction})
        value = row["conditioning"]
        language = proposal.get("language") if proposal else None
        if value is None:
            require(
                language is None and row["condition_id"] == raw_id,
                "Missing or wrong base conditioning",
            )
        else:
            require(language is not None, "Unexpected text intervention")
            require(
                value["requested"]
                == {
                    "guidance_prompt": instruction
                    + "\nGuidance: "
                    + language["target_text"],
                    "alpha": language["scale"],
                    "max_relative_norm": self.protocol["language"][
                        "maximum_relative_norm"
                    ],
                },
                "Guidance differs from accepted proposal",
            )
            require(
                row["condition_id"] == self.conditioning(value, raw_id, observation_id),
                "Wrong generation condition",
            )
            self.conditioning_counts["recorded"] += 1
            self.conditioning_counts["enabled"] += value["enabled"]
            self.conditioning_counts["effect"] += value["has_effect"]
            self.max_text_relative = max(self.max_text_relative, value["relative_rms"])
        self.verified_conditions += 1

    def array(self, name, data):
        require(
            name in self.refs and name not in self.inventory,
            "Unexpected or duplicate archived array",
        )
        value = np.load(io.BytesIO(data), allow_pickle=False)
        ref = self.refs[name]
        require(
            list(value.shape) == ref["shape"] and str(value.dtype) == ref["dtype"],
            "Array shape/dtype mismatch",
        )
        require(
            np.isfinite(value).all() and digest(value) == ref["sha256"],
            "Array content hash mismatch",
        )
        self.inventory[name] = {
            "file_sha256": sha(data),
            "record_digest": ref["sha256"],
        }
        if value.dtype != np.uint8 or value.ndim != 3:
            self.values[name] = value
        for group_key, field in self.consumers.get(name, []):
            group = self.groups[group_key]
            group["values"][field] = value
            if len(group["values"]) == len(group["row"]["observation"]):
                self.observation(group_key, group["row"], group["values"])
                del self.groups[group_key]

    def value(self, ref):
        return self.values[ref["array"]]

    def decode(self, value):
        return (value[0, :, :7] + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01

    def finish(self, extra):
        require(
            set(self.inventory) == set(self.refs) and not self.groups,
            "Incomplete referenced-array coverage",
        )
        require(
            self.verified_conditions == len(self.generations),
            "Incomplete condition verification",
        )
        known, recovered = (
            self.value(self.initial["known_noise"]),
            self.value(self.initial["recovered_noise"]),
        )
        require(
            known.shape == recovered.shape == (1, 10, 32), "Unexpected flow-state shape"
        )
        errors = {}
        for label, left, right in (
            ("noise", "known_noise", "recovered_noise"),
            ("actions", "reference", "roundtrip"),
            ("native_parity", "native_actions", "adapter_actions"),
        ):
            errors[label] = error(
                self.value(self.initial[left]), self.value(self.initial[right])
            )
            require(
                errors[label] == self.initial["errors"][label],
                "Recorded numerical metric differs from arrays",
            )
        require(
            errors["noise"]["max_abs"] <= 0.1
            and errors["actions"]["max_abs"] <= 0.02
            and errors["native_parity"]["max_abs"] <= 1e-5,
            "Initial numerical gate failed",
        )
        require(
            self.initial["errors"]["zero_embedding_hook_parity"]["max_abs"] == 0
            and self.initial["passed"],
            "Initial zero hook gate failed",
        )
        basis = self.value(self.case["basis"])
        require(
            basis.shape == (8, 1, 10, 32) and digest(basis) == self.case["basis_id"],
            "Noise basis binding mismatch",
        )
        flat = basis.reshape(8, -1).astype(np.float64)
        require(
            np.max(np.abs(flat @ flat.T / flat.shape[1] - np.eye(8))) <= 1e-6,
            "Noise basis lost unit-RMS orthogonality",
        )
        words = [self.protocol["seed"], int(digest(self.entry["episode_id"])[:8], 16)]
        expected_known = (
            np.random.default_rng(np.random.SeedSequence(words))
            .standard_normal((1, 10, 32))
            .astype(np.float32)
        )
        require(np.array_equal(known, expected_known), "Known noise seed mismatch")
        fresh_rng = np.random.default_rng(np.random.SeedSequence(words + [2]))
        grouped = defaultdict(list)
        maximum_noise_error, maximum_noise_rms = 0.0, 0.0
        for row in self.generations:
            mode, candidate = row["mode"], row["candidate_id"]
            latent = self.value(row["latent"])
            if mode == "policy_fresh":
                expected = (
                    known
                    if row["observation_step"] == 0
                    else fresh_rng.standard_normal((1, 10, 32)).astype(np.float32)
                )
            elif mode == "known_noise":
                expected = known
            else:
                proposal = self.proposals.get((mode, candidate))
                expected = perturb_noise(
                    recovered, basis, proposal["noise"] if proposal else None
                )
                maximum_noise_rms = max(
                    maximum_noise_rms,
                    float(
                        np.sqrt(np.mean((latent.astype(np.float64) - recovered) ** 2))
                    ),
                )
            delta = error(expected, latent)["max_abs"]
            maximum_noise_error = max(maximum_noise_error, delta)
            require(
                delta <= 1e-7,
                "Generation latent differs from frozen candidate expression",
            )
            require(row["velocity_evaluations"] == 10, "Execution solver cost drift")
            decoded = self.decode(self.value(row["generated_actions"])).astype(
                np.float32
            )
            spec = self.initial["action_spec"]
            clipped = np.clip(decoded, spec["lower"], spec["upper"]).astype(np.float32)
            require(
                np.array_equal(clipped, self.value(row["controller_actions"])),
                "Controller decode or clipping mismatch",
            )
            grouped[(mode, row["iteration"], candidate)].append(row)
        attempts = [row["attempt"] for row in self.rows if row["kind"] == "attempt"]
        reset = attempts[0]["reset_audit"]
        require(
            all(row["reset_audit"] == reset for row in attempts),
            "Candidate reset differs from common scene",
        )
        for attempt in attempts:
            proposal = attempt["proposal"]
            mode = (
                proposal.get("arm", "random_noise")
                if proposal
                else attempt["candidate_id"]
            )
            group = grouped[mode, attempt["iteration"], attempt["candidate_id"]]
            require(
                [row["observation_step"] for row in group]
                == list(range(0, attempt["actions_executed"], 5)),
                "Generation observation steps drift",
            )
            require(
                len(group) == attempt["policy_replans"]
                and attempt["velocity_evaluations"] == 10 * len(group),
                "Rollout generation counts differ",
            )
        require(len(attempts) == len(grouped), "Unpaired generation group")
        requests = {
            row["request"]["request_fingerprint"]: row["request"]
            for row in self.rows
            if row["kind"] == "intervention_request"
        }
        providers = [
            json.loads(line)
            for name, content in extra.items()
            if name.endswith("_provider.jsonl")
            for line in content.splitlines()
            if line
        ]
        require(
            len(providers)
            == len(requests)
            == len({row["request_fingerprint"] for row in providers}),
            "Provider request coverage mismatch",
        )
        accepted, usage_available = 0, 0
        for row in providers:
            request = requests[row["request_fingerprint"]]
            require(
                digest({k: v for k, v in request.items() if k != "request_fingerprint"})
                == row["request_fingerprint"],
                "Request fingerprint mismatch",
            )
            require(
                row["arm"] == request["arm"]
                and row["iteration"] == request["iteration"],
                "Provider arm/iteration mismatch",
            )
            require(
                row["requested_model"] == "azure/openai/gpt-6-astra"
                and row["sampling_settings"]
                == {"max_completion_tokens": 4096, "reasoning_effort": "low"},
                "Provider setting mismatch",
            )
            require(
                row["token_usage"]
                == normalize_usage(row.get("response", {}).get("usage")),
                "Provider usage inconsistency",
            )
            usage_available += row["token_usage"]["total_tokens"] is not None
            if row["accepted"]:
                accepted += 1
                require(
                    row["http_status"] == 200
                    and row["response"]["model"] == row["requested_model"],
                    "Actual provider model mismatch",
                )
                proposal = parse_proposal(
                    row["response"]["choices"][0]["message"]["content"], request
                )
                require(
                    proposal == self.proposals[row["arm"], row["candidate_id"]],
                    "Provider text differs from candidate",
                )
        for arm, value in self.summary["arms"].items():
            require(
                value["status"] == "complete" and len(value["attempts"]) <= 5,
                "Incomplete or over-budget arm",
            )
            curve = success_curve(value["attempts"], 5)
            require(
                all(value["summary"][key] == item for key, item in curve.items()),
                "Success curve disagrees with attempt ledger",
            )
        init_vf = (
            1240 if self.metadata["runtime.json"]["phase"] == "development" else 1230
        )
        require(
            self.initial["velocity_evaluations"] == init_vf, "Initialization cost drift"
        )
        require(
            self.summary["physical_cost"]["velocity_evaluations"]
            == 10 * len(self.generations) + init_vf,
            "Physical solve accounting mismatch",
        )
        probe = self.initial.get("development_embedding_probe")
        probe_metrics = None
        if probe:
            row = next(
                row for row in self.rows if row["kind"] == "development_embedding_probe"
            )
            control = next(
                row
                for row in self.generations
                if row["mode"] == "known_noise" and row["observation_step"] == 0
            )
            require(
                control["condition_id"] == self.initial["condition_id"]
                and np.array_equal(self.value(control["latent"]), known),
                "Probe control does not match initial condition",
            )
            full = self.value(control["generated_actions"])
            require(
                np.array_equal(
                    self.decode(full), self.value(self.initial["native_actions"])
                ),
                "Matched native control parity failed",
            )
            internal = error(full, self.value(row["actions"]))
            require(
                internal == probe["output_difference"] and internal["max_abs"] > 0,
                "Probe output-effect mismatch",
            )
            probe_metrics = {
                "full_internal_difference_recomputed": internal,
                "decoded_control_difference_recomputed": error(
                    self.decode(full), self.decode(self.value(row["actions"]))
                ),
                "simulator_actions": 0,
                "astra_proposal": False,
            }

        def outcome(row):
            return {
                key: row[key]
                for key in ("success", "actions_executed", "policy_replans")
            }

        return {
            "status": "verified",
            "episode_id": self.entry["episode_id"],
            "case": self.name,
            "event_sha256": sha(self.data["events.jsonl"]),
            "summary_sha256": sha(self.data["summary.json"]),
            "arrays_verified": len(self.refs),
            "generations_verified": len(self.generations),
            "rollouts": len(attempts),
            "all_array_condition_reset_provider_bindings_verified": True,
            "initial_errors_recomputed": errors,
            "zero_hook_parity_worker": self.initial["errors"][
                "zero_embedding_hook_parity"
            ],
            "fixed_development_probe": probe_metrics,
            "candidate_text": {
                **self.conditioning_counts,
                "maximum_relative_rms": self.max_text_relative,
            },
            "vision_changed_generations": self.vision_effects,
            "maximum_noise_expression_error": maximum_noise_error,
            "maximum_observed_noise_delta_rms": maximum_noise_rms,
            "provider": {
                "calls": len(providers),
                "accepted": accepted,
                "usage_available_calls": usage_available,
                "http_status": dict(
                    Counter(str(row.get("http_status")) for row in providers)
                ),
            },
            "baseline": outcome(self.summary["baseline"]),
            "controls": {k: outcome(v) for k, v in self.summary["controls"].items()},
            "arms": {
                k: {
                    "first_success_attempt": v["summary"]["first_success_attempt"],
                    "attempts": [
                        {
                            field: a.get(field)
                            for field in (
                                "iteration",
                                "candidate_id",
                                "success",
                                "rollout_executed",
                                "status",
                                "actions_executed",
                            )
                        }
                        for a in v["attempts"]
                    ],
                }
                for k, v in self.summary["arms"].items()
            },
            "limitations": [
                "Array endpoints and controller decoding are recomputed without a model solve.",
                "Prefix embeddings are not stored; token IDs, condition hashes, boundary flags and norm metadata are checked, not independently re-embedded.",
                "Exact alpha0 preflight parity is worker-computed because its endpoint is not separately stored.",
                "Simulator success values are recorded outcomes; the audit does not rerun the environment.",
            ],
        }


def audit_worker(catalog, worker, output, assets):
    import tarfile

    source = Path(__file__).parent
    inventory = json.loads((source / "frozen_source_inventory.json").read_text())
    require(inventory["payload_sha256"] == PAYLOAD, "Frozen source payload mismatch")
    for item in inventory["files"]:
        require(
            sha((source / item["path"]).read_bytes()) == item["sha256"],
            "Frozen audit dependency changed",
        )
    prefix = f"worker_{worker}/"
    archive_key = prefix + "artifacts.tar.gz"
    before = catalog.receipt(archive_key)
    roots = (
        "runtime.json",
        "protocol.json",
        "frozen_plan.json",
        "reset_manifest.json",
        "checkpoint.json",
        "progress.json",
    )
    raw = {name: catalog.read(prefix + name) for name in roots}
    metadata = {name: json.loads(content) for name, content in raw.items()}
    runtime, plan = metadata["runtime.json"], metadata["frozen_plan.json"]
    require(
        runtime["worker"] == worker
        and runtime["payload_sha256"] == PAYLOAD
        and runtime["tf32"] is False,
        "Wrong worker runtime",
    )
    require(metadata["progress.json"]["status"] == "complete", "Worker is incomplete")
    require(
        digest(metadata["protocol.json"]) == plan["protocol_sha256"],
        "Worker protocol hash mismatch",
    )
    manifest = dict(metadata["reset_manifest.json"])
    manifest_hash = manifest.pop("sha256")
    require(
        digest(manifest) == manifest_hash == plan["manifest_sha256"],
        "Reset manifest hash mismatch",
    )
    for name, content in raw.items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        (output / name).write_bytes(content)
    active, extra, results, root_seen = None, {}, [], set()
    validation_seen = False

    def finish_case():
        result = active.finish(extra)
        target = output / active.name
        write_json(target / "array_inventory.json", active.inventory)
        result["array_inventory_sha256"] = sha(
            (target / "array_inventory.json").read_bytes()
        )
        write_json(target / "audit.json", result)
        results.append(result)

    with urllib.request.urlopen(catalog.url(archive_key), timeout=60) as response:
        reader = HashReader(response)
        with gzip.GzipFile(fileobj=reader, mode="rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    require(
                        member.name.startswith("results/")
                        and ".." not in Path(member.name).parts,
                        "Unsafe tar member",
                    )
                    name = member.name[len("results/") :]
                    case_name = name.split("/", 1)[0]
                    if case_name.startswith("case_"):
                        if active is None or active.name != case_name:
                            if active is not None:
                                finish_case()
                            data = {
                                item: catalog.read(prefix + case_name + "/" + item)
                                for item in ("events.jsonl", "summary.json")
                            }
                            active = CaseAudit(case_name, metadata, data, assets)
                            extra = {}
                        relative = name[len(case_name) + 1 :]
                        if relative.endswith(".npy"):
                            active.array(relative, archive.extractfile(member).read())
                        elif relative in active.data:
                            content = archive.extractfile(member).read()
                            require(
                                content == active.data[relative],
                                "Published case record differs from final archive",
                            )
                            target = output / case_name / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(content)
                        elif relative.endswith("_provider.jsonl"):
                            extra[relative] = archive.extractfile(member).read()
                    elif name in raw:
                        require(
                            archive.extractfile(member).read() == raw[name],
                            "Root record differs from final archive",
                        )
                        root_seen.add(name)
                    elif name == "development_validation.json":
                        validation = json.loads(archive.extractfile(member).read())
                        require(
                            all(
                                value
                                for arms in validation.values()
                                for value in arms.values()
                            ),
                            "Development arm validation failed",
                        )
                        validation_seen = True
            for _ in iter(lambda: compressed.read(1024 * 1024), b""):
                pass
        for _ in iter(lambda: reader.read(1024 * 1024), b""):
            pass
    if active is not None:
        finish_case()
    after = catalog.receipt(archive_key)
    require(before == after, "Archive changed during streaming audit")
    require(
        reader.bytes == int(after["content_range"].split("/")[-1]),
        "Compressed archive length mismatch",
    )
    require(root_seen == set(raw), "Missing root archive records")
    require(
        runtime["phase"] != "development" or validation_seen,
        "Missing development arm validation",
    )
    require(
        sorted(row["episode_id"] for row in results)
        == sorted(plan["assigned_episodes"]),
        "Assigned case coverage mismatch",
    )
    report = {
        "status": "complete_verified",
        "workflow": runtime["workflow"],
        "worker": worker,
        "payload_sha256": PAYLOAD,
        "archive": {
            "sha256": reader.hash.hexdigest(),
            "bytes": reader.bytes,
            "receipt_before": before,
            "receipt_after": after,
            "gzip_verified": True,
        },
        "cases": results,
        "source_sha256": sha(Path(__file__).read_bytes()),
        "source_inventory_sha256": sha(
            Path(__file__).with_name("frozen_source_inventory.json").read_bytes()
        ),
        "no_new_model_or_api_calls": True,
    }
    write_json(output / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        result = audit_worker(
            Catalog(args.catalog), args.worker, Path(args.output), Path(args.assets)
        )
    except Exception as exc:
        # urllib exceptions may contain signed URLs: never print exception text.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exception": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                    "check": str(exc) if type(exc) is ValueError else None,
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "status": result["status"],
                "worker": args.worker,
                "cases": len(result["cases"]),
                "archive_sha256": result["archive"]["sha256"],
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
