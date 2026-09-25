"""Publish a completed report with only declared local path prefixes normalized.

The exact absolute-prefix map is kept in --private-receipt. Public provenance
contains its digest, prefix digests, replacement counts, and before/after hashes.
No model, simulator, provider, network, or report-generation call is made.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

FILES = {
    "report.json",
    "report.md",
    "curves.csv",
    "cases.csv",
    "decisions.csv",
    "controls.csv",
    "success_by_budget.png",
    "conditional_rescue_by_budget.png",
    "reported_tokens_by_budget.png",
}
STRING = re.compile(rb'"(?:[^"\\]|\\.)*"')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def receipt(data):
    return {"sha256": sha(data), "bytes": len(data)}


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def normalize(data, prefixes):
    """Patch only JSON string tokens; retain all other lexical bytes exactly."""
    original = decode(data)
    changes = []

    def replacement(value):
        for index, prefix in enumerate(prefixes):
            if value.startswith(prefix):
                return value[len(prefix) :], index
        require(
            not any(prefix in value for prefix in prefixes),
            "Local prefix occurs inside a non-path string",
        )
        require(
            not value.startswith(("/Users/", "/home/")), "Unmapped absolute local path"
        )
        return value, None

    def walk(value, pointer=""):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                require(
                    replacement(key)[1] is None,
                    "Local-root object key needs explicit review",
                )
                result[key] = walk(
                    item, pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                )
            return result
        if isinstance(value, list):
            return [
                walk(item, pointer + f"/{index}") for index, item in enumerate(value)
            ]
        if isinstance(value, str):
            new, index = replacement(value)
            if index is not None:
                changes.append(
                    {
                        "json_pointer": pointer,
                        "prefix_index": index,
                        "original_value": value,
                        "public_value": new,
                    }
                )
            return new
        return value

    expected = walk(original)
    patched_tokens = 0

    def patch(match):
        nonlocal patched_tokens
        value = decode(match.group())
        new, index = replacement(value)
        if index is None:
            return match.group()
        patched_tokens += 1
        return json.dumps(new, ensure_ascii=True).encode()

    public = STRING.sub(patch, data)
    require(
        patched_tokens == len(changes) and decode(public) == expected,
        "JSON normalization changed content beyond path strings",
    )
    return public, changes


def publish(
    source, output, private_receipt, local_roots, phase, expected_report_sha256=None
):
    source, output, private_receipt = (
        Path(path).resolve() for path in (source, output, private_receipt)
    )
    require(
        not output.exists() and not private_receipt.exists(),
        "Publication output or private receipt already exists",
    )
    require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "Input/output directories overlap",
    )
    require(
        not private_receipt.is_relative_to(source)
        and not private_receipt.is_relative_to(output),
        "Private receipt must be outside input and public output directories",
    )
    prefixes = sorted(
        {str(Path(root).absolute()).rstrip("/") + "/" for root in local_roots},
        key=lambda value: (-len(value), value),
    )
    require(
        prefixes and all(prefix != "/" for prefix in prefixes),
        "Declare specific local repository roots",
    )
    require(
        {path.name for path in source.iterdir()} == FILES | {"manifest.json"},
        "Unexpected or missing generated report files",
    )
    original = {
        name: (source / name).read_bytes() for name in FILES | {"manifest.json"}
    }
    manifest = decode(original["manifest.json"])
    report = decode(original["report.json"])
    expected_cases, seed = (3, 19) if phase == "development" else (20, 29)
    require(phase in ("development", "evaluation"), "Declare report phase explicitly")
    require(
        report["schema_version"] == "astra-phase-interpolation-report-1"
        and report["status"] == "complete"
        and report["phase"] == phase
        and report["expected_cases"] == expected_cases
        and report["seed"] == seed,
        "Report is incomplete or belongs to another phase",
    )
    require(
        len(report["cases"])
        == len({row["episode_id"] for row in report["cases"]})
        == expected_cases,
        "Wrong complete-case coverage",
    )
    require(
        manifest["schema"] == "astra-phase-interpolation-report-files-1"
        and manifest["phase"] == phase
        and set(manifest["files"]) == FILES,
        "Unexpected generated file manifest",
    )
    require(
        manifest["postprocessor"] == report["postprocessor"],
        "Reporter identity differs between report and manifest",
    )
    for name in FILES:
        require(
            manifest["files"][name] == receipt(original[name]),
            "Original report file fails its completed manifest",
        )
    if expected_report_sha256 is not None:
        require(
            sha(original["report.json"]) == expected_report_sha256,
            "Report differs from the expected frozen checksum",
        )
    public, changes = {}, {}
    for name in FILES | {"manifest.json"}:
        if name.endswith(".json"):
            public[name], changes[name] = normalize(original[name], prefixes)
        else:
            require(
                not any(prefix.encode() in original[name] for prefix in prefixes),
                "Local prefix outside JSON needs explicit review",
            )
            public[name], changes[name] = original[name], []
    require(any(changes.values()), "No declared local path prefix was found")
    rebuilt = decode(public["manifest.json"])
    rebuilt["files"] = {name: receipt(public[name]) for name in sorted(FILES)}
    public["manifest.json"] = encoded(rebuilt)
    require(
        rebuilt["postprocessor"] == decode(public["report.json"])["postprocessor"],
        "Published reporter identity differs",
    )
    helper = Path(__file__).read_bytes()
    private = {
        "schema_version": "phase-report-private-path-map-1.0",
        "input_directory": str(source),
        "output_directory": str(output),
        "phase": phase,
        "exact_path_prefix_map": [{"from": value, "to": ""} for value in prefixes],
        "changes": changes,
        "original_files": {
            name: receipt(data) for name, data in sorted(original.items())
        },
        "helper_sha256": sha(helper),
    }
    private_data = encoded(private)
    per_file = {
        name: {
            "original": receipt(original[name]),
            "public": receipt(public[name]),
            "byte_identical": original[name] == public[name],
            "path_values_changed": len(changes[name]),
            "transform": "rebuild_file_checksums_after_path_normalization"
            if name == "manifest.json"
            else "declared_path_prefix_normalization"
            if changes[name]
            else "verbatim_copy",
        }
        for name in sorted(original)
    }
    provenance = {
        "schema_version": "phase-report-publication-1.0",
        "phase": phase,
        "private_mapping_receipt_sha256": sha(private_data),
        "private_mapping_receipt_bytes": len(private_data),
        "path_prefix_map": [
            {
                "prefix_index": index,
                "source_prefix_sha256": sha(prefix.encode()),
                "source_prefix_utf8_bytes": len(prefix.encode()),
                "replacement": "",
                "path_values_changed": sum(
                    change["prefix_index"] == index
                    for rows in changes.values()
                    for change in rows
                ),
                "meaning": "absolute local repository root plus slash becomes a repository-relative path",
            }
            for index, prefix in enumerate(prefixes)
        ],
        "files": per_file,
        "path_values_changed": sum(len(rows) for rows in changes.values()),
        "report_path_values_changed": len(changes["report.json"]),
        "byte_identical_original_files": sum(
            row["byte_identical"] for row in per_file.values()
        ),
        "rebuilt_manifest_entries": sum(
            receipt(original[name]) != receipt(public[name]) for name in FILES
        ),
        "source_snapshot": {"name": "source/publish_report.py.txt", **receipt(helper)},
        "validation": {
            "original_file_manifest_verified": True,
            "json_diff_is_only_declared_path_values": True,
            "metrics_and_source_artifact_hashes_unchanged": True,
            "untouched_files_copied_verbatim": True,
            "private_inputs_unchanged": True,
        },
        "limitations": [
            "Publication changes provenance paths, not measurements or the prior audit's scope.",
            "The exact local prefix map and original full path values remain in the private receipt; its SHA-256 binds the public prefix digests and changed counts.",
            "manifest.json is rebuilt from public report-file bytes; original manifest bytes are identified by their original receipt.",
        ],
    }
    public["source/publish_report.py.txt"] = helper
    public["publication.json"] = encoded(provenance)
    require(
        not any(
            prefix.encode() in data for data in public.values() for prefix in prefixes
        ),
        "Local prefix remains in public output",
    )
    for name, data in original.items():
        require(
            (source / name).read_bytes() == data,
            "Private report changed during publication",
        )
    require(
        Path(__file__).read_bytes() == helper,
        "Publication helper changed while running",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    private_receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".report-publication-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        for name, data in public.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        require(not output.exists(), "Publication destination appeared during work")
        with os.fdopen(
            os.open(private_receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        ) as stream:
            stream.write(private_data)
        staging.rename(output)
    return {
        "phase": phase,
        "original_report_sha256": sha(original["report.json"]),
        "public_report_sha256": sha(public["report.json"]),
        "publication_sha256": sha(public["publication.json"]),
        "private_mapping_receipt_sha256": sha(private_data),
        "path_values_changed": provenance["path_values_changed"],
        "byte_identical_original_files": provenance["byte_identical_original_files"],
        "helper_sha256": sha(helper),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--local-root", action="append", required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--expected-report-sha256")
    args = parser.parse_args()
    result = publish(
        args.input,
        args.output,
        args.private_receipt,
        args.local_root,
        args.phase,
        args.expected_report_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
