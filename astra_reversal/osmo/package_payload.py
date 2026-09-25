"""Package this experiment plus its authorized tokenizer, excluding model weights."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

TEST_PATHS = (
    Path("tests/unit/astra"),
    Path("tests/integration/test_astra_lerobot_policy.py"),
    Path("tests/integration/test_astra_interpolation_policy.py"),
    Path("tests/fixtures/astra/lerobot_pi05"),
    Path("tests/fixtures/astra/intervention_audit"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-runtime-probe", action="store_true")
    parser.add_argument("--include-ood-inputs", action="store_true")
    parser.add_argument("--include-ood-manifests", action="store_true")
    parser.add_argument("--include-astra-proposal-replay", action="store_true")
    parser.add_argument("--include-interpolation-banks", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path("astra_reversal")
    output = args.output or root / ".deps/osmo-upload/payload.tar.gz"
    if args.output and output.exists():
        raise FileExistsError(
            "An explicit immutable payload destination already exists"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    def include(info):
        excluded = {".deps", "artifacts", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return None if excluded.intersection(Path(info.name).parts) else info

    with tarfile.open(output, "w:gz") as archive:
        archive.add(root, arcname="astra_reversal", filter=include)
        # Include only this experiment's relocated tests and small native fixture.
        # The standalone worker does not need the repository-wide pytest hooks.
        for path in TEST_PATHS:
            archive.add(path, arcname=str(path), filter=include)
        tokenizer = root / ".deps/tokenizers/paligemma-3b-pt-224"
        inventory = json.loads(
            (root / "checkpoints/paligemma_tokenizer.json").read_text()
        )
        for entry in inventory["files"]:
            path = tokenizer / entry["path"]
            archive.add(path, arcname=str(path))
        for relative in (
            "reference/cpu_libero_probe.npz",
            "reference/pi05_libero/norm_stats.json",
            "reference/pi05_libero/paligemma_tokenizer.model",
        ):
            path = root / ".deps" / relative
            archive.add(path, arcname=str(path))
        if args.include_runtime_probe:
            path = root / ".deps/runtime-probe-input"
            if not (path / "manifest.json").is_file():
                raise FileNotFoundError(
                    "Prepare the verified development probe input first"
                )
            archive.add(path, arcname=str(path))
        if args.include_interpolation_banks:
            path = root / ".deps/interpolation-inputs/bank_inventory.json"
            archive.add(path, arcname=str(path))
        if args.include_astra_proposal_replay:
            from astra_reversal.osmo.astra_proposal_replay import load_inputs

            path = root / ".deps/astra-dev-proposal-replay-input"
            load_inputs(path)
            archive.add(path, arcname=str(path))
        if args.include_ood_inputs or args.include_ood_manifests:
            # Only explicitly named public experiment artifacts are packaged;
            # credentials and presigned download catalogs stay excluded.
            path = root / ".deps/ood-inputs"
            for filename in (
                "libero_goal_ood_manifest.json",
                "libero_spatial_ood_manifest.json",
            ):
                archive.add(path / filename, arcname=str(path / filename))
        if args.include_ood_inputs:
            path = root / ".deps/ood-inputs/runtime_diagnostics.json"
            archive.add(path, arcname=str(path))
            proposal = root / ".deps/astra-proposal-input"
            for filename in ("request.json", "response.json", "provider.jsonl"):
                archive.add(proposal / filename, arcname=str(proposal / filename))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = (
        output.parent / "payload.sha256"
        if args.output
        else root / ".deps/osmo-payload.sha256"
    )
    checksum.write_text(digest + "\n")
    print(
        json.dumps(
            {"archive": str(output), "bytes": output.stat().st_size, "sha256": digest}
        )
    )


if __name__ == "__main__":
    main()
