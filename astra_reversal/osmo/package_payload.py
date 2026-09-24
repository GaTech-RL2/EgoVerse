"""Package this experiment plus its authorized tokenizer, excluding model weights."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-runtime-probe", action="store_true")
    parser.add_argument("--include-ood-inputs", action="store_true")
    parser.add_argument("--include-ood-manifests", action="store_true")
    args = parser.parse_args()
    root = Path("astra_reversal")
    output = root / ".deps/osmo-upload/payload.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)

    def include(info):
        excluded = {".deps", "artifacts", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return None if excluded.intersection(Path(info.name).parts) else info

    with tarfile.open(output, "w:gz") as archive:
        archive.add(root, arcname="astra_reversal", filter=include)
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
    (root / ".deps/osmo-payload.sha256").write_text(digest + "\n")
    print(
        json.dumps(
            {"archive": str(output), "bytes": output.stat().st_size, "sha256": digest}
        )
    )


if __name__ == "__main__":
    main()
