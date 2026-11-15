"""Build an allowlisted, content-addressed source/oracle archive for OSMO."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    package = repo / "reproductions/leact"
    allowed = {".py", ".yaml", ".yml", ".json", ".txt", ".in", ".lock", ".sh", ".md"}
    files = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if set(relative.parts) & {"artifacts", "__pycache__", ".pytest_cache", ".build"}:
            continue
        if path.is_file() and not path.is_symlink() and path.suffix in allowed:
            files.append(path)
    oracle = package / "artifacts/leduc6r2s/oracle.jsonl"
    oracle_meta = oracle.with_suffix(".meta.json")
    for path in [oracle, oracle_meta]:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required immutable oracle input is missing: {path}")
        files.append(path)
    if not (package / "experiment.py").is_file():
        raise FileNotFoundError("experiment.py must be complete before building a launch payload")
    metadata = json.loads(oracle_meta.read_text())
    if hashlib.sha256(oracle.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Oracle content does not match oracle.meta.json")
    sources = {str(p.relative_to(repo)): p.read_bytes() for p in files}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "schema_version": 1, "repository_commit": commit,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in sorted(sources.items())},
    }
    sources["payload-manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed ownership, timestamps and ordering make identical source produce identical archives.
    with args.output.open("wb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(sources.items()):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    member.mode = 0o644
                    member.mtime = 0
                    member.uid = member.gid = 0
                    archive.addfile(member, io.BytesIO(data))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    args.output.with_name("payload-manifest.json").write_bytes(sources["payload-manifest.json"])
    print(json.dumps({"payload": str(args.output.resolve()), "sha256": digest,
                      "bytes": args.output.stat().st_size, "files": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
