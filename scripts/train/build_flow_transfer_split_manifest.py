#!/usr/bin/env python3
"""Build and verify the exact episode-level Flow Transfer split manifest.

This mirrors ``split_dataset_names`` in ``zarr_dataset_multi.py`` without
opening every Zarr array.  ChainGripper identifiers are namespaced by source
index exactly as ``LocalEpisodeResolverManyWithEmbodimentOverride`` does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def episode_records(root: Path, prefix: str = "") -> list[dict[str, str]]:
    root = root.resolve()
    assert root.is_dir(), root
    records: list[dict[str, str]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or not path.name.endswith(".zarr"):
            continue
        name = path.name[: -len(".zarr")]
        records.append(
            {
                "id": f"{prefix}{name}",
                "path": str(path.absolute()),
                "realpath": str(path.resolve()),
            }
        )
    return records


def split_records(
    records: list[dict[str, str]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assert 0.0 <= valid_ratio <= 1.0
    by_id = {record["id"]: record for record in records}
    assert len(by_id) == len(records), "duplicate logical episode identifiers"
    names = sorted(by_id)
    random.Random(seed).shuffle(names)
    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)
    valid_ids = set(names[:n_valid])
    train_ids = set(names[n_valid:])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids | valid_ids == set(names)
    train = [by_id[name] for name in sorted(train_ids)]
    valid = [by_id[name] for name in sorted(valid_ids)]
    train_paths = {record["realpath"] for record in train}
    valid_paths = {record["realpath"] for record in valid}
    assert train_paths.isdisjoint(valid_paths), "train/valid realpath overlap"
    return train, valid


def inventory_sha(records: list[dict[str, str]]) -> str:
    raw = "".join(
        f"{record['id']}\t{record['path']}\t{record['realpath']}\n"
        for record in records
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usocket-root", type=Path, required=True)
    parser.add_argument(
        "--chain-root", type=Path, action="append", required=True
    )
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-usocket", type=int, required=True)
    parser.add_argument(
        "--expected-chain-source", type=int, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert len(args.chain_root) == len(args.expected_chain_source)

    usocket = episode_records(args.usocket_root)
    assert len(usocket) == args.expected_usocket, (
        len(usocket),
        args.expected_usocket,
    )

    chain_by_source: list[list[dict[str, str]]] = []
    for index, (root, expected) in enumerate(
        zip(args.chain_root, args.expected_chain_source)
    ):
        records = episode_records(root, prefix=f"source_{index:03d}/")
        assert len(records) == expected, (root, len(records), expected)
        chain_by_source.append(records)
    chain = [record for source in chain_by_source for record in source]

    u_train, u_valid = split_records(usocket, args.valid_ratio, args.seed)
    c_train, c_valid = split_records(chain, args.valid_ratio, args.seed)

    payload = {
        "status": "PASS",
        "algorithm": "sorted identifiers; random.Random(seed).shuffle; floor ratio",
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "roots": {
            "usocket": str(args.usocket_root.absolute()),
            "chain": [str(path.absolute()) for path in args.chain_root],
        },
        "counts": {
            "usocket_total": len(usocket),
            "usocket_train": len(u_train),
            "usocket_valid": len(u_valid),
            "chain_total": len(chain),
            "chain_train": len(c_train),
            "chain_valid": len(c_valid),
            "chain_by_source": [len(source) for source in chain_by_source],
        },
        "inventory_sha256": {
            "usocket": inventory_sha(usocket),
            "chain": inventory_sha(chain),
        },
        "overlap": {
            "usocket_ids": 0,
            "usocket_realpaths": 0,
            "chain_ids": 0,
            "chain_realpaths": 0,
        },
        "splits": {
            "usocket": {"train": u_train, "valid": u_valid},
            "chain": {"train": c_train, "valid": c_valid},
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        "SPLIT_PASS "
        f"u={len(u_train)}/{len(u_valid)} "
        f"chain={len(c_train)}/{len(c_valid)} output={args.output}"
    )


if __name__ == "__main__":
    main()
