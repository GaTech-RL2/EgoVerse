"""Audited native compatibility fix for the pinned RoboCasa counter sampler.

The upstream set contains XML Element objects, whose hashes depend on process
addresses. Preserve first occurrence instead so an environment seed addresses a
stable list of the same eligible placement geometries. The patch is explicit in
source and runtime receipts; no other upstream source change is accepted.
"""

import argparse
import hashlib
from pathlib import Path

from .common import write_json

COUNTER_PATH = "robocasa/models/fixtures/counter.py"
BEFORE_SHA256 = "77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a"
AFTER_SHA256 = "cc72124e9d05b39f0f50617c9f94ed5abe73ab6205636eb42ea367dfda521a74"
BEFORE = b"valid_geoms = list(set(valid_geoms))"
AFTER = b"valid_geoms = list(dict.fromkeys(valid_geoms))"


def patch_receipt():
    return {
        "name": "stable-counter-region-order-v1",
        "path": COUNTER_PATH,
        "upstream_sha256": BEFORE_SHA256,
        "patched_sha256": AFTER_SHA256,
        "change": "Preserve first occurrence when deduplicating candidate XML geometry objects before seeded region sampling.",
    }


def verify_fixture_order_fix(root):
    if (
        hashlib.sha256((Path(root) / COUNTER_PATH).read_bytes()).hexdigest()
        != AFTER_SHA256
    ):
        raise ValueError("robocasa_fixture_order_fix_missing_or_changed")
    return patch_receipt()


def apply_fixture_order_fix(root):
    path = Path(root) / COUNTER_PATH
    source = path.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    if sha == AFTER_SHA256:
        return patch_receipt()
    if sha != BEFORE_SHA256 or source.count(BEFORE) != 1:
        raise ValueError("robocasa_counter_source_not_pinned")
    patched = source.replace(BEFORE, AFTER)
    if hashlib.sha256(patched).hexdigest() != AFTER_SHA256:
        raise ValueError("robocasa_counter_patch_hash_mismatch")
    path.write_bytes(patched)
    return verify_fixture_order_fix(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    write_json(args.receipt, apply_fixture_order_fix(args.root))


if __name__ == "__main__":
    main()
