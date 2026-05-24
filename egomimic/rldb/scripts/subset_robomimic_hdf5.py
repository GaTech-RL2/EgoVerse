#!/usr/bin/env python
"""Create a smaller robomimic-style HDF5 dataset by copying demo groups."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py


def demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-demos", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "w") as dst:
        if "data" not in src:
            raise KeyError(f"{args.input} does not contain a top-level 'data' group")

        copy_attrs(src.attrs, dst.attrs)

        for key in src.keys():
            if key != "data":
                src.copy(key, dst, name=key)

        dst_data = dst.create_group("data")
        copy_attrs(src["data"].attrs, dst_data.attrs)

        demo_names = sorted(src["data"].keys(), key=demo_sort_key)
        selected = demo_names[: args.num_demos]
        if len(selected) != args.num_demos:
            raise ValueError(
                f"requested {args.num_demos} demos, found only {len(selected)} in {args.input}"
            )

        total = 0
        for demo_name in selected:
            src["data"].copy(demo_name, dst_data, name=demo_name)
            total += int(src["data"][demo_name].attrs.get("num_samples", 0))

        if "total" in dst_data.attrs:
            dst_data.attrs["total"] = total

    print(f"Wrote {args.output} with {len(selected)} demos")


if __name__ == "__main__":
    main()
