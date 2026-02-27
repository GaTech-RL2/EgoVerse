#!/usr/bin/env python

"""
Utility script to combine selected successful demos from multiple robomimic-style
HDF5 files into a single HDF5 file.

Assumptions (explicit by design):
- Each source HDF5 uses robomimic layout with a top-level "data" group containing
  episode groups named "demo_<int>" (e.g. "demo_0").
- The runs listed in RUNS below live under DATA_ROOT and are stored as either
  "<run_name>.hdf5" or "<run_name>.h5".
- For runs without an explicit "demos" dict, all groups under "data" whose names
  start with "demo_" are treated as part of that run and are copied if the run's
  "use" flag is True.
- For runs with a "demos" dict (e.g. "teleop_1771993878"), only demo groups with
  "use=True" are copied.
- In the combined output file, all selected demos are reindexed sequentially as
  "demo_0", "demo_1", ..., in the order they are encountered based on RUNS.

If any of these assumptions are wrong for your data, adjust RUNS or the logic
below before running.

python egomimic/rldb/scripts/combine_successful_demos.py \
  --data-root /coc/flash7/zhenyang/RBY1_0224_data \
  --output /coc/flash7/zhenyang/RBY1_0224_data/combined_success_demos.hdf5 \
  --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


# Root directory containing the raw HDF5 files.
DATA_ROOT = Path("/coc/flash7/zhenyang/RBY1_0224_data")

# Configuration for which runs / demos to include.
# Notes are kept for human reference; "use" controls behavior.
RUNS: dict[str, dict] = {
    "RBY1_BC_test3_20260225_002620": {
        "status": "fail",
        "use": False,
        "note": "RBY1_BC_test3_20260225_002620 (fail)",
    },
    "RBY1_BC_test3_20260225_002744": {
        "status": "fail",
        "use": False,
        "note": "RBY1_BC_test3_20260225_002744 (fail)",
    },
    "RBY1_BC_test1_20260225_001619": {
        "status": "success",
        "use": True,
        "note": "success with failure recover won't help",
        # No explicit demos dict: copy all demo_* groups under data
    },
    "RBY1_BC_test1_20260225_001425": {
        "status": "success",
        "use": True,
        "note": "success failure recover won't help",
    },
    "teleop_1771994586": {
        "status": "success_base_moved",
        "use": True,
        "note": "success, but base seems moved",
        # Assumed to contain one or more demo_* groups under data
    },
    "teleop_1771993878": {
        "status": "mixed",
        "use": True,
        "note": "demo_0 fail; demo_1 and demo_2 success but base moved",
        # Here we select demos explicitly by ID.
        "demos": {
            "demo_0": {
                "status": "fail",
                "use": False,
                "note": "teleop_1771993878 demo_0 (fail)",
            },
            "demo_1": {
                "status": "success_base_moved",
                "use": True,
                "note": "teleop_1771993878 demo_1 (success but base moved)",
            },
            "demo_2": {
                "status": "success_base_moved",
                "use": True,
                "note": "teleop_1771993878 demo_2 (success but base moved)",
            },
        },
    },
}


def find_hdf5_file(run_name: str, data_root: Path) -> Path:
    """
    Resolve the HDF5 path for a given run name by trying common extensions.
    """
    candidates = [
        data_root / f"{run_name}.hdf5",
        data_root / f"{run_name}.h5",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find HDF5 file for run '{run_name}' in {data_root} "
        f"(tried: {', '.join(str(c) for c in candidates)})"
    )


def collect_selected_demos(
    runs: dict[str, dict], data_root: Path
) -> list[tuple[Path, str]]:
    """
    Build a flat list of (hdf5_path, demo_key) pairs to copy.
    The order in this list defines the ordering in the combined output.
    """
    selected: list[tuple[Path, str]] = []

    for run_name, cfg in runs.items():
        if not cfg.get("use", False):
            continue

        hdf5_path = find_hdf5_file(run_name, data_root)
        demos_cfg = cfg.get("demos")

        if demos_cfg is not None:
            # Explicit per-demo selection.
            for demo_key, demo_meta in demos_cfg.items():
                if demo_meta.get("use", False):
                    selected.append((hdf5_path, demo_key))
        else:
            # No explicit demo list: take all demo_* under data.
            with h5py.File(hdf5_path, "r") as f:
                data_group = f["data"]
                demo_keys = [
                    key for key in data_group.keys() if key.startswith("demo_")
                ]
                for demo_key in sorted(demo_keys):
                    selected.append((hdf5_path, demo_key))

    return selected


def combine_demos_to_single_hdf5(
    selected_demos: list[tuple[Path, str]],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Copy the specified (file, demo_key) groups into a single HDF5 file,
    reindexing them as demo_0, demo_1, ... under a top-level "data" group.
    """
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file {output_path} already exists. "
                f"Use --overwrite to replace it."
            )
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as out_f:
        out_data = out_f.create_group("data")

        next_idx = 0
        for hdf5_path, demo_key in selected_demos:
            with h5py.File(hdf5_path, "r") as in_f:
                in_data = in_f["data"]
                if demo_key not in in_data:
                    raise KeyError(
                        f"Demo key '{demo_key}' not found in file {hdf5_path}"
                    )

                new_demo_name = f"demo_{next_idx}"
                # Copy the entire group (datasets, subgroups, attributes).
                in_data.copy(demo_key, out_data, name=new_demo_name)
                print(f"Copied {hdf5_path.name}::data/{demo_key} -> data/{new_demo_name}")
                next_idx += 1

        print(f"\nWrote {next_idx} demos to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine selected successful demos from multiple robomimic-style "
            "HDF5 files into a single HDF5."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing the source HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "combined_success_demos.hdf5",
        help="Path to output combined HDF5 file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output HDF5 if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = collect_selected_demos(RUNS, args.data_root)
    if not selected:
        raise RuntimeError(
            "No demos selected. Check RUNS configuration and 'use' flags."
        )

    combine_demos_to_single_hdf5(
        selected_demos=selected,
        output_path=args.output,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()

