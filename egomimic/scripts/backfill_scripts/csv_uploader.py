#!/usr/bin/env python3
"""
CSV group uploader

Scans a directory (recursively) for per-recording CSV metadata files named
with a suffix like `_meta.csv`, e.g.:

  fold_cloth_eth_scene_1_recording_1_meta.csv
  fold_cloth_eth_scene_1_recording_1.vrs
  fold_cloth_eth_scene_1_recording_1.vrs.json

For each CSV found the uploader will:
 - read the first non-empty CSV row and map columns to TableRow fields
 - generate a metadata JSON (episode row) using CSV values where present
 - upload the metadata JSON and the corresponding .vrs and .vrs.json files to S3

This script builds on the existing `Uploader` in `abstract_upload.py` but
is fully non-interactive: CSV values are used to populate metadata fields
and missing fields are left empty/defaulted. Only files that have a matching
.vrs (and optionally .vrs.json) are uploaded.

Usage: run without flags and you'll be prompted for the directory and
embodiment. Example:

  python3 egomimic/scripts/csv_uploader.py

"""

from __future__ import annotations

import asyncio
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from abstract_upload import Uploader

from egomimic.utils.aws.aws_sql import TableRow


class CSVGroupUploader(Uploader):
    """Uploader that sources metadata from per-recording CSV files.

    csv_map maps a recording base name -> dict of header->value
    groups is a list of tuples: (vrs_path: Path, json_path: Optional[Path], csv_path: Path)
    """

    def __init__(
        self,
        embodiment: str,
        datatype: str,
        csv_map: Dict[str, Dict],
        groups: List[Tuple[Path, Optional[Path], Path]],
    ):
        # collect_files will be provided but we still pass a placeholder; we'll set local_dir externally
        super().__init__(
            embodiment=embodiment, datatype=datatype, collect_files=lambda _: []
        )
        self.csv_map = csv_map
        self.groups = groups

    def collect_files(self, local_dir: Path):
        # return list of tuples (vrs, json) expected by the base run() loop
        out = []
        for vrs, jsn, csvp in self.groups:
            if jsn is not None and jsn.exists():
                out.append((vrs, jsn))
            else:
                out.append((vrs,))
        return out

    def collect_metadata(self, file_path: Path):
        """Build metadata JSON from the CSV row for this recording.

        file_path is the main file (the .vrs file) passed from the run loop.
        """
        # Determine base name to lookup in csv_map
        base = file_path.stem
        # handle .vrs suffix already included in stem; for .vrs files, stem is without extension
        # try direct match then fallback to variations
        row = self.csv_map.get(base)
        if row is None:
            # try without known suffix patterns
            for k in list(self.csv_map.keys()):
                if base.startswith(k) or k.startswith(base):
                    row = self.csv_map[k]
                    break

        # Build metadata using TableRow fields
        submitted_metadata = {"embodiment": self.embodiment, "episode_hash": ""}
        for key in self.metadata_keys:
            val = row.get(key) if row else None

            # Coerce types for a few well-known fields
            if key == "objects":
                # Ensure objects is always a list
                if isinstance(val, list):
                    submitted_metadata[key] = val
                elif val is None or val == "":
                    submitted_metadata[key] = []
                else:
                    # If a string slipped through, split on commas; otherwise wrap single value
                    if isinstance(val, str) and "," in val:
                        submitted_metadata[key] = [
                            v.strip() for v in val.split(",") if v.strip()
                        ]
                    else:
                        submitted_metadata[key] = [val] if val is not None else []
            elif key in ("num_frames",):
                try:
                    submitted_metadata[key] = int(val) if val not in (None, "") else ""
                except Exception:
                    submitted_metadata[key] = ""
            elif key in ("eval_score",):
                try:
                    submitted_metadata[key] = (
                        float(val) if val not in (None, "") else ""
                    )
                except Exception:
                    submitted_metadata[key] = ""
            elif key in ("is_deleted", "is_eval", "eval_success"):
                if val is None or val == "":
                    submitted_metadata[key] = ""
                else:
                    v = str(val).strip().lower()
                    submitted_metadata[key] = v in ("1", "true", "yes", "y", "t")
            else:
                submitted_metadata[key] = val if val is not None else ""

        # Ensure ALL TableRow fields are present in the metadata JSON (user requested)
        for fld in TableRow.__dataclass_fields__.keys():
            if fld not in submitted_metadata:
                if fld == "objects":
                    submitted_metadata[fld] = []
                elif fld == "episode_hash":
                    submitted_metadata[fld] = ""
                elif fld == "embodiment":
                    submitted_metadata[fld] = self.embodiment
                else:
                    submitted_metadata[fld] = ""

        # Use file timestamp as episode_hash (same behavior as base Uploader)
        timestamp_ms = int(self.get_timestamp_name(file_path))
        submitted_metadata["episode_hash"] = timestamp_ms
        submitted_metadata["embodiment"] = self.embodiment

        # Hardcode task/robot/task_description as requested
        submitted_metadata["task_description"] = "sorting utensils"
        submitted_metadata["robot_name"] = "aria_bimanual"
        submitted_metadata["task"] = "sort utensils"

        # Write JSON tempfile
        metadata_tempfile = tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".json"
        )
        json.dump(submitted_metadata, metadata_tempfile, indent=2)
        metadata_tempfile.close()
        return Path(metadata_tempfile.name)


def find_groups(
    root: Path,
) -> Tuple[Dict[str, Dict], List[Tuple[Path, Optional[Path], Path]]]:
    """Scan root for *_meta.csv files and pair them with .vrs and .vrs.json files.

    Returns (csv_map, groups)
    csv_map: base -> row dict
    groups: list of (vrs_path, json_path_or_None, csv_path)
    """
    csv_map: Dict[str, Dict] = {}
    groups: List[Tuple[Path, Optional[Path], Path]] = []

    # alias map: map common CSV headers to TableRow fields
    alias_map = {
        "collector": "operator",
        "clothes_info": "objects",
        "clothes": "objects",
        "scene_number": "scene",
        "recording_number": "recording_number",
        "arm": "arm",
        # add more aliases here as needed
    }

    for csv_path in sorted(root.rglob("*_meta.csv")):
        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                row = None
                for r in reader:
                    # take first non-empty row
                    if any(
                        (v is not None and str(v).strip() != "") for v in r.values()
                    ):
                        row = {
                            k.strip(): (v.strip() if isinstance(v, str) else v)
                            for k, v in r.items()
                            if k
                        }
                        break
                if row is None:
                    # empty csv; skip
                    continue
                # normalize headers using alias_map and coerce certain fields
                norm_row = {}
                for k, v in row.items():
                    key = k.strip()
                    mapped = alias_map.get(key, key)
                    # normalize objects into a list immediately
                    if mapped == "objects":
                        if v is None or (isinstance(v, str) and v.strip() == ""):
                            norm_row[mapped] = []
                        elif isinstance(v, list):
                            norm_row[mapped] = v
                        else:
                            # split on commas, but treat single token as single-item list
                            if isinstance(v, str) and "," in v:
                                norm_row[mapped] = [
                                    x.strip() for x in v.split(",") if x.strip()
                                ]
                            else:
                                norm_row[mapped] = (
                                    [v.strip()] if isinstance(v, str) else [v]
                                )
                    else:
                        norm_row[mapped] = v
                row = norm_row
        except Exception:
            continue

        # derive base name (remove trailing _meta)
        name = csv_path.stem
        if name.endswith("_meta"):
            base = name[:-5]
        else:
            # fallback: use stem
            base = name

        # map csv keys to TableRow keys where possible (e.g. collector -> operator)
        mapped_row = {}
        for k, v in row.items():
            # if header already matches a TableRow field, keep it
            if k in TableRow.__dataclass_fields__:
                mapped_row[k] = v
            else:
                # common patterns mapping
                if k == "operator":
                    mapped_row["operator"] = v
                elif k == "lab":
                    mapped_row["lab"] = v
                elif k == "clothes_info" or k == "clothes" or k == "objects":
                    mapped_row["objects"] = v
                elif k == "scene" or k == "scene_number":
                    mapped_row["scene"] = v
                elif k == "collector":
                    mapped_row["operator"] = v
                elif k == "arm":
                    # include arm info in task_description (concise)
                    prev = mapped_row.get("task_description", "")
                    mapped_row["task_description"] = (prev + " " + str(v)).strip()
                else:
                    # store unknowns as-is under their header name for potential use
                    mapped_row[k] = v

        csv_map[base] = mapped_row

        # find vrs file with same base (may have extension already). Search for files matching base.* with .vrs
        vrs_candidate = None
        for ext in (".vrs", ".VRS"):
            p = csv_path.with_name(base + ext)
            if p.exists():
                vrs_candidate = p
                break
        if vrs_candidate is None:
            # try find any file that startswith base and endswith .vrs (case-insensitive)
            for p in csv_path.parent.iterdir():
                if (
                    p.is_file()
                    and p.suffix.lower() == ".vrs"
                    and p.stem.startswith(base)
                ):
                    vrs_candidate = p
                    break

        if vrs_candidate is None:
            print(
                f"Skipping CSV {csv_path} — no matching .vrs file found for base '{base}'"
            )
            continue

        # expected json companion: <name>.vrs.json
        json_candidate = vrs_candidate.with_suffix(vrs_candidate.suffix + ".json")
        if not json_candidate.exists():
            # warn but still allow (user asked to upload vrs and json if present)
            json_candidate = None

        groups.append((vrs_candidate, json_candidate, csv_path))

    return csv_map, groups


def prompt_for(prompt: str, default: Optional[str] = None) -> str:
    while True:
        resp = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        if resp:
            return resp
        if default is not None:
            return default


def main():
    print(
        "CSV group uploader — finds per-recording *_meta.csv files and uploads the matching vrs/json files using CSV-provided metadata."
    )

    dirpath = prompt_for("Directory to scan for recordings (root)")
    root = Path(dirpath).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Directory not found: {root}")
        return

    embodiment = prompt_for("Embodiment name (e.g. aria, eve)")
    datatype = prompt_for("Main file extension (e.g. .vrs, .hdf5)", default=".vrs")

    print(f"Scanning {root} for *_meta.csv files...")
    csv_map, groups = find_groups(root)
    if not groups:
        print("No recording groups found (no *_meta.csv with matching .vrs). Exiting.")
        return

    print(f"Found {len(groups)} recording group(s). Preparing uploader...")

    uploader = CSVGroupUploader(
        embodiment=embodiment, datatype=datatype, csv_map=csv_map, groups=groups
    )
    # set directory so uploader won't prompt for it
    uploader.local_dir = root
    uploader.directory_prompted = True
    # avoid batch interactive prompt in run(); we supply per-file metadata from CSV
    uploader.batch_metadata_asked = True

    # prevent the Uploader.set_directory prompt (we already set local_dir)
    uploader.set_directory = lambda: None

    # bind collect_files to call the instance method with the desired root
    uploader.collect_files = lambda _: CSVGroupUploader.collect_files(uploader, root)

    asyncio.run(uploader.run())


if __name__ == "__main__":
    main()
