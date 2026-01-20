#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def iter_hdf5_files(root: Path, recursive: bool):
  if root.is_file() and root.suffix in [".hdf5", ".h5"]:
    yield root
    return
  if not root.is_dir():
    raise FileNotFoundError(f"Path not found: {root}")

  if recursive:
    for p in root.rglob("*.hdf5"):
      yield p
    for p in root.rglob("*.h5"):
      yield p
  else:
    for p in root.glob("*.hdf5"):
      yield p
    for p in root.glob("*.h5"):
      yield p


def swap_rgb_bgr_in_dataset(dset: h5py.Dataset, batch: int):
  shape = dset.shape
  if len(shape) != 4 or shape[-1] != 3:
    return 0
  # Expect uint8 images; still works for other dtypes, but this is the common case.
  n = shape[0]
  if n == 0:
    return 0

  for i in range(0, n, batch):
    j = min(i + batch, n)
    x = dset[i:j]                # (B, H, W, 3)
    x = x[..., ::-1]             # swap channels RGB <-> BGR
    dset[i:j] = x

  return n


def find_image_groups(f: h5py.File):
  # Common layout: /observations/images/{front_img_1,left_wrist_img,right_wrist_img}
  groups = []
  if "observations" in f and "images" in f["observations"]:
    groups.append(f["observations"]["images"])

  # Also handle multi-demo layout: /demo_*/observations/images/...
  def visitor(name, obj):
    if isinstance(obj, h5py.Group) and name.endswith("observations/images"):
      groups.append(obj)

  f.visititems(visitor)

  # Dedup by object id
  uniq = []
  seen = set()
  for g in groups:
    oid = g.id.__hash__()
    if oid not in seen:
      seen.add(oid)
      uniq.append(g)
  return uniq


def process_file(path: Path, batch: int, backup: bool):
  if backup:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
      shutil.copy2(path, bak)

  with h5py.File(path, "r+") as f:
    img_groups = find_image_groups(f)
    if not img_groups:
      print(f"[skip] {path} (no observations/images group found)")
      return

    total_frames = 0
    total_dsets = 0
    for g in img_groups:
      for k in list(g.keys()):
        dset = g[k]
        if not isinstance(dset, h5py.Dataset):
          continue
        changed = swap_rgb_bgr_in_dataset(dset, batch=batch)
        if changed > 0:
          total_frames += changed
          total_dsets += 1

    print(f"[ok] {path} (swapped {total_dsets} dataset(s), {total_frames} frame(s))")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--path", type=str, required=True, help="File or directory containing .h5/.hdf5")
  ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
  ap.add_argument("--batch", type=int, default=64, help="Frames per write batch")
  ap.add_argument("--no-backup", action="store_true", help="Disable .bak backup")
  args = ap.parse_args()

  root = Path(args.path)
  files = list(iter_hdf5_files(root, recursive=args.recursive))
  if not files:
    raise FileNotFoundError(f"No .h5/.hdf5 files found under: {root}")

  for fp in files:
    process_file(fp, batch=args.batch, backup=(not args.no_backup))


if __name__ == "__main__":
  main()
