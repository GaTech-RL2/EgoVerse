"""Normalize a pushshapes zarr dataset to a target dtype / length / schema.

Like ``rechunk_zarr_dataset.py`` but with three extra transformations:

* ``--trim-to-total-frames`` slices every array's leading dim down to
  ``attrs["total_frames"]`` (strips the historical zero-pad tail that
  the old ``ZarrWriter`` baked in for chunk alignment).
* ``--upcast-numeric-to {f32,f64}`` converts every non-image numeric
  array to the requested dtype (only ``upcast`` is supported — narrowing
  is rejected because it would break replay determinism).
* ``--drop-keys k1,k2`` skips the listed top-level array/group keys
  entirely (useful for matching schemas across datasets).

Writes one-chunk-per-array (same as the rechunk script), preserves all
attrs, and verifies a sample of random frames per array.  Source is
never mutated; output goes to ``--dst``.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/normalize_zarr_dataset.py \\
        --src /coc/.../circle/basic \\
        --dst /coc/.../circle/basic_normalized \\
        --trim-to-total-frames --upcast-numeric-to f64 --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthBytes


_DTYPE_MAP = {"f32": np.float32, "f64": np.float64}
_DTYPE_RANK = {"f32": 32, "f64": 64}


def _numeric_target_dtype(src_dtype: np.dtype, upcast_to: str | None) -> np.dtype:
    """Return the dtype to use for a numeric array given the upcast flag."""
    if upcast_to is None:
        return src_dtype
    if not np.issubdtype(src_dtype, np.floating):
        return src_dtype
    src_bits = np.dtype(src_dtype).itemsize * 8
    tgt_bits = _DTYPE_RANK[upcast_to]
    if tgt_bits < src_bits:
        raise ValueError(
            f"refusing to narrow {src_dtype} -> {upcast_to}; only upcast is allowed"
        )
    return _DTYPE_MAP[upcast_to]


def _copy_array(
    src_arr: zarr.Array,
    dst_group: zarr.Group,
    name: str,
    *,
    trim_to: int | None,
    upcast_to: str | None,
) -> None:
    src_shape = tuple(src_arr.shape)
    if trim_to is not None and len(src_shape) >= 1:
        new_lead = min(int(trim_to), src_shape[0]) if src_shape[0] > 0 else 0
        shape = (new_lead,) + src_shape[1:]
    else:
        shape = src_shape

    if len(shape) == 0:
        new_chunks = ()
    else:
        new_chunks = (max(shape[0], 1),) + shape[1:]

    is_vlb = isinstance(src_arr.metadata.data_type, VariableLengthBytes)
    if is_vlb:
        dtype = VariableLengthBytes()
    else:
        dtype = _numeric_target_dtype(src_arr.dtype, upcast_to)

    create_kwargs = dict(name=name, shape=shape, chunks=new_chunks, dtype=dtype)
    if not is_vlb:
        create_kwargs["fill_value"] = src_arr.fill_value
    dst_arr = dst_group.create_array(**create_kwargs)

    if shape and shape[0] > 0:
        data = src_arr[: shape[0]]
        if not is_vlb and upcast_to is not None and data.dtype != dtype:
            data = data.astype(dtype, copy=False)
        dst_arr[:] = data

    for k, v in dict(src_arr.attrs).items():
        dst_arr.attrs[k] = v


def _copy_group(
    src_group: zarr.Group,
    dst_group: zarr.Group,
    *,
    trim_to: int | None,
    upcast_to: str | None,
    drop_keys: frozenset[str],
) -> None:
    for k, v in dict(src_group.attrs).items():
        dst_group.attrs[k] = v

    for name in src_group.array_keys():
        if name in drop_keys:
            continue
        _copy_array(src_group[name], dst_group, name, trim_to=trim_to, upcast_to=upcast_to)

    for name in src_group.group_keys():
        if name in drop_keys:
            continue
        child = dst_group.create_group(name)
        _copy_group(src_group[name], child, trim_to=trim_to, upcast_to=upcast_to, drop_keys=drop_keys)


def _verify(
    src_path: Path,
    dst_path: Path,
    *,
    n_samples: int,
    trim_to: int | None,
    upcast_to: str | None,
    drop_keys: frozenset[str],
) -> None:
    """For each kept array, sample frames in the kept range and compare.

    Accounts for both trimming (only compares first ``trim_to`` frames)
    and upcasting (compares src.astype(target) == dst for numeric).
    """
    src = zarr.open_group(str(src_path), mode="r")
    dst = zarr.open_group(str(dst_path), mode="r")
    rng = np.random.default_rng(seed=hash(str(src_path)) & 0xFFFFFFFF)

    def _walk(s, d, prefix=""):
        for name in s.array_keys():
            if name in drop_keys:
                if name in d.array_keys():
                    raise AssertionError(f"dropped {prefix}{name} still present in dst")
                continue
            s_arr, d_arr = s[name], d[name]
            kept = (
                min(trim_to, s_arr.shape[0]) if (trim_to is not None and s_arr.shape) else (s_arr.shape[0] if s_arr.shape else 0)
            )
            if d_arr.shape and d_arr.shape[0] != kept:
                raise AssertionError(
                    f"{prefix}{name}: dst leading dim {d_arr.shape[0]} != expected {kept}"
                )
            if kept == 0:
                continue
            k = min(n_samples, kept)
            idxs = rng.choice(kept, size=k, replace=False)
            for i in map(int, idxs):
                a = s_arr[i : i + 1]
                b = d_arr[i : i + 1]
                if a.dtype == object:
                    if bytes(a[0]) != bytes(b[0]):
                        raise AssertionError(f"vlb mismatch {prefix}{name}[{i}]")
                else:
                    if upcast_to is not None and np.issubdtype(a.dtype, np.floating):
                        a_cmp = a.astype(b.dtype, copy=False)
                    else:
                        a_cmp = a
                    if not np.array_equal(a_cmp, b):
                        raise AssertionError(f"data mismatch {prefix}{name}[{i}]")
        for name in s.group_keys():
            if name in drop_keys:
                continue
            _walk(s[name], d[name], prefix=f"{prefix}{name}/")

    _walk(src, dst)


def _normalize_one(args) -> tuple[Path, str | None]:
    src_path, dst_path, verify_samples, trim_flag, upcast_to, drop_keys = args
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    try:
        if dst_path.exists():
            return src_path, "skipped: dst already exists"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        src = zarr.open_group(str(src_path), mode="r")
        trim_to = None
        if trim_flag:
            tf = dict(src.attrs).get("total_frames", None)
            if tf is not None:
                trim_to = int(tf)

        dst = zarr.create_group(store=str(tmp_path), overwrite=True)
        _copy_group(src, dst, trim_to=trim_to, upcast_to=upcast_to, drop_keys=drop_keys)
        _verify(src_path, tmp_path, n_samples=verify_samples,
                trim_to=trim_to, upcast_to=upcast_to, drop_keys=drop_keys)
        tmp_path.rename(dst_path)
        return src_path, None
    except Exception as e:
        return src_path, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--verify-samples", type=int, default=3)
    p.add_argument("--pattern", default="*.zarr")
    p.add_argument("--trim-to-total-frames", action="store_true",
                   help="Slice every array's leading dim to attrs['total_frames'].")
    p.add_argument("--upcast-numeric-to", choices=("f32", "f64"), default=None,
                   help="Upcast all non-image float arrays to this dtype (no narrowing).")
    p.add_argument("--drop-keys", default="",
                   help="Comma-separated top-level array/group names to skip in the output.")
    args = p.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    if not src_root.is_dir():
        print(f"--src not a dir: {src_root}", file=sys.stderr); return 2

    episodes = sorted(p for p in src_root.glob(args.pattern) if p.is_dir())
    if not episodes:
        print(f"no episodes matched {src_root}/{args.pattern}", file=sys.stderr); return 2

    drop_keys = frozenset(k for k in args.drop_keys.split(",") if k)
    dst_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (ep, dst_root / ep.name, args.verify_samples, args.trim_to_total_frames,
         args.upcast_numeric_to, drop_keys)
        for ep in episodes
    ]
    print(
        f"normalizing {len(jobs)} episodes  {src_root} -> {dst_root}",
        flush=True,
    )
    print(
        f"  trim_to_total_frames={args.trim_to_total_frames}  "
        f"upcast_numeric_to={args.upcast_numeric_to}  "
        f"drop_keys={sorted(drop_keys)}  workers={args.workers}",
        flush=True,
    )

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    fails: list[tuple[Path, str]] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (sp, err) in enumerate(pool.imap_unordered(_normalize_one, jobs), start=1):
            if err is None:
                n_ok += 1
            elif err.startswith("skipped"):
                n_skip += 1
            else:
                n_fail += 1
                fails.append((sp, err))
                print(f"  FAIL {sp.name}: {err}", flush=True)
            if i % 25 == 0 or i == len(jobs):
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1e-9)
                eta = (len(jobs) - i) / max(rate, 1e-9)
                print(
                    f"  [{i}/{len(jobs)}] ok={n_ok} fail={n_fail} skip={n_skip} "
                    f"| {rate:.1f} ep/s | eta {eta/60:.1f} min",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed/60:.1f} min  ok={n_ok}  fail={n_fail}  skip={n_skip}")
    if fails:
        print("\nFailures:", file=sys.stderr)
        for sp, err in fails[:20]:
            print(f"  {sp.name}: {err}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
