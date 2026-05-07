"""Copy a per-episode zarr sub-array from each "source" episode into its paired
"destination" episode. Each `--pair dst_hash:src_hash` says "copy from
src_hash into dst_hash". Useful, for example, when human episodes ship without
language annotations and you want to reuse the robot-side language
augmentations for paired analysis.

Usage:
    # copy `annotations` between two episodes
    python -m egomimic.scripts.language_process.copy_pair_annotations \\
        --pair dst_hash:src_hash

    # copy a different sub-array (e.g. `dino.front_1`)
    python -m egomimic.scripts.language_process.copy_pair_annotations \\
        --name dino.front_1 \\
        --pair dst_hash:src_hash

    # cross-folder copy, multiple pairs
    python -m egomimic.scripts.language_process.copy_pair_annotations \\
        --src-root /some/other/dataset \\
        --dst-root /my/dataset \\
        --name annotations \\
        --pair dst1:src1 --pair dst2:src2
"""

import argparse
import os
import shutil

import zarr


def _parse_pair(s):
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--pair must be 'dst_hash:src_hash', got {s!r}"
        )
    return (parts[0], parts[1])


def _entries(path):
    """Length of zarr array at path, or -1 if it can't be read."""
    try:
        return len(zarr.open(path, mode="r")[:])
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/storage/project/r-dxu345-0/agao81/pick_place",
        help=(
            "Folder containing per-episode dirs. Used as both source and "
            "destination unless --src-root / --dst-root are set."
        ),
    )
    parser.add_argument(
        "--src-root",
        default=None,
        help="Override --root for the source side.",
    )
    parser.add_argument(
        "--dst-root",
        default=None,
        help="Override --root for the destination side.",
    )
    parser.add_argument(
        "--name",
        default="annotations",
        help=(
            "Sub-array directory name to copy (e.g. 'annotations', "
            "'dino.front_1', 'images.front_1'). Defaults to 'annotations'."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        help="dst_hash:src_hash; repeat for multiple pairs. Required.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing.",
    )
    args = parser.parse_args()

    if not args.pair:
        parser.error(
            "at least one --pair dst_hash:src_hash is required "
            "(e.g. --pair 2026-04-14-03-39-11-000000:2026-04-14-03-36-19-145000)"
        )

    src_root = args.src_root or args.root
    dst_root = args.dst_root or args.root
    pairs = args.pair

    for dst_hash, src_hash in pairs:
        src = os.path.join(src_root, src_hash, args.name)
        dst = os.path.join(dst_root, dst_hash, args.name)

        if not os.path.exists(src):
            raise SystemExit(f"Source not found: {src}")

        src_n = _entries(src)
        dst_n_before = _entries(dst) if os.path.exists(dst) else 0

        print(
            f"[{'dry' if args.dry_run else 'copy'}] {args.name}: "
            f"{src_hash}({src_n}) -> {dst_hash} (had {dst_n_before})"
        )
        if args.dry_run:
            continue

        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        verify_n = _entries(dst)
        print(f"  verified: n_entries={verify_n}")
        if verify_n != src_n:
            raise SystemExit(f"Mismatch after copy: src={src_n} dst={verify_n}")


if __name__ == "__main__":
    main()
