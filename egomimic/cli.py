"""`egoverse` console script: episode-format tools.

egoverse validate <path>...   check episode dirs (or dirs of episodes); exit 1 on errors
egoverse schema               print the attrs schema as markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _episode_dirs(path: Path) -> list[Path]:
    if (path / "zarr.json").exists():
        return [path]
    return sorted(
        p for p in path.iterdir() if p.is_dir() and (p / "zarr.json").exists()
    )


def _validate(paths: list[str]) -> int:
    from egomimic.rldb.zarr.schema import validate_episode

    failed = 0
    for raw in paths:
        root = Path(raw)
        dirs = _episode_dirs(root) if root.is_dir() else []
        if not dirs:
            print(f"FAIL {raw}: not an episode directory or a directory of episodes")
            failed += 1
            continue
        for d in dirs:
            name = d.name[:-5] if d.name.endswith(".zarr") else d.name
            try:
                rep = validate_episode(d)
            except Exception as e:  # one unreadable episode must not end the run
                failed += 1
                print(f"FAIL {name}: validator crashed: {type(e).__name__}: {e}")
                continue
            if rep.errors:
                failed += 1
                print(f"FAIL {name}")
                for e in rep.errors:
                    print(f"     error: {e}")
            elif rep.warnings:
                print(f"WARN {name}")
            else:
                print(f"OK   {name}")
            for w in rep.warnings:
                print(f"     warning: {w}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="egoverse",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="validate episode directories")
    v.add_argument("paths", nargs="+")
    sub.add_parser("schema", help="print the episode attrs schema as markdown")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.paths)
    from egomimic.rldb.zarr.schema import schema_markdown

    sys.stdout.write(schema_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
