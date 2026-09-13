"""Apply / verify openpi's ``transformers_replace`` overlay.

openpi's PyTorch pi0 needs four patched transformers modules (gemma config +
modeling, paligemma modeling, siglip modeling) and a marker
``models/siglip/check.py`` copied over the transformers version that marker
names. Any transformers reinstall (a lock change under ``uv sync``,
``--reinstall``, a fresh venv) restores the originals. ``PI.__init__`` calls
:func:`check` and its error names the fix:

    python -m egomimic.openpi_patch apply

``apply`` writes each file to a temp file beside its target and renames it over
the target: uv hardlinks site-packages into its cache, and writing through the
link would patch the cached wheel for every project.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import re
import stat
import sys
import tempfile
import warnings
from pathlib import Path

APPLY_CMD = "python -m egomimic.openpi_patch apply"
INSTALL_CMD = "uv pip install --no-deps -e external/openpi"
MARKER = Path("models/siglip/check.py")


def source_dir() -> Path:
    """``transformers_replace`` inside the installed openpi package."""
    spec = importlib.util.find_spec("openpi")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"openpi is not installed; run: {INSTALL_CMD}")
    pkg = Path(next(iter(spec.submodule_search_locations)))
    return pkg / "models_pytorch" / "transformers_replace"


def target_dir() -> Path:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("transformers is not installed")
    return Path(spec.origin).parent


def installed_transformers_version() -> str | None:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def patch_files(src: Path) -> list[Path]:
    files = sorted(p.relative_to(src) for p in src.rglob("*.py"))
    if not files:
        raise RuntimeError(
            f"openpi's transformers_replace overlay is missing or empty at {src}; "
            f"reinstall openpi: {INSTALL_CMD}"
        )
    return files


def required_transformers(src: Path) -> str:
    """The transformers version the overlay targets, as its marker checks it."""
    marker = src / MARKER
    found = marker.is_file() and re.search(
        r"__version__\s*==\s*[\"']([^\"']+)[\"']", marker.read_text()
    )
    if not found:
        raise RuntimeError(f"no transformers version check found in {marker}")
    return found.group(1)


def _check_version(src: Path) -> None:
    required, version = required_transformers(src), installed_transformers_version()
    if version != required:
        raise RuntimeError(
            f"openpi needs transformers=={required}, found {version}; "
            "run `uv sync --inexact` (see README, Installation)"
        )


def status(src: Path, dst: Path) -> tuple[list[Path], list[Path]]:
    """``(stale, linked)``: overlay files missing from or differing in the target,
    and applied ones whose target is still a hardlink or symlink."""
    stale, linked = [], []
    for rel in patch_files(src):
        target = dst / rel
        if not (target.is_file() and target.read_bytes() == (src / rel).read_bytes()):
            stale.append(rel)
        elif target.is_symlink() or target.stat().st_nlink > 1:
            linked.append(rel)
    return stale, linked


def _names(rels: list[Path]) -> str:
    return ", ".join(map(str, rels[:3])) + (" ..." if len(rels) > 3 else "")


def _warn_linked(linked: list[Path]) -> None:
    # Replacing the venv file cannot help: the other link, normally uv's cache
    # patched by the old `cp -r` step, keeps serving the patched bytes to every
    # later transformers install.
    if linked:
        warnings.warn(
            f"{len(linked)} applied overlay file(s) still share their inode with "
            f"another path, most likely uv's cache, which then holds openpi's "
            f"patch ({_names(linked)}). Run once: uv cache clean transformers",
            stacklevel=3,
        )


def _replace(src_file: Path, target: Path) -> None:
    """Atomically replace ``target`` (never writing through a link), keeping its mode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE((target if target.is_file() else src_file).stat().st_mode)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(src_file.read_bytes())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def apply(src: Path | None = None, dst: Path | None = None) -> list[Path]:
    """Write every stale overlay file into transformers; returns what was written."""
    src = src or source_dir()
    dst = dst or target_dir()
    stale, linked = status(src, dst)
    _check_version(src)
    for rel in stale:
        _replace(src / rel, dst / rel)
    _warn_linked(linked)
    return stale


def check(src: Path | None = None, dst: Path | None = None) -> None:
    """Raise ``RuntimeError`` unless transformers is the overlay's version with
    the full overlay applied; warn if the uv cache looks patched too."""
    src = src or source_dir()
    dst = dst or target_dir()
    stale, linked = status(src, dst)
    _check_version(src)
    if stale:
        raise RuntimeError(
            f"openpi's transformers_replace overlay is missing or stale for "
            f"{len(stale)} file(s): {_names(stale)}. Run: {APPLY_CMD}"
        )
    _warn_linked(linked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m egomimic.openpi_patch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("apply", help="copy the overlay into the installed transformers")
    sub.add_parser("check", help="exit 1 unless the overlay is fully applied")
    args = parser.parse_args(argv)
    code = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if args.command == "apply":
                changed = apply()
                for rel in changed:
                    print(f"patched {rel}")
                print(f"{len(changed)} file(s) updated in {target_dir()}")
            else:
                check()
                print(f"openpi transformers_replace overlay: OK ({target_dir()})")
        except (RuntimeError, ModuleNotFoundError) as e:
            print(e)
            code = 1
    for w in caught:
        print(f"WARNING: {w.message}")
    return code


if __name__ == "__main__":
    sys.exit(main())
