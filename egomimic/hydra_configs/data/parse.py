from __future__ import annotations

from pathlib import Path


def episode_hashes_from_sample(sample_path: str | Path | None = None) -> list[str]:
    """
    Load episode hashes from `sample.txt` (one per line).

    Ignores blank lines and lines starting with '#'.
    """
    if sample_path is None:
        sample_path = Path(__file__).with_name("sample.txt")
    else:
        sample_path = Path(sample_path)

    hashes: list[str] = []
    seen: set[str] = set()
    for line in sample_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            continue
        seen.add(s)
        hashes.append(s)

    return hashes
