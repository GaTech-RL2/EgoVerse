"""Split the Mecka private set's IDs out of a generated Hydra config.

The repo is public and Mecka's episode and operator IDs are not, so every ID in a
``mecka_private_*`` config becomes a ``${mecka_private:<config>,<key>}`` reference
(egomimic/utils/hydra_resolvers.py) and the values go to
``$EGOVERSE_MECKA_PRIVATE_DIR/splits/<config>.json``. IDs in comments are dropped.
"""

import json
import os
import re
from pathlib import Path

ID = r"[0-9a-f]{24}"
_ID = re.compile(rf"\b{ID}\b")
_KEY = re.compile(r"^(\s*)([A-Za-z_]\w*):(.*)$")
_ITEM = re.compile(rf"^\s*- ({ID})\s*(#.*)?$")
_FLOW = re.compile(rf"^\[\s*{ID}(\s*,\s*{ID})*\s*\]$")

PRIVATE_DIR_ENV = "EGOVERSE_MECKA_PRIVATE_DIR"
ZARR_DIR = "${oc.env:" + PRIVATE_DIR_ENV + "}/zarr"


def private_dir() -> Path:
    return Path(os.environ[PRIVATE_DIR_ENV])


def _strip_comment_ids(line: str) -> str:
    code, sep, comment = line.partition("#")
    if not sep or not _ID.search(comment):
        return line
    comment = re.sub(r"\s+", " ", _ID.sub("", comment)).rstrip()
    return (code + sep + comment).rstrip() if comment.strip() else code.rstrip()


def split_ids(text: str, name: str) -> tuple[str, dict]:
    """Return ``text`` with every ID replaced by a resolver reference, and the IDs by key."""
    lines = text.splitlines()
    out, ids, stack, counts = [], {}, [], {}

    def ref(key):
        return "${mecka_private:" + name + "," + key + "}"

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _KEY.match(line)
        if m:
            indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, key))
            path = ".".join(k for _, k in stack)
            value = rest.split("#")[0].strip()
            if not value:
                j, items = i + 1, []
                while j < len(lines) and (im := _ITEM.match(lines[j])):
                    items.append(im.group(1))
                    j += 1
                if items:
                    if j < len(lines) and lines[j].lstrip().startswith("- "):
                        raise ValueError(f"{name}: {path} mixes IDs with other items")
                    ids[path] = items
                    out.append(f"{m.group(1)}{key}: {ref(path)}")
                    i = j
                    continue
            elif _FLOW.match(value):
                ids[path] = _ID.findall(value)
                out.append(f"{m.group(1)}{key}: {ref(path)}")
                i += 1
                continue
        else:
            path = ".".join(k for _, k in stack)
        code, sep, comment = line.partition("#")
        if _ID.search(code):

            def sub(match):
                n = counts.get(path, 0)
                counts[path] = n + 1
                ids[f"{path}.{n}"] = match.group(0)
                return ref(f"{path}.{n}")

            code = _ID.sub(sub, code)
        out.append(_strip_comment_ids(code + sep + comment))
        i += 1
    public = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if _ID.search(public):
        raise ValueError(f"{name}: an ID survived the split")
    return public, ids


def write_config(path: Path, text: str, splits_dir: Path) -> None:
    """Write the ID-free config to ``path`` and its IDs to ``splits_dir/<stem>.json``."""
    public, ids = split_ids(text, path.stem)
    splits = splits_dir / f"{path.stem}.json"
    if ids:
        splits_dir.mkdir(parents=True, exist_ok=True)
        splits.write_text(json.dumps(ids, indent=1) + "\n")
    else:
        splits.unlink(missing_ok=True)
    path.write_text(public)
