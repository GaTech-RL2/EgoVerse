"""Encode exact string values for Hydra command-line overrides.

Use this module instead of interpolating paths directly into an override.  In
particular, an unquoted ``=`` in a checkpoint path is meaningful to Hydra's
override parser and can change or invalidate the override.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Sequence

_KEY_PATTERN = re.compile(r"(?:\+\+|\+)?[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*\Z")
_UNSAFE_SEPARATOR_CATEGORIES = frozenset({"Zl", "Zp"})


def _validate_value(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("Hydra override value must be a string")
    if "${" in value:
        raise ValueError("Hydra override value must not contain interpolation '${'")

    for character in value:
        category = unicodedata.category(character)
        if category == "Cc" or category in _UNSAFE_SEPARATOR_CATEGORIES:
            raise ValueError(
                "Hydra override value contains an unsafe control or line-separator "
                f"character U+{ord(character):04X}"
            )


def quote_hydra_string(value: str) -> str:
    """Return a JSON-quoted string adapted to Hydra's quoted-value grammar.

    Hydra uses JSON-style double quotes and ``\"`` for an embedded quote, but
    unlike a JSON decoder it preserves ordinary internal backslashes.  It only
    unescapes backslashes immediately before a quote.  Starting from
    :func:`json.dumps` and normalizing only those ordinary internal runs keeps
    all characters byte-for-character identical after Hydra/OmegaConf parsing.

    Values containing control characters or ``${`` are rejected rather than
    escaped because either would make a generated command unsafe or change the
    resolved OmegaConf identity.
    """

    _validate_value(value)
    json_quoted = json.dumps(value, ensure_ascii=False)
    inner = json_quoted[1:-1]
    quoted_parts = ['"']
    index = 0

    while index < len(inner):
        if inner[index] != "\\":
            quoted_parts.append(inner[index])
            index += 1
            continue

        run_end = index
        while run_end < len(inner) and inner[run_end] == "\\":
            run_end += 1
        run_length = run_end - index

        # JSON and Hydra use the same spelling for a run before an embedded
        # quote and for a trailing run before the closing quote.  Elsewhere,
        # JSON doubles each literal backslash while Hydra preserves it.
        if run_end == len(inner) or inner[run_end] == '"':
            quoted_parts.append("\\" * run_length)
        else:
            if run_length % 2 != 0:  # pragma: no cover - json.dumps invariant
                raise AssertionError("unexpected JSON backslash escape")
            quoted_parts.append("\\" * (run_length // 2))
        index = run_end

    quoted_parts.append('"')
    return "".join(quoted_parts)


def encode_hydra_string_override(key: str, value: str) -> str:
    """Return one exact ``key="value"`` Hydra override token.

    ``key`` may be a dot-separated config key and may start with Hydra's ``+``
    or ``++`` add prefixes.  Config-group overrides are intentionally outside
    this helper's string-scalar contract.
    """

    if not isinstance(key, str):
        raise TypeError("Hydra override key must be a string")
    if _KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(
            "Hydra override key must be a dot-separated scalar key, optionally "
            "prefixed by '+' or '++'"
        )
    return f"{key}={quote_hydra_string(value)}"


def main(argv: Sequence[str] | None = None) -> int:
    """Print one encoded override token for shell command substitution."""

    parser = argparse.ArgumentParser(
        description="Encode an exact string as one Hydra CLI override token."
    )
    parser.add_argument("key", help="Dot-separated Hydra scalar key")
    parser.add_argument("value", help="Exact string value to encode")
    args = parser.parse_args(argv)
    try:
        override = encode_hydra_string_override(args.key, args.value)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
