"""The submit script's hydra overrides must not parse as SWEEPS.

A comma inside an override VALUE turns it into a sweep — a list of values —
which hydra rejects outside `--multirun`:

    ConfigCompositionException: Ambiguous value for argument
    'description=... | train tight,loose,laggy,sticky | ...'

The rejection comes from `validate_sweep_overrides_legal`, not from parsing, so
the string parses fine and only blows up during config composition. That
happens in PHASE 1 — after the image pull, uv sync, the R2 pull and staging,
roughly 40 minutes in — and presents as a training failure.

Bash quoting does not help: the value reaches hydra intact and hydra is the one
that objects.

This is a CLI-composition error, not a config error, so instantiating configs
locally cannot catch it — `tests/test_control_mode_configs.py` passed on the
node minutes before this failed. Hence a separate gate that asks hydra's own
parser.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts/control_modes/submit.sh"


def _dry_run_overrides() -> list[str]:
    """Every `key=value` the submit script would hand to `osmo --set`."""
    out = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run", "small", "large"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    # The dry run prints a shell-quoted command; recover the --set pairs.
    pairs = []
    for line in out.stdout.splitlines():
        if "osmo workflow submit" not in line:
            continue
        after = line.split("--set", 1)[1] if "--set" in line else ""
        # Unescape the shell quoting printf %q produced.
        after = after.replace("\\ ", " ").replace("\\|", "|").replace("\\,", ",")
        after = after.replace("\\'", "'").replace("\\+", "+")
        pairs.extend(re.findall(r"(?:^|\s)([a-z_]+=[^\s].*?)(?=\s+[a-z_]+=|$)",
                                after))
    assert pairs, "no --set overrides recovered from the dry run"
    return pairs


def test_submit_script_dry_run_has_no_commas_in_any_value():
    """The cheap, blunt version of the check the script itself now enforces."""
    for pair in _dry_run_overrides():
        key, _, value = pair.partition("=")
        assert "," not in value, (
            f"comma in override value for {key!r}: hydra will read it as a "
            f"list and abort composition in phase 1 -> {pair!r}")


def test_no_submitted_override_parses_as_a_sweep():
    """Ask hydra's own parser rather than reasoning about its grammar.

    `is_sweep_override()` is the predicate that actually matters:
    `parse_overrides` accepts a comma happily and returns a sweep, and it is
    `validate_sweep_overrides_legal` that then refuses it in single-run mode.
    """
    parser = pytest.importorskip(
        "hydra.core.override_parser.overrides_parser"
    ).OverridesParser.create()

    for pair in _dry_run_overrides():
        # `run_desc` etc. are osmo template params, but they are interpolated
        # verbatim into hydra overrides inside the launcher, so check them all.
        parsed = parser.parse_overrides([pair])
        sweeps = [o for o in parsed if o.is_sweep_override()]
        assert not sweeps, (
            f"{pair!r} parses as a hydra SWEEP; illegal without --multirun and "
            f"fatal during phase-1 config composition")


def test_the_known_bad_string_is_still_detected():
    """Guards the gate itself.

    A check that cannot fail is not a check — if hydra's behaviour changed and
    commas became benign, this would tell us rather than the gate quietly
    passing everything forever.
    """
    parser = pytest.importorskip(
        "hydra.core.override_parser.overrides_parser"
    ).OverridesParser.create()
    bad = ("description=ARM2 CONTROL bidirectional | "
           "train tight,loose,laggy,sticky | holdout ideal,jittery")
    parsed = parser.parse_overrides([bad])
    assert any(o.is_sweep_override() for o in parsed), (
        "the comma-as-sweep behaviour this gate exists for is no longer "
        "reproducible; re-derive the gate before trusting it")
