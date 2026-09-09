#!/usr/bin/env python3
"""Validate the deterministic UNITE save/requeue/resume integration trace."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TERMINAL_CHECKPOINT_RE = re.compile(
    r"integration-terminal-epoch(?:=|-)(?P<epoch>[0-9]+)-step(?:=|-)(?P<step>[0-9]+)\.ckpt"
)


def normalize_restart_count(value: str | None) -> int:
    """Normalize only Slurm's absent first-attempt value to exact zero."""

    if value is None:
        return 0
    if not value or re.fullmatch(r"[0-9]+", value, flags=re.ASCII) is None:
        raise ValueError("restart count must be ASCII base-10 digits or absent")
    return int(value)


def terminal_checkpoint_step(name: str) -> int:
    match = TERMINAL_CHECKPOINT_RE.fullmatch(name)
    if match is None:
        raise ValueError("terminal checkpoint filename is not canonical")
    return int(match.group("step"))


def expected_plan(
    *, seed_step: int = 3, pre_requeue_steps: int = 5, resumed_steps: int = 3
) -> dict[str, int]:
    for label, value in (
        ("seed_step", seed_step),
        ("pre_requeue_steps", pre_requeue_steps),
        ("resumed_steps", resumed_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    signal_step = seed_step + pre_requeue_steps
    terminal_step = signal_step + resumed_steps
    return {
        "seed_step": seed_step,
        "signal_step": signal_step,
        "terminal_step": terminal_step,
        # Lightning logs the validation batch before the terminal optimizer
        # step is reflected in its displayed global step.
        "validation_logged_step": terminal_step - 1,
        "validation_interval": terminal_step,
    }


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def validate_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a trace unless it proves the exact runner-owned sequence."""

    required = {
        "schema_version",
        "seed_step",
        "first_attempt_optimizer_steps",
        "signal_checkpoint_step",
        "step_after_signal_executed",
        "save_acknowledged_ranks",
        "world_size",
        "slurm_restarts",
        "resume_checkpoint_step",
        "resumed_optimizer_steps",
        "validation_logged_step",
        "terminal_checkpoint",
        "child_complete",
        "complete_sentinel",
    }
    if not isinstance(trace, Mapping) or set(trace) != required:
        raise ValueError("integration trace keys are not canonical")
    if trace["schema_version"] != 1:
        raise ValueError("unsupported integration trace schema")
    plan = expected_plan(seed_step=_integer(trace["seed_step"], "seed step"))
    if trace["first_attempt_optimizer_steps"] != list(
        range(plan["seed_step"] + 1, plan["signal_step"] + 1)
    ):
        raise ValueError("first attempt did not execute exactly five new steps")
    if trace["signal_checkpoint_step"] != plan["signal_step"]:
        raise ValueError("save-only checkpoint was not published at step 8")
    if trace["step_after_signal_executed"] is not False:
        raise ValueError("optimizer step after the signal boundary executed")
    world_size = _integer(trace["world_size"], "world size")
    if world_size < 1 or trace["save_acknowledged_ranks"] != list(range(world_size)):
        raise ValueError("not every rank acknowledged save completion")
    if trace["slurm_restarts"] != 1:
        raise ValueError("integration requires exactly one real Slurm restart")
    if trace["resume_checkpoint_step"] != plan["signal_step"]:
        raise ValueError("restart did not select the exact signal checkpoint")
    if trace["resumed_optimizer_steps"] != list(
        range(plan["signal_step"] + 1, plan["terminal_step"] + 1)
    ):
        raise ValueError("restart did not execute exactly three new steps")
    if trace["validation_logged_step"] != plan["validation_logged_step"]:
        raise ValueError("validation ran before all three resumed steps")
    if terminal_checkpoint_step(str(trace["terminal_checkpoint"])) != plan[
        "terminal_step"
    ]:
        raise ValueError("terminal checkpoint does not represent step 11")
    if trace["child_complete"] is not True or trace["complete_sentinel"] is not True:
        raise ValueError("integration completion receipts are absent")
    return {"schema_version": 1, "status": "PASS", **plan, "world_size": world_size}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    print(json.dumps(validate_trace(trace), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
