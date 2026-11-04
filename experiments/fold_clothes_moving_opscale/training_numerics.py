"""Pure numeric-evidence contract shared in behavior with the operations observer.

These helpers mirror the reviewed observer schema and bounded parsing. They are
kept in the pinned training source so workers need not import mutable run-root code.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool)


# These are operational alarms, not optimizer controls or statistical MAD flags.
TRAINING_NUMERIC_LIMIT = 1000.0
TRAINING_LOG_READ_LIMIT = 131072
_NUMERIC_TOKEN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)"
_NUMERIC_END = r"(?=$|[\s,;\]\)}])"
_GRADIENT_VALUE = re.compile(
    r"\[GRAD_NORM_SPIKE\][^\r\n]*?\bgrad_norm=(" + _NUMERIC_TOKEN + ")" + _NUMERIC_END,
    re.IGNORECASE,
)
_LOSS_VALUE = re.compile(
    r"(?:^|[\s{,])[\"']?Train/Loss[\"']?\s*[:=]\s*("
    + _NUMERIC_TOKEN
    + ")"
    + _NUMERIC_END,
    re.IGNORECASE,
)
_STEP_VALUE = re.compile(r"\b(?:global_step|step)\s*[:=]\s*(\d+)\b")


def training_numeric_tail(path):
    """Return whitelisted numeric evidence only, from a bounded complete-line tail.

    The current trainer prints selected MAD gradient diagnostics, not every
    gradient or step loss. Absence of an alarm is not complete numerical coverage.
    """
    try:
        with Path(path).open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(0, size - TRAINING_LOG_READ_LIMIT)
            stream.seek(start)
            raw = stream.read(TRAINING_LOG_READ_LIMIT)
    except OSError:
        return {"status": "UNAVAILABLE", "evidence": []}
    # A partial leading record cannot safely identify which metric it belongs to.
    text = raw.decode(errors="replace")
    if start:
        text = (
            re.split(r"[\r\n]", text, maxsplit=1)[-1]
            if re.search(r"[\r\n]", text)
            else ""
        )
    evidence = {}
    counts = {"raw_gradient": 0, "train_loss": 0}
    for line in text.splitlines():
        for metric, expression in (
            ("raw_gradient", _GRADIENT_VALUE),
            ("train_loss", _LOSS_VALUE),
        ):
            for match in expression.finditer(line):
                counts[metric] += 1
                value = float(match.group(1))
                if math.isfinite(value) and abs(value) < TRAINING_NUMERIC_LIMIT:
                    continue
                if metric in evidence:
                    continue
                step = _STEP_VALUE.search(line)
                evidence[metric] = {
                    "metric": metric,
                    "classification": "magnitude"
                    if math.isfinite(value)
                    else "nonfinite",
                    "value": value if math.isfinite(value) else None,
                    "nonfinite": None
                    if math.isfinite(value)
                    else (
                        "NaN"
                        if math.isnan(value)
                        else ("+Infinity" if value > 0 else "-Infinity")
                    ),
                    "step": int(step.group(1)) if step else None,
                }
    return {
        "status": "READ",
        "bytes_read": len(raw),
        "tail_truncated": start > 0,
        "numeric_records": counts,
        "evidence": list(evidence.values()),
    }


def training_numeric_identity(run, source_commit, root):
    """Bind evidence to a real run and its log under this run root, not its job ID."""
    wandb_id = run.get("wandb_id")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        return None
    if not isinstance(wandb_id, str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,128}", wandb_id
    ):
        return None
    directory = run.get("directory")
    if not isinstance(directory, str):
        return None
    path = Path(directory) / "training.log"
    if not path.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        return None
    return source_commit + ":" + wandb_id, path


def load_training_numeric_latches(root):
    """Fail closed on a damaged receipt rather than erase a previously seen alarm."""
    path = root / "operations/training_numerics.json"
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or not isinstance(
        document.get("incidents"), dict
    ):
        raise ValueError("Invalid training numeric receipt")
    for key, record in document["incidents"].items():
        if not isinstance(record, dict) or set(record) != {
            "source_commit",
            "wandb_id",
            "first_observed_epoch",
            "evidence",
        }:
            raise ValueError("Invalid training numeric identity")
        if key != str(record["source_commit"]) + ":" + str(
            record["wandb_id"]
        ) or not re.fullmatch(r"[0-9a-f]{40}:[a-zA-Z0-9_-]{1,128}", key):
            raise ValueError("Invalid training numeric identity")
        if not numeric(record["first_observed_epoch"]) or not math.isfinite(
            record["first_observed_epoch"]
        ):
            raise ValueError("Invalid training numeric timestamp")
        evidence = record["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 2:
            raise ValueError("Invalid training numeric evidence")
        metrics = set()
        for item in evidence:
            if (
                not isinstance(item, dict)
                or set(item)
                != {"metric", "classification", "value", "nonfinite", "step"}
                or item["metric"] not in ("raw_gradient", "train_loss")
                or item["metric"] in metrics
            ):
                raise ValueError("Invalid training numeric evidence")
            metrics.add(item["metric"])
            if item["step"] is not None and (
                type(item["step"]) is not int or item["step"] < 0
            ):
                raise ValueError("Invalid training numeric step")
            if item["classification"] == "magnitude":
                if (
                    not numeric(item["value"])
                    or not math.isfinite(item["value"])
                    or abs(item["value"]) < TRAINING_NUMERIC_LIMIT
                    or item["nonfinite"] is not None
                ):
                    raise ValueError("Invalid training numeric magnitude")
            elif (
                item["classification"] != "nonfinite"
                or item["value"] is not None
                or item["nonfinite"] not in ("NaN", "+Infinity", "-Infinity")
            ):
                raise ValueError("Invalid training numeric classification")
    return document["incidents"]
