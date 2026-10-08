"""The ``~/.egoverse_env`` loader, kept free of cloud dependencies.

This lives here rather than in ``egomimic.utils.aws.aws_data_utils`` so that a
purely local run (``LocalEpisodeResolver``, no S3/SQL) can still pick up the
env file -- ``WANDB_API_KEY`` in particular -- without importing boto3 /
cloudpathlib. ``aws_data_utils`` re-exports ``load_env`` so every existing
caller and monkeypatch target keeps working.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path


def load_env(path="~/.egoverse_env", required: bool = False):
    p = Path(path).expanduser()
    if not p.exists():
        if required:
            raise ValueError(
                f"Env file {p} does not exist, run ./egomimic/utils/aws/setup_secret.sh"
            )
        warnings.warn(
            f"Env file {p} does not exist; AWS/R2 env vars not set. "
            "Run ./egomimic/utils/aws/setup_secret.sh if you need S3/R2.",
            UserWarning,
            stacklevel=2,
        )
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
