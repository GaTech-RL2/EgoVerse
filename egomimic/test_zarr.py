"""Compatibility entry point for episode validation.

Use ``python -m egomimic.rldb.zarr.validate`` for validation reports and options.
The legacy Python function retains its ``(errors, successes)`` return format.
"""

from egomimic.rldb.zarr.validate import OK, main
from egomimic.rldb.zarr.validate import validate_episode as _validate_episode


def validate_episode(zarr_path: str) -> tuple[list[str], list[str]]:
    """Return error and passing-check messages from the canonical validator.

    Use the canonical report API to inspect warnings and delivery eligibility.
    """
    report = _validate_episode(zarr_path)
    return (
        [str(finding) for finding in report.errors],
        [str(finding) for finding in report.findings if finding.level == OK],
    )


if __name__ == "__main__":
    raise SystemExit(main())
