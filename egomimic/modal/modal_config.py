"""Re-exports from run.py for backwards compatibility.

All Modal config (image, app, zarr_volume, CFG, REPO_ROOT) now lives in
run.py so that file is fully self-contained when Modal mounts it before
the repo is cloned.
"""
from egomimic.modal.run import CFG, REPO_ROOT, app, zarr_volume  # noqa: F401
