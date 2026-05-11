"""Re-exports from test_run.py for backwards compatibility.

All Modal config (image, app, zarr_volume, CFG, REPO_ROOT) now lives in
test_run.py so that file is fully self-contained when Modal mounts it as
/root/test_run.py (before the repo is cloned).
"""
from egomimic.modal.test_run import CFG, REPO_ROOT, app, zarr_volume  # noqa: F401
