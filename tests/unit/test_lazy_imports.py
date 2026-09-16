import subprocess
import sys

from fixtures.synthetic_episodes import write_episode

BLOCK = "boto3", "botocore", "sqlalchemy", "cloudpathlib", "scaleapi", "psycopg"


def test_local_path_imports_without_cloud_packages(tmp_path) -> None:
    write_episode(tmp_path, "aria", seed=0)
    code = f"""
import sys
for m in {BLOCK!r}:
    sys.modules[m] = None
import egomimic.trainHydra
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
from egomimic.rldb.embodiment.human import Human
r = LocalEpisodeResolver({str(tmp_path)!r}, key_map=Human.get_keymap(keymap_mode="cartesian"),
                         transform_list=Human.get_transform_list(mode="cartesian", allow_legacy_rotation=True))
ds = r.resolve(filters=DatasetFilter(episode_hashes=["aria_00"]))
assert set(ds) == {{"aria_00"}}, ds
print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_s3_resolver_names_missing_package(tmp_path) -> None:
    # A real (empty) tmp dir, never "/nonexistent": the constructor mkdirs its
    # folder_path, and "/nonexistent" makes that a write to "/".
    sync_dir = tmp_path / "sync"
    code = f"""
import sys
sys.modules["sqlalchemy"] = None
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.rldb.filters import DatasetFilter
try:
    S3EpisodeResolver({str(sync_dir)!r}).resolve_paths(DatasetFilter())
except ImportError as e:
    assert "sqlalchemy" in str(e), e
    print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_env_loader_is_cloud_free_and_reexported() -> None:
    """`trainHydra` calls `load_env()` unconditionally (WANDB_API_KEY lives in
    ~/.egoverse_env), so the loader must import with the cloud packages blocked,
    and `aws_data_utils.load_env` must stay the very same object so existing
    callers and monkeypatch targets keep working."""
    code = f"""
import sys
for m in {BLOCK!r}:
    sys.modules[m] = None
import egomimic.utils.env as env
assert callable(env.load_env)
for m in {BLOCK!r}:
    del sys.modules[m]
from egomimic.utils.aws import aws_data_utils
assert aws_data_utils.load_env is env.load_env
print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
