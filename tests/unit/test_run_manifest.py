import json

import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import write_catalog  # noqa: E402

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import steps_path  # noqa: E402
from egomimic.rldb.multi import MultiDataset  # noqa: E402
from egomimic.utils import run_manifest  # noqa: E402

STEPS = steps_path("rl2", "human_bimanual", "hand")
KEY_MAP = {
    "left.action_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.obs_ee_pose",
        "horizon": 3,
    },
}


def _dataset(root, **kw):
    resolver = LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}, **kw)
    return MultiDataset._from_resolver(resolver, mode="total", valid_ratio=0.5)


def test_manifest_names_the_split_and_the_versions_read(tmp_path, caplog):
    root = tmp_path / "lance"
    recs = write_catalog(root)
    with caplog.at_level("WARNING"):
        ds = _dataset(root)
    assert "untagged" in caplog.text
    files = run_manifest.run_manifest({"human_bimanual": ds}, {"human_bimanual": ds})
    split = files[run_manifest.SPLIT_FILE]["human_bimanual"]
    assert sorted(split["train"] + split["valid"]) == sorted(r for r, _, _ in recs)
    read = files[run_manifest.LANCE_FILE]["train/human_bimanual"]
    assert read["versions"][STEPS] == 1
    assert read["tags"]["recordings.lance"] == []

    commit.tag(root, "t1")
    caplog.clear()
    with caplog.at_level("WARNING"):
        tagged = _dataset(root, tag="t1")
    assert "untagged" not in caplog.text
    assert tagged.source_manifest["tags"][STEPS] == ["t1"]


def test_write_keeps_the_first_launchs_record(tmp_path):
    run_manifest.write(tmp_path, {"split.json": {"a": 1}})
    run_manifest.write(tmp_path, {"split.json": {"a": 1}})
    assert [p.name for p in tmp_path.iterdir()] == ["split.json"]
    run_manifest.write(tmp_path, {"split.json": {"a": 2}})
    assert json.loads((tmp_path / "split.json").read_text()) == {"a": 1}
    assert len(list(tmp_path.glob("split.*.json"))) == 1


def test_resume_warns_when_old_val_episodes_now_train(tmp_path, caplog):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "last.ckpt"
    ckpt.write_bytes(b"")
    run_manifest.write(run, {"split.json": {"h": {"train": ["a"], "valid": ["b"]}}})
    same = {"h": {"train": ["a", "c"], "valid": ["b"]}}
    assert run_manifest.warn_on_split_drift(str(ckpt), same) == []
    moved = {"h": {"train": ["a", "b"], "valid": ["c"]}}
    with caplog.at_level("WARNING"):
        assert run_manifest.warn_on_split_drift(str(ckpt), moved) == ["b"]
    assert "validated on" in caplog.text
    assert run_manifest.warn_on_split_drift(None, moved) == []
