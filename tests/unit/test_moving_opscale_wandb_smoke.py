"""W&B completion checks use genuine fresh metrics and exact uploaded videos."""

import copy
import sys
from types import SimpleNamespace

import pytest

from experiments.fold_clothes_moving_opscale import run_preflight as preflight

KEYS = [
    "videos_open_loop/seen_matched/operator/episode-a",
    "videos_open_loop/unseen/operator/episode-b",
]
PATHS = ["media/videos/episode-a.mp4", "media/videos/episode-b.mp4"]
STATUS = {"videos": [{"wandb_key": key} for key in KEYS]}


def summary():
    return {
        "Train/Loss": 1.14467,
        "Train/action_loss": 1.14467,
        "trainer/global_step": 1,
        "val/seen_matched/Valid/action_loss": 0.5,
        "val/seen_perlevel/Valid/action_loss": 0.6,
        "val/unseen/Valid/action_loss": 0.7,
        KEYS[0]: {
            "_type": "videos",
            "count": 1,
            "videos": [{"_type": "video-file", "path": PATHS[0]}],
        },
        KEYS[1]: {"_type": "video-file", "path": PATHS[1]},
    }


class FakeRun:
    def __init__(self, data=None, state="finished", sizes=None):
        self.summary = SimpleNamespace(
            _json_dict=copy.deepcopy(data if data is not None else summary())
        )
        self.state = state
        self.url = "https://wandb.invalid/synthetic"
        self.sizes = sizes if sizes is not None else dict(zip(PATHS, [100, 200]))
        self.file_reads = []

    def file(self, path):
        self.file_reads.append(path)
        return SimpleNamespace(size=self.sizes.get(path, 0))


class CachedApi:
    """Match public Api.run's local cache; only flush observes later updates."""

    def __init__(self, versions):
        self.versions = versions
        self.remote_index = 0
        self.cached = None
        self.flushes = 0
        self.paths = []

    def flush(self):
        self.flushes += 1
        self.cached = None

    def run(self, path):
        self.paths.append(path)
        if self.cached is None:
            self.cached = self.versions[min(self.remote_index, len(self.versions) - 1)]
        return self.cached


def setup_api(monkeypatch, versions):
    api = CachedApi(versions)
    clock = [0]
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return api

    def sleep(seconds):
        assert seconds == 20
        api.remote_index += 1
        clock[0] += 1200 / len(versions)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=create))
    monkeypatch.setattr(preflight.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(preflight.time, "sleep", sleep)
    return api, calls


def test_grouped_and_legacy_videos_require_real_remote_files(monkeypatch):
    run = FakeRun()
    api, calls = setup_api(monkeypatch, [run])
    result = preflight.verify_wandb("synthetic", STATUS, 2)
    assert result["state"] == "finished"
    assert result["backend_step"] == 1
    assert result["verified_videos"] == 2
    assert result["finite_losses"] == [1.14467, 1.14467]
    assert run.file_reads == PATHS
    assert api.paths == ["rl2-group/egoverse/synthetic"]
    assert api.flushes == 1
    assert calls == [{"timeout": 45}]


def test_retries_flush_stale_run_state_and_summary(monkeypatch):
    first = FakeRun(data={}, state="running")
    second = FakeRun()
    api, _ = setup_api(monkeypatch, [first, second])
    result = preflight.verify_wandb("synthetic", STATUS, 2)
    assert result["verified_videos"] == 2
    assert api.flushes == 2
    assert len(api.paths) == 2
    assert first.file_reads == []
    assert second.file_reads == PATHS


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        [],
        {"_type": "videos", "count": 0, "videos": []},
        {
            "_type": "videos",
            "count": 2,
            "videos": [{"path": PATHS[0]}, {"path": PATHS[1]}],
        },
        {"_type": "videos", "count": 2, "videos": [{"path": PATHS[0]}]},
        {"_type": "videos", "count": True, "videos": [{"path": PATHS[0]}]},
        {"_type": "videos", "count": 1, "videos": []},
        {"_type": "videos", "count": 1, "videos": {"path": PATHS[0]}},
        {"_type": "videos", "count": 1, "videos": [{}]},
        {"_type": "videos", "count": 1, "videos": [None]},
        {
            "_type": "videos",
            "count": 1,
            "videos": [{"path": PATHS[0]}],
            "path": PATHS[1],
        },
        {"_type": "videos", "count": 1, "videos": [{"path": 123}]},
        {"_type": "table-file", "path": PATHS[0]},
        {"path": ""},
        {"path": "  "},
        {"path": "/absolute.mp4"},
        {"path": "../other.mp4"},
    ],
)
def test_malformed_or_ambiguous_video_metadata_cannot_pass(monkeypatch, record):
    data = summary()
    data[KEYS[0]] = record
    run = FakeRun(data)
    setup_api(monkeypatch, [run])
    with pytest.raises(
        RuntimeError, match="W&B smoke evidence did not become available"
    ):
        preflight.verify_wandb("synthetic", STATUS, 2)
    assert run.file_reads == []


@pytest.mark.parametrize(
    "videos",
    [
        None,
        [],
        [{"wandb_key": ""}],
        [{"wandb_key": 7}],
        [{}],
        [None],
        [{"wandb_key": KEYS[0]}, {"wandb_key": KEYS[0]}],
    ],
)
def test_expected_keys_must_be_nonempty_valid_and_unique(monkeypatch, videos):
    api, calls = setup_api(monkeypatch, [FakeRun()])
    with pytest.raises(RuntimeError, match="Expected W&B"):
        preflight.verify_wandb("synthetic", {"videos": videos}, 2)
    assert calls == []
    assert api.paths == []


def test_two_expected_videos_cannot_reuse_one_upload(monkeypatch):
    data = summary()
    data[KEYS[1]]["path"] = PATHS[0]
    run = FakeRun(data)
    setup_api(monkeypatch, [run])
    with pytest.raises(RuntimeError, match="distinct uploaded files"):
        preflight.verify_wandb("synthetic", STATUS, 2)
    assert run.file_reads == []


@pytest.mark.parametrize("size", [0, -1, None, "100", True, float("nan")])
def test_summary_metadata_does_not_replace_nonempty_remote_file(monkeypatch, size):
    data = summary()
    data[KEYS[0]]["videos"][0]["size"] = 100  # Claimed summary size is not enough.
    setup_api(monkeypatch, [FakeRun(data, sizes={PATHS[0]: size, PATHS[1]: 200})])
    with pytest.raises(RuntimeError, match="video not uploaded"):
        preflight.verify_wandb("synthetic", STATUS, 2)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"Train/Loss": float("nan")}, "finite training loss missing"),
        ({"Train/action_loss": float("inf")}, "finite training loss missing"),
        (
            {"Train/Loss": None, "Train/action_loss": None},
            "finite training loss missing",
        ),
        ({"trainer/global_step": 0}, "step is behind"),
        (
            {"val/seen_matched/Valid/action_loss": None},
            "missing finite metrics: seen_matched",
        ),
        (
            {"val/seen_perlevel/Valid/action_loss": float("nan")},
            "missing finite metrics: seen_perlevel",
        ),
        (
            {"val/unseen/Valid/action_loss": float("inf")},
            "missing finite metrics: unseen",
        ),
    ],
)
def test_video_fix_preserves_scalar_and_step_requirements(monkeypatch, change, error):
    data = summary()
    data.update(change)
    run = FakeRun(data)
    setup_api(monkeypatch, [run])
    with pytest.raises(RuntimeError, match=error):
        preflight.verify_wandb("synthetic", STATUS, 2)
    assert run.file_reads == []


def test_finished_state_is_required_even_with_all_uploads(monkeypatch):
    setup_api(monkeypatch, [FakeRun(state="crashed")])
    with pytest.raises(RuntimeError, match="run state is crashed"):
        preflight.verify_wandb("synthetic", STATUS, 2)
