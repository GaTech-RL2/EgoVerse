"""Lane A: split each val head into a subsampled METRIC loader and a
contiguous pinned-episode VIDEO loader.

Before this change the unshuffled val loaders were truncated by
``limit_val_batches`` to their leading hash-sorted slice, so the metric saw a
third of the seen-op split and effectively one of the two held-out operators
(docs/hpt-experiments/supervisors/S1/reports/lane0-coverage.md). Now:

* metric loader = ``EvenStrideDataset(split, frames_per_episode=K)``, K from
  the data config, so every episode contributes and nothing renders;
* video loader = the head's split restricted to the pinned ``episode_hash``es,
  contiguous, logging no metrics.

A data config with neither key must behave exactly as before.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lightning import LightningModule
from omegaconf import OmegaConf

import egomimic.trainHydra as th
from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EvenStrideDataset,
    MultiDataset,
    PinError,
    pinned_episode_subset,
)

HUMAN = "human_bimanual"
FLAGSHIP = "train_zarr_mecka_flagship_6d_hpt"
LIMIT_VAL_BATCHES = 80
BATCH = 64

# (data config, {head: K}, {head: [pinned hashes]}) as set on this branch.
CONFIG_EXPECTATIONS = {
    "mecka_fold_flagship_opsplit_hpt_6d": (
        {"valid": 73, "train_viz": 3, "unseen_op_valid": 26},
        {
            "valid": ["692e771c39719ab57395b408"],
            "train_viz": ["692e9bda21fc595fbe6f941e"],
            "unseen_op_valid": [
                "692eadbaaec602a46af10686",
                "692ea49a727c13b350cb7c48",
            ],
        },
    ),
    "mecka_fold_flagship_topop_hpt_6d": (
        {"valid": 256, "train_viz": 13, "unseen_op_valid": 26},
        {
            "valid": ["692e83ad567c97525f4011a5"],
            "train_viz": ["692e9bda21fc595fbe6f941e"],
            "unseen_op_valid": [
                "692eadbaaec602a46af10686",
                "692ea49a727c13b350cb7c48",
            ],
        },
    ),
    "mecka_fold_flagship_top3_hpt_6d": (
        {"valid": 96, "train_viz": 4, "unseen_op_valid": 26},
        {
            # one seen-val pin per train operator (683785ac / 690366b2 /
            # 6903686e), 191 + 223 + 226 = 640f = the whole video budget
            "valid": [
                "692ea44dc621d7f4aac3aae6",
                "692eb08997ffd8ec90340290",
                "692fe79f34a99e18e25289ad",
            ],
            "train_viz": ["692e9bda21fc595fbe6f941e"],
            "unseen_op_valid": [
                "692eadbaaec602a46af10686",
                "692ea49a727c13b350cb7c48",
            ],
        },
    ),
}

# Episode counts these K values were computed from (measured 2026-09-15).
SPLIT_EPISODES = {
    "mecka_fold_flagship_opsplit_hpt_6d": {
        "valid": 70,
        "train_viz": 1346,
        "unseen_op_valid": 191,
    },
    "mecka_fold_flagship_topop_hpt_6d": {
        "valid": 20,
        "train_viz": 385,
        "unseen_op_valid": 191,
    },
    "mecka_fold_flagship_top3_hpt_6d": {
        "valid": 53,
        "train_viz": 1026,
        "unseen_op_valid": 191,
    },
}


# --------------------------------------------------------------- fixtures
class _Episode:
    """Minimal leaf dataset: n frames, each sample tagged with its frame idx."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"frame": np.float32(i)}


def _split(sizes: dict[str, int]) -> MultiDataset:
    return MultiDataset(
        datasets={h: _Episode(n) for h, n in sizes.items()}, mode="total"
    )


def _cfg(metric=None, video=None, **data):
    node = dict(data)
    if metric is not None:
        node["metric_frames_per_episode"] = metric
    if video is not None:
        node["video_episodes"] = video
    return OmegaConf.create({"data": node})


# ------------------------------------------------------------ 1. config keys
@pytest.mark.parametrize("data_config", sorted(CONFIG_EXPECTATIONS))
def test_flagship_configs_declare_k_and_pins(data_config, compose_resolve):
    """Both flagship configs compose with the new keys, K per head matches
    floor(5120 / episodes-in-split), and each head pins its video episodes."""
    expected_k, expected_pins = CONFIG_EXPECTATIONS[data_config]
    cfg = compose_resolve(FLAGSHIP, [f"data={data_config}"])
    assert int(cfg.trainer.limit_val_batches) == LIMIT_VAL_BATCHES
    assert int(cfg.data.valid_dataloader_params[HUMAN].batch_size) == BATCH

    k_table = OmegaConf.to_container(cfg.data.metric_frames_per_episode)
    assert k_table == expected_k
    pins = OmegaConf.to_container(cfg.data.video_episodes)
    assert pins == expected_pins

    for head, k in k_table.items():
        n_eps = SPLIT_EPISODES[data_config][head]
        assert k == LIMIT_VAL_BATCHES * BATCH // n_eps, (head, k, n_eps)
        # the subsampled set still fits the limit_val_batches window
        assert k * n_eps <= LIMIT_VAL_BATCHES * BATCH, (head, k, n_eps)
    # the unseen head is the same split in every flagship config, so its pins
    # and K are shared
    assert k_table["unseen_op_valid"] == 26
    assert len(pins["unseen_op_valid"]) == 2, "one pin per held-out operator"

    # and trainHydra reads exactly these
    for head, k in expected_k.items():
        assert th._metric_frames_per_episode(cfg, head) == k


# The 3 largest flagship operators (SQL app.episodes, 2026-09-15), by frames.
TOP3_OPERATORS = [
    "6903686e0e94ce070afd1f24",
    "690366b20e94ce070afd1e8a",
    "683785ac01ca734152093448",
]


def test_top3_config_trains_on_exactly_the_three_largest_operators(compose_resolve):
    """mecka_fold_flagship_top3_hpt_6d composes under the flagship recipe, its
    train filter names exactly the 3 top operators (none of them held out), the
    seen-op val follows that filter, and both lane A tables cover all 3 heads."""
    cfg = compose_resolve(FLAGSHIP, ["data=mecka_fold_flagship_top3_hpt_6d"])
    parent = compose_resolve(FLAGSHIP, ["data=mecka_fold_flagship_opsplit_hpt_6d"])
    d, pd = cfg.data, parent.data
    train = d.train_datasets[HUMAN]

    # the operator lambda names exactly the three ids, largest first
    operator_lambda = train.filters.filter_lambdas[-1]
    assert "row.get('operator'" in operator_lambda
    assert re.findall(r"[0-9a-f]{24}", operator_lambda) == TOP3_OPERATORS
    held_out = [str(o) for o in d.held_out_operators]
    assert not set(TOP3_OPERATORS) & set(held_out), "a train operator is held out"

    # the first three (lab / task / flagship-path) lambdas are the parent's
    assert list(train.filters.filter_lambdas[:3]) == list(
        pd.train_datasets[HUMAN].filters.filter_lambdas[:3]
    )

    # seen-op val interpolates the train filters, so it follows this override
    valid = d.valid_datasets[HUMAN]
    assert list(valid.filters.filter_lambdas) == list(train.filters.filter_lambdas)
    assert valid.valid_ratio == train.valid_ratio == 0.05
    assert (train.mode, valid.mode) == ("train", "valid")
    assert d.valid_prefix == "seen_op_valid"
    assert d.get("train_viz_datasets") is None

    # the unseen-operator head is untouched
    unseen = d.unseen_op_valid_datasets[HUMAN]
    assert list(unseen.filters.filter_lambdas) == list(
        pd.unseen_op_valid_datasets[HUMAN].filters.filter_lambdas
    )
    assert unseen.mode == "total"

    # lane A keys cover all three heads
    heads = {"valid", "train_viz", "unseen_op_valid"}
    k_table = OmegaConf.to_container(d.metric_frames_per_episode)
    pins = OmegaConf.to_container(d.video_episodes)
    assert set(k_table) == heads
    assert set(pins) == heads
    assert all(isinstance(v, int) and v > 0 for v in k_table.values())
    assert len(pins["valid"]) == 3, "one seen-val video pin per train operator"
    assert len(set(pins["valid"])) == 3


def test_config_without_the_keys_is_unchanged(compose_resolve):
    """Backward compatibility: a data config that declares neither key gets the
    old loader names and the unwrapped dataset objects."""
    cfg = compose_resolve("train_zarr_cartesian", ["data=mecka_all_6d"])
    assert cfg.data.get("metric_frames_per_episode") is None
    assert cfg.data.get("video_episodes") is None
    assert th._metric_frames_per_episode(cfg, "valid") is None

    split = _split({"ep0": 10, "ep1": 10})
    datasets = {HUMAN: split}
    assert th._subsample_val_datasets(cfg, "valid", datasets) is datasets
    assert th._video_datasets(cfg, {"valid": datasets}) == {}

    params = {HUMAN: {"batch_size": 2, "num_workers": 0}}
    dm = MultiDataModuleWrapper(
        train_datasets=datasets,
        valid_datasets=datasets,
        train_dataloader_params=params,
        valid_dataloader_params=params,
        train_viz_datasets=datasets,
        train_viz_dataloader_params=params,
        unseen_op_valid_datasets=datasets,
        unseen_op_valid_dataloader_params=params,
    )
    assert dm.val_loader_names() == ["valid", "train_viz", "unseen_op_valid"]
    assert dm.valid_datasets[HUMAN] is split


# ------------------------------------------------------------ 2. metric loaders
def test_metric_loader_is_even_stride_over_every_episode():
    sizes = {"ep0": 500, "ep1": 300, "ep2": 40}
    split = _split(sizes)
    cfg = _cfg(metric={"valid": 73})
    wrapped = th._subsample_val_datasets(cfg, "valid", {HUMAN: split})[HUMAN]

    assert isinstance(wrapped, EvenStrideDataset)
    assert isinstance(wrapped, MultiDataset), "trainHydra's isinstance checks"
    assert wrapped.frames_per_episode == 73
    assert wrapped.base is split

    per_episode = {name: 0 for name in sizes}
    starts = {}
    offset = 0
    for name, n in sizes.items():
        starts[name] = offset
        offset += n
    for global_idx in wrapped.indices:
        name, _ = split.index_map[global_idx]
        per_episode[name] += 1
    # K frames from each episode, or the whole episode when it is shorter
    assert per_episode == {"ep0": 73, "ep1": 73, "ep2": 40}
    assert len(wrapped) == sum(per_episode.values())
    assert len(wrapped) <= LIMIT_VAL_BATCHES * BATCH
    # ... and the frames really are spread across each episode, not its head
    ep0 = [
        split.index_map[g][1] for g in wrapped.indices if split.index_map[g][0] == "ep0"
    ]
    assert ep0[0] == 0 and ep0[-1] == 499

    # samples still come through (the wrapper indexes into the base)
    assert wrapped[0]["frame"] == split[wrapped.indices[0]]["frame"]


def test_metric_wrapper_forwards_norm_stats_to_the_base():
    """__getitem__ delegates to base, so the stats must land on the base or the
    subsampled val loader would be unnormalized."""
    split = _split({"ep0": 10})
    wrapped = EvenStrideDataset(split, frames_per_episode=5)
    source = MultiDataset(state={})
    source.norm_stats = {0: {"actions_cartesian": {"mean": np.zeros(3)}}}
    wrapped.set_norm_stats_from(source)
    assert split.norm_stats is source.norm_stats


def test_every_head_can_be_subsampled_independently():
    cfg = _cfg(metric={"valid": 5, "unseen_op_valid": 2})
    heads = {h: {HUMAN: _split({"ep0": 100})} for h in ("valid", "train_viz")}
    heads["unseen_op_valid"] = {HUMAN: _split({"ep0": 100})}
    assert len(th._subsample_val_datasets(cfg, "valid", heads["valid"])[HUMAN]) == 5
    assert (
        len(
            th._subsample_val_datasets(
                cfg, "unseen_op_valid", heads["unseen_op_valid"]
            )[HUMAN]
        )
        == 2
    )
    # train_viz has no entry -> untouched
    assert (
        th._subsample_val_datasets(cfg, "train_viz", heads["train_viz"])
        is heads["train_viz"]
    )
    with pytest.raises(ValueError, match="positive int"):
        th._metric_frames_per_episode(_cfg(metric={"valid": 0}), "valid")


def test_k_scales_with_the_world_size():
    """K in the config is the single-rank value. DistributedSampler strides,
    so W ranks between them read W times the frames in the same number of
    batches each; without the scaling a 4-GPU run scores a quarter of what the
    same config scores on one GPU."""
    cfg = _cfg(metric={"valid": 7})
    for devices, nodes, expected in (
        (1, 1, 7),
        (4, 1, 28),
        (8, 2, 112),
        ([0, 1], 1, 14),
        ("auto", 1, 7),  # unknown at config time: stay single-rank
        (-1, 1, 7),
    ):
        cfg.trainer = {"devices": devices, "num_nodes": nodes}
        assert th._metric_frames_per_episode(cfg, "valid") == expected, devices
    # a split shorter than K is kept whole, so scaling cannot truncate
    cfg.trainer = {"devices": 4, "num_nodes": 1}
    split = {HUMAN: _split({"ep0": 20})}
    assert len(th._subsample_val_datasets(cfg, "valid", split)[HUMAN]) == 20


def test_eval_mode_is_single_rank():
    """Eval runs are forced onto one device after the loaders are built, so
    their val split must be subsampled for one rank, not the training W."""
    cfg = _cfg(metric={"valid": 7})
    cfg.trainer = {"devices": 4, "num_nodes": 2}
    assert th._metric_frames_per_episode(cfg, "valid") == 56
    cfg.mode = "eval"
    assert th._metric_frames_per_episode(cfg, "valid") == 7
    del cfg["mode"]
    cfg.eval = True
    assert th._metric_frames_per_episode(cfg, "valid") == 7


@pytest.mark.parametrize("trainer_cfg", ["ddp", "ddp_pi"])
def test_ddp_devices_are_per_node(trainer_cfg, compose_resolve):
    """Lightning reads devices per node; devices * num_nodes is the world."""
    cfg = compose_resolve(
        "train_zarr_cartesian",
        [
            f"trainer={trainer_cfg}",
            "launch_params.gpus_per_node=8",
            "launch_params.nodes=2",
        ],
    )
    assert cfg.trainer.devices == 8 and cfg.trainer.num_nodes == 2
    assert th._trainer_world_size(cfg) == 16


def test_auto_k_is_derived_from_the_resolved_split():
    """`auto` = floor(limit_val_batches * batch_size / episodes), per dataset,
    from the head's own loader batch size, then scaled by the world size."""
    cfg = _cfg(metric={"valid": "auto"})
    cfg.trainer = {"limit_val_batches": 4, "devices": 2}
    split = {HUMAN: _split({"ep0": 100, "ep1": 100, "ep2": 100})}
    params = {HUMAN: {"batch_size": 5}}
    wrapped = th._subsample_val_datasets(cfg, "valid", split, params)[HUMAN]
    assert wrapped.frames_per_episode == (4 * 5 // 3) * 2
    for limit in (1.0, None, True):
        cfg.trainer.limit_val_batches = limit
        with pytest.raises(ValueError, match="auto"):
            th._subsample_val_datasets(cfg, "valid", split, params)
    cfg.trainer.limit_val_batches = 0  # validation off, as the bench scripts run
    assert th._subsample_val_datasets(cfg, "valid", split, params) == split
    cfg.trainer.limit_val_batches = 4
    with pytest.raises(ValueError, match="batch_size"):
        th._subsample_val_datasets(cfg, "valid", split, {})


# ------------------------------------------------------------- 3. video loaders
def test_video_dataset_holds_only_the_pins_in_frame_order():
    split = _split({"ep0": 30, "ep1": 20, "ep2": 25})
    sub = pinned_episode_subset(split, ["ep2", "ep0"], dataset_name=HUMAN)

    assert list(sub.datasets) == ["ep2", "ep0"], "config order is playback order"
    assert len(sub) == 55
    # contiguous: frame indices run 0..n-1 inside each episode, in order
    by_episode: dict[str, list[int]] = {}
    for name, local in sub.index_map:
        by_episode.setdefault(name, []).append(local)
    assert by_episode == {"ep2": list(range(25)), "ep0": list(range(30))}

    with pytest.raises(PinError, match="not in dataset"):
        pinned_episode_subset(split, ["nope"], dataset_name=HUMAN)


def test_video_datasets_are_built_per_head_from_the_resolved_split():
    valid = _split({"a": 10, "b": 10})
    unseen = _split({"c": 10, "d": 10})
    cfg = _cfg(video={"valid": ["b"], "unseen_op_valid": ["c", "d"]})
    built = th._video_datasets(
        cfg,
        {
            "valid": {HUMAN: valid},
            "train_viz": {HUMAN: _split({"t": 10})},
            "unseen_op_valid": {HUMAN: unseen},
        },
    )
    assert set(built) == {"valid", "unseen_op_valid"}, "no pins -> no video loader"
    assert list(built["valid"][HUMAN].datasets) == ["b"]
    assert list(built["unseen_op_valid"][HUMAN].datasets) == ["c", "d"]
    # the video dataset reuses the SAME leaf objects: no second resolve
    assert built["valid"][HUMAN].datasets["b"] is valid.datasets["b"]

    with pytest.raises(PinError, match="none of its datasets"):
        th._video_datasets(_cfg(video={"valid": ["c"]}), {"valid": {HUMAN: valid}})


def test_val_loader_names_append_the_video_loaders():
    params = {HUMAN: {"batch_size": 2, "num_workers": 0, "shuffle": False}}
    datasets = {HUMAN: _split({"ep0": 4})}
    dm = MultiDataModuleWrapper(
        train_datasets=datasets,
        valid_datasets=datasets,
        train_dataloader_params=params,
        valid_dataloader_params=params,
        train_viz_datasets=datasets,
        train_viz_dataloader_params=params,
        unseen_op_valid_datasets=datasets,
        unseen_op_valid_dataloader_params=params,
        video_datasets={
            "valid": {HUMAN: _split({"pin": 4})},
            "unseen_op_valid": {HUMAN: _split({"pin2": 4})},
        },
    )
    # metric loaders keep their historical indices; video loaders follow
    assert dm.val_loader_names() == [
        "valid",
        "train_viz",
        "unseen_op_valid",
        "valid_video",
        "unseen_op_valid_video",
    ]
    loaders = dm.val_dataloader()
    assert len(loaders) == 5
    # the video loader really iterates the pinned dataset, unshuffled
    batch, _, _ = next(iter(loaders[3]))
    assert [float(f) for f in batch[HUMAN]["frame"]] == [0.0, 1.0]

    # a head whose video datasets were dropped gets no name
    dm.video_datasets = {}
    assert dm.val_loader_names() == ["valid", "train_viz", "unseen_op_valid"]


# ----------------------------------------------------------------- 4. dispatch
class _StubEval:
    """Records how validation_step called it, with EvalVideo's viz gating."""

    def __init__(self, should_viz=True):
        self.calls: list[tuple] = []
        self._viz = should_viz

    def _should_viz(self):
        return self._viz

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0, mode="both"):
        self.calls.append((batch_idx, dataloader_idx, mode))


def _wrapper(names, heads):
    w = ModelWrapper.__new__(ModelWrapper)
    LightningModule.__init__(w)
    w.model = SimpleNamespace(process_batch_for_training=lambda b: b)
    w.evaluator = heads.get("valid")
    w.train_viz_evaluator = heads.get("train_viz")
    w.unseen_op_valid_evaluator = heads.get("unseen_op_valid")
    w.val_loader_names = names
    return w


def test_validation_step_splits_metrics_from_video():
    valid, unseen = _StubEval(), _StubEval()
    names = ["valid", "unseen_op_valid", "valid_video", "unseen_op_valid_video"]
    w = _wrapper(names, {"valid": valid, "unseen_op_valid": unseen})

    for idx in range(4):
        w.validation_step({"k": idx}, 0, dataloader_idx=idx)

    assert valid.calls == [(0, 0, "metrics"), (0, 2, "video")]
    assert unseen.calls == [(0, 1, "metrics"), (0, 3, "video")]


def test_non_video_epoch_skips_the_video_loader_before_the_forward():
    valid = _StubEval(should_viz=False)
    processed = []
    w = _wrapper(["valid", "valid_video"], {"valid": valid})
    w.model = SimpleNamespace(
        process_batch_for_training=lambda b: processed.append(b) or b
    )
    w.validation_step({"k": 0}, 0, dataloader_idx=0)
    w.validation_step({"k": 1}, 0, dataloader_idx=1)
    assert valid.calls == [(0, 0, "metrics")], "video loader skipped"
    assert processed == [{"k": 0}], "no forward pass for the skipped loader"


def test_head_without_a_video_loader_keeps_the_old_combined_call():
    """Evaluators that predate the split (and the stubs in older tests) are
    called with no ``mode`` kwarg at all."""
    seen = []

    class _Old:
        def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
            seen.append((batch_idx, dataloader_idx))

    w = _wrapper(["valid"], {"valid": _Old()})
    w.validation_step({"k": 0}, 7, dataloader_idx=0)
    assert seen == [(7, 0)]


# ----------------------------------------------- EvalVideo's two modes
class _Trainer:
    def __init__(self, epoch=0, max_epochs=1):
        self.current_epoch = epoch
        self.max_epochs = max_epochs
        self.world_size = 1
        self.is_global_zero = True
        self.logged: list[dict] = []
        outer = self

        class _LM:
            device = "cpu"

            def log_dict(self, metrics, **kw):
                outer.logged.append(dict(metrics))

        self.lightning_module = _LM()


def _eval_video(viz_max_batches=2):
    from egomimic.eval.eval_video import EvalVideo

    class _Impl(EvalVideo):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.rendered: list[int] = []
            self.forwards = 0

        def compute_metrics_and_viz(self, batch, do_viz=True):
            self.forwards += 1
            if do_viz:
                self.rendered.append(batch["i"])
                return {"Valid/mse": 1.0}, {HUMAN: np.zeros((1, 2, 2, 3), np.uint8)}
            return {"Valid/mse": 1.0}, {}

    ev = _Impl(viz_every_n_epochs=1, viz_max_batches=viz_max_batches)
    ev.trainer = _Trainer()
    return ev


def test_metric_mode_never_renders_and_video_mode_never_logs():
    ev = _eval_video()
    for i in range(3):
        ev.on_validation_step({"i": i}, i, 0, mode="metrics")
    assert ev.rendered == [], "no frame rendered on the metric loader"
    assert ev.val_image_buffer == {}, "no frame buffered either"
    assert len(ev.trainer.logged) == 3, "metrics logged for every metric batch"

    ev.trainer.logged.clear()
    for i in range(4):
        ev.on_validation_step({"i": i}, i, 1, mode="video")
    assert ev.rendered == [0, 1], "rendered up to viz_max_batches"
    assert ev.trainer.logged == [], "video loader logs no metric key"
    # past viz_max_batches the video loader does not even run the forward
    assert ev.forwards == 3 + 2

    with pytest.raises(ValueError, match="unknown validation mode"):
        ev.on_validation_step({"i": 0}, 0, 0, mode="bogus")


def test_default_mode_is_todays_behaviour():
    ev = _eval_video()
    ev.on_validation_step({"i": 0}, 0, 0)
    assert ev.rendered == [0] and len(ev.trainer.logged) == 1


# --------------------------------------------- video writes happen on rank 0 only
def test_only_rank_zero_writes_the_video(monkeypatch, tmp_path):
    """Every rank iterates every val batch and renders the same frames, so
    without a guard N ranks race to write one mp4 path."""
    import egomimic.eval.eval_video as ev_mod

    written: list[str] = []
    monkeypatch.setattr(
        ev_mod.tvio, "write_video", lambda path, *a, **kw: written.append(path)
    )

    buffers = {}
    for rank_zero in (True, False):
        ev = _eval_video(viz_max_batches=None)
        ev.trainer.is_global_zero = rank_zero
        monkeypatch.setattr(type(ev), "video_dir", lambda self: str(tmp_path))
        for i in range(3):
            ev.on_validation_step({"i": i}, i, 0, mode="video")
        ev.on_validation_end()
        # rendering, buffering and counters are identical on every rank...
        buffers[rank_zero] = (ev.rendered, dict(ev.val_counter), ev.val_image_buffer)

    assert buffers[True] == buffers[False], "only the write differs between ranks"
    assert len(written) == 1, "one file, from rank 0"


# ------------------------------------------- video loaders are gated on non-viz
class _GateTrainer:
    """Just enough trainer for MultiDataModuleWrapper._video_gate."""

    def __init__(self, evaluators):
        self.lightning_module = SimpleNamespace(_val_heads=lambda: evaluators)
        self.is_global_zero = True


def _dm():
    """A datamodule with one metric head and its pinned video companion."""
    params = {HUMAN: {"batch_size": 2, "num_workers": 0, "shuffle": False}}
    datasets = {HUMAN: _split({"ep0": 4})}
    return MultiDataModuleWrapper(
        train_datasets=datasets,
        valid_datasets=datasets,
        train_dataloader_params=params,
        valid_dataloader_params=params,
        video_datasets={"valid": {HUMAN: _split({"pin": 4})}},
    )


def test_video_loader_yields_nothing_when_the_gate_is_closed():
    from egomimic.pl_utils.pl_data_utils import _VizGatedLoader

    dm = _dm()
    metric, video = dm.val_dataloader()
    assert not isinstance(metric, _VizGatedLoader), "metric loaders run every pass"
    assert isinstance(video, _VizGatedLoader)

    viz = _StubEval(should_viz=True)
    dm.trainer = _GateTrainer({"valid": viz})
    assert len(list(iter(video))) == 2, "viz pass: the pinned episode is read"

    viz._viz = False
    assert list(iter(video)) == [], "non-viz pass: not one batch is built"
    # the length Lightning caches at setup does not move with the gate
    assert len(video) == 2


def test_gate_is_open_without_a_trainer_or_an_evaluator():
    dm = _dm()
    _, video = dm.val_dataloader()
    assert len(list(iter(video))) == 2, "no trainer attached -> gate open"

    dm.trainer = _GateTrainer({"valid": None})
    assert len(list(iter(video))) == 2, "head with no evaluator -> gate open"

    dm.trainer = _GateTrainer({"valid": object()})
    assert len(list(iter(video))) == 2, "evaluator without _should_viz -> gate open"


def test_gated_loader_length_survives_a_closed_gate_at_setup():
    """Lightning reads len() once, in setup_data, which can land on a non-viz
    epoch: CombinedLoader.__len__ needs a live iterator the gate never made."""
    dm = _dm()
    _, video = dm.val_dataloader()
    dm.trainer = _GateTrainer({"valid": _StubEval(should_viz=False)})
    assert list(iter(video)) == []
    assert len(video) == 2


def test_gated_loader_length_is_lightnings_own_and_opens_no_iterator():
    from lightning.pytorch.utilities.combined_loader import _MaxSizeCycle

    dm = _dm()
    _, video = dm.val_dataloader()
    video.limits = [1]
    assert len(video) == len(_MaxSizeCycle(video.flattened, [1])) == 1
    assert video._iterator is None


def test_video_head_without_a_viz_cap_is_rejected():
    """Rank 0 renders a pinned video loader alone while the other ranks wait, so
    an uncapped head is bounded only by the process-group timeout."""
    capped = SimpleNamespace(viz_func=print, viz_max_batches=10)
    uncapped = SimpleNamespace(viz_func=print, viz_max_batches=None)
    no_viz = SimpleNamespace(viz_func=None, viz_max_batches=None)

    def _run(heads, video):
        model = SimpleNamespace(_val_heads=lambda: heads)
        th._require_capped_video_heads(model, SimpleNamespace(video_datasets=video))

    _run({"valid": capped}, {"valid": {HUMAN: object()}})
    _run({"valid": uncapped}, {"valid": {}})
    _run({"valid": uncapped}, {})
    _run({"valid": no_viz}, {"valid": {HUMAN: object()}})
    with pytest.raises(ValueError, match="viz_max_batches"):
        _run({"valid": capped, "train_viz": uncapped}, {"train_viz": {HUMAN: object()}})


# sanity steps on: the loaders (and their len()) are set up on a pass whose gate
# is already closed, which is what a real run does -- trainer/default.yaml sets no
# num_sanity_val_steps, so lightning's default of 2 applies.
@pytest.mark.parametrize("sanity_steps", [0, 2])
def test_a_real_fit_builds_video_batches_only_on_viz_passes(tmp_path, sanity_steps):
    """The gate against the real Lightning evaluation loop: a val dataloader
    that yields nothing mid-sequence must not derail the pass, and the pinned
    episode must not be touched at all on a non-viz epoch."""
    import torch
    from lightning import Trainer

    class _Counting(_Episode):
        def __init__(self, n):
            super().__init__(n)
            self.reads = 0

        def __getitem__(self, i):
            self.reads += 1
            return super().__getitem__(i)

    pinned, metric = _Counting(4), _Counting(4)
    params = {HUMAN: {"batch_size": 2, "num_workers": 0, "shuffle": False}}
    dm = MultiDataModuleWrapper(
        train_datasets={HUMAN: _split({"ep0": 4})},
        valid_datasets={HUMAN: MultiDataset(datasets={"ep0": metric}, mode="total")},
        train_dataloader_params={HUMAN: {"batch_size": 2, "num_workers": 0}},
        valid_dataloader_params=params,
        video_datasets={
            "valid": {HUMAN: MultiDataset(datasets={"pin": pinned}, mode="total")}
        },
    )

    class _Module(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            self.val_loaders_seen: list[tuple[int, int]] = []
            self.pinned_reads: dict[int, int] = {}

        # viz only on the second validation pass
        def _val_heads(self):
            return {
                "valid": SimpleNamespace(_should_viz=lambda: self.current_epoch == 1)
            }

        def training_step(self, batch, batch_idx):
            return self.layer(batch[HUMAN]["frame"].reshape(-1, 1)).sum()

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            if not self.trainer.sanity_checking:
                self.val_loaders_seen.append((self.current_epoch, dataloader_idx))

        def on_validation_epoch_end(self):
            self.pinned_reads[self.current_epoch] = pinned.reads

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    model = _Module()
    Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        limit_train_batches=2,
        limit_val_batches=2,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=sanity_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    ).fit(model, datamodule=dm)

    assert model.pinned_reads[0] == 0, "non-viz pass never touched the pinned episode"
    assert model.pinned_reads[1] > 0, "viz pass still reads it"
    # the metric loader is unaffected, and dataloader_idx stays stable
    by_epoch = {e: [i for ep, i in model.val_loaders_seen if ep == e] for e in (0, 1)}
    assert by_epoch[0] == [0, 0], "epoch 0: metric loader only"
    assert by_epoch[1] == [0, 0, 1, 1], "epoch 1: metric loader then video loader"


def test_video_loader_is_closed_off_rank_zero():
    """Only rank 0 writes the overlay, and the val loaders carry no
    DistributedSampler, so every other rank was decoding, forwarding and
    rendering the same frames for nothing."""
    dm = _dm()
    _, video = dm.val_dataloader()
    viz = _StubEval(should_viz=True)

    dm.trainer = _GateTrainer({"valid": viz})
    dm.trainer.is_global_zero = True
    assert len(list(iter(video))) == 2

    dm.trainer.is_global_zero = False
    assert list(iter(video)) == [], "rank 1 builds no video batch at all"


def test_two_rank_ddp_fit_survives_the_rank_divergence(tmp_path):
    """The gate makes ranks iterate *different* numbers of val batches. Proven
    against real DDP (gloo, 2 CPU processes) with ModelWrapper's collective
    pattern -- sync_dist metrics off the metric loader, nothing off the video
    one -- because a desync here would hang a campaign, not fail it."""
    import json
    import subprocess

    script = Path(__file__).resolve().parents[1] / "fixtures" / "ddp_video_gate.py"
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    ranks = {
        int(p.stem.removeprefix("rank")): json.loads(p.read_text())
        for p in tmp_path.glob("rank*.json")
    }
    assert set(ranks) == {0, 1}, "both ranks finished; neither hung"
    # epoch 0 renders nothing, epoch 1 does, and only on rank 0
    assert ranks[0]["video_steps"] == [1, 1]
    assert ranks[0]["pinned_reads"] >= 4, "rank 0 read the frames it stepped"
    # exactly 0, not "few": a gate that still builds an iterator makes the
    # loader's workers prefetch, which is the cost this whole change is about
    assert ranks[1]["video_steps"] == [] and ranks[1]["pinned_reads"] == 0
