"""Validation cohorts must stay separate and cover their complete episode pools."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.eval.open_loop_sim import OpenLoopSimEval
from egomimic.eval.sampled_trajectory_mse import trajectory_mse, trajectory_mse_thirds
from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import EvenStrideDataset, MultiDataset
from egomimic.trainHydra import (
    _attach_named_validation_evaluators,
    _subsample_val_datasets,
)


class Episode:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"frame_idx": index, "x": torch.tensor([float(index)])}


def split(sizes):
    return MultiDataset({key: Episode(n) for key, n in sizes.items()}, mode="total")


def test_random_samples_cover_all_episodes_and_are_reproducible():
    base = split({"a": 1000, "b": 700, "short": 7})
    a = EvenStrideDataset(base, frames_per_episode=128, random_seed=42026)
    b = EvenStrideDataset(base, frames_per_episode=128, random_seed=42026)
    c = EvenStrideDataset(base, frames_per_episode=128, random_seed=123)
    assert a.indices == b.indices and a.indices != c.indices
    assert len(a.indices) == len(set(a.indices)) == 263
    assert Counter(base.index_map[i][0] for i in a.indices) == {
        "a": 128,
        "b": 128,
        "short": 7,
    }
    # Sample identity survives reading and never enters training outlier retries.
    base._check_bounds = lambda *args: (_ for _ in ()).throw(
        AssertionError("training bounds used")
    )
    assert a[0]["frame_idx"] == base.index_map[a.indices[0]][1]


def test_random_sampling_budget_is_independent_of_gpu_count():
    cfg = OmegaConf.create(
        {
            "data": {
                "metric_sampling": {
                    "method": "random",
                    "seed": 42,
                    "frames_per_episode": 128,
                }
            },
            "trainer": {"devices": 1, "limit_val_batches": 8},
        }
    )
    base = split({"a": 1000, "b": 1000})
    params = {"human": {"batch_size": 32}}
    one = _subsample_val_datasets(cfg, "valid", {"human": base}, params)["human"]
    cfg.trainer.devices = 8
    eight = _subsample_val_datasets(cfg, "valid", {"human": base}, params)["human"]
    assert one.indices == eight.indices
    cfg.trainer.limit_val_batches = 7
    with pytest.raises(ValueError, match="exceed validation budget"):
        _subsample_val_datasets(cfg, "valid", {"human": base}, params)


def test_named_loaders_do_not_cycle_short_sets():
    dm = MultiDataModuleWrapper(
        {},
        {"short": split({"a": 4}), "long": split({"b": 12})},
        {},
        {key: {"batch_size": 2, "num_workers": 0} for key in ("short", "long")},
        validation_dataset_options={
            key: {"embodiment": "human_bimanual", "prefix": key}
            for key in ("short", "long")
        },
    )
    assert dm.val_loader_names() == ["valid::short", "valid::long"]
    assert [len(list(loader)) for loader in dm.val_dataloader()] == [2, 6]


def test_mse_ignores_padding_and_gives_each_trajectory_one_vote():
    pred = torch.tensor([[[1.0], [999.0]], [[3.0], [3.0]]])
    mask = torch.tensor([[1, 0], [1, 1]])
    actual = trajectory_mse(pred, torch.zeros_like(pred), mask)
    assert torch.equal(actual, torch.tensor([1.0, 9.0]))
    assert actual.mean() == 5


def test_mse_thirds_exclude_padding_and_empty_trajectory_sections():
    pred = torch.tensor(
        [
            [[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]],
            [[4.0], [4.0], [5.0], [5.0], [999.0], [999.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    parts = trajectory_mse_thirds(pred, torch.zeros_like(pred), mask)
    assert parts == {
        "early": torch.tensor(8.5),
        "mid": torch.tensor(14.5),
        "late": torch.tensor(9.0),
    }


@pytest.mark.parametrize("rung", [1, 4, 8, 12, 16])
def test_rung_config_has_one_video_per_set_and_separate_metric_prefixes(
    compose_resolve, rung
):
    cfg = compose_resolve(f"experiments/fixed_budget_opscale/rdt_1b_opscale{rung}", [])
    data = cfg.data
    names = list(data.valid_datasets) + list(data.unseen_op_valid_datasets)
    assert len(names) == (3 if rung in (1, 4) else 4)
    assert len(names) == len(set(names))
    assert len({data.validation_dataset_options[n].prefix for n in names}) == len(names)
    assert len(data.open_loop_episodes.seen) + len(
        data.open_loop_episodes.unseen
    ) == len(names)
    assert data.metric_sampling.frames_per_episode == 128
    assert cfg.trainer.limit_val_batches * 32 >= 120 * 128
    assert (
        cfg.open_loop_evaluator.final_only and not cfg.open_loop_evaluator.log_metrics
    )
    assert cfg.evaluator._target_.endswith("SampledTrajectoryMSE")
    for n in names:
        assert n in data.open_loop_dataloader_params


def test_manifest_cohorts_are_disjoint():
    path = (
        Path(__file__).resolve().parents[2]
        / "egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json"
    )
    m = json.loads(path.read_text())
    train = {
        e["episode_id"] for t in [m["op1"], *m["tiers"].values()] for e in t["episodes"]
    }
    unseen = m["flagship_unseen_val"]
    assert set(Counter(e["user_id"] for e in unseen["episodes"]).values()) == {20}
    assert len(unseen["operators"]) == 4
    assert set(unseen["operators"]).isdisjoint(m["tiers"]["op16"]["operators"])
    seen_ops, val_ids = set(), set()
    for k, group in m["seen_val_by_rung"].items():
        ops = set(group["operators"])
        prev = f"op{int(k[2:])-4}"
        assert ops == set(m["tiers"][k]["operators"]) - set(
            m["tiers"][prev]["operators"]
        )
        assert not ops & seen_ops
        seen_ops |= ops
        assert set(Counter(e["user_id"] for e in group["episodes"]).values()) == {30}
    for group in [
        m["seen_val"],
        m["unseen_val"],
        unseen,
        *m["seen_val_by_rung"].values(),
    ]:
        ids = {e["episode_id"] for e in group["episodes"]}
        assert not ids & train and not ids & val_ids
        val_ids |= ids


def test_named_dispatch_uses_canonical_embodiment():
    received = []
    evaluator = SimpleNamespace(
        on_validation_step=lambda batch, *args, **kwargs: received.append(batch)
    )
    resets = []
    model = SimpleNamespace(
        process_batch_for_training=lambda batch: batch,
        reset_eval_pass_counter=lambda: resets.append(True),
    )
    wrapper = SimpleNamespace(
        val_loader_names=["valid::added"],
        named_val_embodiments={"valid::added": "human_bimanual"},
        model=model,
        global_rank=0,
        _val_heads=lambda: {"valid::added": evaluator},
    )
    ModelWrapper.validation_step(wrapper, {"added": {"x": 1}}, 0)
    assert received == [{"human_bimanual": {"x": 1}}]
    ModelWrapper.validation_step(wrapper, {"added": {"x": 2}}, 1)
    assert len(resets) == 1


def test_video_only_evaluator_logs_no_metrics_and_runs_only_at_end():
    logged = []
    ev = OpenLoopSimEval(log_metrics=False, final_only=True)
    ev.trainer = SimpleNamespace(
        current_epoch=0,
        max_epochs=2,
        lightning_module=SimpleNamespace(log_dict=lambda *a, **k: logged.append(a)),
    )
    called = []
    ev.compute_metrics_and_viz = lambda *a, **k: (
        called.append(k) or {"mse": torch.tensor(1.0)},
        {},
    )
    ev.on_validation_step({}, 0)
    assert not called
    ev.trainer.current_epoch = 1
    ev.on_validation_step({}, 0)
    assert called and not logged


def test_named_evaluators_keep_prefix_and_embodiment_separate(compose_resolve):
    cfg = compose_resolve("experiments/fixed_budget_opscale/rdt_1b_opscale8", [])
    names = [
        "valid::human_bimanual",
        "valid::op8_seen_human_bimanual",
        "unseen_op_valid::flagship_human_bimanual",
        "open_loop::flagship_human_bimanual",
    ]
    dm = SimpleNamespace(
        validation_dataset_options=cfg.data.validation_dataset_options,
        val_loader_names=lambda: names,
    )
    model = SimpleNamespace(model=object())
    _attach_named_validation_evaluators(cfg, model, SimpleNamespace(), dm)
    assert list(model.named_val_evaluators) == names
    assert set(model.named_val_embodiments.values()) == {"human_bimanual"}
    assert model.named_val_evaluators[names[1]].prefix == "op8_added_seen_op_valid"
    assert model.named_val_evaluators[names[2]].prefix == "flagship_unseen_op_valid"
    video = model.named_val_evaluators[names[3]]
    assert video.video_group == "flagship_unseen_op_valid" and not video.log_metrics
