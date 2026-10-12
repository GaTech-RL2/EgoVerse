"""Open-loop prefix MSE: executed steps only, split into xyz / ypr / grip."""

import numpy as np
import torch

from egomimic.eval.open_loop_sim import (
    executed_control_steps,
    open_loop_channel_mse,
)
from egomimic.utils.action_utils import _ypr_to_matrix


def test_execute_fraction_is_a_prefix_of_the_chunk():
    assert executed_control_steps(100, 0.25) == 25
    assert executed_control_steps(100, 1.0) == 100
    assert executed_control_steps(1, 0.25) == 1
    assert executed_control_steps(100, 0.25, execute_steps=30) == 30
    assert executed_control_steps(20, 0.25, execute_steps=30) == 20


def test_execute_steps_ignores_the_tail_past_the_cap():
    pred = torch.zeros(1, 100, 14)
    gt = torch.zeros(1, 100, 14)
    gt[:, 30:, 0] = 9.0
    metrics = open_loop_channel_mse(pred, gt, execute_fraction=0.25, execute_steps=30)
    assert float(metrics["XYZ_MSE"]) == 0.0


def test_prefix_mse_ignores_the_unexecuted_tail():
    pred = torch.zeros(1, 8, 14)
    gt = torch.zeros(1, 8, 14)
    gt[:, 4:, 0] = 5.0  # past the first half
    metrics = open_loop_channel_mse(pred, gt, execute_fraction=0.5)
    assert float(metrics["XYZ_MSE"]) == 0.0
    assert float(metrics["YPR_MSE"]) == 0.0
    assert float(metrics["Grip_MSE"]) == 0.0
    assert float(metrics["Paired_MSE"]) == 0.0


def test_channel_mse_splits_a_14d_command():
    pred = torch.zeros(1, 4, 14)
    gt = torch.zeros(1, 4, 14)
    gt[:, :2, 0] = 1.0  # left x, both executed steps
    gt[:, :2, 6] = 2.0  # left gripper
    gt[:, :2, 3] = 0.5  # left yaw
    metrics = open_loop_channel_mse(pred, gt, execute_fraction=0.5)
    # prefix is 2 steps. xyz has 6 channels, one of them off by 1.
    assert float(metrics["XYZ_MSE"]) == torch.tensor(1.0 / 6)
    # ypr has 6 channels, one of them off by 0.5.
    assert float(metrics["YPR_MSE"]) == torch.tensor(0.25 / 6)
    # grip has 2 channels, one of them off by 2.
    assert float(metrics["Grip_MSE"]) == torch.tensor(4.0 / 2)
    # paired is xyz + grip (8 channels): one 1^2 and one 2^2.
    assert float(metrics["Paired_MSE"]) == torch.tensor((1.0 + 4.0) / 8)


def test_18d_rotation_is_scored_as_yaw_pitch_roll():
    gt = torch.zeros(1, 4, 18)
    pred = gt.clone()
    yaw = torch.tensor([[[0.3, 0.0, 0.0]]])
    rotation = _ypr_to_matrix(yaw)[0, 0]  # (3, 3)
    pred[:, :, 3:6] = rotation[:, 0]
    pred[:, :, 6:9] = rotation[:, 1]
    metrics = open_loop_channel_mse(pred, gt, execute_fraction=1.0)
    assert "Grip_MSE" not in metrics
    assert float(metrics["XYZ_MSE"]) < 1e-8
    assert float(metrics["Paired_MSE"]) < 1e-8
    assert float(metrics["YPR_MSE"]) > 1e-4


def test_pad_mask_drops_repeat_last_steps():
    pred = torch.zeros(1, 4, 14)
    gt = torch.zeros(1, 4, 14)
    gt[:, 2:, 0] = 3.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    metrics = open_loop_channel_mse(pred, gt, execute_fraction=1.0, valid_steps=mask)
    assert float(metrics["XYZ_MSE"]) == 0.0


def test_keypoint_width_is_not_an_open_loop_command():
    pred = torch.zeros(2, 8, 144)
    assert open_loop_channel_mse(pred, pred, execute_fraction=0.25) == {}


class _Leaf:
    def __init__(self, name, n):
        self.name = name
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"episode_hash": self.name, "i": idx}


def test_open_loop_loader_walks_seen_then_unseen_at_the_stride():
    from omegaconf import OmegaConf

    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    from egomimic.trainHydra import _open_loop_datasets

    def folder(names, n=90):
        return MultiDataset(
            datasets={name: _Leaf(name, n) for name in names}, mode="total"
        )

    cfg = OmegaConf.create(
        {
            "data": {
                "open_loop_episodes": {
                    "seen": ["s1", "s2"],
                    "unseen": ["u1"],
                },
                "open_loop_stride": 30,
            },
            "open_loop_evaluator": {"execute_steps": 30},
        }
    )
    built = _open_loop_datasets(
        cfg,
        {
            "valid": {"human": folder(["s1", "s2", "other"])},
            "unseen_op_valid": {"human": folder(["u1", "left_out"])},
        },
    )
    walked = [built["human"][i]["episode_hash"] for i in range(len(built["human"]))]
    assert walked == ["s1", "s1", "s1", "s2", "s2", "s2", "u1", "u1", "u1"]


def test_open_loop_eval_saves_every_episode_and_uploads_one_per_group(
    tmp_path, monkeypatch
):
    from lightning.pytorch.loggers import WandbLogger

    from egomimic.eval.open_loop_sim import OpenLoopSimEval
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    written = []

    def fake_write(path, frames, fps, video_codec):
        written.append((path, tuple(frames.shape), fps, video_codec))

    monkeypatch.setattr("egomimic.eval.open_loop_sim.tvio.write_video", fake_write)

    class _Logger(WandbLogger):
        def __init__(self):
            self.calls = []

        def log_video(self, name, paths, step=None, caption=None, format=None):
            self.calls.append((name, list(paths)))

    class _Algo:
        def __init__(self, emb_id):
            self.ac_keys = {emb_id: "actions_cartesian"}

        def forward_eval(self, batch):
            gt = next(iter(batch.values()))["actions_cartesian"]
            return {"human_bimanual_actions_cartesian": torch.zeros_like(gt)}

        class norm_stats:
            @staticmethod
            def unnormalize(batch, embodiment_id):
                return batch

    logger = _Logger()
    logged = []

    class _Module:
        device = torch.device("cpu")

        def log_dict(self, metrics, **kwargs):
            logged.append(set(metrics))

    class _Trainer:
        is_global_zero = True
        current_epoch = 0
        global_step = 4
        max_epochs = 10
        default_root_dir = str(tmp_path)
        loggers = [logger]
        lightning_module = _Module()

    emb = get_embodiment_id("human_bimanual")
    seen = ["seen_a", "seen_b"]
    unseen = ["unseen_a", "unseen_b"]
    ev = OpenLoopSimEval(
        execute_steps=30,
        seen_episodes=seen,
        unseen_episodes=unseen,
        wandb_videos=1,
        viz_every_n_epochs=1,
        viz_mode="overlay",
        viz_func={
            "human_bimanual": lambda predictions, batch: np.zeros(
                (batch["actions_cartesian"].shape[0], 4, 4, 3), dtype=np.uint8
            )
        },
    )
    ev.trainer = _Trainer()
    ev.model = _Algo(emb)

    def step(hashes):
        gt = torch.zeros(len(hashes), 100, 14)
        gt[:, 0, 0] = 1.0
        gt[:, 40, 0] = 9.0
        ev.on_validation_step(
            {
                emb: {
                    "actions_cartesian": gt,
                    "episode_hash": list(hashes),
                    "embodiment": torch.tensor([emb] * len(hashes)),
                }
            },
            0,
        )

    ev.on_validation_start()
    step(seen)
    step(unseen)
    ev.on_validation_end()

    assert any("Valid/open_loop_sim/XYZ_MSE" in keys for keys in logged)
    assert any("unseen_op_valid/Valid/open_loop_sim/XYZ_MSE" in keys for keys in logged)
    assert "Valid/human_bimanual_loss" not in set().union(*logged)
    names = [path.split("/")[-1] for path, *_rest in written]
    assert names == [
        "seen_a.mp4",
        "seen_b.mp4",
        "unseen_a.mp4",
        "unseen_b.mp4",
    ]
    assert {fps for _path, _shape, fps, _codec in written} == {30}
    # No keypoint action in this batch, so each sample is one frame.
    assert {shape[0] for _path, shape, _fps, _codec in written} == {1}
    assert [name for name, _paths in logger.calls] == [
        "videos_open_loop/seen",
        "videos_open_loop/unseen",
    ]
    assert logger.calls[0][1][0].endswith("seen/seen_a.mp4")
    assert logger.calls[1][1][0].endswith("unseen/unseen_a.mp4")


def test_chunk_step_matches_the_source_frame_not_the_vector_index():
    from egomimic.eval.replay_viz import chunk_step_for_video_frame

    # 30 source frames resampled to 100 steps: the ends meet, and a later
    # source frame is not the same index in the chunk.
    assert chunk_step_for_video_frame(0, 30, 100) == 0
    assert chunk_step_for_video_frame(29, 30, 100) == 99
    assert chunk_step_for_video_frame(15, 30, 100) != 15


def _hand(xyz):
    return np.tile(np.asarray(xyz, dtype=np.float32), 42)


def test_overlay_draws_only_the_current_frames_skeleton():
    from egomimic.eval.replay_viz import (
        chunk_step_for_video_frame,
        overlay_executed_chunk,
    )
    from egomimic.rldb.embodiment.human import Human

    span, steps = 30, 100
    images = np.zeros((span, 64, 64, 3), dtype=np.uint8)
    gt = np.stack([_hand((0.0, -0.25, 1.0)) for _ in range(steps)])
    pred = np.stack([_hand((0.0, 0.0, 1.0)) for _ in range(steps)])
    k = chunk_step_for_video_frame(15, span, steps)
    pred[k] = _hand((0.5, 0.0, 1.0))
    gt_frames = np.stack([_hand((0.0, -0.25, 1.0)) for _ in range(span)])
    K = np.array([[40, 0, 32, 0], [0, 40, 32, 0], [0, 0, 1, 0]], dtype=np.float32)
    frames = overlay_executed_chunk(
        images,
        gt,
        pred,
        gt_frames,
        K,
        Human.FINGER_EDGES,
        n_window=span,
    )
    assert frames.shape == (span, 64, 64, 3)
    # Step 0 projects to (32, 32); the later step projects to (52, 32).
    # Green ground truth is at (32, 22) on every frame.
    assert frames[0, 32, 32, 0] > 200
    assert frames[0, 32, 52, 0] == 0
    assert frames[0, 22, 32, 1] > 200
    assert frames[15, 32, 52, 0] > 200
    assert frames[15, 32, 32, 0] == 0


def test_keypoint_window_writes_one_source_frame_per_executed_step(
    tmp_path, monkeypatch
):
    from egomimic.eval.open_loop_sim import OpenLoopSimEval
    from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment_id

    written = []
    monkeypatch.setattr(
        "egomimic.eval.open_loop_sim.tvio.write_video",
        lambda path, frames, fps, video_codec: written.append(
            (tuple(frames.shape), fps)
        ),
    )
    read = []

    class _Leaf:
        total_frames = 80

        def __getitem__(self, idx):
            read.append(idx)
            return {
                "actions_keypoints": torch.zeros(100, 144),
                "observations.images.front_img_1": torch.zeros(3, 4, 4),
                "intrinsics": torch.tensor(
                    [[1, 0, 2, 0], [0, 1, 2, 0], [0, 0, 1, 0]], dtype=torch.float32
                ),
            }

    def fake_apply(cls, batch, transform_list):
        b = batch["actions_keypoints"].shape[0]
        return {"actions_keypoints": torch.zeros(b, 100, 126)}

    monkeypatch.setattr(Embodiment, "apply_transform", classmethod(fake_apply))

    class _Algo:
        def __init__(self, emb_id):
            self.ac_keys = {emb_id: "actions_cartesian"}

        def forward_eval(self, batch):
            item = next(iter(batch.values()))
            return {
                "human_bimanual_actions_cartesian": torch.zeros_like(
                    item["actions_cartesian"]
                ),
                "human_bimanual_actions_keypoints": torch.zeros_like(
                    item["actions_keypoints"]
                ),
            }

        class norm_stats:
            @staticmethod
            def unnormalize(batch, embodiment_id):
                return batch

    class _Module:
        device = torch.device("cpu")

        def log_dict(self, metrics, **kwargs):
            pass

    class _Trainer:
        is_global_zero = True
        current_epoch = 0
        global_step = 1
        max_epochs = 1
        default_root_dir = str(tmp_path)
        loggers = []
        lightning_module = _Module()
        datamodule = type("DM", (), {})()

    emb = get_embodiment_id("human_bimanual")
    ev = OpenLoopSimEval(
        execute_steps=30,
        seen_episodes=["ep"],
        wandb_videos=0,
        viz_every_n_epochs=1,
        viz_mode="overlay",
        viz_func={"human_bimanual": lambda predictions, batch: None},
        transform_lists={"human_bimanual": ["revert"]},
    )
    ev.trainer = _Trainer()
    ev.trainer.datamodule.open_loop_datasets = {"human": type("DS", (), {})()}
    ev.trainer.datamodule.open_loop_datasets["human"].datasets = {"ep": _Leaf()}
    ev.model = _Algo(emb)

    ev.on_validation_start()
    ev.on_validation_step(
        {
            emb: {
                "actions_cartesian": torch.zeros(1, 100, 14),
                "actions_keypoints": torch.zeros(1, 100, 144),
                "episode_hash": ["ep"],
                "frame_idx": torch.tensor([10]),
                "embodiment": torch.tensor([emb]),
            }
        },
        0,
    )
    ev.on_validation_end()

    assert read == list(range(10, 40))
    assert written == [((30, 4, 4, 3), 30)]


def test_named_video_metric_scores_full_chunk_thirds():
    from egomimic.eval.open_loop_sim import OpenLoopSimEval

    ev = OpenLoopSimEval(
        execute_steps=2,
        seen_episodes=["episode"],
        metric_prefix="episode/Valid/open_loop_sim",
    )
    pred = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).view(1, 6, 1).expand(1, 6, 14)
    metrics = {}
    ev._score_groups(metrics, ["episode"], pred, torch.zeros_like(pred), None)
    prefix = "episode/Valid/open_loop_sim/XYZ_MSE"
    assert metrics[prefix] == 1
    assert metrics[f"{prefix}/early"] == 1
    assert metrics[f"{prefix}/mid"] == 4
    assert metrics[f"{prefix}/late"] == 9
