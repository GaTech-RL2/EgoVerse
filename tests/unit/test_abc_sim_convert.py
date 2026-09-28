"""ABC sim_224 episode -> EgoVerse zarr: states / actions land in the joint
keymap's keys in ABC order, the three stacked cameras are split onto
front/left/right, prompts follow ABC's sim rule, and the joint-mode dataset
reads the converted episode with ABC's own train/val split filterable."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from egomimic.rldb.embodiment.eva import JOINT_ACTION_KEY, JOINT_STATE_KEY, Eva
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.scripts.abc_sim import convert_sim_to_zarr as conv

T, H, W = 60, 16, 24
TASK = "sim_put_the_plastic_bottles_in_the_bin"


def _write_abc_episode(root, split, name, meta_extra=None):
    ep = root / f"{split}_sim" / name
    ep.mkdir(parents=True)
    t = np.arange(T, dtype=np.float64)[:, None]
    states = np.tile(np.arange(14, dtype=np.float64), (T, 1)) + 0 * t
    actions = states + 0.01 * t
    np.concatenate([states, actions], axis=1).astype(np.float64).tofile(
        ep / "states_actions.bin"
    )
    # Three cameras stacked vertically, each a flat colour so the split is checkable.
    with av.open(str(ep / "combined_camera-images-rgb.mp4"), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = W, 3 * H, "yuv420p"
        s.options = {"crf": "0"}
        for _ in range(T):
            img = np.zeros((3 * H, W, 3), np.uint8)
            img[:H] = (200, 30, 30)
            img[H : 2 * H] = (30, 200, 30)
            img[2 * H :] = (30, 30, 200)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    meta = {"task_name": TASK, "num_steps": T, "cameras": ["top", "left", "right"]}
    meta.update(meta_extra or {})
    (ep / "episode_metadata.json").write_text(json.dumps(meta))
    return ep


def test_prompt_rule_matches_abc():
    assert conv.prompt_spans({"task_name": TASK}, 10) == [
        ("sim put the plastic bottles in the bin", 0, 10)
    ]
    spans = conv.prompt_spans(
        {
            "task_name": "multi_drawer_search",
            "prompt_timeline": [
                {"frame": 0, "prompt": "find the cup"},
                {"frame": 4, "prompt": "find the ball"},
            ],
        },
        10,
    )
    assert spans == [("sim find the cup", 0, 4), ("sim find the ball", 4, 10)]
    spans = conv.prompt_spans(
        {"task_name": "put_relative", "instruction": "put the cube left of the mug"}, 5
    )
    assert spans == [("sim put the cube left of the mug", 0, 5)]


def test_intrinsics_are_the_sim_pinhole():
    K = conv.pinhole_intrinsics(168, 224)
    assert K.shape == (3, 4) and K[0, 2] == 112 and K[1, 2] == 84
    assert K[1, 1] == pytest.approx(84 / np.tan(np.radians(29)))


def test_convert_and_read_back(tmp_path):
    src, out = tmp_path / "cache", tmp_path / "zarr"
    _write_abc_episode(src, "train", "episode_0001")
    _write_abc_episode(src, "val", "episode_0002")
    _write_abc_episode(src, "train", "episode_0003", {"task_name": "sim_pouring_beads"})
    assert conv.main(["--src", str(src), "--task", TASK, "--out", str(out), "--workers", "1"]) == 0
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["written"] == 2 and not report["failed"]
    # Idempotent: a rerun skips both.
    conv.main(["--src", str(src), "--task", TASK, "--out", str(out), "--workers", "1"])
    assert json.loads((out / "conversion_report.json").read_text())["skipped"] == 2

    resolver = LocalEpisodeResolver(
        out,
        key_map=Eva.get_keymap("joints", annotation_key="annotations"),
        transform_list=Eva.get_transform_list("joints"),
    )
    filters = DatasetFilter(filter_lambdas=["lambda row: row.get('split') == 'train'"])
    ds = MultiDataset._from_resolver(resolver, filters=filters, mode="total")
    assert len(ds.datasets) == 1
    leaf = next(iter(ds.datasets.values()))
    sample = leaf[5]
    assert torch.allclose(sample[JOINT_STATE_KEY].float(), torch.arange(14.0))
    assert torch.allclose(
        sample[JOINT_ACTION_KEY][0].float(), torch.arange(14.0) + 0.05, atol=1e-5
    )
    assert sample["annotations"] == ["sim put the plastic bottles in the bin"]
    front = sample[Eva.VIZ_IMAGE_KEY]
    assert front.shape == (3, H, W)
    # top camera is red-dominant, left wrist green, right wrist blue
    assert front[0].mean() > front[1].mean()
    assert sample["observations.images.left_wrist_img"][1].mean() > 0.5
    assert sample["observations.images.right_wrist_img"][2].mean() > 0.5
    assert sample["fps"].item() == 30
