"""policy_server builds the model input exactly as ZarrDataset builds a
training sample (same keymap, transforms, history frame, fps), normalizes it
with the checkpoint's norm stats, and speaks the sim client's HTTP protocol."""

from __future__ import annotations

import base64
import json
import urllib.request

import numpy as np
import torch
from fixtures.synthetic_episodes import write_episode
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.eva import JOINT_ACTION_KEY, JOINT_STATE_KEY, Eva
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.scripts.abc_sim import policy_server as ps

CAM = Eva.VIZ_IMAGE_KEY
CFG = OmegaConf.create({"data": {"train_datasets": {"eva_bimanual": {"resolver": {
    "key_map": {"_target_": "x", "keymap_mode": "joints_front", "annotation_key": "annotations", "image_history_gap_s": 0.1},
    "transform_list": {"_target_": "x", "mode": "joints"},
}}}}})


class _FakeNormStats:
    """Scales the state by 2 so the test can see normalization happened."""

    def normalize(self, data, emb_id):
        out = dict(data)
        out[JOINT_STATE_KEY] = data[JOINT_STATE_KEY] * 2
        return out


class _FakeAlgo:
    """Records the batch it was given; predicts the (normalized) state tiled."""

    norm_stats = _FakeNormStats()
    annotation_key = "annotations"

    def process_batch_for_training(self, batch):
        self.seen = batch["eva_bimanual"]
        return batch

    def forward_eval(self, batch):
        state = batch["eva_bimanual"][JOINT_STATE_KEY]
        return {f"eva_bimanual_{JOINT_ACTION_KEY}": state[:, None, :].repeat(1, 100, 1)}


def _policy():
    pol = ps.JointPolicy.__new__(ps.JointPolicy)
    pol.ckpt_path, pol.algo = "fake.ckpt", _FakeAlgo()
    pol.configure(CFG)
    return pol


def _episode_frames(root):
    """Raw frames/states of the synthetic eva episode as the sim would send
    them, read through the dataset's own reader and JPEG decode."""
    import simplejpeg

    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode

    ep = ZarrEpisode(next(root.glob("*.zarr")))
    n = ep.metadata["total_frames"]
    keys = ["images.front_1", "left.obs_joints", "left.obs_gripper", "right.obs_joints", "right.obs_gripper"]
    got = ep.read_intervals({k: [(0, n)] for k in keys})
    rows = {k: got[k][0][1] for k in keys}
    frames = [simplejpeg.decode_jpeg(rows["images.front_1"][i], colorspace="RGB") for i in range(n)]
    state = np.concatenate([np.asarray(rows[k], dtype=np.float32) for k in keys[1:]], axis=1)
    return frames, state


def test_server_sample_matches_the_dataset_sample(tmp_path):
    write_episode(tmp_path, "eva", T=48)
    frames, state = _episode_frames(tmp_path)
    key_map = Eva.get_keymap("joints_front", image_history_gap_s=0.1)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=Eva.get_transform_list("joints"))
    leaf = next(iter(MultiDataset._from_resolver(resolver, mode="total").datasets.values()))
    pol = _policy()
    for t in range(0, 12):  # feed the frames in order so the history fills like an episode
        sample = pol.build_sample("ep", t / 30.0, state[t], {"top": np.moveaxis(frames[t], -1, 0).astype(np.float32) / 255})
        ref = leaf[t]
        assert torch.equal(sample[CAM], ref[CAM]), t
        assert torch.equal(sample[f"{CAM}_hist"], ref[f"{CAM}_hist"]), t  # 3 frames back, clamped at the start
        assert torch.allclose(sample[JOINT_STATE_KEY], ref[JOINT_STATE_KEY].float()), t
        assert sample[JOINT_ACTION_KEY].shape == ref[JOINT_ACTION_KEY].shape == (100, 14)
        assert sample["fps"].item() == ref["fps"].item() == 30
    assert "observations.images.left_wrist_img" not in sample


def test_infer_normalizes_then_returns_unnormalized_chunk(tmp_path):
    pol = _policy()
    img = np.zeros((3, 16, 24), np.float32)
    state = np.arange(14, dtype=np.float32)
    out = pol.infer("ep", 0.0, state, {"top": img}, "sim put the bottles in the bin")
    assert out.shape == (100, 14)
    seen = pol.algo.seen
    assert torch.allclose(seen[JOINT_STATE_KEY][0], torch.arange(14.0) * 2)  # normalized before the model
    assert seen["annotations"] == [["sim put the bottles in the bin"]]
    assert seen["fps"].shape == (1,)


def test_http_protocol_round_trip():
    pol = _policy()
    server = ps.serve(pol, "127.0.0.1", 0)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
            assert json.load(r)["mode"] == "joints"
        hwc = np.random.default_rng(0).integers(0, 255, (16, 24, 3), np.uint8)
        payload = {"episode": "e", "t": 0.0, "state": list(range(14)), "prompt": "sim x",
                   "images": {"top": {"shape": [16, 24, 3], "b64": base64.b64encode(hwc.tobytes()).decode()}}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/infer", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.load(r)
        assert np.asarray(out["actions"]).shape == (100, 14) and out["dt"] == 1 / 30
        assert torch.allclose(pol.algo.seen[CAM][0], torch.from_numpy(np.moveaxis(hwc, -1, 0).astype(np.float32) / 255))
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/infer", data=json.dumps({"episode": "e", "t": 0, "state": [0], "images": {}}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(bad)
            assert False, "expected 500"
        except urllib.error.HTTPError as e:
            assert e.code == 500 and "error" in json.load(e)
    finally:
        server.shutdown()
