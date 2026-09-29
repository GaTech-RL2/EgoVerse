"""policy_server builds the model input exactly as ZarrDataset builds a
training sample (keymap, transforms, the 0.1 s history frame, fps), normalizes
it with the checkpoint's norm stats, returns the chunk on the raw 30 Hz grid,
and speaks the sim client's HTTP protocol."""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import numpy as np
import simplejpeg
import torch
from fixtures.synthetic_episodes import write_episode
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.eva import JOINT_ACTION_KEY, JOINT_STATE_KEY, Eva
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset, ZarrEpisode
from egomimic.scripts.abc_sim import policy_server as ps

CAM = Eva.VIZ_IMAGE_KEY
CFG = OmegaConf.create({"data": {"train_datasets": {"eva_bimanual": {"resolver": {
    "key_map": {"keymap_mode": "joints_front", "annotation_key": "annotations", "image_history_gap_s": 0.1},
    "transform_list": {"mode": "joints"},
}}}}})


class _FakeNormStats:
    """Scales the state by 2 so the test can see normalization happened."""

    def normalize(self, data, emb_id):
        return {**data, JOINT_STATE_KEY: data[JOINT_STATE_KEY] * 2}


class _FakeAlgo:
    """Records the batch; predicts a ramp in time so the resampling is visible."""

    norm_stats = _FakeNormStats()

    def process_batch_for_training(self, batch):
        self.seen = batch["eva_bimanual"]
        return batch

    def forward_eval(self, batch):
        ramp = torch.linspace(0, 99, 100)[None, :, None].repeat(1, 1, 14)
        return {f"eva_bimanual_{JOINT_ACTION_KEY}": ramp}


def _policy():
    pol = ps.JointPolicy.__new__(ps.JointPolicy)
    pol.ckpt_path, pol.algo = "fake.ckpt", _FakeAlgo()
    pol.configure(CFG)
    return pol


def _chw(hwc: np.ndarray) -> np.ndarray:
    return np.moveaxis(hwc, -1, 0).astype(np.float32) / np.float32(255.0)


def test_server_sample_matches_the_dataset_sample(tmp_path):
    write_episode(tmp_path, "eva", T=48)
    ep = ZarrEpisode(next(tmp_path.glob("*.zarr")))
    keys = ["images.front_1", "left.obs_joints", "left.obs_gripper", "right.obs_joints", "right.obs_gripper"]
    rows = {k: g[0][1] for k, g in ep.read_intervals({k: [(0, 48)] for k in keys}).items()}
    frames = [simplejpeg.decode_jpeg(rows["images.front_1"][i], colorspace="RGB") for i in range(48)]
    state = np.concatenate([np.asarray(rows[k], np.float32) for k in keys[1:]], axis=1)
    key_map = Eva.get_keymap("joints_front", image_history_gap_s=0.1)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=Eva.get_transform_list("joints"))
    leaf = next(iter(MultiDataset._from_resolver(resolver, mode="total").datasets.values()))
    pol = _policy()
    assert pol.cameras == ["top"] and pol.lag_frames == 3
    for t in (0, 1, 3, 10):
        images = {"top": _chw(frames[t])}
        if t > 0:  # the frame lag_frames back, clamped to the episode start; at t=0 none is sent
            images["top_hist"] = _chw(frames[max(0, t - pol.lag_frames)])
        sample, ref = pol.build_sample(state[t], images), leaf[t]
        assert torch.equal(sample[CAM], ref[CAM]), t
        assert torch.equal(sample[f"{CAM}_hist"], ref[f"{CAM}_hist"]), t
        assert torch.allclose(sample[JOINT_STATE_KEY], ref[JOINT_STATE_KEY].float()), t
        assert sample[JOINT_ACTION_KEY].shape == ref[JOINT_ACTION_KEY].shape == (100, 14)
        assert sample["fps"].item() == ref["fps"].item() == 30


def test_infer_normalizes_and_returns_the_raw_30hz_grid():
    pol = _policy()
    out = pol.infer(np.arange(14, dtype=np.float32), {"top": np.zeros((3, 16, 24), np.float32)}, "sim x")
    # 100 model steps span 45 raw frames: raw frame k sits at model step 99k/44
    assert out.shape == (45, 14)
    assert np.allclose(out[:, 0], np.arange(45) * 99 / 44, atol=1e-4)
    assert torch.allclose(pol.algo.seen[JOINT_STATE_KEY][0], torch.arange(14.0) * 2)
    assert pol.algo.seen["annotations"] == [["sim x"]]


def test_http_protocol_round_trip():
    pol = _policy()
    server = HTTPServer(("127.0.0.1", 0), ps.make_handler(pol))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(payload):
        req = urllib.request.Request(f"{url}/infer", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    try:
        with urllib.request.urlopen(f"{url}/health") as r:
            health = json.load(r)
        assert health["cameras"] == ["top"] and health["lag_frames"] == 3 and health["horizon"] == 45
        hwc = np.random.default_rng(0).integers(0, 255, (16, 24, 3), np.uint8)
        out = post({"state": list(range(14)), "prompt": "sim x",
                    "images": {"top": {"shape": [16, 24, 3], "b64": base64.b64encode(hwc.tobytes()).decode()}}})
        assert np.asarray(out["actions"]).shape == (45, 14) and out["dt"] == 1 / 30
        assert torch.allclose(pol.algo.seen[CAM][0], torch.from_numpy(_chw(hwc)))
        try:
            post({"state": [0], "images": {}})
            raise AssertionError("expected 500")
        except urllib.error.HTTPError as e:
            assert e.code == 500 and "error" in json.load(e)
    finally:
        server.shutdown()
