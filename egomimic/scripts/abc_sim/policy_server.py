"""Serve an EgoVerse joint-space checkpoint over HTTP for the abc_sim client.

The simulator's Python (amazon-far/abc: py3.12, MuJoCo 3.8, torch 2.11) cannot
share the training venv, so the policy runs here, in the training venv, and
the sim talks to it over plain HTTP + JSON (stdlib on both ends).

    python -m egomimic.scripts.abc_sim.policy_server --ckpt <epoch_N.ckpt> --port 8765

POST /infer  {"episode": str, "t": float seconds, "state": [14], "prompt": str,
              "images": {"top": {"shape": [H, W, 3], "b64": <raw uint8 bytes>}, ...}}
         ->  {"actions": [[14] x 100], "dt": 1/30}
GET  /health ->  {"ok": true, "ckpt": ..., "mode": "joints"}

The inputs are built EXACTLY as ZarrDataset builds a training sample for the
checkpoint's data config -- the same Eva keymap and transform list, then the
checkpoint's own norm stats (the dataset normalizes samples in __getitem__;
``process_batch_for_training`` does not, which the old robot rollout missed).
The ``_hist`` camera is the frame ``image_history_gap_s`` earlier, kept per
episode; at an episode start it is the current frame, as in training.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from egomimic.rldb.embodiment.embodiment import IMAGE_HISTORY_SUFFIX, get_embodiment_id
from egomimic.rldb.embodiment.eva import (
    JOINT_ACTION_KEY,
    JOINT_MODE,
    JOINT_RAW_HORIZON,
    Eva,
)

log = logging.getLogger("abc_sim.policy_server")

EMB = "eva_bimanual"
FPS = 30.0
# abc_sim camera -> the zarr key the converter wrote it to
CAMERA_ZARR = {"top": "images.front_1", "left": "images.left_wrist", "right": "images.right_wrist"}


def decode_image(spec: dict) -> np.ndarray:
    """{"shape": [H, W, 3], "b64": raw uint8} -> (3, H, W) float32 in [0, 1], RGB."""
    h, w, c = spec["shape"]
    raw = np.frombuffer(base64.b64decode(spec["b64"]), dtype=np.uint8).reshape(h, w, c)
    return np.moveaxis(raw, -1, -3).astype(np.float32) / np.float32(255.0)


def split_state(state) -> dict[str, np.ndarray]:
    """ABC 14-D state -> the zarr proprio keys of Eva's joint keymap."""
    s = np.asarray(state, dtype=np.float32).reshape(14)
    return {
        "left.obs_joints": s[0:6],
        "left.obs_gripper": s[6:7],
        "right.obs_joints": s[7:13],
        "right.obs_gripper": s[13:14],
    }


class JointPolicy:
    """The checkpoint plus the data pipeline it was trained with."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from egomimic.pl_utils.pl_model import ModelWrapper

        self.ckpt_path = ckpt_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        wrapper = ModelWrapper.load_from_checkpoint(ckpt_path, weights_only=False, map_location="cpu")
        self.algo = wrapper.model.to(self.device).eval()
        self.algo.device = self.device
        cfg = wrapper._as_config(getattr(wrapper.hparams, "config_tree", None))
        self.configure(cfg)

    def configure(self, cfg) -> None:
        """Keymap + transform list from the checkpoint's data config."""
        resolver = OmegaConf.select(cfg, f"data.train_datasets.{EMB}.resolver")
        if resolver is None:
            raise ValueError(f"checkpoint config has no data.train_datasets.{EMB}")
        km = {k: v for k, v in OmegaConf.to_container(resolver.key_map, resolve=True).items() if k != "_target_"}
        tl = {k: v for k, v in OmegaConf.to_container(resolver.transform_list, resolve=True).items() if k != "_target_"}
        if tl.get("mode") != JOINT_MODE:
            raise ValueError(f"policy_server serves Eva mode '{JOINT_MODE}', checkpoint has {tl.get('mode')!r}")
        self.key_map = Eva.get_keymap(**{k: v for k, v in km.items() if k != "annotation_key"})
        self.annotation_key = km.get("annotation_key")
        self.transforms = Eva.get_transform_list(**tl)
        self.gap_s = km.get("image_history_gap_s")
        self.cameras = {
            k: v["zarr_key"]
            for k, v in self.key_map.items()
            if v.get("key_type") == "camera_keys" and not k.endswith(IMAGE_HISTORY_SUFFIX)
        }
        self.emb_id = get_embodiment_id(EMB)
        self.history: dict[str, deque] = {}
        self.lock = threading.Lock()

    def _past_frame(self, episode: str, t: float, frame: np.ndarray) -> np.ndarray:
        """The front frame round(gap_s * fps) frames before t -- the dataset's
        _seconds_to_frames lag, in frames not seconds -- clamped to the oldest
        frame this episode (an episode start: the current frame)."""
        lag = max(1, int(round(self.gap_s * FPS)))
        hist = self.history.setdefault(episode, deque(maxlen=lag + 1))
        hist.append((int(round(t * FPS)), frame))
        idx = hist[-1][0] - lag
        return next((f for (i, f) in hist if i == idx), hist[0][1])

    def build_sample(self, episode: str, t: float, state, images: dict[str, np.ndarray]) -> dict:
        """One un-normalized sample with the dataset's keys (the 45-step cmd
        chunk is faked from the current pose: only its shape reaches the model)."""
        raw = split_state(state)
        raw.update({f"{arm}.cmd_{part}": np.tile(raw[f"{arm}.obs_{part}"], (JOINT_RAW_HORIZON, 1))
                    for arm in ("left", "right") for part in ("joints", "gripper")})
        sample = {}
        for key, spec in self.key_map.items():
            zk = spec["zarr_key"]
            if spec.get("key_type") == "camera_keys":
                cam = next(c for c, z in CAMERA_ZARR.items() if z == zk)
                if cam not in images:
                    raise ValueError(f"camera '{cam}' missing from the request (have {sorted(images)})")
                frame = images[cam]
                if key.endswith(IMAGE_HISTORY_SUFFIX):
                    frame = self._past_frame(episode, t, images[cam])
                sample[key] = frame
            elif spec.get("key_type") in ("proprio_keys", "action_keys"):
                sample[key] = raw[zk]
        for tf in self.transforms:
            sample = tf.transform(sample)
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = torch.from_numpy(v).to(torch.float32)
        sample["fps"] = torch.tensor(FPS, dtype=torch.float32)
        return sample

    @torch.no_grad()
    def infer(self, episode: str, t: float, state, images: dict[str, np.ndarray], prompt: str) -> np.ndarray:
        with self.lock:
            sample = self.build_sample(episode, t, state, images)
            sample = self.algo.norm_stats.normalize(sample, self.emb_id)
            batch = default_collate([{k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}])
            if self.annotation_key:
                batch[self.annotation_key] = [[prompt]]
            processed = self.algo.process_batch_for_training({EMB: batch})
            preds = self.algo.forward_eval(processed)
        return preds[f"{EMB}_{JOINT_ACTION_KEY}"][0].float().cpu().numpy()

    def reset(self, episode: str) -> None:
        with self.lock:
            self.history.pop(episode, None)


def make_handler(policy: JointPolicy):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": policy.ckpt_path, "mode": JOINT_MODE})
            else:
                self._send(404, {"error": self.path})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            try:
                if self.path == "/reset":
                    policy.reset(req["episode"])
                    self._send(200, {"ok": True})
                elif self.path == "/infer":
                    images = {k: decode_image(v) for k, v in req["images"].items()}
                    actions = policy.infer(req["episode"], float(req["t"]), req["state"], images, req.get("prompt", ""))
                    self._send(200, {"actions": actions.tolist(), "dt": 1.0 / FPS})
                else:
                    self._send(404, {"error": self.path})
            except Exception as e:  # the client must see the reason, not a hang
                log.exception("request failed")
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):  # quiet
            pass

    return Handler


def serve(policy: JointPolicy, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(policy))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    policy = JointPolicy(a.ckpt, a.device)
    server = serve(policy, a.host, a.port)
    log.info("serving %s on http://%s:%d (cameras %s, gap %s s)", a.ckpt, a.host, a.port, sorted(policy.cameras), policy.gap_s)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
