"""Serve an EgoVerse joint-space checkpoint over HTTP for the abc_sim client.

The simulator's Python (amazon-far/abc: py3.12, MuJoCo 3.8, torch 2.11) cannot
share the training venv, so the policy runs here and the sim talks to it over
plain HTTP + JSON (stdlib on both ends).

    python -m egomimic.scripts.abc_sim.policy_server --ckpt <epoch_N.ckpt> --port 8765

POST /infer  {"state": [14], "prompt": str,
              "images": {"top": {"shape": [H, W, 3], "b64": <raw uint8 RGB>},
                         "top_hist": {...}}}      # optional: the frame lag_frames back
         ->  {"actions": [[14] x 45], "dt": 1/30}
GET  /health ->  {"ok": true, "ckpt", "cameras", "lag_frames", "horizon"}

The input is built exactly as ZarrDataset builds a training sample for the
checkpoint's data config (keymap, transform list, fps) and normalized with
the checkpoint's norm stats: the dataset normalizes in __getitem__, and
process_batch_for_training does not. ``top_hist`` is the frame
``image_history_gap_s`` earlier; without it the current frame stands in, which
is what an episode start looks like in training. The model's 100-step chunk
spans 45 raw 30 Hz frames, so it is resampled back onto that grid: action k
is the command k frames ahead.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from egomimic.rldb.embodiment.embodiment import IMAGE_HISTORY_SUFFIX, get_embodiment_id
from egomimic.rldb.embodiment.eva import JOINT_ACTION_KEY, JOINT_MODE, JOINT_RAW_HORIZON, Eva
from egomimic.scripts.abc_sim.convert_sim_to_zarr import CAMERA_TO_ZARR, split_arms
from egomimic.utils.pose_utils import _interpolate_linear

log = logging.getLogger("abc_sim.policy_server")

EMB = "eva_bimanual"
FPS = 30.0
ZARR_TO_CAMERA = {z: c for c, z in CAMERA_TO_ZARR.items()}


def decode_image(spec: dict) -> np.ndarray:
    """{"shape": [H, W, 3], "b64": raw uint8} -> (3, H, W) float32 in [0, 1]."""
    raw = np.frombuffer(base64.b64decode(spec["b64"]), dtype=np.uint8).reshape(spec["shape"])
    return np.moveaxis(raw, -1, -3).astype(np.float32) / np.float32(255.0)


class JointPolicy:
    """The checkpoint plus the data pipeline it was trained with."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from egomimic.pl_utils.pl_model import ModelWrapper

        self.ckpt_path = ckpt_path
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        # The algo is a plain object; the Lightning wrapper owns (and moves) its nets.
        self.wrapper = ModelWrapper.load_from_checkpoint(ckpt_path, weights_only=False, map_location="cpu")
        self.wrapper.to(device).eval()
        self.algo = self.wrapper.model
        self.algo.device = device
        self.configure(self.wrapper._as_config(self.wrapper.hparams.config_tree))

    def configure(self, cfg) -> None:
        """Keymap + transform list from the checkpoint's data config."""
        resolver = OmegaConf.select(cfg, f"data.train_datasets.{EMB}.resolver")
        if resolver is None:
            raise ValueError(f"checkpoint config has no data.train_datasets.{EMB}")
        km = {k: v for k, v in OmegaConf.to_container(resolver.key_map, resolve=True).items() if k != "_target_"}
        tl = {k: v for k, v in OmegaConf.to_container(resolver.transform_list, resolve=True).items() if k != "_target_"}
        if tl.get("mode") != JOINT_MODE:
            raise ValueError(f"policy_server serves Eva mode '{JOINT_MODE}', checkpoint has {tl.get('mode')!r}")
        self.annotation_key = km.pop("annotation_key", None)
        self.key_map = Eva.get_keymap(**km)
        self.transforms = Eva.get_transform_list(**tl)
        self.lag_frames = max(1, round(km["image_history_gap_s"] * FPS)) if km.get("image_history_gap_s") else 0
        self.cameras = sorted({ZARR_TO_CAMERA[v["zarr_key"]] for v in self.key_map.values() if v.get("key_type") == "camera_keys"})
        self.emb_id = get_embodiment_id(EMB)

    def build_sample(self, state, images: dict[str, np.ndarray]) -> dict:
        """One un-normalized sample with the dataset's keys. The 45-step cmd
        chunk is the current pose tiled: only its shape reaches the model."""
        raw = {k: v[0] for k, v in split_arms(np.asarray(state, np.float32).reshape(1, 14), "obs").items()}
        raw.update({k.replace("obs_", "cmd_"): np.tile(v, (JOINT_RAW_HORIZON, 1)) for k, v in list(raw.items())})
        sample = {}
        for key, spec in self.key_map.items():
            if spec.get("key_type") == "camera_keys":
                cam = ZARR_TO_CAMERA[spec["zarr_key"]]
                if cam not in images:
                    raise ValueError(f"camera '{cam}' missing from the request (have {sorted(images)})")
                past = images.get(cam + IMAGE_HISTORY_SUFFIX, images[cam])
                sample[key] = past if key.endswith(IMAGE_HISTORY_SUFFIX) else images[cam]
            elif spec.get("key_type") in ("proprio_keys", "action_keys"):
                sample[key] = raw[spec["zarr_key"]]
        for tf in self.transforms:
            sample = tf.transform(sample)
        sample = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in sample.items() if isinstance(v, (np.ndarray, torch.Tensor))}
        sample["fps"] = torch.tensor(FPS)
        return sample

    @torch.no_grad()
    def infer(self, state, images: dict[str, np.ndarray], prompt: str) -> np.ndarray:
        """(45, 14) absolute joint + gripper commands, one per 30 Hz step."""
        sample = self.algo.norm_stats.normalize(self.build_sample(state, images), self.emb_id)
        batch = default_collate([sample])
        if self.annotation_key:
            batch[self.annotation_key] = [[prompt]]
        preds = self.algo.forward_eval(self.algo.process_batch_for_training({EMB: batch}))
        chunk = preds[f"{EMB}_{JOINT_ACTION_KEY}"][0].float().cpu().numpy()
        return _interpolate_linear(chunk, JOINT_RAW_HORIZON)


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
            if self.path != "/health":
                return self._send(404, {"error": self.path})
            self._send(200, {"ok": True, "ckpt": policy.ckpt_path, "cameras": policy.cameras,
                             "lag_frames": policy.lag_frames, "horizon": JOINT_RAW_HORIZON})

        def do_POST(self):
            if self.path != "/infer":
                return self._send(404, {"error": self.path})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                images = {k: decode_image(v) for k, v in req["images"].items()}
                actions = policy.infer(req["state"], images, req.get("prompt", ""))
                self._send(200, {"actions": actions.tolist(), "dt": 1.0 / FPS})
            except Exception as e:  # the client must see the reason, not a hang
                log.exception("request failed")
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    policy = JointPolicy(a.ckpt, a.device)
    log.info("serving %s on http://%s:%d (cameras %s, lag %d frames)", a.ckpt, a.host, a.port, policy.cameras, policy.lag_frames)
    HTTPServer((a.host, a.port), make_handler(policy)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
