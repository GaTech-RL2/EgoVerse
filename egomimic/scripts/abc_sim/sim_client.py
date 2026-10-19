"""Roll a served EgoVerse policy out in abc_sim (amazon-far/abc) through ABC's
own eval loop and summary (abc_minimal.eval_policy.rollout_worlds /
build_summary). Runs in ABC's venv; the policy lives behind policy_server.py.

    <abc venv>/bin/python egomimic/scripts/abc_sim/sim_client.py \
        --task sim_put_the_plastic_bottles_in_the_bin --server http://127.0.0.1:8765 \
        --num-worlds 50 --out <dir>

ABC's SimEvalConfig defaults (seeded worlds seed+i with its unplaceable-seed
retry, 15 executed steps per inference, 236 chunks, early stop on success,
top/left/right at 168x224) with RTC off. ``--task`` is any abc_sim name or
alias -- the sim_224 task_name works. The prompt defaults to the one the
converter trained on (the task name with _ -> space); ``--prompt env`` keeps
the env's own (the prompt-randomized tasks). ``--policy hold`` needs no
server: it repeats the current state, the no-motion floor.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np

from abc_minimal.config import SimEvalConfig
from abc_minimal.dit import task_name_to_prompt
from abc_minimal.eval_policy import build_summary, resolved_physics, rollout_worlds
from abc_minimal.sim_env import SimTaskEnv, task_prompt

CAMERAS = ("top", "left", "right")
HORIZON = 45  # the served chunk: 1.5 s of 30 Hz commands


class HistEnv(SimTaskEnv):
    """Renders the frame ``lag`` steps before each re-plan (ABC's loop only
    renders at the re-plan itself), for the policy's 2-frame image history."""

    def __init__(self, *args, lag: int, execute: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_at, self.hist, self._since_obs = execute - lag, None, 0

    def reset(self, seed, options=None):
        self.hist, self._since_obs = None, 0
        return super().reset(seed, options)

    def obs(self):
        self._since_obs = 0
        return super().obs()

    def step_one(self, action):
        super().step_one(action)
        self._since_obs += 1
        if self.capture_at > 0 and self._since_obs == self.capture_at:
            self.hist = self.render_cameras()


class ServedPolicy:
    """abc_minimal's SimPolicy.infer surface over HTTP (noise and prefix unused)."""

    def __init__(self, server: str):
        self.server, self.env = server.rstrip("/"), None
        with urllib.request.urlopen(f"{self.server}/health", timeout=30) as r:
            self.health = json.load(r)

    def infer(self, obs, noise=None, action_prefix=None, prefix_length=0) -> np.ndarray:
        images = {c: obs["images"][c] for c in self.health["cameras"]}
        if self.env is not None and self.env.hist is not None:
            images.update({f"{c}_hist": self.env.hist[c] for c in self.health["cameras"]})
        payload = {"state": np.asarray(obs["state"], np.float32).tolist(), "prompt": obs["prompt"], "images": {
            k: {"shape": list(hwc.shape), "b64": base64.b64encode(hwc.tobytes()).decode()}
            for k, hwc in ((k, np.ascontiguousarray(np.moveaxis(np.asarray(v, np.uint8), 0, -1))) for k, v in images.items())}}
        req = urllib.request.Request(f"{self.server}/infer", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            return np.asarray(json.load(r)["actions"], dtype=np.float32)


class HoldPolicy:
    health = {"ckpt": "hold", "cameras": [], "lag_frames": 0}

    def infer(self, obs, noise=None, action_prefix=None, prefix_length=0) -> np.ndarray:
        return np.tile(np.asarray(obs["state"], np.float32)[None], (HORIZON, 1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--policy", choices=["served", "hold"], default="served")
    p.add_argument("--prompt", default=None, help="default: the training prompt; 'env' = the env's own")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-worlds", type=int, default=50)
    p.add_argument("--save-video", action="store_true")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    policy = HoldPolicy() if a.policy == "hold" else ServedPolicy(a.server)
    prompt = task_prompt(a.task) if a.prompt == "env" else (a.prompt or task_name_to_prompt(a.task))
    cfg = SimEvalConfig(checkpoint=str(policy.health["ckpt"]), task=a.task, num_worlds=a.num_worlds,
                        rtc=False, fast_inference=False, prefix_length=0, save_video=a.save_video,
                        output_dir=str(a.out), prompt=prompt)
    model_config = replace(cfg.model, chunk_length=HORIZON, action_dim=14, camera_keys=CAMERAS)
    env = HistEnv(task=a.task, height=cfg.camera_height, width=cfg.camera_width, camera_keys=CAMERAS,
                  prompt=prompt, camera_backend=cfg.camera_backend, gpu_id=cfg.gpu_id,
                  lag=int(policy.health["lag_frames"]), execute=cfg.execute_chunk_dim)
    policy.env = env
    physics = resolved_physics(env)
    worlds = rollout_worlds(cfg, policy, env, 0, None, a.out, model_config)
    build_summary(config=cfg, ckpt_path=Path(cfg.checkpoint), device="served", worlds=worlds, out_dir=a.out, physics=physics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
