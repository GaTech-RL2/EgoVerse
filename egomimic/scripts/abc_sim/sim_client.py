"""Roll a served EgoVerse policy out in abc_sim (amazon-far/abc) and score it
with ABC's own task evaluators. Runs in ABC's venv: stdlib + numpy + abc_minimal
only (no egomimic import; the policy lives behind policy_server.py).

    <abc venv>/bin/python egomimic/scripts/abc_sim/sim_client.py \
        --task put_plastic_bottles_in_bin --server http://127.0.0.1:8765 \
        --num-worlds 50 --out <dir>

Protocol = ABC's SimEvalConfig defaults without RTC: worlds seeded seed+i,
one inference per 15 executed steps (execute_chunk_dim) out of the policy's
100-step chunk, 236 chunks max (~118 s at 30 Hz), early stop on success
(rollout_over), cameras top/left/right at 168x224. summary.json carries the
same fields ABC's build_summary writes (success_rate, num_success,
mean_reward, mean_max_progress, worlds[]) so the two are comparable side by
side. ``--policy hold`` needs no server: it repeats the current state (the
no-motion floor and a protocol smoke).
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from abc_minimal.eval_policy import jsonable, rollout_over, video_frame
from abc_minimal.sim_env import SimTaskEnv, task_prompt

CAMERAS = ("top", "left", "right")


class ServedPolicy:
    def __init__(self, server: str):
        self.server = server.rstrip("/")
        with urllib.request.urlopen(f"{self.server}/health", timeout=30) as r:
            self.health = json.load(r)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.server}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def reset(self, episode: str) -> None:
        self._post("/reset", {"episode": episode})

    def infer(self, episode: str, t: float, obs: dict) -> np.ndarray:
        images = {}
        for cam, img in obs["images"].items():
            hwc = np.ascontiguousarray(np.moveaxis(np.asarray(img, dtype=np.uint8), 0, -1))  # (3,H,W)->(H,W,3)
            images[cam] = {"shape": list(hwc.shape), "b64": base64.b64encode(hwc.tobytes()).decode()}
        out = self._post("/infer", {
            "episode": episode, "t": t, "state": np.asarray(obs["state"], dtype=np.float32).tolist(),
            "prompt": obs["prompt"], "images": images,
        })
        return np.asarray(out["actions"], dtype=np.float32)


class HoldPolicy:
    health = {"ckpt": "hold", "mode": "hold"}

    def reset(self, episode: str) -> None:
        pass

    def infer(self, episode: str, t: float, obs: dict) -> np.ndarray:
        return np.tile(np.asarray(obs["state"], dtype=np.float32)[None], (100, 1))


def rollout(env: SimTaskEnv, policy, *, seed: int, num_chunks: int, execute: int, video_path=None) -> dict:
    episode = f"seed{seed}"
    policy.reset(episode)
    obs = env.reset(seed=seed)
    video = None
    if video_path is not None:
        import imageio.v2 as imageio

        video = imageio.get_writer(str(video_path), fps=30, macro_block_size=1)
        video.append_data(video_frame(obs["images"], CAMERAS))
    result = env.evaluate()
    max_reward = float(result.get("reward", 0.0))
    steps, infer_s, t0 = 0, [], time.perf_counter()
    try:
        for _ in range(num_chunks):
            t_inf = time.perf_counter()
            actions = policy.infer(episode, steps / 30.0, obs)
            infer_s.append(time.perf_counter() - t_inf)
            for action in actions[:execute]:
                env.step_one(action)
                result = env.evaluate()
                max_reward = max(max_reward, float(result.get("reward", 0.0)))
                steps += 1
                if video is not None:
                    video.append_data(video_frame(env.render_cameras(), CAMERAS))
                if rollout_over(result):
                    break
            if rollout_over(result):
                break
            obs = env.obs()
    finally:
        if video is not None:
            video.close()
    return {
        "world_seed": seed,
        "success": bool(result["ever_success"]),
        "final_success": bool(result["success"]),
        "reward": float(result["reward"]),
        "max_reward": max_reward,
        "steps": steps,
        "wall_s": time.perf_counter() - t0,
        "infer_s_mean": float(np.mean(infer_s)) if infer_s else None,
        "randomization": env.randomization,
        "final_task_eval": result,
        "video_path": str(video_path) if video_path is not None else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True, help="abc_sim task name / alias (e.g. put_plastic_bottles_in_bin)")
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--policy", choices=["served", "hold"], default="served")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-worlds", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260511, help="ABC's SimEvalConfig default")
    p.add_argument("--num-chunks", type=int, default=236)
    p.add_argument("--execute", type=int, default=15, help="steps executed per inference (ABC's execute_chunk_dim)")
    p.add_argument("--camera-backend", default="mjwarp")
    p.add_argument("--gpu-id", type=int, default=None)
    p.add_argument("--save-videos", type=int, default=2, help="record the first N worlds")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    policy = HoldPolicy() if a.policy == "hold" else ServedPolicy(a.server)
    env = SimTaskEnv(task=a.task, height=168, width=224, camera_keys=CAMERAS, prompt=task_prompt(a.task),
                     camera_backend=a.camera_backend, gpu_id=a.gpu_id)
    worlds = []
    try:
        for i in range(a.num_worlds):
            video = a.out / f"world_{i:03d}.mp4" if i < a.save_videos else None
            w = rollout(env, policy, seed=a.seed + i, num_chunks=a.num_chunks, execute=a.execute, video_path=video)
            w["world_index"] = i
            worlds.append(w)
            print(f"world={i:03d} success={w['success']} max_progress={w['max_reward']:.3f} steps={w['steps']} "
                  f"infer={1000 * (w['infer_s_mean'] or 0):.0f}ms", flush=True)
            (a.out / "worlds.jsonl").open("a").write(json.dumps(jsonable(w)) + "\n")
    finally:
        env.close()
    succ = np.array([w["success"] for w in worlds], dtype=bool)
    summary = {
        "format": "egoverse_abc_sim_rollout/v1",
        "policy": policy.health,
        "task": env.spec.name,
        "prompt": env.prompt,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "success_rate": float(succ.mean()) if len(succ) else None,
        "num_success": int(succ.sum()),
        "num_worlds": len(worlds),
        "mean_reward": float(np.mean([w["reward"] for w in worlds])) if worlds else None,
        "mean_max_progress": float(np.mean([w["max_reward"] for w in worlds])) if worlds else None,
        "worlds": worlds,
    }
    (a.out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print(f"summary: success_rate={summary['success_rate']} num_success={summary['num_success']}/{len(worlds)} "
          f"mean_max_progress={summary['mean_max_progress']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
