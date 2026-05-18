"""Evaluation rollout for the PushShapes (`tsim`) sim, saving per-episode MP4s
+ per-step state sidecars (`.npz`) that `scripts/rollout_viewer.py` plays
back live as the eval runs.

Runs N open-loop episodes in parallel headless workers; writes MP4s,
per-step `.npz` sidecars, per-episode JSONL metrics, and an aggregate
summary. No network / streaming — the viewer just watches the output dir.

Open-loop semantics: the H-Net policy is behavior-cloning with full-episode
action chunking — it predicts the entire `action_horizon` chunk from the
*initial* observation. We reset the env, query the policy once, then replay
every predicted action through `env.step` with no re-query. This matches how
the model is trained and validated.

See scripts/README.md for invocation + viewer instructions.

Quick self-test (no checkpoint needed — uses the random-policy fallback to
exercise the whole harness end-to-end):

    python scripts/eval_rollout.py --num-episodes 3 --num-workers 2 \\
        --max-steps 40
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# Headless rendering: the compute node has no display. Must be set before
# pygame is imported anywhere down the env import chain.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Running `python scripts/eval_rollout.py` puts scripts/ on sys.path[0], not
# the repo root, so `Tsimulation` / `egomimic` wouldn't import. Add the repo
# root explicitly (also covers spawned Pool workers).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

LOG = logging.getLogger("eval_rollout")

_ERROR_RATE_FAIL_THRESHOLD = 0.05  # exit nonzero if >5% of episodes errored

# The `.npz` sidecar is written AFTER the MP4 is released, so its presence
# is a safe sentinel for "episode is ready to play" — rollout_viewer.py
# watches for these.
_STATE_SIDECAR_SUFFIX = ".npz"


# --------------------------------------------------------------------------- #
# Episode result
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class EpisodeResult:
    """One episode's metrics. Serialized (sorted by episode_idx) to
    episodes.jsonl. `error` is None on success, else a traceback string."""

    episode_idx: int
    seed: int
    worker_id: int
    scenario: dict
    success: bool = False
    episode_length: int = 0
    episode_return: float = 0.0
    final_coverage: float = 0.0
    max_coverage: float = 0.0
    final_distance_to_goal: float = float("nan")
    min_distance_to_goal: float = float("nan")
    time_to_first_success: int | None = None
    video_path: str | None = None
    state_path: str | None = None  # .npz sidecar consumed by rollout_viewer.py
    error: str | None = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Policy loading (generic ModelWrapper) + open-loop action-chunk runner
# --------------------------------------------------------------------------- #


class RandomPolicy:
    """Harness self-test fallback. Emits a uniformly-random action chunk so
    the full pipeline (env, MP4, metrics, determinism) can be tested
    without a trained checkpoint. NOT a real policy."""

    def __init__(self, world_size: float, action_dim: int = 2):
        self.world_size = world_size
        self.action_dim = action_dim

    def action_chunk(
        self, env_obs: dict, rng: np.random.Generator, horizon: int
    ) -> np.ndarray:
        return rng.uniform(
            0.0, self.world_size, size=(horizon, self.action_dim)
        ).astype(np.float32)


class PolicyRunner:
    """Wraps a loaded ModelWrapper algo and turns one env observation into a
    full open-loop action chunk by reusing the algo's own
    `process_batch_for_training` + `forward_eval` path (so normalization /
    keyname-resolution is identical to validation — we don't reimplement it).
    """

    # Batch key names produced by the pushshapes ZarrDataset keymap
    # (egomimic/rldb/embodiment/pushshapes.get_keymap).
    _STATE_KEY = "state_agent_obj"
    _IMAGE_KEY = "front_img_1"
    _ACTION_KEY = "actions"
    _EMB_NAME = "pushshapes_sim"

    def __init__(self, algo: Any, device: str):
        self.algo = algo
        self.device = device
        import torch

        self._torch = torch
        self.emb_id = self._emb_id()
        self.ac_key = self.algo.resolved_ac_keys[self.emb_id]
        # Action chunk length the policy will produce.
        self.horizon = int(self.algo.nets["policy"].action_horizon)

    def _emb_id(self) -> int:
        from egomimic.rldb.embodiment.embodiment import get_embodiment_id

        return get_embodiment_id(self._EMB_NAME)

    def action_chunk(
        self, env_obs: dict, rng: np.random.Generator, horizon: int
    ) -> np.ndarray:
        """env_obs (raw PushShapesEnv obs) -> (horizon, 2) action chunk in
        raw 512-px world coords. `rng`/`horizon` accepted for a uniform
        interface with RandomPolicy; the policy's own action_horizon governs
        the real chunk length, then we trim/pad to `horizon`."""
        torch = self._torch
        # state = [agent_x, agent_y, obj_x, obj_y, obj_theta] — exactly the
        # 5-D observations.state the writer stored and the loader feeds.
        state = np.concatenate([env_obs["agent_pos"], env_obs["object_pose"]]).astype(
            np.float32
        )
        image = np.asarray(env_obs["image"])  # (H, W, 3) uint8
        # ZarrDataset hands images back as float (C, H, W); match that.
        image_chw = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Build a single-sample, single-timestep batch in the same shape the
        # dataloader would yield. Dummy GT actions: forward_eval only uses
        # their leading dim (B) and packed-detection, never their values.
        raw = {
            self._STATE_KEY: torch.from_numpy(state)[None],  # (1, 5)
            self._IMAGE_KEY: torch.from_numpy(image_chw)[None],  # (1, 3, H, W)
            self._ACTION_KEY: torch.zeros(1, self.horizon, 2),  # (1, T, 2)
        }
        batch = {self._EMB_NAME: raw}
        with torch.no_grad():
            processed = self.algo.process_batch_for_training(batch)
            preds = self.algo.forward_eval(processed)

        pred_key = f"emb{self.emb_id}_{self.ac_key}"
        chunk = preds.get(pred_key)
        if chunk is None:
            raise RuntimeError(
                f"policy produced no prediction under {pred_key!r}; "
                f"got keys {list(preds)}"
            )
        chunk = chunk.detach().cpu().float().numpy()[0]  # (T, action_dim)
        return _fit_horizon(chunk, horizon)


def _fit_horizon(chunk: np.ndarray, horizon: int) -> np.ndarray:
    """Trim or last-value-pad an action chunk to exactly `horizon` rows."""
    if chunk.shape[0] >= horizon:
        return chunk[:horizon]
    pad = np.repeat(chunk[-1:], horizon - chunk.shape[0], axis=0)
    return np.concatenate([chunk, pad], axis=0)


def load_policy(policy_path: str | None, device: str):
    """Generic ModelWrapper checkpoint loader: reads `config_tree` +
    `norm_stats_state` out of the Lightning checkpoint's hyper_parameters
    and re-instantiates exactly as ModelWrapper / trainHydra eval mode does.
    Works for any ModelWrapper ckpt (HNet / HPT / variants).

    Returns (runner, is_real_policy)."""
    if policy_path in (None, "", "none"):
        LOG.warning(
            "No --policy-path given: using RandomPolicy fallback. This "
            "exercises the harness only; metrics are meaningless."
        )
        return None, False  # caller substitutes RandomPolicy (needs world size)

    import torch

    from egomimic.pl_utils.pl_model import ModelWrapper

    ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "config_tree" not in hp or "norm_stats_state" not in hp:
        raise RuntimeError(
            f"{policy_path} is missing config_tree/norm_stats_state in "
            "hyper_parameters — not a ModelWrapper checkpoint?"
        )
    model = ModelWrapper(
        config_tree=hp["config_tree"],
        norm_stats_state=hp["norm_stats_state"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    algo = model.model
    algo.device = device
    algo.nets = algo.nets.to(device).eval()
    LOG.info("Loaded policy from %s", policy_path)
    return PolicyRunner(algo, device), True


# --------------------------------------------------------------------------- #
# Per-episode rollout (open-loop)
# --------------------------------------------------------------------------- #


def _distance_to_goal(obs: dict) -> float:
    """Object-centroid → goal-centroid euclidean distance (world px). The env
    has a single object so this is the mean-across-blocks degenerate case."""
    return float(np.linalg.norm(obs["object_pose"][:2] - obs["goal_pose"][:2]))


def _write_state_sidecar(
    sidecar: Path,
    *,
    pusher_obs: np.ndarray,
    object_obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    coverages: np.ndarray,
    goal_pose: np.ndarray,
    metadata: dict,
) -> None:
    """Write the per-step `.npz` consumed by rollout_viewer.py. This is the
    last thing run_episode does so the sidecar's existence is the "episode
    is ready" sentinel the viewer polls for."""
    np.savez(
        sidecar,
        pusher_obs=pusher_obs,
        object_obs=object_obs,
        actions=actions,
        rewards=rewards,
        coverages=coverages,
        goal_pose=goal_pose,
        metadata_json=np.array(json.dumps(metadata), dtype=object),
    )


def run_episode(
    *,
    episode_idx: int,
    seed: int,
    worker_id: int,
    args: argparse.Namespace,
    policy,
    record_video: bool,
) -> EpisodeResult:
    """One open-loop episode. Never raises — errors are captured into the
    result so one bad episode can't crash the run."""
    from Tsimulation.pushshapes.env import PushShapesEnv

    scenario = dict(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
    )
    res = EpisodeResult(
        episode_idx=episode_idx,
        seed=seed,
        worker_id=worker_id,
        scenario=scenario,
    )
    env = None
    vw = None
    # Per-step buffers — parallel arrays of length T. We record the
    # pre-step state alongside the action we're about to apply, matching
    # Tsimulation/visualize_episode.py's convention so the viewer's HUD
    # reads naturally next to each rendered frame.
    pusher_buf: list[np.ndarray] = []
    object_buf: list[np.ndarray] = []
    action_buf: list[np.ndarray] = []
    reward_buf: list[float] = []
    coverage_buf: list[float] = []
    goal_pose: np.ndarray | None = None
    try:
        env = PushShapesEnv(
            object_shape=args.object,
            pusher_shape=args.pusher,
            obstacle_level=args.obstacles,
            image_size=args.image_size,
            render_mode="rgb_array" if record_video else None,
        )
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        goal_pose = np.asarray(obs["goal_pose"], dtype=np.float32).copy()

        # Open-loop: one policy query on the initial observation. Both
        # RandomPolicy and PolicyRunner expose the same action_chunk(obs,
        # rng, horizon) signature.
        chunk = policy.action_chunk(obs, rng, args.max_steps)

        if record_video:
            vid_dir = Path(args.output_dir) / "videos"
            vid_dir.mkdir(parents=True, exist_ok=True)
            vpath = vid_dir / f"episode_{episode_idx:04d}.mp4"
            vw = cv2.VideoWriter(
                str(vpath),
                cv2.VideoWriter_fourcc(*args.video_codec),
                args.video_fps,
                (int(env.WORLD_SIZE), int(env.WORLD_SIZE)),
            )
            if not vw.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {vpath}")
            res.video_path = str(vpath)

        ep_return = 0.0
        max_cov = 0.0
        min_dist = _distance_to_goal(obs)
        steps = 0
        for t in range(min(args.max_steps, chunk.shape[0])):
            # Snapshot pre-step state alongside the action we're about to
            # issue — the resulting frame is what env.render() returns next.
            pusher_buf.append(np.asarray(obs["agent_pos"], dtype=np.float32).copy())
            object_buf.append(np.asarray(obs["object_pose"], dtype=np.float32).copy())
            action_buf.append(np.asarray(chunk[t], dtype=np.float32).copy())

            obs, reward, terminated, truncated, info = env.step(chunk[t])
            steps += 1
            ep_return += float(reward)
            cov = float(info.get("coverage", 0.0))
            max_cov = max(max_cov, cov)
            min_dist = min(min_dist, _distance_to_goal(obs))
            reward_buf.append(float(reward))
            coverage_buf.append(cov)
            success_now = terminated  # coverage >= SUCCESS_THRESHOLD
            if success_now and res.time_to_first_success is None:
                res.time_to_first_success = steps

            if record_video:
                frame = env.render()  # (512,512,3) uint8 RGB
                if vw is not None and frame is not None:
                    vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                break

        res.episode_length = steps
        res.episode_return = ep_return
        res.final_coverage = float(info.get("coverage", 0.0))
        res.max_coverage = max_cov
        res.final_distance_to_goal = _distance_to_goal(obs)
        res.min_distance_to_goal = min_dist
        res.success = res.time_to_first_success is not None
    except Exception:
        res.error = traceback.format_exc()
        LOG.error("episode %d errored:\n%s", episode_idx, res.error)
    finally:
        if vw is not None:
            vw.release()
        if env is not None:
            env.close()

    # Sentinel write — only after VideoWriter.release() so the MP4 is
    # finalized on disk. Skip if nothing useful got recorded (errored
    # before step 0, or the user disabled video).
    if res.video_path is not None and pusher_buf:
        sidecar = Path(res.video_path).with_suffix(_STATE_SIDECAR_SUFFIX)
        try:
            _write_state_sidecar(
                sidecar,
                pusher_obs=np.stack(pusher_buf),
                object_obs=np.stack(object_buf),
                actions=np.stack(action_buf),
                rewards=np.asarray(reward_buf, dtype=np.float32),
                coverages=np.asarray(coverage_buf, dtype=np.float32),
                goal_pose=goal_pose
                if goal_pose is not None
                else np.zeros(3, dtype=np.float32),
                metadata=dict(
                    episode_idx=episode_idx,
                    seed=seed,
                    worker_id=worker_id,
                    scenario=scenario,
                    fps=int(args.video_fps),
                    world_size=512,
                    episode_return=res.episode_return,
                    success=res.success,
                    final_coverage=res.final_coverage,
                    max_coverage=res.max_coverage,
                    error=res.error,
                ),
            )
            res.state_path = str(sidecar)
        except Exception:
            # Sidecar failure is non-fatal: the MP4 still plays; the
            # viewer just won't show the HUD overlay for this episode.
            LOG.exception("episode %d: failed to write state sidecar", episode_idx)
    return res


# --------------------------------------------------------------------------- #
# Worker orchestration
# --------------------------------------------------------------------------- #


def _headless_worker(payload: dict) -> EpisodeResult:
    """Pool worker: one episode. SDL dummy is inherited via the module-level
    os.environ.setdefault, re-asserted here for spawn-start."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    logging.basicConfig(level=logging.INFO)
    args = payload["args"]
    policy = _make_worker_policy(args)
    return run_episode(
        episode_idx=payload["episode_idx"],
        seed=payload["seed"],
        worker_id=payload["worker_id"],
        args=args,
        policy=policy,
        record_video=(args.record_video == "all"),
    )


# Each spawned worker loads its own policy once (lazily cached per process).
_WORKER_POLICY = None


def _make_worker_policy(args: argparse.Namespace):
    global _WORKER_POLICY
    if _WORKER_POLICY is not None:
        return _WORKER_POLICY
    runner, is_real = load_policy(args.policy_path, args.device)
    if not is_real:
        runner = RandomPolicy(world_size=512.0)
    _WORKER_POLICY = runner
    return runner


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _summarize(
    results: list[EpisodeResult], args: argparse.Namespace, wall_s: float
) -> dict:
    ok = [r for r in results if r.error is None]
    n = len(ok)
    succ = sum(r.success for r in ok)
    returns = np.array([r.episode_return for r in ok]) if ok else np.array([0.0])
    lengths = np.array([r.episode_length for r in ok]) if ok else np.array([0])
    fdist = (
        np.array([r.final_distance_to_goal for r in ok])
        if ok
        else np.array([float("nan")])
    )
    lo, hi = _wilson_ci(succ, n)

    def _stats(a: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "stderr": float(np.std(a) / math.sqrt(len(a))) if len(a) else 0.0,
        }

    return {
        "num_episodes": len(results),
        "num_ok": n,
        "num_errored": len(results) - n,
        "success_rate": (succ / n) if n else 0.0,
        "success_rate_wilson95": [lo, hi],
        "return": _stats(returns),
        "episode_length": _stats(lengths),
        "final_distance_to_goal": _stats(fdist),
        "scenario": dict(
            object_shape=args.object,
            pusher_shape=args.pusher,
            obstacle_level=args.obstacles,
        ),
        "wall_clock_s": wall_s,
        "episodes_per_s": (len(results) / wall_s) if wall_s > 0 else 0.0,
        "cli_args": vars(args),
        "git_sha": _git_sha(),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    out = Path(args.output_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / "rollout.log"),
        ],
    )
    LOG.info("output dir: %s", out)
    (out / "config_resolved.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n"
    )

    n_total = args.num_episodes
    base_seed = args.seed
    # Episode i always gets seed base+i regardless of which worker runs it →
    # byte-identical episodes.jsonl for a fixed seed + policy.
    episodes = [
        dict(
            episode_idx=i, seed=base_seed + i, worker_id=i % args.num_workers, args=args
        )
        for i in range(n_total)
    ]

    results: list[EpisodeResult] = []
    stop = threading.Event()

    def _sigint(_sig, _frm):
        LOG.warning(
            "SIGINT — finishing in-flight episode, then writing partial results"
        )
        stop.set()

    signal.signal(signal.SIGINT, _sigint)

    t0 = time.monotonic()

    # All episodes fan out to a process pool. rollout_viewer.py polls the
    # output dir and picks up each `.npz` sidecar as it lands, then plays
    # the matching MP4. No in-process display-episode special case anymore.
    if episodes and not stop.is_set():
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            for r in pool.imap_unordered(_headless_worker, episodes):
                results.append(r)
                LOG.info(
                    "episode %d done (success=%s, ret=%.3f)%s",
                    r.episode_idx,
                    r.success,
                    r.episode_return,
                    " [ERROR]" if r.error else "",
                )
                if stop.is_set():
                    pool.terminate()
                    break

    wall = time.monotonic() - t0

    # Deterministic output: sort by episode_idx before writing.
    results.sort(key=lambda r: r.episode_idx)
    with (out / "episodes.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_json(), sort_keys=True) + "\n")
    summary = _summarize(results, args, wall)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    LOG.info(
        "done: %d episodes, %d errored, success_rate=%.3f, %.1fs",
        summary["num_episodes"],
        summary["num_errored"],
        summary["success_rate"],
        wall,
    )
    err_rate = summary["num_errored"] / max(1, summary["num_episodes"])
    return 1 if err_rate > _ERROR_RATE_FAIL_THRESHOLD else 0


def _build_parser() -> argparse.ArgumentParser:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy-path",
        default=None,
        help="ModelWrapper .ckpt; omit/`none` for random-policy self-test",
    )
    p.add_argument(
        "--config",
        default=None,
        help="optional env/scenario config (reserved; scenario is "
        "set via --object/--pusher/--obstacles today)",
    )
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="per-episode horizon (caps the open-loop chunk replay)",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="base seed; episode i uses seed+i"
    )
    p.add_argument("--output-dir", default=f"./eval_runs/{ts}")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="force argmax/mean action (policy-dependent; no-op for "
        "the deterministic H-Net generate path)",
    )
    p.add_argument(
        "--record-video",
        choices=["all", "none"],
        default="all",
        help="`all` (default): write MP4 + .npz per episode; `none`: skip "
        "both. rollout_viewer.py needs the MP4 + .npz to play episodes.",
    )
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-codec", default="mp4v")
    p.add_argument("--device", default="cpu")
    # Scenario (this env's notion of "scenario").
    p.add_argument("--object", default="T", choices=["T", "U", "Z"])
    p.add_argument("--pusher", default="circle", choices=["circle", "stick"])
    p.add_argument("--obstacles", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--image-size", type=int, default=96)
    return p


if __name__ == "__main__":
    sys.exit(main())
