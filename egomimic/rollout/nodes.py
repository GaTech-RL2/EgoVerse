"""Concrete rollout nodes. Each is `node(state) -> state`.

The two decisions that were previously hardcoded are now their own nodes:
  ObsCadence   -- WHEN to query the policy
  ChunkCommit  -- WHICH predicted actions actually get executed
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from egomimic.rollout.core import RolloutNode

# Mirrors egomimic/robot/rollout.py:154 (QUERY_FREQUENCY = 30, DEFAULT_FREQUENCY
# = 30 Hz) so the rollout nodes have the robot's real cadence as their default
# off-robot too. Stated once, here, so a change on the robot side shows up as a
# visible mismatch rather than two silently different replan rates.
QUERY_FREQUENCY = 30          # frames between inferences (30 Hz loop -> 1 Hz)


# --------------------------------------------------------------------------- #
# 1. WHEN do we query the policy
# --------------------------------------------------------------------------- #
class ObsCadence(RolloutNode):
    """Decide whether to run the policy this env step.

    modes
      on_queue_empty : replan only when there are no committed actions left.
                       The natural default for a chunked policy -- with
                       chunk_len 100 and n_keep 25 that is one query per 25 env
                       steps instead of the current one PER STEP.
      every_n        : fixed cadence, regardless of the queue.
      always         : query every step (what replan_stride=1 does today; at
                       num_inference_steps=100 this is ~100 denoiser passes per
                       env step, which is why episodes took ~66 min).

    DEFAULTS come from the live robot loop (egomimic/robot/rollout.py:153-154):
        DEFAULT_FREQUENCY = 30 Hz      # control loop
        QUERY_FREQUENCY   = 30 frames  # `if i % query_frequency == 0`
    i.e. ONE inference per second, 30 actions executed per plan. mode="every_n"
    with every_n=30 reproduces that modulo exactly. "on_queue_empty" is
    equivalent while a replan fits its budget and degrades better when one
    overruns -- it waits for the queue to drain instead of skipping a beat.

    ``min_interval`` is a hard floor in env steps -- a safety rail for hardware,
    where a replan that overruns the control period is worse than a stale plan.
    """

    reads = ("t", "queue")
    writes = ("should_query",)

    def __init__(self, mode: str = "every_n", every_n: int = QUERY_FREQUENCY,
                 min_interval: int = 0):
        if mode not in ("on_queue_empty", "every_n", "always"):
            raise ValueError(f"ObsCadence: bad mode {mode!r}")
        self.mode, self.every_n = mode, int(every_n)
        self.min_interval = int(min_interval)
        self._last_query_t: Optional[int] = None

    def reset(self, state):
        self._last_query_t = None

    def __call__(self, state):
        t = int(state["t"])
        if self.mode == "always":
            want = True
        elif self.mode == "every_n":
            want = (t % self.every_n) == 0
        else:
            want = len(state.get("queue") or []) == 0
        if want and self._last_query_t is not None and self.min_interval > 0:
            if t - self._last_query_t < self.min_interval:
                want = False
        state["should_query"] = bool(want)
        if want:
            self._last_query_t = t
        return state


# --------------------------------------------------------------------------- #
# 2. obs -> model-ready tensors
# --------------------------------------------------------------------------- #
class ObsAssemble(RolloutNode):
    """Env obs -> normalized model input, via the SAME transforms training used.

    Passing the training transform_list in (rather than reimplementing the
    conversion here) is the point: a rollout that preprocesses differently from
    training is the classic silent train/deploy skew.
    """

    reads = ("obs", "should_query")
    writes = ("obs_norm",)

    def __init__(self, transforms: Sequence[Any], norm_stats, emb_id: int,
                 image_keys: Sequence[str] = ()):
        self.transforms = list(transforms or [])
        self.norm_stats = norm_stats
        self.emb_id = int(emb_id)
        self.image_keys = list(image_keys)

    def __call__(self, state):
        if not state.get("should_query"):
            return state                      # cheap: skip on non-query steps
        b = dict(state["obs"])
        for tf in self.transforms:
            b = tf.transform(b)
        state["obs_norm"] = self.norm_stats.normalize(b, self.emb_id)
        return state


# --------------------------------------------------------------------------- #
# 3. run the model graph
# --------------------------------------------------------------------------- #
class PolicyStep(RolloutNode):
    """Query PipelineAlgo -> a (C, D) action chunk.

    Wraps init_step_state/step, which is the streaming rollout interface --
    NOT forward_eval, which is teacher-forced and expects a full batch with
    ground-truth actions. rollout.py calls forward_eval today; that is the
    main reason it cannot drive this algo.
    """

    reads = ("should_query", "obs_norm", "t")
    writes = ("chunk", "chunk_t", "prev_chunk", "policy_state")

    def __init__(self, algo, emb_name: str, device=None, dtype=None,
                 t_max: Optional[int] = None):
        self.algo, self.emb_name = algo, str(emb_name)
        self.device, self.dtype, self.t_max = device, dtype, t_max

    def reset(self, state):
        p = next(self.algo.nets.parameters())
        state["policy_state"] = self.algo.init_step_state(
            batch_size=1, T_max=int(self.t_max or self.algo.action_horizon),
            device=self.device or p.device, dtype=self.dtype or p.dtype)

    @torch.no_grad()
    def __call__(self, state):
        if not state.get("should_query"):
            return state
        if state.get("policy_state") is None:
            self.reset(state)
        chunk = self.algo.step(state["policy_state"], state["obs_norm"],
                               int(state["t"]), embodiment_id=self.emb_name)
        state["prev_chunk"] = state.get("chunk")
        state["chunk"] = chunk                     # (C, D)
        state["chunk_t"] = int(state["t"])
        return state


# --------------------------------------------------------------------------- #
# 4. WHICH predicted actions get executed
# --------------------------------------------------------------------------- #
class ChunkCommit(RolloutNode):
    """Push the executable slice of the chunk onto the queue.

    n_keep      how many of the C predicted steps to actually execute before
                replanning. n_keep == C is open-loop; n_keep == 1 is fully
                closed-loop (and 100x the inference cost at C=100).
                Default 30 = QUERY_FREQUENCY from the live robot loop, which at
                30 Hz is one second of actions per plan. NOTE with chunk_len 100
                the model predicts 100 steps and only the first 30 execute; rows
                30..99 are discarded each replan (rollout.py does the same via
                `act_i = i % query_frequency`). Those unused rows are exactly
                what `blend` recycles.
    blend       temporal ensembling in the DP/ACT sense. The previous plan
                already forecast this window at rows [n_keep : 2*n_keep], so
                    commit[j] = blend*prev[n_keep+j] + (1-blend)*new[j]
                which directly attacks replan-seam discontinuity. This was a
                PUSHSHAPES_PLAN_BLEND env var read inside inference_step; here
                it is a declared parameter that lands in the run config.
    """

    reads = ("should_query", "chunk", "chunk_t", "prev_chunk", "queue")
    writes = ("queue",)

    def __init__(self, n_keep: int = QUERY_FREQUENCY, blend: float = 0.0,
                 replace_queue: bool = True):
        self.n_keep, self.blend = int(n_keep), float(blend)
        self.replace_queue = bool(replace_queue)
        if not (0.0 <= self.blend <= 1.0):
            raise ValueError(f"ChunkCommit: blend must be in [0,1], got {blend}")

    def __call__(self, state):
        if not state.get("should_query") or state.get("chunk") is None:
            return state
        chunk = state["chunk"]
        if chunk.dim() != 2:
            raise ValueError(
                f"ChunkCommit: expected a (C, D) chunk, got {tuple(chunk.shape)}. "
                f"A 3-D tensor means the per-token chunk was not selected -- "
                f"PolicyStep should return pred_action at the LAST token.")
        C = chunk.shape[0]
        n = max(1, min(self.n_keep, C))
        commit = chunk[:n]

        prev = state.get("prev_chunk")
        if self.blend > 0 and prev is not None and prev.shape[0] >= 2 * n \
                and prev.shape[-1] == chunk.shape[-1]:
            commit = (self.blend * prev[n:2 * n] + (1 - self.blend) * commit)

        rows = [commit[i] for i in range(commit.shape[0])]
        state["queue"] = rows if self.replace_queue else list(state["queue"]) + rows
        return state


# --------------------------------------------------------------------------- #
# 5. one action per env step
# --------------------------------------------------------------------------- #
class ActionDequeue(RolloutNode):
    """Pop the action to execute this step.

    on_empty="hold" repeats the last action rather than crashing -- on hardware
    a dropped control cycle is worse than a stale command. It is logged, since
    silently holding is also how a dead policy looks healthy.
    """

    reads = ("queue",)
    writes = ("action",)

    def __init__(self, on_empty: str = "hold"):
        if on_empty not in ("hold", "raise"):
            raise ValueError(f"ActionDequeue: bad on_empty {on_empty!r}")
        self.on_empty = on_empty
        self._last = None
        self._holds = 0

    def reset(self, state):
        self._last, self._holds = None, 0

    def __call__(self, state):
        q = state.get("queue") or []
        if q:
            state["action"] = q.pop(0)
            state["queue"] = q
            self._last = state["action"]
            return state
        if self.on_empty == "raise" or self._last is None:
            raise RuntimeError(
                "ActionDequeue: action queue empty and no previous action. "
                "ObsCadence never fired, or ChunkCommit committed nothing.")
        self._holds += 1
        if self._holds in (1, 10, 100):
            print(f"[ActionDequeue] queue empty, holding last action "
                  f"(count={self._holds})", flush=True)
        state["action"] = self._last
        return state


# --------------------------------------------------------------------------- #
# 6. action vector -> robot/env command
# --------------------------------------------------------------------------- #
class ActionDecode(RolloutNode):
    """Unnormalize and destructure the action into a per-arm command.

    ``layout`` is declared, not hardcoded. The current rollout.py assumes the
    old 14-D quat space with literal slices (actions[:, :7], gripper at index
    6); the space is now 20-D [xyz(3) rot6d(6) grip(1)] x 2 with grippers at 9
    and 19, so every one of those slices is silently wrong. Declaring the
    layout means a change of action space is a config edit and a mismatch is an
    error rather than a wrong number.
    """

    reads = ("action",)
    writes = ("command",)

    def __init__(self, norm_stats, emb_id: int, ac_key: str,
                 per_arm: Sequence[str] = ("xyz:3", "rot6d:6", "grip:1"),
                 arms: Sequence[str] = ("left", "right"),
                 rot_out: str = "rot6d"):
        self.norm_stats, self.emb_id, self.ac_key = norm_stats, int(emb_id), ac_key
        self.fields = [(f.split(":")[0], int(f.split(":")[1])) for f in per_arm]
        self.arms = list(arms)
        self.width = sum(w for _, w in self.fields)
        self.rot_out = str(rot_out)

    def __call__(self, state):
        a = state.get("action")
        if a is None:
            return state
        if torch.is_tensor(a):
            a = a.detach().float().cpu()
        un = self.norm_stats.unnormalize({self.ac_key: a[None, :]}, self.emb_id)
        vec = np.asarray(un[self.ac_key])[0]

        expect = self.width * len(self.arms)
        if vec.shape[-1] != expect:
            raise ValueError(
                f"ActionDecode: action is {vec.shape[-1]}-D but the declared "
                f"layout {self.fields} x {len(self.arms)} arms = {expect}-D. "
                f"Update `per_arm` to match the model's action space.")

        cmd: Dict[str, Dict[str, np.ndarray]] = {}
        for i, arm in enumerate(self.arms):
            off, out = i * self.width, {}
            for name, w in self.fields:
                out[name] = vec[off:off + w]
                off += w
            if "rot6d" in out and self.rot_out == "matrix":
                out["R"] = _rot6d_to_matrix(out["rot6d"])
            cmd[arm] = out
        state["command"] = cmd
        return state


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(6,) first-two-columns -> (3,3) via Gram-Schmidt (Zhou et al. 2019)."""
    a1, a2 = np.asarray(d6[:3], float), np.asarray(d6[3:6], float)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


# --------------------------------------------------------------------------- #
# 6b. MODEL action space -> ROBOT action interface
# --------------------------------------------------------------------------- #
class ActionToRobot(RolloutNode):
    """Convert OUR action format to the one the robot interface accepts.

        model : 2 x [xyz(3), rot6d(6), grip(1)] = 20
        robot : 2 x [xyz(3), ypr(3),   grip(1)] = 14

    The conversion is exactly the sequence PolicyRollout.rollout_step already
    performs, and it REUSES those functions rather than restating them, so the
    deploy path cannot drift from the robot path:

        rot6d -> R (Gram-Schmidt) -> ZYX euler
        [optional] cam_frame_to_base_frame(pose6, extrinsics[arm])
        [optional] rot_ee_frame_to_ee_pose_batch(pose6)
        concat gripper

    apply_frame_transforms DEFAULTS TO FALSE, and that is load-bearing.
    The two frame ops exist to UNDO the cam/wrist encoding that
    Eva.get_transform_list applies:

        "cartesian"                -> world -> CAMFRAME
                                      (ActionChunkCoordinateFrameTransform)
        "cartesian_wristframe_ypr" -> world -> camframe -> WRIST frame,
                                      i.e. actions RELATIVE to the current EEF

    The fold pipeline (eva_normal_transforms) applies NEITHER: it runs
    PoseToRot6D on the raw cmd_ee_pose, so the model predicts ABSOLUTE poses in
    the frame the zarr stores. Undoing an encoding that was never applied
    silently corrupts the command -- the arm moves smoothly to the wrong place,
    which is the hardest failure to spot on hardware. Set this True only for a
    checkpoint actually TRAINED through the cam/wrist transform list.

    ``rot_ee_frame_to_ee_pose_batch`` and its R_t_e calibration constant live in
    egomimic.robot.rollout, which imports robot_utils and therefore only
    imports ON the robot. It is resolved lazily; off-robot, inject it via
    ``ee_frame_fn`` (the test harness does). It is deliberately NOT copied here
    -- duplicating a calibration constant is how the two paths silently diverge.
    """

    reads = ("action",)
    writes = ("command",)

    def __init__(self, norm_stats, emb_id: int, ac_key: str, extrinsics: dict,
                 arms: Sequence[str] = ("left", "right"),
                 per_arm_in: Sequence[str] = ("xyz:3", "rot6d:6", "grip:1"),
                 ee_frame_fn=None, cam_to_base_fn=None,
                 apply_frame_transforms: bool = False):
        self.norm_stats, self.emb_id, self.ac_key = norm_stats, int(emb_id), ac_key
        self.extrinsics, self.arms = extrinsics, list(arms)
        self.fields = [(f.split(":")[0], int(f.split(":")[1])) for f in per_arm_in]
        self.width_in = sum(w for _, w in self.fields)
        self._ee_fn, self._cam_fn = ee_frame_fn, cam_to_base_fn
        self.apply_frame_transforms = bool(apply_frame_transforms)

    def _resolve(self):
        if self._ee_fn is None:
            try:
                from egomimic.robot.rollout import rot_ee_frame_to_ee_pose_batch
                self._ee_fn = rot_ee_frame_to_ee_pose_batch
            except Exception as e:
                raise ImportError(
                    "ActionToRobot: could not import rot_ee_frame_to_ee_pose_batch "
                    f"from egomimic.robot.rollout ({type(e).__name__}: {e}). That "
                    "module needs robot_utils, so off-robot you must pass "
                    "ee_frame_fn=... explicitly. Not copied here on purpose: the "
                    "R_t_e calibration must have exactly one definition.") from e
        if self._cam_fn is None:
            from egomimic.utils.egomimicUtils import cam_frame_to_base_frame
            self._cam_fn = cam_frame_to_base_frame

    def __call__(self, state):
        a = state.get("action")
        if a is None:
            return state
        if self.apply_frame_transforms:
            self._resolve()
        if torch.is_tensor(a):
            a = a.detach().float().cpu()
        un = self.norm_stats.unnormalize({self.ac_key: a[None, :]}, self.emb_id)
        vec = np.asarray(un[self.ac_key], dtype=np.float64)[0]

        expect = self.width_in * len(self.arms)
        if vec.shape[-1] != expect:
            raise ValueError(
                f"ActionToRobot: model action is {vec.shape[-1]}-D but the "
                f"declared input layout {self.fields} x {len(self.arms)} arms = "
                f"{expect}-D. Update per_arm_in when the action space changes.")

        out = []
        for i, arm in enumerate(self.arms):
            off, f = i * self.width_in, {}
            for name, w in self.fields:
                f[name] = vec[off:off + w]
                off += w
            ypr = _rot6d_to_ypr_np(f["rot6d"])
            pose6 = np.concatenate([f["xyz"], ypr])[None, :]        # (1,6)
            if self.apply_frame_transforms:
                pose6 = self._cam_fn(pose6.copy(), self.extrinsics[arm])
                pose6 = self._ee_fn(pose6)
            out.append(np.concatenate([pose6[0], f["grip"]]))       # (7,)

        cmd = np.concatenate(out).astype(np.float32)                # (14,)
        if cmd.shape[-1] != 7 * len(self.arms):
            raise ValueError(
                f"ActionToRobot: produced {cmd.shape[-1]}-D, robot expects "
                f"{7 * len(self.arms)}-D.")
        state["command"] = cmd
        return state


def _rot6d_to_ypr_np(d6: np.ndarray) -> np.ndarray:
    """(6,) first-two-columns rotation -> (3,) ZYX euler, matching the robot's
    convention (R.from_euler('ZYX', ypr) round-trips)."""
    from scipy.spatial.transform import Rotation as _R
    return _R.from_matrix(_rot6d_to_matrix(np.asarray(d6, float))).as_euler("ZYX")
