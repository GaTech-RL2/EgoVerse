"""Concrete rollout nodes. Each is `node(state) -> state`.

The two decisions that were previously hardcoded are now their own nodes:
  ObsCadence   -- WHEN to query the policy
  ChunkCommit  -- WHICH predicted actions actually get executed
"""

from typing import Optional

import torch

from egomimic.rollout.core import RolloutNode

# Mirrors egomimic/robot/rollout.py:154 (QUERY_FREQUENCY = 30, DEFAULT_FREQUENCY
# = 30 Hz) so the rollout nodes have the robot's real cadence as their default
# off-robot too. Stated once, here, so a change on the robot side shows up as a
# visible mismatch rather than two silently different replan rates.
QUERY_FREQUENCY = 30  # frames between inferences (30 Hz loop -> 1 Hz)


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

    def __init__(
        self,
        mode: str = "every_n",
        every_n: int = QUERY_FREQUENCY,
        min_interval: int = 0,
    ):
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
# 2. run the model graph
# --------------------------------------------------------------------------- #
class PolicyStep(RolloutNode):
    """Query ``PipelineAlgo.forward_rollout`` for one action chunk."""

    reads = ("should_query", "obs_batch", "t")
    writes = ("chunk", "chunk_t", "prev_chunk")

    def __init__(self, algo, emb_name: str, observation_is_normalized: bool = False):
        self.algo = algo
        self.emb_name = str(emb_name)
        self.observation_is_normalized = bool(observation_is_normalized)

    def reset(self, state):
        return None

    @torch.no_grad()
    def __call__(self, state):
        if not state.get("should_query"):
            return state
        chunk = self.algo.forward_rollout(
            self.emb_name,
            state["obs_batch"],
            rollout_t=int(state["t"]),
            observation_is_normalized=self.observation_is_normalized,
        )
        if chunk.ndim != 3 or chunk.shape[0] != 1:
            raise ValueError(
                "PolicyStep expected a single-batch (1,C,D) prediction, got "
                f"{tuple(chunk.shape)}"
            )
        chunk = chunk[0]
        state["prev_chunk"] = state.get("chunk")
        state["chunk"] = chunk  # (C, D)
        state["chunk_t"] = int(state["t"])
        return state


# --------------------------------------------------------------------------- #
# 3. WHICH predicted actions get executed
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

    def __init__(
        self,
        n_keep: int = QUERY_FREQUENCY,
        blend: float = 0.0,
        replace_queue: bool = True,
    ):
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
                f"PolicyStep should return pred_action at the LAST token."
            )
        C = chunk.shape[0]
        n = max(1, min(self.n_keep, C))
        commit = chunk[:n]

        prev = state.get("prev_chunk")
        if (
            self.blend > 0
            and prev is not None
            and prev.shape[0] >= 2 * n
            and prev.shape[-1] == chunk.shape[-1]
        ):
            commit = self.blend * prev[n : 2 * n] + (1 - self.blend) * commit

        rows = [commit[i] for i in range(commit.shape[0])]
        state["queue"] = rows if self.replace_queue else list(state["queue"]) + rows
        return state


# --------------------------------------------------------------------------- #
# 4. one action per env step
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
                "ObsCadence never fired, or ChunkCommit committed nothing."
            )
        self._holds += 1
        if self._holds in (1, 10, 100):
            print(
                f"[ActionDequeue] queue empty, holding last action "
                f"(count={self._holds})",
                flush=True,
            )
        state["action"] = self._last
        return state
