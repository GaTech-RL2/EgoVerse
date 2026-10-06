"""Keep continuous execution, reset episodes and incomplete probes distinct."""

from dataclasses import asdict, dataclass

import numpy as np


def checked_chunk(actions, *, horizon, action_dim):
    values = np.asarray(actions)
    if values.shape != (horizon, action_dim):
        raise ValueError(
            f"Checkpoint returned {values.shape}; expected {(horizon, action_dim)}"
        )
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("Action predictions must be finite floating-point values")
    return values


@dataclass
class EpisodeCounts:
    episode_id: str
    action_limit: int
    execution_prefix: int
    reset_episodes_started: int = 1
    executed_actions: int = 0
    reset_free_segments: int = 0
    assisted_segments: int = 0
    generated_chunks: int = 0
    generated_candidates: int = 0

    def execute(self, steps, *, assisted=False):
        if type(steps) is not int or not 1 <= steps <= self.execution_prefix:
            raise ValueError("Count only a nonempty, actually executed prefix")
        if self.executed_actions + steps > self.action_limit:
            raise ValueError("Executed prefix exceeds the declared episode horizon")
        self.executed_actions += steps
        self.reset_free_segments += 1
        self.assisted_segments += int(assisted)

    def result(self, *, success, stop_reason):
        if type(success) is not bool:
            raise ValueError("Success must come from the benchmark's binary predicate")
        complete = success or stop_reason in {"horizon", "environment_terminated"}
        if stop_reason == "horizon" and self.executed_actions != self.action_limit:
            raise ValueError(
                "A shorter smoke test cannot be reported as a full horizon"
            )
        if success and self.executed_actions == 0:
            raise ValueError("An initially satisfied reset is not an executed success")
        return {
            **asdict(self),
            "success": success,
            "stop_reason": stop_reason,
            "episode_complete": complete,
            "eligible_for_completed_episode_sr": complete,
            "physical_candidate_retries": 0,
        }


def summarize_episodes(rows):
    identifiers = [r["episode_id"] for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "Repeated reset attempts require distinct physical episode IDs"
        )
    complete = [r for r in rows if r["eligible_for_completed_episode_sr"]]
    return {
        "reset_episodes_started": sum(r["reset_episodes_started"] for r in rows),
        "completed_episodes": len(complete),
        "incomplete_episodes": len(rows) - len(complete),
        "reset_free_segments": sum(r["reset_free_segments"] for r in rows),
        "assisted_segments": sum(r["assisted_segments"] for r in rows),
        "executed_actions": sum(r["executed_actions"] for r in rows),
        "completed_successes": sum(r["success"] for r in complete),
        "completed_episode_sr": (
            sum(r["success"] for r in complete) / len(complete) if complete else None
        ),
        "intended_batch_sr": None
        if len(complete) != len(rows)
        else (
            sum(r["success"] for r in complete) / len(complete) if complete else None
        ),
        "sr_note": "Incomplete probes stay visible; they do not establish a full-batch SR.",
    }
