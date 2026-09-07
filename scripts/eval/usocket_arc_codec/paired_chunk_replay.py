#!/usr/bin/env python3
"""Paired U-Socket ARC replay: free-running and 8-step state-reset.

This is a NON-PROTOCOL tokenizer diagnostic, not a learned-policy score.
The state-reset branch reconstructs each demonstrated chunk boundary by
replaying the exact demonstrated prefix into two fresh simulator instances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

import arc_replay_local_velocity as base
from egomimic.pipeline.stages_arc import ArcDetokenizeStage
from egomimic.rldb.zarr.planar_arc import TokenizeUSocketArcVelocity


LABEL = "NON-PROTOCOL_PAIRED_TOKENIZER_REPLAY_DIAGNOSTIC_NOT_COMPARABLE"
DISTANCE = 40.0
DEGREES = 6.0
STATE_RESET_THRESHOLDS = {
    "action_xy_error_mean_max": 3.0,
    "action_angle_error_mean_deg_max": 2.0,
    "agent_endpoint_position_error_mean_max": 3.0,
    "agent_endpoint_angle_error_mean_deg_max": 2.0,
    "object_endpoint_position_error_mean_max": 2.0,
    "object_endpoint_position_error_p95_max": 5.0,
    "object_endpoint_angle_error_mean_deg_max": 2.0,
    "coverage_absolute_delta_mean_max": 0.02,
}


def execute_exact(env, actions: np.ndarray) -> dict[str, Any]:
    peak = 0.0
    final = 0.0
    terminated = False
    for action in actions:
        _, _, term, trunc, info = env.step(action)
        final = float(info["coverage"])
        peak = max(peak, final)
        terminated = bool(term or trunc)
    return {"peak_coverage": peak, "final_coverage": final, "terminated": terminated}


def snapshot(env) -> dict[str, np.ndarray]:
    obs = env._get_obs()
    return {
        "agent_pos": np.asarray(obs["agent_pos"], dtype=np.float64),
        "agent_angle": np.asarray(obs["agent_angle"], dtype=np.float64),
        "object_pose": np.asarray(obs["object_pose"], dtype=np.float64),
    }


def angle_error_deg(a: float, b: float) -> float:
    return float(np.degrees(base.circular_abs(np.asarray(a), np.asarray(b))))


def build_codec():
    tokenizer = TokenizeUSocketArcVelocity(
        min_distance_unit=DISTANCE,
        resampled_vector_length=base.M,
        dt=base.DT,
        rotation_distance_unit=base.angle_unit(DEGREES),
    )
    decoder = ArcDetokenizeStage(
        resampled_vector_length=base.M,
        action_horizon=base.DECODE_HORIZON,
        dt=base.DT,
        native_action_dim=3,
    )
    return tokenizer, decoder


def decode_chunk(tokenizer, decoder, actions: np.ndarray, start: int, execute: int) -> np.ndarray:
    future = base.pad_future(actions, start)
    token = tokenizer.tokenize(future)
    decoded = decoder.forward(
        {"pred_action": torch.as_tensor(token[None], dtype=torch.float64)}
    )["pred_action_native"][0].detach().cpu().numpy()
    return decoded[:execute]


def load_recorded_states(path: Path, total_frames: int) -> np.ndarray:
    states = np.asarray(zarr.open_group(str(path), mode="r")["observations.state"][:total_frames], dtype=np.float64)
    if states.shape != (total_frames, 6) or not np.isfinite(states).all():
        raise ValueError(f"{path.name}: expected finite U-Socket states ({total_frames},6), got {states.shape}")
    return states


def reset_to_recorded_state(env, state: np.ndarray, init: dict[str, Any]) -> None:
    env.set_state(
        agent_pos=(float(state[0]), float(state[1])),
        agent_angle=float(state[2]),
        object_pose=(float(state[3]), float(state[4]), float(state[5])),
        goal_pose=tuple(init["goal_pose"]),
    )


def state_reset_replay(actions, states, init, env_args, raw_steps: int) -> dict[str, Any]:
    tokenizer, decoder = build_codec()
    chunks: list[dict[str, Any]] = []
    action_xy_errors: list[float] = []
    action_angle_errors: list[float] = []

    for start in range(0, raw_steps, base.REPLAN_EVERY):
        execute = min(base.REPLAN_EVERY, raw_steps - start)
        target = actions[start : start + execute]
        decoded = decode_chunk(tokenizer, decoder, actions, start, execute)
        action_xy_errors.extend(np.linalg.norm(decoded[:, :2] - target[:, :2], axis=1).tolist())
        action_angle_errors.extend(base.circular_abs(decoded[:, 2], target[:, 2]).tolist())

        raw_env = base.make_env(env_args, init)
        codec_env = base.make_env(env_args, init)
        try:
            # The dataset stores pre-action state[t]. Reset both branches to
            # that exact demonstrated pose. set_state zeroes hidden velocities
            # and clears latch state equally in both branches, deliberately
            # isolating local codec distortion from accumulated contact history.
            reset_to_recorded_state(raw_env, states[start], init)
            reset_to_recorded_state(codec_env, states[start], init)
            raw_reset = snapshot(raw_env)
            codec_reset = snapshot(codec_env)
            reset_repro_error = max(
                float(np.max(np.abs(raw_reset[key] - codec_reset[key])))
                for key in raw_reset
            )

            raw_result = execute_exact(raw_env, target)
            codec_result = execute_exact(codec_env, decoded)
            raw_end = snapshot(raw_env)
            codec_end = snapshot(codec_env)
            chunks.append({
                "start": start,
                "steps": execute,
                "reset_state_reproducibility_max_abs_error": reset_repro_error,
                "agent_endpoint_position_error": float(np.linalg.norm(codec_end["agent_pos"] - raw_end["agent_pos"])),
                "agent_endpoint_angle_error_deg": angle_error_deg(codec_end["agent_angle"][0], raw_end["agent_angle"][0]),
                "object_endpoint_position_error": float(np.linalg.norm(codec_end["object_pose"][:2] - raw_end["object_pose"][:2])),
                "object_endpoint_angle_error_deg": angle_error_deg(codec_end["object_pose"][2], raw_end["object_pose"][2]),
                "raw_peak_coverage": raw_result["peak_coverage"],
                "codec_peak_coverage": codec_result["peak_coverage"],
                "raw_final_coverage": raw_result["final_coverage"],
                "codec_final_coverage": codec_result["final_coverage"],
                "coverage_delta": codec_result["final_coverage"] - raw_result["final_coverage"],
                "coverage_absolute_delta": abs(codec_result["final_coverage"] - raw_result["final_coverage"]),
            })
        finally:
            raw_env.close()
            codec_env.close()

    return {
        "chunks": chunks,
        "chunk_count": len(chunks),
        "action_xy_error_mean": float(np.mean(action_xy_errors)),
        "action_angle_error_mean_deg": float(np.degrees(np.mean(action_angle_errors))),
    }


def aggregate_state_reset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = [chunk for row in rows for chunk in row["chunks"]]
    def mean(key: str) -> float:
        return float(np.mean([float(chunk[key]) for chunk in chunks]))
    def p95(key: str) -> float:
        return float(np.quantile([float(chunk[key]) for chunk in chunks], 0.95))
    return {
        "episodes": len(rows),
        "chunks": len(chunks),
        "action_steps": int(sum(sum(chunk["steps"] for chunk in row["chunks"]) for row in rows)),
        "action_xy_error_mean": float(np.mean([row["action_xy_error_mean"] for row in rows])),
        "action_angle_error_mean_deg": float(np.mean([row["action_angle_error_mean_deg"] for row in rows])),
        "reset_state_reproducibility_max_abs_error": float(max(chunk["reset_state_reproducibility_max_abs_error"] for chunk in chunks)),
        "agent_endpoint_position_error_mean": mean("agent_endpoint_position_error"),
        "agent_endpoint_angle_error_mean_deg": mean("agent_endpoint_angle_error_deg"),
        "object_endpoint_position_error_mean": mean("object_endpoint_position_error"),
        "object_endpoint_position_error_p95": p95("object_endpoint_position_error"),
        "object_endpoint_angle_error_mean_deg": mean("object_endpoint_angle_error_deg"),
        "coverage_delta_mean": mean("coverage_delta"),
        "coverage_absolute_delta_mean": mean("coverage_absolute_delta"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    ids = base.load_validation_ids(args.split_manifest)
    paths = [args.dataset / f"{episode_id}.zarr" for episode_id in ids]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing validation episodes: {missing}")

    first_actions, first_init, first_env_args, _ = base.load_episode(paths[0])
    preflight = {
        "label": LABEL,
        "dataset": str(args.dataset.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": base.sha256(args.split_manifest),
        "validation_episodes": len(ids),
        "selected_variant": base.SELECTED_VARIANT,
        "replan_every": base.REPLAN_EVERY,
        "state_reset_method": "two fresh envs reset from recorded pre-action observations.state at every 8-step boundary",
        "state_reset_scope": "positions and angles only; velocities are zeroed and socket latch is cleared equally in both branches",
        "free_running_gate_thresholds": base.GATE_THRESHOLDS,
        "state_reset_gate_thresholds": STATE_RESET_THRESHOLDS,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "simulator_commit": os.environ.get("SIMULATOR_COMMIT"),
        "protocol_sha256": os.environ.get("PROTOCOL_SHA256"),
        "canonical_launcher_sha256": os.environ.get("CANONICAL_LAUNCHER_SHA256"),
    }
    if args.preflight_only:
        env = base.make_env(first_env_args, first_init)
        env.close()
        tokenizer, decoder = build_codec()
        decoded = decode_chunk(tokenizer, decoder, first_actions, 0, base.REPLAN_EVERY)
        if decoded.shape != (base.REPLAN_EVERY, 3) or not np.isfinite(decoded).all():
            raise ValueError(f"invalid decoded preflight chunk {decoded.shape}")
        preflight["environment_construct_reset_close"] = "PASS"
        preflight["codec_decode_shape_and_finite"] = "PASS"
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    raw_rows: list[dict[str, Any]] = []
    free_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    for index, (episode_id, path) in enumerate(zip(ids, paths), 1):
        actions, init, env_args, stored_peak = base.load_episode(path)
        raw = base.raw_replay(actions, init, env_args)
        free = base.arc_replay(actions, init, env_args, DISTANCE, DEGREES)
        free["delta_vs_raw"] = free["peak_coverage"] - raw["peak_coverage"]
        states = load_recorded_states(path, len(actions))
        reset = state_reset_replay(actions, states, init, env_args, int(raw["steps"]))
        raw_rows.append(raw)
        free_rows.append(free)
        reset_rows.append(reset)
        per_episode.append({"episode_id": episode_id, "stored_peak_coverage": stored_peak, "raw": raw, "free_running": free, "state_reset": reset})
        print(f"completed {index}/{len(paths)} {episode_id}", flush=True)

    raw_summary = base.aggregate(raw_rows)
    free_summary = base.aggregate(free_rows)
    reset_summary = aggregate_state_reset(reset_rows)
    numeric_values = [
        value for summary in (raw_summary, free_summary, reset_summary)
        for value in summary.values() if isinstance(value, (int, float))
    ]
    free_checks = {
        "exact_episode_count": free_summary["episodes"] == len(ids) == 29,
        "delta_vs_raw_mean": free_summary["delta_vs_raw_mean"] >= base.GATE_THRESHOLDS["delta_vs_raw_mean_min"],
        "delta_vs_raw_median": free_summary["delta_vs_raw_median"] >= base.GATE_THRESHOLDS["delta_vs_raw_median_min"],
        "success_rate_0.80": free_summary["success_rate_0.80"] >= base.GATE_THRESHOLDS["success_rate_0.80_min"],
        "xy_error_mean": free_summary["xy_error_mean_mean"] <= base.GATE_THRESHOLDS["xy_error_mean_mean_max"],
        "angle_error_mean": free_summary["angle_error_mean_deg_mean"] <= base.GATE_THRESHOLDS["angle_error_mean_deg_mean_max"],
    }
    reset_checks = {
        "exact_episode_count": reset_summary["episodes"] == len(ids) == 29,
        "reset_state_reproducible": reset_summary["reset_state_reproducibility_max_abs_error"] <= 1e-9,
        "action_xy_error_mean": reset_summary["action_xy_error_mean"] <= STATE_RESET_THRESHOLDS["action_xy_error_mean_max"],
        "action_angle_error_mean": reset_summary["action_angle_error_mean_deg"] <= STATE_RESET_THRESHOLDS["action_angle_error_mean_deg_max"],
        "agent_position_error_mean": reset_summary["agent_endpoint_position_error_mean"] <= STATE_RESET_THRESHOLDS["agent_endpoint_position_error_mean_max"],
        "agent_angle_error_mean": reset_summary["agent_endpoint_angle_error_mean_deg"] <= STATE_RESET_THRESHOLDS["agent_endpoint_angle_error_mean_deg_max"],
        "object_position_error_mean": reset_summary["object_endpoint_position_error_mean"] <= STATE_RESET_THRESHOLDS["object_endpoint_position_error_mean_max"],
        "object_position_error_p95": reset_summary["object_endpoint_position_error_p95"] <= STATE_RESET_THRESHOLDS["object_endpoint_position_error_p95_max"],
        "object_angle_error_mean": reset_summary["object_endpoint_angle_error_mean_deg"] <= STATE_RESET_THRESHOLDS["object_endpoint_angle_error_mean_deg_max"],
        "coverage_absolute_delta_mean": reset_summary["coverage_absolute_delta_mean"] <= STATE_RESET_THRESHOLDS["coverage_absolute_delta_mean_max"],
    }
    finite = all(math.isfinite(float(value)) for value in numeric_values)
    gate = {
        "status": "PASS" if finite and all(free_checks.values()) and all(reset_checks.values()) else "FAIL",
        "all_summary_metrics_finite": finite,
        "free_running": {"status": "PASS" if all(free_checks.values()) else "FAIL", "checks": free_checks, "thresholds": base.GATE_THRESHOLDS, "summary": free_summary},
        "state_reset": {"status": "PASS" if all(reset_checks.values()) else "FAIL", "checks": reset_checks, "thresholds": STATE_RESET_THRESHOLDS, "summary": reset_summary},
    }
    payload = {"preflight": preflight, "raw_summary": raw_summary, "gate": gate, "per_episode": per_episode}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    gate_path = args.output.with_name("PAIRED_REPLAY_GATE.json")
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "gate_path": str(gate_path), "gate": gate, "raw_summary": raw_summary}, indent=2, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
