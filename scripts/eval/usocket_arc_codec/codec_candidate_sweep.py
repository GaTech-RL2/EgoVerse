#!/usr/bin/env python3
"""Compare corrected ARC candidates with an identity H16 action control.

NON-PROTOCOL tokenizer diagnostic. This is not a learned-policy evaluation.
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

import arc_replay_local_velocity as base
import paired_chunk_replay as paired
from egomimic.pipeline.stages_arc import ArcDetokenizeStage
from egomimic.rldb.zarr.planar_arc import TokenizeUSocketArcVelocity


LABEL = "NON-PROTOCOL_CODEC_CANDIDATE_SWEEP_NOT_COMPARABLE"
CANDIDATES = {
    "D80_M32_R12deg": {"kind": "arc", "distance": 80.0, "M": 32, "degrees": 12.0},
    "D80_M64_R12deg": {"kind": "arc", "distance": 80.0, "M": 64, "degrees": 12.0},
    "RAW_H16_IDENTITY": {"kind": "identity", "M": 16},
}


def build_arc_codec(config: dict[str, Any]):
    tokenizer = TokenizeUSocketArcVelocity(
        min_distance_unit=float(config["distance"]),
        resampled_vector_length=int(config["M"]),
        dt=base.DT,
        rotation_distance_unit=base.angle_unit(float(config["degrees"])),
    )
    decoder = ArcDetokenizeStage(
        resampled_vector_length=int(config["M"]),
        action_horizon=base.DECODE_HORIZON,
        dt=base.DT,
        native_action_dim=3,
    )
    return tokenizer, decoder


def decode(config: dict[str, Any], codec, actions: np.ndarray, start: int, execute: int) -> np.ndarray:
    if config["kind"] == "identity":
        return actions[start : start + execute].copy()
    tokenizer, decoder = codec
    chunk = base.pad_future(actions, start)
    token = tokenizer.tokenize(chunk)
    output = decoder.forward(
        {"pred_action": torch.as_tensor(token[None], dtype=torch.float64)}
    )["pred_action_native"][0].detach().cpu().numpy()
    return output[:execute]


def free_running(actions, init, env_args, config: dict[str, Any]) -> dict[str, Any]:
    codec = None if config["kind"] == "identity" else build_arc_codec(config)
    env = base.make_env(env_args, init)
    peak = final = 0.0
    total_steps = 0
    xy_errors: list[float] = []
    angle_errors: list[float] = []
    terminated = False
    try:
        for start in range(0, len(actions), base.REPLAN_EVERY):
            execute = min(base.REPLAN_EVERY, len(actions) - start)
            decoded = decode(config, codec, actions, start, execute)
            target = actions[start : start + execute]
            xy_errors.extend(np.linalg.norm(decoded[:, :2] - target[:, :2], axis=1).tolist())
            angle_errors.extend(base.circular_abs(decoded[:, 2], target[:, 2]).tolist())
            local_peak, final, terminated, count = base.step_actions(env, decoded)
            peak = max(peak, local_peak)
            total_steps += count
            if terminated:
                break
        return {
            "peak_coverage": peak,
            "final_coverage": final,
            "terminated": terminated,
            "steps": total_steps,
            "xy_error_mean": float(np.mean(xy_errors)),
            "xy_error_p95": float(np.quantile(xy_errors, 0.95)),
            "angle_error_mean_deg": float(np.degrees(np.mean(angle_errors))),
            "angle_error_p95_deg": float(np.degrees(np.quantile(angle_errors, 0.95))),
        }
    finally:
        env.close()


def state_reset(actions, states, init, env_args, raw_steps: int, config: dict[str, Any]) -> dict[str, Any]:
    codec = None if config["kind"] == "identity" else build_arc_codec(config)
    chunks: list[dict[str, Any]] = []
    xy_errors: list[float] = []
    angle_errors: list[float] = []
    for start in range(0, raw_steps, base.REPLAN_EVERY):
        execute = min(base.REPLAN_EVERY, raw_steps - start)
        target = actions[start : start + execute]
        decoded = decode(config, codec, actions, start, execute)
        xy_errors.extend(np.linalg.norm(decoded[:, :2] - target[:, :2], axis=1).tolist())
        angle_errors.extend(base.circular_abs(decoded[:, 2], target[:, 2]).tolist())
        raw_env = base.make_env(env_args, init)
        codec_env = base.make_env(env_args, init)
        try:
            paired.reset_to_recorded_state(raw_env, states[start], init)
            paired.reset_to_recorded_state(codec_env, states[start], init)
            reset_a = paired.snapshot(raw_env)
            reset_b = paired.snapshot(codec_env)
            reset_error = max(float(np.max(np.abs(reset_a[key] - reset_b[key]))) for key in reset_a)
            raw_result = paired.execute_exact(raw_env, target)
            codec_result = paired.execute_exact(codec_env, decoded)
            raw_end = paired.snapshot(raw_env)
            codec_end = paired.snapshot(codec_env)
            chunks.append({
                "start": start,
                "steps": execute,
                "reset_state_reproducibility_max_abs_error": reset_error,
                "agent_endpoint_position_error": float(np.linalg.norm(codec_end["agent_pos"] - raw_end["agent_pos"])),
                "agent_endpoint_angle_error_deg": paired.angle_error_deg(codec_end["agent_angle"][0], raw_end["agent_angle"][0]),
                "object_endpoint_position_error": float(np.linalg.norm(codec_end["object_pose"][:2] - raw_end["object_pose"][:2])),
                "object_endpoint_angle_error_deg": paired.angle_error_deg(codec_end["object_pose"][2], raw_end["object_pose"][2]),
                "coverage_delta": codec_result["final_coverage"] - raw_result["final_coverage"],
                "coverage_absolute_delta": abs(codec_result["final_coverage"] - raw_result["final_coverage"]),
            })
        finally:
            raw_env.close()
            codec_env.close()
    return {
        "chunks": chunks,
        "chunk_count": len(chunks),
        "action_xy_error_mean": float(np.mean(xy_errors)),
        "action_angle_error_mean_deg": float(np.degrees(np.mean(angle_errors))),
    }


def gate_candidate(free_summary: dict[str, Any], reset_summary: dict[str, Any], episode_count: int) -> dict[str, Any]:
    free_checks = {
        "exact_episode_count": free_summary["episodes"] == episode_count,
        "delta_vs_raw_mean": free_summary["delta_vs_raw_mean"] >= base.GATE_THRESHOLDS["delta_vs_raw_mean_min"],
        "delta_vs_raw_median": free_summary["delta_vs_raw_median"] >= base.GATE_THRESHOLDS["delta_vs_raw_median_min"],
        "success_rate_0.80": free_summary["success_rate_0.80"] >= base.GATE_THRESHOLDS["success_rate_0.80_min"],
        "xy_error_mean": free_summary["xy_error_mean_mean"] <= base.GATE_THRESHOLDS["xy_error_mean_mean_max"],
        "angle_error_mean": free_summary["angle_error_mean_deg_mean"] <= base.GATE_THRESHOLDS["angle_error_mean_deg_mean_max"],
    }
    thresholds = paired.STATE_RESET_THRESHOLDS
    reset_checks = {
        "exact_episode_count": reset_summary["episodes"] == episode_count,
        "reset_state_reproducible": reset_summary["reset_state_reproducibility_max_abs_error"] <= 1e-9,
        "action_xy_error_mean": reset_summary["action_xy_error_mean"] <= thresholds["action_xy_error_mean_max"],
        "action_angle_error_mean": reset_summary["action_angle_error_mean_deg"] <= thresholds["action_angle_error_mean_deg_max"],
        "agent_position_error_mean": reset_summary["agent_endpoint_position_error_mean"] <= thresholds["agent_endpoint_position_error_mean_max"],
        "agent_angle_error_mean": reset_summary["agent_endpoint_angle_error_mean_deg"] <= thresholds["agent_endpoint_angle_error_mean_deg_max"],
        "object_position_error_mean": reset_summary["object_endpoint_position_error_mean"] <= thresholds["object_endpoint_position_error_mean_max"],
        "object_position_error_p95": reset_summary["object_endpoint_position_error_p95"] <= thresholds["object_endpoint_position_error_p95_max"],
        "object_angle_error_mean": reset_summary["object_endpoint_angle_error_mean_deg"] <= thresholds["object_endpoint_angle_error_mean_deg_max"],
        "coverage_absolute_delta_mean": reset_summary["coverage_absolute_delta_mean"] <= thresholds["coverage_absolute_delta_mean_max"],
    }
    values = [value for summary in (free_summary, reset_summary) for value in summary.values() if isinstance(value, (int, float))]
    finite = all(math.isfinite(float(value)) for value in values)
    return {
        "status": "PASS" if finite and all(free_checks.values()) and all(reset_checks.values()) else "FAIL",
        "all_summary_metrics_finite": finite,
        "free_running": {"status": "PASS" if all(free_checks.values()) else "FAIL", "checks": free_checks, "summary": free_summary},
        "state_reset": {"status": "PASS" if all(reset_checks.values()) else "FAIL", "checks": reset_checks, "summary": reset_summary},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=29)
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    ids = base.load_validation_ids(args.split_manifest)
    if len(ids) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} replay episodes, manifest contains {len(ids)}"
        )
    paths = [args.dataset / f"{episode_id}.zarr" for episode_id in ids]
    if any(not path.is_dir() for path in paths):
        raise FileNotFoundError("one or more held-out episodes are missing")
    actions, init, env_args, _ = base.load_episode(paths[0])
    states = paired.load_recorded_states(paths[0], len(actions))
    preflight = {
        "label": LABEL,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "simulator_commit": os.environ.get("SIMULATOR_COMMIT"),
        "protocol_sha256": os.environ.get("PROTOCOL_SHA256"),
        "canonical_launcher_sha256": os.environ.get("CANONICAL_LAUNCHER_SHA256"),
        "split_manifest_sha256": base.sha256(args.split_manifest),
        "validation_episodes": len(ids),
        "expected_episodes": args.expected_episodes,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "replan_every": base.REPLAN_EVERY,
        "candidates": CANDIDATES,
        "state_reset_scope": "recorded positions/angles; velocities zeroed and latch cleared equally",
    }
    if args.preflight_only:
        for name, config in CANDIDATES.items():
            codec = None if config["kind"] == "identity" else build_arc_codec(config)
            decoded = decode(config, codec, actions, 0, base.REPLAN_EVERY)
            if decoded.shape != (base.REPLAN_EVERY, 3) or not np.isfinite(decoded).all():
                raise ValueError(f"{name}: invalid decode {decoded.shape}")
        env = base.make_env(env_args, init)
        paired.reset_to_recorded_state(env, states[0], init)
        env.close()
        preflight["codec_shapes_finite_and_recorded_state_reset"] = "PASS"
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    per_episode: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    free_rows = {name: [] for name in CANDIDATES}
    reset_rows = {name: [] for name in CANDIDATES}
    for index, (episode_id, path) in enumerate(zip(ids, paths), 1):
        actions, init, env_args, stored_peak = base.load_episode(path)
        states = paired.load_recorded_states(path, len(actions))
        raw = base.raw_replay(actions, init, env_args)
        raw_rows.append(raw)
        row: dict[str, Any] = {"episode_id": episode_id, "stored_peak_coverage": stored_peak, "raw": raw, "candidates": {}}
        for name, config in CANDIDATES.items():
            free = free_running(actions, init, env_args, config)
            free["delta_vs_raw"] = free["peak_coverage"] - raw["peak_coverage"]
            reset = state_reset(actions, states, init, env_args, int(raw["steps"]), config)
            free_rows[name].append(free)
            reset_rows[name].append(reset)
            row["candidates"][name] = {"free_running": free, "state_reset": reset}
        per_episode.append(row)
        print(f"completed {index}/{len(paths)} {episode_id}", flush=True)

    summaries: dict[str, Any] = {}
    for name in CANDIDATES:
        free_summary = base.aggregate(free_rows[name])
        reset_summary = paired.aggregate_state_reset(reset_rows[name])
        summaries[name] = {
            "config": CANDIDATES[name],
            "gate": gate_candidate(free_summary, reset_summary, len(ids)),
        }
    gate = {
        "status": "PASS" if all(row["gate"]["status"] == "PASS" for row in summaries.values()) else "FAIL",
        "candidate_status": {name: row["gate"]["status"] for name, row in summaries.items()},
        "candidates": summaries,
    }
    payload = {"preflight": preflight, "raw_summary": base.aggregate(raw_rows), "gate": gate, "per_episode": per_episode}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    gate_path = args.output.with_name("CODEC_CANDIDATE_GATE.json")
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "gate_path": str(gate_path), "gate": gate}, indent=2, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
