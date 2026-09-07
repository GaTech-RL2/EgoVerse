#!/usr/bin/env python3
"""Oracle receding replay for U-Socket ARC encode/decode distortion.

This is a NON-PROTOCOL tokenizer diagnostic, not a learned-policy score.
Each ARC variant receives the recorded future action chunk at every replan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from egomimic.pipeline.stages_arc import ArcDetokenizeStage
from egomimic.rldb.zarr.planar_arc import (
    TokenizeUSocketArcVelocity,
    planar_step_distance,
)
from Tsimulation.collect.zarr_writer import ACTION_KEY
from Tsimulation.pushshapes import get_env


LABEL = "NON-PROTOCOL_TOKENIZER_REPLAY_DIAGNOSTIC_NOT_COMPARABLE"
DT = 1.0 / 30.0
M = 32
RAW_HORIZON = 200
DECODE_HORIZON = 16
REPLAN_EVERY = 8
D_VALUES = (20.0, 40.0, 80.0)
ANGLE_DEGREES = (6.0, 12.0, 24.0)
SELECTED_VARIANT = "D40_M32_R6deg"
GATE_THRESHOLDS = {
    "delta_vs_raw_mean_min": -0.05,
    "delta_vs_raw_median_min": -0.02,
    "success_rate_0.80_min": 0.90,
    "xy_error_mean_mean_max": 3.0,
    "angle_error_mean_deg_mean_max": 2.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def angle_unit(degrees: float) -> float:
    return math.radians(degrees)


def circular_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def load_validation_ids(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    domains = payload.get("domains", {})
    if len(domains) != 1:
        raise ValueError(f"expected exactly one split domain, got {list(domains)}")
    row = next(iter(domains.values()))
    ids = list(row["valid_ids"])
    if len(ids) != int(row["valid_count"]):
        raise ValueError("validation ID count does not match manifest")
    return ids


def load_episode(path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], float]:
    store = zarr.open_group(str(path), mode="r")
    attrs = dict(store.attrs)
    actions = np.asarray(store[ACTION_KEY][:], dtype=np.float64)
    total_frames = int(attrs.get("total_frames", len(actions)))
    actions = actions[:total_frames]
    if actions.ndim != 2 or actions.shape[1] != 3 or len(actions) < 2:
        raise ValueError(f"{path.name}: expected finite (T,3) actions, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{path.name}: non-finite actions")
    init = json.loads(attrs["episode_init"])
    env_args = json.loads(attrs["task_description"])["env_args"]
    reward = np.asarray(store["reward"][:total_frames]).reshape(-1)
    stored_peak = float(np.max(reward)) if len(reward) else float("nan")
    return actions, init, env_args, stored_peak


def make_env(env_args: dict[str, Any], init: dict[str, Any]):
    env = get_env("v2")(
        object_shape=env_args["object_shape"],
        pusher_shape=env_args["pusher_shape"],
        obstacle_level=env_args.get("obstacle_level", 0),
        image_size=env_args.get("image_size", 96),
        render_mode=None,
    )
    env._skip_obs_render = True
    env.reset(seed=init.get("reset_seed"))
    if "obstacles" in init:
        env.set_obstacles(init["obstacles"])
    env.set_state(
        agent_pos=tuple(init["agent_pos"]),
        agent_angle=float(init.get("agent_angle", 0.0)),
        object_pose=tuple(init["object_pose"]),
        goal_pose=tuple(init["goal_pose"]),
    )
    return env


def step_actions(env, actions: np.ndarray) -> tuple[float, float, bool, int]:
    peak = 0.0
    final = 0.0
    terminated = False
    count = 0
    for action in actions:
        _, _, term, trunc, info = env.step(action)
        count += 1
        final = float(info["coverage"])
        peak = max(peak, final)
        terminated = bool(term or trunc)
        if terminated:
            break
    return peak, final, terminated, count


def pad_future(actions: np.ndarray, start: int) -> np.ndarray:
    chunk = actions[start : start + RAW_HORIZON]
    if len(chunk) < RAW_HORIZON:
        chunk = np.concatenate(
            (chunk, np.repeat(chunk[-1:], RAW_HORIZON - len(chunk), axis=0)), axis=0
        )
    return chunk


def window_stats(chunk: np.ndarray, distance: float, unit: float) -> dict[str, Any]:
    translation_arc = np.concatenate(
        (np.zeros(1), np.cumsum(planar_step_distance(chunk[:, :2])))
    )
    theta = np.unwrap(chunk[:, 2])
    rotation_arc = np.concatenate(
        (np.zeros(1), np.cumsum(np.abs(np.diff(theta))))
    )
    translation_end = min(distance, float(translation_arc[-1]))
    rotation_end = min(unit, float(rotation_arc[-1]))
    translation_last = int(np.searchsorted(translation_arc, translation_end, side="left"))
    rotation_last = int(np.searchsorted(rotation_arc, rotation_end, side="left"))
    return {
        "translation_source_last_index": translation_last,
        "rotation_source_last_index": rotation_last,
        "translation_early_saturation": translation_last < REPLAN_EVERY - 1,
        "rotation_early_saturation": rotation_last < REPLAN_EVERY - 1,
        "translation_budget_limited": translation_end + 1e-8 < float(translation_arc[-1]),
        "rotation_budget_limited": rotation_end + 1e-8 < float(rotation_arc[-1]),
        "translation_window_distance": translation_end,
        "rotation_window_distance": rotation_end,
    }


def raw_replay(actions, init, env_args) -> dict[str, Any]:
    env = make_env(env_args, init)
    try:
        peak, final, terminated, count = step_actions(env, actions)
        return {"peak_coverage": peak, "final_coverage": final, "terminated": terminated, "steps": count}
    finally:
        env.close()


def arc_replay(actions, init, env_args, distance: float, degrees: float) -> dict[str, Any]:
    unit = angle_unit(degrees)
    tokenizer = TokenizeUSocketArcVelocity(
        min_distance_unit=distance,
        resampled_vector_length=M,
        dt=DT,
        rotation_distance_unit=unit,
    )
    decoder = ArcDetokenizeStage(
        resampled_vector_length=M,
        action_horizon=DECODE_HORIZON,
        dt=DT,
        native_action_dim=3,
    )
    env = make_env(env_args, init)
    peak = 0.0
    final = 0.0
    total_steps = 0
    xy_errors: list[float] = []
    angle_errors: list[float] = []
    windows: list[dict[str, Any]] = []
    terminated = False
    try:
        for start in range(0, len(actions), REPLAN_EVERY):
            chunk = pad_future(actions, start)
            token = tokenizer.tokenize(chunk)
            output = decoder.forward(
                {"pred_action": torch.as_tensor(token[None], dtype=torch.float64)}
            )["pred_action_native"][0].detach().cpu().numpy()
            execute = min(REPLAN_EVERY, len(actions) - start)
            decoded = output[:execute]
            target = actions[start : start + execute]
            xy_errors.extend(np.linalg.norm(decoded[:, :2] - target[:, :2], axis=1).tolist())
            angle_errors.extend(circular_abs(decoded[:, 2], target[:, 2]).tolist())
            windows.append(window_stats(chunk, distance, unit))
            local_peak, final, terminated, count = step_actions(env, decoded)
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
            "windows": len(windows),
            "translation_early_saturation_fraction": float(np.mean([w["translation_early_saturation"] for w in windows])),
            "rotation_early_saturation_fraction": float(np.mean([w["rotation_early_saturation"] for w in windows])),
            "translation_budget_limited_fraction": float(np.mean([w["translation_budget_limited"] for w in windows])),
            "rotation_budget_limited_fraction": float(np.mean([w["rotation_budget_limited"] for w in windows])),
            "translation_source_last_index_mean": float(np.mean([w["translation_source_last_index"] for w in windows])),
            "rotation_source_last_index_mean": float(np.mean([w["rotation_source_last_index"] for w in windows])),
        }
    finally:
        env.close()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "peak_coverage", "final_coverage", "steps", "xy_error_mean", "xy_error_p95",
        "angle_error_mean_deg", "angle_error_p95_deg",
        "translation_early_saturation_fraction", "rotation_early_saturation_fraction",
        "translation_budget_limited_fraction", "rotation_budget_limited_fraction",
        "translation_source_last_index_mean", "rotation_source_last_index_mean",
        "delta_vs_raw",
    ]
    out: dict[str, Any] = {"episodes": len(rows)}
    for key in keys:
        values = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_median"] = float(np.median(values))
    for threshold in (0.80, 0.95):
        out[f"success_rate_{threshold:.2f}"] = float(np.mean([r["peak_coverage"] >= threshold for r in rows]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    ids = load_validation_ids(args.split_manifest)
    paths = [args.dataset / f"{episode_id}.zarr" for episode_id in ids]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing validation episodes: {missing}")
    actions, init, env_args, stored_peak = load_episode(paths[0])
    preflight = {
        "label": LABEL,
        "dataset": str(args.dataset.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256(args.split_manifest),
        "validation_episodes": len(ids),
        "first_episode": ids[0],
        "first_episode_frames": len(actions),
        "first_episode_stored_peak": stored_peak,
        "pusher_shape": env_args.get("pusher_shape"),
        "sim_version": init.get("sim_version"),
        "M": M,
        "raw_horizon": RAW_HORIZON,
        "decode_horizon": DECODE_HORIZON,
        "replan_every": REPLAN_EVERY,
        "D_values": D_VALUES,
        "angle_degrees": ANGLE_DEGREES,
        "angle_units": {str(v): angle_unit(v) for v in ANGLE_DEGREES},
        "tokenizer": "TokenizeUSocketArcVelocity",
        "token_schema": "64x5: [x,y,0,0,v_xy] + [0,0,cos(theta),sin(theta),omega]",
        "selected_variant": SELECTED_VARIANT,
        "gate_thresholds": GATE_THRESHOLDS,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "simulator_commit": os.environ.get("SIMULATOR_COMMIT"),
        "protocol_sha256": os.environ.get("PROTOCOL_SHA256"),
        "canonical_launcher_sha256": os.environ.get("CANONICAL_LAUNCHER_SHA256"),
    }
    if preflight["pusher_shape"] != "u_socket":
        raise ValueError(f"expected u_socket pusher, got {preflight['pusher_shape']!r}")
    if args.preflight_only:
        # Construct and reset the exact runtime environment. Import-only checks
        # do not catch invalid simulator-version aliases or reset incompatibility.
        env = make_env(env_args, init)
        env.close()
        preflight["environment_construct_reset_close"] = "PASS"
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    raw_rows: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {
        f"D{int(d)}_M{M}_R{int(a)}deg": [] for d in D_VALUES for a in ANGLE_DEGREES
    }
    per_episode: list[dict[str, Any]] = []
    for index, (episode_id, path) in enumerate(zip(ids, paths), 1):
        actions, init, env_args, stored_peak = load_episode(path)
        raw = raw_replay(actions, init, env_args)
        raw["stored_peak_coverage"] = stored_peak
        raw["delta_vs_stored"] = raw["peak_coverage"] - stored_peak
        raw_rows.append(raw)
        episode_row: dict[str, Any] = {"episode_id": episode_id, "stored_peak_coverage": stored_peak, "raw": raw, "variants": {}}
        for distance in D_VALUES:
            for degrees in ANGLE_DEGREES:
                name = f"D{int(distance)}_M{M}_R{int(degrees)}deg"
                row = arc_replay(actions, init, env_args, distance, degrees)
                row["delta_vs_raw"] = row["peak_coverage"] - raw["peak_coverage"]
                variant_rows[name].append(row)
                episode_row["variants"][name] = row
        per_episode.append(episode_row)
        print(f"completed {index}/{len(paths)} {episode_id}", flush=True)

    raw_summary = aggregate(raw_rows)
    raw_summary["stored_peak_coverage_mean"] = float(np.mean([r["stored_peak_coverage"] for r in raw_rows]))
    raw_summary["delta_vs_stored_mean"] = float(np.mean([r["delta_vs_stored"] for r in raw_rows]))
    summary = {name: aggregate(rows) for name, rows in variant_rows.items()}
    selected = summary[SELECTED_VARIANT]
    numeric_values = [
        value
        for row in summary.values()
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    checks = {
        "exact_episode_count": raw_summary["episodes"] == len(ids) == 29,
        "all_summary_metrics_finite": all(math.isfinite(float(v)) for v in numeric_values),
        "delta_vs_raw_mean": selected["delta_vs_raw_mean"] >= GATE_THRESHOLDS["delta_vs_raw_mean_min"],
        "delta_vs_raw_median": selected["delta_vs_raw_median"] >= GATE_THRESHOLDS["delta_vs_raw_median_min"],
        "success_rate_0.80": selected["success_rate_0.80"] >= GATE_THRESHOLDS["success_rate_0.80_min"],
        "xy_error_mean": selected["xy_error_mean_mean"] <= GATE_THRESHOLDS["xy_error_mean_mean_max"],
        "angle_error_mean": selected["angle_error_mean_deg_mean"] <= GATE_THRESHOLDS["angle_error_mean_deg_mean_max"],
    }
    gate = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selected_variant": SELECTED_VARIANT,
        "thresholds": GATE_THRESHOLDS,
        "checks": checks,
        "selected_summary": selected,
    }
    payload = {"preflight": preflight, "raw_summary": raw_summary, "variant_summary": summary, "gate": gate, "per_episode": per_episode}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        fields = ["variant", "episodes", "peak_coverage_mean", "delta_vs_raw_mean", "success_rate_0.80", "success_rate_0.95", "xy_error_mean_mean", "angle_error_mean_deg_mean", "translation_early_saturation_fraction_mean", "rotation_early_saturation_fraction_mean", "translation_budget_limited_fraction_mean", "rotation_budget_limited_fraction_mean", "translation_source_last_index_mean_mean", "rotation_source_last_index_mean_mean"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, row in summary.items():
            writer.writerow({key: name if key == "variant" else row.get(key) for key in fields})
    gate_path = args.output.with_name("REPLAY_GATE.json")
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"label": LABEL, "output": str(args.output), "csv": str(csv_path), "gate_path": str(gate_path), "gate": gate, "raw_summary": raw_summary, "variant_summary": summary}, indent=2, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
