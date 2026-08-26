"""Deterministic ChainGripper FK/IK boundary microbenchmark.

This reports numerical and CPU-throughput measurements only. It does not run a
policy or make simulator-success claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from egomimic.rldb.zarr.action_chunk_transforms import (
    ChainGripperNative4ToPoints6,
    ChainGripperPoints6ToNative4,
    _to_float64_numpy,
)


def _controls(seed: int, trajectories: int, horizon: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (trajectories, horizon)
    return np.stack(
        [
            rng.uniform(64.0, 448.0, size=shape),
            rng.uniform(64.0, 448.0, size=shape),
            rng.uniform(-np.pi, np.pi, size=shape),
            rng.uniform(0.05, 0.95, size=shape),
        ],
        axis=-1,
    )


def _as_input(value: np.ndarray, dtype_name: str):
    if dtype_name == "fp64":
        return value.astype(np.float64)
    if dtype_name == "fp32":
        return value.astype(np.float32)
    if dtype_name == "bf16":
        return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
    raise ValueError(f"Unsupported dtype {dtype_name!r}")


def _clone(value):
    return value.clone() if torch.is_tensor(value) else value.copy()


def _control_error(predicted, target) -> dict[str, float | list[float]]:
    predicted_np = _to_float64_numpy(predicted)
    target_np = _to_float64_numpy(target)
    delta = predicted_np - target_np
    delta[..., 2] = (delta[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "rmse_all": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs_all": float(np.max(np.abs(delta))),
        "rmse_by_native_dim": np.sqrt(
            np.mean(np.square(delta), axis=tuple(range(delta.ndim - 1)))
        ).tolist(),
    }


def _median_timed(callable_, repeats: int):
    elapsed = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        elapsed.append(time.perf_counter() - start)
    return result, float(np.median(elapsed))


def _benchmark_dtype(
    controls: np.ndarray,
    noise: np.ndarray,
    dtype_name: str,
    repeats: int,
) -> dict:
    source = _as_input(controls, dtype_name)
    fk = ChainGripperNative4ToPoints6(keys=["actions"])
    ik = ChainGripperPoints6ToNative4(keys=["actions"])

    def exact_call():
        batch = fk.transform({"actions": _clone(source)})
        return ik.transform(batch)["actions"]

    exact_result, exact_elapsed = _median_timed(exact_call, repeats)
    exact_diagnostics = dict(ik.last_projection_diagnostics or {})

    base_points = fk.transform({"actions": _clone(source)})["actions"]
    noisy_points_np = _to_float64_numpy(base_points) + noise
    noisy_points = _as_input(noisy_points_np, dtype_name)

    def noisy_call():
        return ik.transform({"actions": _clone(noisy_points)})["actions"]

    noisy_result, noisy_elapsed = _median_timed(noisy_call, repeats)
    noisy_diagnostics = dict(ik.last_projection_diagnostics or {})
    fitted_points = fk.transform({"actions": _clone(noisy_result)})["actions"]
    fitted_delta = _to_float64_numpy(fitted_points) - _to_float64_numpy(noisy_points)
    fitted_point_rmse = np.sqrt(np.mean(np.square(fitted_delta), axis=-1))
    sample_count = int(np.prod(controls.shape[:-1]))

    return {
        "input_kind": "torch" if torch.is_tensor(source) else "numpy",
        "on_manifold_fk_ik": {
            "seconds_median": exact_elapsed,
            "actions_per_second": sample_count / exact_elapsed,
            "control_error": _control_error(exact_result, source),
            "mean_point_rmse": float(exact_diagnostics["mean_point_rmse"]),
            "max_point_rmse": float(exact_diagnostics["max_point_rmse"]),
            "exact_inverse_fraction": float(
                np.mean(exact_diagnostics["used_exact_inverse"])
            ),
        },
        "noisy_off_manifold_ik": {
            "seconds_median": noisy_elapsed,
            "actions_per_second": sample_count / noisy_elapsed,
            "control_error_to_source": _control_error(noisy_result, source),
            "mean_point_rmse": float(np.mean(fitted_point_rmse)),
            "max_point_rmse": float(np.max(fitted_point_rmse)),
            "wrong_chirality_count": int(noisy_diagnostics["wrong_chirality_count"]),
            "degenerate_count": int(noisy_diagnostics["degenerate_count"]),
        },
    }


def run_benchmark(
    *,
    seed: int = 20260826,
    trajectories: int = 8,
    horizon: int = 16,
    repeats: int = 3,
    noise_std: float = 2.0,
) -> dict:
    if trajectories <= 0 or horizon <= 0 or repeats <= 0:
        raise ValueError("trajectories, horizon, and repeats must be positive")
    controls = _controls(seed, trajectories, horizon)
    noise_rng = np.random.default_rng(seed + 1)
    noise = noise_rng.normal(0.0, noise_std, size=(*controls.shape[:-1], 6))
    return {
        "benchmark": "chain_gripper_fk_ik_boundary",
        "scope": "numerical_and_cpu_throughput_only",
        "seed": seed,
        "trajectories": trajectories,
        "horizon": horizon,
        "sample_count": trajectories * horizon,
        "repeats": repeats,
        "noise_std_points": noise_std,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "results": {
            dtype_name: _benchmark_dtype(controls, noise, dtype_name, repeats)
            for dtype_name in ("fp64", "fp32", "bf16")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        seed=args.seed,
        trajectories=args.trajectories,
        horizon=args.horizon,
        repeats=args.repeats,
        noise_std=args.noise_std,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
