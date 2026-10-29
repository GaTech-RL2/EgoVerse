"""Offline endpoint IK comparison; imports no robot, camera or CAN drivers.

Run with --xml-path pointing to the same model used by the station driver.
Synthetic FK targets are not evidence of physical fold completion.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.robot.yam.kinematics import MujocoArmKinematics
from egomimic.robot.yam.tracik import TracIKEndpoint


def make_cases(k, count, seed):
    rng = np.random.default_rng(seed)
    lo, hi = k.model.jnt_range[k.joints].T
    pool = rng.uniform(lo, hi, (max(2000, 20 * count), 6))
    height, singular = [], []
    for q in pool:
        pose = k.fk(q)
        jp, jr = np.zeros((3, k.model.nv)), np.zeros((3, k.model.nv))
        k.mj.mj_jacSite(k.model, k.data, jp, jr, k.site)
        # Normalize meters and radians before ranking conditioning.
        jac = np.vstack([jp[:, k.didx] / 0.05, jr[:, k.didx] / 0.5])
        singular.append(np.linalg.svd(jac, compute_uv=False)[-1])
        height.append(pose[2, 3])
    groups = {
        "nearby": pool[:count].copy(),
        "near_limits": pool[count : 2 * count].copy(),
        "near_singular": pool[np.argsort(singular)[:count]].copy(),
        "high_tcp": pool[np.argsort(height)[-count:]].copy(),
        "distant_seed": pool[2 * count : 3 * count].copy(),
    }
    for i, q in enumerate(groups["near_limits"]):
        j = i % 6
        q[j] = lo[j] + 1e-5 if (i // 6) % 2 else hi[j] - 1e-5
    cases = []
    for group, targets in groups.items():
        for q in targets:
            radius = 0.8 if group == "distant_seed" else 0.2
            initial = np.clip(q + rng.uniform(-radius, radius, 6), lo, hi)
            cases.append((group, k.fk(q).copy(), initial, q.copy()))
    for q in pool[3 * count : 4 * count]:
        pose = k.fk(q).copy()
        pose[:3, 3] += [3, 3, 3]  # Provably beyond this arm's reach.
        cases.append(("unreachable", pose, q.copy(), None))
    return cases


def compare(k, endpoint, cases, repeats, max_step):
    rows = []
    for repeat in range(repeats):
        for case_id, (group, target, initial, witness) in enumerate(cases):
            row = {
                "repeat": repeat,
                "case": case_id,
                "group": group,
                "target": target.tolist(),
                "seed": initial.tolist(),
                "reachable_witness": None if witness is None else witness.tolist(),
            }
            # Alternate order to avoid always warming one backend first.
            solvers = [("mujoco", k), ("pytracik", endpoint)]
            if (repeat + case_id) % 2:
                solvers.reverse()
            for name, solver in solvers:
                started = time.perf_counter()
                try:
                    q = solver.ik(target, initial)
                except ValueError as error:
                    result = {
                        "solved": False,
                        "within_step": False,
                        "error": str(error),
                    }
                else:
                    pose = k.fk(q)
                    pe = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
                    re = float(
                        Rotation.from_matrix(
                            target[:3, :3] @ pose[:3, :3].T
                        ).magnitude()
                    )
                    lo, hi = k.model.jnt_range[k.joints].T
                    valid = bool(
                        np.isfinite(q).all()
                        and np.all(q >= lo)
                        and np.all(q <= hi)
                        and pe <= k.position_tolerance
                        and re <= k.rotation_tolerance
                    )
                    delta = float(np.max(np.abs(q - initial)))
                    result = {
                        "solved": valid,
                        "within_step": valid and delta <= max_step,
                        "max_step_rad": delta,
                        "position_error_m": pe,
                        "rotation_error_rad": re,
                        "joints": q.tolist(),
                    }
                result["ms"] = 1000 * (time.perf_counter() - started)
                row[name] = result
            rows.append(row)
    summary = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        selected = [row for row in rows if row["group"] == group]
        summary[group] = {"attempts": len(selected)}
        for name in ("mujoco", "pytracik"):
            values = [row[name] for row in selected]
            summary[group][name] = {
                "solved": sum(v["solved"] for v in values),
                "within_step": sum(v["within_step"] for v in values),
                "ms_median": float(np.median([v["ms"] for v in values])),
                "ms_p95": float(np.percentile([v["ms"] for v in values], 95)),
                "ms_max": max(v["ms"] for v in values),
            }
        summary[group]["tracik_rescued"] = sum(
            r["pytracik"]["solved"] and not r["mujoco"]["solved"] for r in selected
        )
        summary[group]["tracik_lost"] = sum(
            r["mujoco"]["solved"] and not r["pytracik"]["solved"] for r in selected
        )
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--site-name", default="tcp_site")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--timeout-s", type=float, default=0.005)
    parser.add_argument(
        "--solver-type",
        choices=("Speed", "Distance", "Manip1", "Manip2"),
        default="Distance",
    )
    parser.add_argument("--max-step", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.repeats <= 0 or args.max_step <= 0:
        parser.error("count, repeats and max-step must be positive")
    k = MujocoArmKinematics(
        args.xml_path, [f"joint{i}" for i in range(1, 7)], site_name=args.site_name
    )
    endpoint = TracIKEndpoint(k, timeout_s=args.timeout_s, solver_type=args.solver_type)
    cases = make_cases(k, args.count, args.seed)
    summary, rows = compare(k, endpoint, cases, args.repeats, args.max_step)
    report = {
        "settings": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "xml_sha256": hashlib.sha256(args.xml_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("pytracik", "mujoco", "numpy", "scipy")
        },
        "mujoco_max_iterations": k.max_iterations,
        "position_tolerance_m": k.position_tolerance,
        "rotation_tolerance_rad": k.rotation_tolerance,
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
