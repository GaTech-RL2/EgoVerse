#!/usr/bin/env python3
"""
Test script for EvaMinkKinematicsSolver.

This script demonstrates how to use the mink-based IK solver for the Eva robot
and compares TracIK vs Mink (MuJoCo) solvers to quantify FK/IK, cross gaps,
and diagnose the remaining configuration-dependent residuals.
"""

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import egomimic
from egomimic.robot.eva.eva_kinematics import (
    EvaMinkKinematicsSolver,
)

EVA_XML_PATH = os.path.join(
    os.path.dirname(egomimic.__file__), "resources/model_x5.xml"
)

# -------------------- Helpers (robust to Rotation vs ndarray) --------------------


def _as_rotation(x):
    """Return a scipy Rotation from either Rotation or 3x3 ndarray."""
    return x if isinstance(x, R) else R.from_matrix(np.asarray(x))


def _as_matrix(x):
    """Return a 3x3 ndarray from either Rotation or 3x3 ndarray."""
    return x.as_matrix() if isinstance(x, R) else np.asarray(x)


def _rot_geodesic(a, b):
    """Return geodesic angle [rad] between two rotations (Rotation or ndarray)."""
    if isinstance(a, np.ndarray):
        a = R.from_matrix(a)
    if isinstance(b, np.ndarray):
        b = R.from_matrix(b)
    return (a.inv() * b).magnitude()


# -------------------- Original single-solver test (slightly hardened) --------------------


def test_eva_mink_ik():
    """Test Eva mink IK solver."""
    # Path to MuJoCo XML scene (you'll need to create this)
    xml_path = Path(EVA_XML_PATH)

    if not xml_path.exists():
        print(f"Error: Scene file not found at {xml_path}")
        print("Please create a MuJoCo XML scene file for the X5 robot.")
        return

    print("=" * 60)
    print("Testing EvaMinkKinematicsSolver")
    print("=" * 60)

    # Initialize solver
    print("\n1. Initializing solver...")
    solver = EvaMinkKinematicsSolver(
        model_path=str(xml_path),
        eef_link_name="tcp_match_trac",
        eef_frame_type="site",
        max_iterations=100,
        position_tolerance=1e-3,
        orientation_tolerance=1e-3,
    )
    print("   [OK] Solver initialized successfully")

    # Test forward kinematics
    print("\n2. Testing forward kinematics...")
    home_joints = np.array([0.0, 1.57, 1.57, 0.0, 0.0, 0.0])
    pos, rot = solver.fk(home_joints)
    rot = _as_rotation(rot)
    print(f"   Home position: {pos}")
    print(f"   Home orientation (euler): {rot.as_euler('xyz', degrees=True)}")

    # Test inverse kinematics
    print("\n3. Testing inverse kinematics...")

    # Target: Move 10cm forward from home
    target_pos = pos + np.array([0.1, 0.0, 0.0])
    target_rot = rot.as_matrix()

    print(f"   Target position: {target_pos}")

    solution = solver.ik(target_pos, target_rot, home_joints)

    if solution is not None:
        print("   [OK] IK converged!")
        print(f"   Solution joints: {solution}")

        # Verify with FK
        achieved_pos, achieved_rot = solver.fk(solution)
        achieved_rot = _as_rotation(achieved_rot)
        pos_error = np.linalg.norm(achieved_pos - target_pos)
        rot_error = _rot_geodesic(achieved_rot, target_rot)

        print(f"   Achieved position: {achieved_pos}")
        print(f"   Position error: {pos_error:.6f} m")
        print(f"   Rotation error: {rot_error:.6f} rad")
    else:
        print("   [FAILED] IK did not converge")

    # Test multiple targets
    print("\n4. Testing multiple target poses...")
    test_targets = [
        ("Move up", pos + np.array([0.0, 0.0, 0.1])),
        ("Move right", pos + np.array([0.0, 0.1, 0.0])),
        ("Move diagonal", pos + np.array([0.05, 0.05, 0.05])),
    ]

    for name, target_pos_i in test_targets:
        print(f"\n   {name}: {target_pos_i}")
        solution_i = solver.ik(target_pos_i, target_rot, home_joints)

        if solution_i is not None:
            achieved_pos_i, _ = solver.fk(solution_i)
            error = np.linalg.norm(achieved_pos_i - target_pos_i)
            print(f"   [OK] Converged (error: {error:.6f} m)")
        else:
            print("   [FAILED] Failed to converge")

    # Test IK with retries
    print("\n5. Testing IK with retries...")
    difficult_target = pos + np.array([0.15, 0.15, 0.0])
    solution_retry = solver.ik_with_retries(
        difficult_target, target_rot, home_joints, num_retries=3
    )

    if solution_retry is not None:
        print("   [OK] IK with retries succeeded!")
        achieved_pos_r, _ = solver.fk(solution_retry)
        error = np.linalg.norm(achieved_pos_r - difficult_target)
        print(f"   Position error: {error:.6f} m")
    else:
        print("   [FAILED] IK with retries failed")

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


# -------------------- TracIK vs Mink comparison with FK/IK/CROSS metrics --------------------


# def compare_solvers():
#     """Compare TracIK solver vs Mink solver and quantify FK/IK/cross gaps."""
#     urdf_path = (
#         Path(__file__).parent.parent / "eva" / "stanford_repo" / "models" / "X5.urdf"
#     )
#     xml_path = Path(EVA_XML_PATH)
#     if not urdf_path.exists() or not xml_path.exists():
#         print("Error: URDF or XML file not found")
#         print(f"  URDF: {urdf_path.exists()} @ {urdf_path}")
#         print(f"  XML : {xml_path.exists()} @ {xml_path}")
#         return

#     print("\n" + "=" * 60)
#     print("Comparing TracIK vs Mink Solver (with FK/IK/cross-gap metrics)")
#     print("=" * 60)

#     # Initialize both solvers
#     print("\nInitializing solvers...")
#     trac_solver = EvaKinematicsSolver(str(urdf_path))
#     mink_solver = EvaMinkKinematicsSolver(
#         model_path=str(xml_path),
#         eef_link_name="tcp_match_trac",
#         eef_frame_type="site",
#         max_iterations=100,
#         position_tolerance=1e-3,
#         orientation_tolerance=1e-3,
#     )
#     print("[OK] Both solvers initialized")

#     # Test configuration
#     home_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#     # ----------------------------------------------------------------------
#     # 1) FK GAP at the same joint config(s)
#     # ----------------------------------------------------------------------
#     print("\n[1] FK GAP @ home_joints")
#     pos_t, rot_t = trac_solver.fk(home_joints)
#     rot_t = _as_rotation(rot_t)
#     pos_m, rot_m = mink_solver.fk(home_joints)
#     rot_m = _as_rotation(rot_m)

#     fk_pos_err = np.linalg.norm(pos_t - pos_m)
#     fk_rot_err = _rot_geodesic(rot_t, rot_m)
#     print(
#         f"   Trac FK @ home: pos={pos_t}, euler(deg)={rot_t.as_euler('xyz', degrees=True)}"
#     )
#     print(
#         f"   Mink FK @ home: pos={pos_m}, euler(deg)={rot_m.as_euler('xyz', degrees=True)}"
#     )
#     print(f"   FK position gap [m]: {fk_pos_err:.6e}")
#     print(f"   FK rotation gap [rad]: {fk_rot_err:.6e}")

#     # ----------------------------------------------------------------------
#     # 2) IK GAP to a shared target (10 cm forward from TRAC FK pose)
#     # ----------------------------------------------------------------------
#     print("\n[2] IK GAP to a shared target (10 cm forward from Trac FK pose)")
#     target_pos = pos_t + np.array([0.1, 0.0, 0.0])
#     target_rot = rot_t  # keep orientation identical to Trac FK

#     print(f"   Target position (TRAC-based): {target_pos}")
#     start = time.time()
#     trac_solution = trac_solver.ik(target_pos, _as_matrix(target_rot), home_joints)
#     trac_time = (time.time() - start) * 1000.0

#     start = time.time()
#     mink_solution = mink_solver.ik(target_pos, _as_matrix(target_rot), home_joints)
#     mink_time = (time.time() - start) * 1000.0

#     print(f"   TracIK time: {trac_time:.2f} ms")
#     print(f"   Mink time:  {mink_time:.2f} ms")

#     if trac_solution is not None:
#         pA, rA = trac_solver.fk(trac_solution)
#         rA = _as_rotation(rA)
#         pos_errA = np.linalg.norm(pA - target_pos)
#         rot_errA = _rot_geodesic(rA, target_rot)
#         print(
#             f"   TracIK achieved pos: {pA} | pos_err={pos_errA:.6e} m | rot_err={rot_errA:.6e} rad"
#         )
#     else:
#         print("   TracIK failed on TRAC-based shared target")

#     if mink_solution is not None:
#         pB, rB = mink_solver.fk(mink_solution)
#         rB = _as_rotation(rB)
#         pos_errB = np.linalg.norm(pB - target_pos)
#         rot_errB = _rot_geodesic(rB, target_rot)
#         print(
#             f"   Mink achieved pos:  {pB} | pos_err={pos_errB:.6e} m | rot_err={rot_errB:.6e} rad"
#         )
#     else:
#         print("   Mink failed on TRAC-based shared target")

#     if trac_solution is not None and mink_solution is not None:
#         jdelta = np.linalg.norm(trac_solution - mink_solution)
#         print(
#             f"   Joint-space Δ between solvers (TRAC-based target) [L2]: {jdelta:.6e}"
#         )

#     # ----------------------------------------------------------------------
#     # 2b) IK GAP to a shared target (10 cm forward from MINK FK pose)
#     # ----------------------------------------------------------------------
#     print("\n[2b] IK GAP to a shared target (10 cm forward from MINK FK pose)")
#     target_pos2 = pos_m + np.array([0.1, 0.0, 0.0])
#     target_rot2 = rot_m  # keep orientation identical to MINK FK

#     print(f"   Target position (MINK-based): {target_pos2}")
#     start = time.time()
#     trac_solution2 = trac_solver.ik(target_pos2, _as_matrix(target_rot2), home_joints)
#     trac_time2 = (time.time() - start) * 1000.0

#     start = time.time()
#     mink_solution2 = mink_solver.ik(target_pos2, _as_matrix(target_rot2), home_joints)
#     mink_time2 = (time.time() - start) * 1000.0

#     print(f"   TracIK time: {trac_time2:.2f} ms")
#     print(f"   Mink time:  {mink_time2:.2f} ms")

#     if trac_solution2 is not None:
#         pA2, rA2 = trac_solver.fk(trac_solution2)
#         rA2 = _as_rotation(rA2)
#         pos_errA2 = np.linalg.norm(pA2 - target_pos2)
#         rot_errA2 = _rot_geodesic(rA2, target_rot2)
#         print(
#             f"   TracIK achieved pos: {pA2} | pos_err={pos_errA2:.6e} m | rot_err={rot_errA2:.6e} rad"
#         )
#     else:
#         print("   TracIK failed on MINK-based shared target")

#     if mink_solution2 is not None:
#         pB2, rB2 = mink_solver.fk(mink_solution2)
#         rB2 = _as_rotation(rB2)
#         pos_errB2 = np.linalg.norm(pB2 - target_pos2)
#         rot_errB2 = _rot_geodesic(rB2, target_rot2)
#         print(
#             f"   Mink achieved pos:  {pB2} | pos_err={pos_errB2:.6e} m | rot_err={rot_errB2:.6e} rad"
#         )
#     else:
#         print("   Mink failed on MINK-based shared target")

#     if trac_solution2 is not None and mink_solution2 is not None:
#         jdelta2 = np.linalg.norm(trac_solution2 - mink_solution2)
#         print(
#             f"   Joint-space Δ between solvers (MINK-based target) [L2]: {jdelta2:.6e}"
#         )

#     # ----------------------------------------------------------------------
#     # 3) CROSS-GAP: “FK of one” → “IK of the other” (both directions).
#     # ----------------------------------------------------------------------
#     print("\n[3] CROSS-GAP (targets from each solver’s FK, solved by the other)")

#     # 3a) Use TRAC FK pose as the target → solve with MINK
#     print("\n   [3a] TRAC FK pose as target → MINK IK")
#     trac_target_pos, trac_target_rot = pos_t, rot_t
#     mink_on_trac = mink_solver.ik(
#         trac_target_pos, _as_matrix(trac_target_rot), home_joints
#     )
#     if mink_on_trac is not None:
#         p_m_on_t, r_m_on_t = mink_solver.fk(mink_on_trac)
#         r_m_on_t = _as_rotation(r_m_on_t)
#         pos_err_mt = np.linalg.norm(p_m_on_t - trac_target_pos)
#         rot_err_mt = _rot_geodesic(r_m_on_t, trac_target_rot)
#         print(
#             f"      Mink→(Trac target) pos_err={pos_err_mt:.6e} m | rot_err={rot_err_mt:.6e} rad"
#         )
#     else:
#         print("      Mink failed to hit Trac FK pose")

#     # 3b) Use MINK FK pose as the target → solve with TRAC
#     print("\n   [3b] MINK FK pose as target → TRAC IK")
#     mink_target_pos, mink_target_rot = pos_m, rot_m
#     trac_on_mink = trac_solver.ik(
#         mink_target_pos, _as_matrix(mink_target_rot), home_joints
#     )
#     if trac_on_mink is not None:
#         p_t_on_m, r_t_on_m = trac_solver.fk(trac_on_mink)
#         r_t_on_m = _as_rotation(r_t_on_m)
#         pos_err_tm = np.linalg.norm(p_t_on_m - mink_target_pos)
#         rot_err_tm = _rot_geodesic(r_t_on_m, mink_target_rot)
#         print(
#             f"      Trac→(Mink target) pos_err={pos_err_tm:.6e} m | rot_err={rot_err_tm:.6e} rad"
#         )
#     else:
#         print("      Trac failed to hit Mink FK pose")

#     # ----------------------------------------------------------------------
#     # 4) Additional shared targets (from TRAC FK pose)
#     # ----------------------------------------------------------------------
#     print("\n[4] Additional shared targets (up/right/diagonal from TRAC FK pose)")
#     extra_targets = [
#         ("Move up", pos_t + np.array([0.0, 0.0, 0.1])),
#         ("Move right", pos_t + np.array([0.0, 0.1, 0.0])),
#         ("Move diagonal", pos_t + np.array([0.05, 0.05, 0.05])),
#     ]

#     for name, tpos in extra_targets:
#         print(f"\n   {name}: {tpos}")
#         tA = trac_solver.ik(tpos, _as_matrix(rot_t), home_joints)
#         tB = mink_solver.ik(tpos, _as_matrix(rot_t), home_joints)

#         if tA is not None:
#             pA, rA = trac_solver.fk(tA)
#             rA = _as_rotation(rA)
#             eposA = np.linalg.norm(pA - tpos)
#             erotA = _rot_geodesic(rA, rot_t)
#             print(
#                 f"      Trac achieved pos_err={eposA:.6e} m | rot_err={erotA:.6e} rad"
#             )
#         else:
#             print("      Trac failed")

#         if tB is not None:
#             pB, rB = mink_solver.fk(tB)
#             rB = _as_rotation(rB)
#             eposB = np.linalg.norm(pB - tpos)
#             erotB = _rot_geodesic(rB, rot_t)
#             print(
#                 f"      Mink achieved pos_err={eposB:.6e} m | rot_err={erotB:.6e} rad"
#             )
#         else:
#             print("      Mink failed")

#         if tA is not None and tB is not None:
#             jdelta = np.linalg.norm(tA - tB)
#             print(f"      Joint-space Δ (same target) [L2]: {jdelta:.6e}")

#     # ----------------------------------------------------------------------
#     # 4b) Additional shared targets (from MINK FK pose)  ← reverse sweep
#     # ----------------------------------------------------------------------
#     print("\n[4b] Additional shared targets (up/right/diagonal from MINK FK pose)")
#     extra_targets_mink = [
#         ("Move up", pos_m + np.array([0.0, 0.0, 0.1])),
#         ("Move right", pos_m + np.array([0.0, 0.1, 0.0])),
#         ("Move diagonal", pos_m + np.array([0.05, 0.05, 0.05])),
#     ]

#     for name, tpos in extra_targets_mink:
#         print(f"\n   {name}: {tpos}")
#         tA = trac_solver.ik(tpos, _as_matrix(rot_m), home_joints)
#         tB = mink_solver.ik(tpos, _as_matrix(rot_m), home_joints)

#         if tA is not None:
#             pA, rA = trac_solver.fk(tA)
#             rA = _as_rotation(rA)
#             eposA = np.linalg.norm(pA - tpos)
#             erotA = _rot_geodesic(rA, rot_m)
#             print(
#                 f"      Trac achieved pos_err={eposA:.6e} m | rot_err={erotA:.6e} rad"
#             )
#         else:
#             print("      Trac failed")

#         if tB is not None:
#             pB, rB = mink_solver.fk(tB)
#             rB = _as_rotation(rB)
#             eposB = np.linalg.norm(pB - tpos)
#             erotB = _rot_geodesic(rB, rot_m)
#             print(
#                 f"      Mink achieved pos_err={eposB:.6e} m | rot_err={erotB:.6e} rad"
#             )
#         else:
#             print("      Mink failed")

#         if tA is not None and tB is not None:
#             jdelta = np.linalg.norm(tA - tB)
#             print(f"      Joint-space Δ (same target) [L2]: {jdelta:.6e}")

#     # ----------------------------------------------------------------------
#     # 5) Estimate a constant Trac→Mink SE(3) and validate it
#     # ----------------------------------------------------------------------
#     R_off, t_off = estimate_and_validate_offset(
#         trac_solver,
#         mink_solver,
#         pos_t,
#         rot_t,  # TRAC FK at home
#         pos_m,
#         rot_m,  # MINK FK at home
#         home_joints,
#     )

#     # ----------------------------------------------------------------------
#     # 6) Deep-dive diagnostics: variance, PCA, joint correlations, Jacobians
#     # ----------------------------------------------------------------------
#     run_gap_diagnostics(trac_solver, mink_solver, R_off, t_off, seed=3)

#     print("\n" + "=" * 60)
#     print("Summary / Reading the metrics")
#     print("=" * 60)
#     print(
#         "- FK GAP: pure forward mismatch at identical joints; near-constant bias ⇒ frame/site/tool offset."
#     )
#     print(
#         "- IK GAP ([2] TRAC-based, [2b] MINK-based): how close each gets to the same Cartesian target; compare joint Δ too."
#     )
#     print(
#         "- CROSS-GAP ([3]): using one solver’s FK as target for the other isolates frame/orientation convention differences."
#     )
#     print(
#         "- [4] and [4b]: directional consistency checks around each solver’s native FK pose."
#     )
#     print(
#         "- [5],[6]: estimate constant SE(3) and diagnose residual variance to find which joints/links cause the gap."
#     )


# -------------------- SE(3) estimation + validation --------------------


def _estimate_se3_offset(trac_solver, mink_solver, n_samples=60, rng_seed=0):
    """
    Estimate a fixed SE(3) transform T_off = (R_off, t_off) such that:
        R_M ≈ R_off * R_T
        p_M ≈ R_off * p_T + t_off
    using random joint samples. Returns (R_off: Rotation, t_off: (3,), stats: dict).
    """
    rng = np.random.default_rng(rng_seed)

    # Conservative joint limits if your solver doesn't expose them:
    # (Eva is 6-DoF; adjust if needed)
    lim = np.deg2rad(170.0)
    q_min = -lim * np.ones(6)
    q_max = lim * np.ones(6)
    Q = q_min + (q_max - q_min) * rng.random((n_samples, 6))

    rotvecs = []
    pts_T, pts_M = [], []

    for q in Q:
        pT, RT = trac_solver.fk(q)
        RT = _as_rotation(RT)
        pM, RM = mink_solver.fk(q)
        RM = _as_rotation(RM)

        # Relative rotation RT^T * RM
        R_rel = RT.inv() * RM
        rotvecs.append(R_rel.as_rotvec())

        pts_T.append(np.asarray(pT, float))
        pts_M.append(np.asarray(pM, float))

    rotvecs = np.asarray(rotvecs)  # (N,3)
    pts_T = np.asarray(pts_T)  # (N,3)
    pts_M = np.asarray(pts_M)  # (N,3)

    # Rotation averaging in Lie algebra: mean of rotation vectors → exp
    r_mean = rotvecs.mean(axis=0)
    R_off = R.from_rotvec(r_mean)

    # Translation from least-squares (here: simple mean, since constant offset)
    # p_M ≈ R_off p_T + t_off  ⇒  t_off ≈ mean_i (p_Mi - R_off p_Ti)
    t_off = (pts_M - R_off.apply(pts_T)).mean(axis=0)

    # Residuals with/without compensation (for sanity)
    pre_pos = np.linalg.norm(pts_M - pts_T, axis=1)
    pre_rot = np.linalg.norm(rotvecs, axis=1)  # angle magnitude of RT^T RM

    pts_T_comp = R_off.apply(pts_T) + t_off
    RM_pred = [R_off * _as_rotation(trac_solver.fk(q)[1]) for q in Q]
    rotvecs_post = []
    for i, q in enumerate(Q):
        # true RM:
        _, RM_true = mink_solver.fk(q)
        RM_true = _as_rotation(RM_true)
        rotvecs_post.append((RM_pred[i].inv() * RM_true).as_rotvec())
    rotvecs_post = np.asarray(rotvecs_post)

    post_pos = np.linalg.norm(pts_M - pts_T_comp, axis=1)
    post_rot = np.linalg.norm(rotvecs_post, axis=1)

    stats = {
        "pre_pos_mean": pre_pos.mean(),
        "pre_pos_med": np.median(pre_pos),
        "post_pos_mean": post_pos.mean(),
        "post_pos_med": np.median(post_pos),
        "pre_rot_mean": pre_rot.mean(),
        "pre_rot_med": np.median(pre_rot),
        "post_rot_mean": post_rot.mean(),
        "post_rot_med": np.median(post_rot),
    }
    return R_off, t_off, stats


def _apply_trac_to_mink_target(p_T, R_T, R_off, t_off):
    """
    Map a Trac target (p_T, R_T) to an equivalent Mink-frame target:
        p_M = R_off p_T + t_off
        R_M = R_off R_T
    """
    R_T = _as_rotation(R_T)
    p_M = R_off.apply(np.asarray(p_T, float)) + t_off
    R_M = R_off * R_T
    return p_M, R_M


def _apply_mink_to_trac_target(p_M, R_M, R_off, t_off):
    """
    Map a Mink target (p_M, R_M) to an equivalent Trac-frame target:
        p_T = R_off^T (p_M - t_off)
        R_T = R_off^T R_M
    """
    R_M = _as_rotation(R_M)
    R_off_inv = R_off.inv()
    p_T = R_off_inv.apply(np.asarray(p_M, float) - t_off)
    R_T = R_off_inv * R_M
    return p_T, R_T


def estimate_and_validate_offset(
    trac_solver, mink_solver, pos_t, rot_t, pos_m, rot_m, home_joints
):
    """
    Convenience wrapper:
      - estimates T_off
      - prints residual stats before/after
      - validates by transforming your TRAC-based and MINK-based shared targets
        into the other frame and solving IK there.
    """
    print("\n[5] Estimating fixed SE(3) offset between TRAC EEF and MINK EEF")
    R_off, t_off, stats = _estimate_se3_offset(
        trac_solver, mink_solver, n_samples=80, rng_seed=1
    )

    print("   Pre/Post residuals over random FK samples:")
    print(
        f"      pos  mean (pre→post): {stats['pre_pos_mean']:.6f} → {stats['post_pos_mean']:.6f} m"
    )
    print(
        f"      pos median (pre→post): {stats['pre_pos_med']:.6f} → {stats['post_pos_med']:.6f} m"
    )
    print(
        f"      rot  mean (pre→post): {stats['pre_rot_mean']:.6f} → {stats['post_rot_mean']:.6f} rad"
    )
    print(
        f"      rot median (pre→post): {stats['pre_rot_med']:.6f} → {stats['post_rot_med']:.6f} rad"
    )

    # Pretty-print T_off
    T = np.eye(4)
    T[:3, :3] = R_off.as_matrix()
    T[:3, 3] = t_off
    print("   Estimated T_off (Trac → Mink):")
    np.set_printoptions(precision=6, suppress=True)
    print(T)

    # ---- Validate on your two shared targets ([2] and [2b]) ----
    print(
        "\n   Validation: transform TRAC-based shared target into MINK frame and solve:"
    )
    target_pos_T = pos_t + np.array([0.1, 0.0, 0.0])
    target_rot_T = rot_t
    targM_pos, targM_rot = _apply_trac_to_mink_target(
        target_pos_T, target_rot_T, R_off, t_off
    )
    mink_sol_adj = mink_solver.ik(targM_pos, _as_matrix(targM_rot), home_joints)
    if mink_sol_adj is not None:
        pM_hit, RM_hit = mink_solver.fk(mink_sol_adj)
        RM_hit = _as_rotation(RM_hit)
        print(
            f"      Mink (transformed target) pos_err={np.linalg.norm(pM_hit - targM_pos):.6e} m | rot_err={_rot_geodesic(RM_hit, targM_rot):.6e} rad"
        )
    else:
        print("      Mink failed even after transforming the TRAC target.")

    print(
        "\n   Validation: transform MINK-based shared target into TRAC frame and solve:"
    )
    target_pos_M = pos_m + np.array([0.1, 0.0, 0.0])
    target_rot_M = rot_m
    targT_pos, targT_rot = _apply_mink_to_trac_target(
        target_pos_M, target_rot_M, R_off, t_off
    )
    trac_sol_adj = trac_solver.ik(targT_pos, _as_matrix(targT_rot), home_joints)
    if trac_sol_adj is not None:
        pT_hit, RT_hit = trac_solver.fk(trac_sol_adj)
        RT_hit = _as_rotation(RT_hit)
        print(
            f"      Trac (transformed target) pos_err={np.linalg.norm(pT_hit - targT_pos):.6e} m | rot_err={_rot_geodesic(RT_hit, targT_rot):.6e} rad"
        )
    else:
        print("      Trac failed even after transforming the MINK target.")

    return R_off, t_off


# -------------------- Diagnostics: variance, PCA, joint correlations, Jacobians --------------------


def _sample_qs(n=200, seed=0, dof=6, deg_lim=170.0):
    rng = np.random.default_rng(seed)
    lim = np.deg2rad(deg_lim)
    qmin = -lim * np.ones(dof)
    qmax = lim * np.ones(dof)
    return qmin + (qmax - qmin) * rng.random((n, dof))


def _collect_fk_pairs(trac_solver, mink_solver, Q):
    """Return arrays: pT (N,3), RT (N Rotation), pM (N,3), RM (N Rotation)."""
    pT, pM, RT, RM = [], [], [], []
    for q in Q:
        p_t, r_t = trac_solver.fk(q)
        r_t = _as_rotation(r_t)
        p_m, r_m = mink_solver.fk(q)
        r_m = _as_rotation(r_m)
        pT.append(np.asarray(p_t, float))
        RT.append(r_t)
        pM.append(np.asarray(p_m, float))
        RM.append(r_m)
    return np.asarray(pT), RT, np.asarray(pM), RM


def _pca_dir(X):
    """Principal direction (unit vec) and explained variances of 3D samples."""
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc / max(len(Xc) - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vecs[:, 0], vals  # principal axis, variances


def _spearman(xs, ys):
    # tiny, dependency-free Spearman (ranks + Pearson)
    xr = np.argsort(np.argsort(xs))
    yr = np.argsort(np.argsort(ys))
    xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    return float(np.dot(xr, yr) / (len(xs) - 1))


def _numeric_jacobian_fk_pos(solver, q, eps=1e-6):
    """Numeric Jacobian of EEF position wrt joints: Jpos (3,dof)."""
    q = np.asarray(q, float)
    dof = q.size
    J = np.zeros((3, dof))
    p0, _ = solver.fk(q)
    p0 = np.asarray(p0, float)
    for j in range(dof):
        dq = np.zeros_like(q)
        dq[j] = eps
        p1, _ = solver.fk(q + dq)
        p1 = np.asarray(p1, float)
        J[:, j] = (p1 - p0) / eps
    return J


def _numeric_jacobian_fk_rotvec(solver, q, eps=1e-6):
    """Numeric Jacobian of orientation (as rotvec) wrt joints: Jrot (3,dof)."""
    q = np.asarray(q, float)
    dof = q.size
    J = np.zeros((3, dof))
    _, R0 = solver.fk(q)
    R0 = _as_rotation(R0)
    for j in range(dof):
        dq = np.zeros_like(q)
        dq[j] = eps
        _, R1 = solver.fk(q + dq)
        R1 = _as_rotation(R1)
        dR = R0.inv() * R1
        J[:, j] = dR.as_rotvec() / eps
    return J


def _jacobian_gap_stats(trac_solver, mink_solver, Q, report_k=5):
    """Compare Trac vs Mink numeric Jacobians; return per-joint norms & a quick print."""
    pos_diffs = []
    rot_diffs = []
    for q in Q:
        Jt_p = _numeric_jacobian_fk_pos(trac_solver, q)
        Jm_p = _numeric_jacobian_fk_pos(mink_solver, q)
        Jt_r = _numeric_jacobian_fk_rotvec(trac_solver, q)
        Jm_r = _numeric_jacobian_fk_rotvec(mink_solver, q)
        pos_diffs.append(np.linalg.norm(Jt_p - Jm_p, axis=0))  # (dof,)
        rot_diffs.append(np.linalg.norm(Jt_r - Jm_r, axis=0))
    pos_diffs = np.asarray(pos_diffs)  # (N,dof)
    rot_diffs = np.asarray(rot_diffs)

    pos_mean = pos_diffs.mean(axis=0)  # per-joint averages
    rot_mean = rot_diffs.mean(axis=0)

    print(
        "\n[Diag] Mean ||ΔJ_pos|| per joint:",
        "  ".join(f"j{j + 1}:{v:.4f}" for j, v in enumerate(pos_mean)),
    )
    print(
        "[Diag] Mean ||ΔJ_rot|| per joint:",
        "  ".join(f"j{j + 1}:{v:.4f}" for j, v in enumerate(rot_mean)),
    )

    # Biggest gaps
    jp = np.argsort(-pos_mean)[:report_k]
    jr = np.argsort(-rot_mean)[:report_k]
    print(
        f"[Diag] Top-{report_k} J_pos gaps at joints:",
        [int(i + 1) for i in jp],
        "values:",
        [float(pos_mean[i]) for i in jp],
    )
    print(
        f"[Diag] Top-{report_k} J_rot gaps at joints:",
        [int(i + 1) for i in jr],
        "values:",
        [float(rot_mean[i]) for i in jr],
    )
    return pos_mean, rot_mean


def _fit_joint_zero_offsets_on_rot(trac_solver, mink_solver, Q):
    """
    Quick test: do constant joint angle offsets (delta_q) reduce orientation gap?
    Linearize orientation residual in rotvec space: r ≈ J_rot * delta_q.
    Solve least squares for delta_q. Report pre/post orientation residual norms.
    """
    _ = Q.shape[1]
    Rs = []
    Jall = []
    for q in Q:
        _, RT = trac_solver.fk(q)
        RT = _as_rotation(RT)
        _, RM = mink_solver.fk(q)
        RM = _as_rotation(RM)
        Rrel = (RT.inv() * RM).as_rotvec()  # desired ~ J * delta_q
        Jt_r = _numeric_jacobian_fk_rotvec(trac_solver, q)  # (3,dof)
        Rs.append(Rrel)
        Jall.append(Jt_r)
    r = np.concatenate(Rs, axis=0)  # (3N,)
    J = np.concatenate(Jall, axis=0)  # (3N,dof)

    # Solve min ||J * dq - r||_2
    dq, *_ = np.linalg.lstsq(J, r, rcond=None)
    # Evaluate post residual
    r_post = r - J @ dq
    pre = np.linalg.norm(r) / np.sqrt(len(r))
    post = np.linalg.norm(r_post) / np.sqrt(len(r_post))
    print("\n[Diag] Joint zero-offset LS on orientation:")
    print(f"       pre rms(rotvec)={pre:.6e}  →  post rms(rotvec)={post:.6e}")
    print(f"       estimated delta_q (deg): {np.rad2deg(dq)}")
    return dq, pre, post


def run_gap_diagnostics(trac_solver, mink_solver, R_off, t_off, seed=2):
    """
    Master diagnostic: variance, PCA, correlations, Jacobians, and LS zero-offsets.
    """
    print(
        "\n[6] Residual variance, PCA, joint correlations, Jacobian & zero-offset tests"
    )

    Q = _sample_qs(n=200, seed=seed)
    pT, RT, pM, RM = _collect_fk_pairs(trac_solver, mink_solver, Q)

    # Residuals pre- and post- SE3 compensation
    # Pre:
    dP_pre = pM - pT
    _ = np.array([(RT[i].inv() * RM[i]).as_rotvec() for i in range(len(Q))])  # (N,3)
    # Post (apply Trac→Mink offset):
    pT_comp = R_off.apply(pT) + t_off
    dP_post = pM - pT_comp
    _ = np.array([((R_off * RT[i]).inv() * RM[i]).as_rotvec() for i in range(len(Q))])

    # Variance & PCA (position part)
    pre_var = dP_pre.var(axis=0)
    post_var = dP_post.var(axis=0)
    vdir_pre, evals_pre = _pca_dir(dP_pre)
    vdir_post, evals_post = _pca_dir(dP_post)
    print(f"   Pos var pre:  {pre_var}   (sum={pre_var.sum():.6f})")
    print(f"   Pos var post: {post_var}  (sum={post_var.sum():.6f})")
    print(f"   PCA pre:  principal dir ~ {vdir_pre}  eigenvals={evals_pre}")
    print(f"   PCA post: principal dir ~ {vdir_post} eigenvals={evals_post}")

    # Norm residuals and per-joint correlations
    r_pre = np.linalg.norm(dP_pre, axis=1)  # (N,)
    r_post = np.linalg.norm(dP_post, axis=1)

    print("\n   Spearman corr(residual, q_j) per joint (pre / post):")
    for j in range(Q.shape[1]):
        c_pre = _spearman(Q[:, j], r_pre)
        c_post = _spearman(Q[:, j], r_post)
        print(f"      j{j + 1}: {c_pre:+.3f} / {c_post:+.3f}")

    # Simple linear regression of residual magnitude on joints (pre/post)
    X = np.c_[Q, np.ones(len(Q))]  # add bias
    w_pre, *_ = np.linalg.lstsq(X, r_pre, rcond=None)
    w_post, *_ = np.linalg.lstsq(X, r_post, rcond=None)
    print("\n   Linear reg weights on residual magnitude (deg units helpful to read):")
    print("      pre :", [f"{wi:.4f}" for wi in w_pre[:-1]], "bias", f"{w_pre[-1]:.4f}")
    print(
        "      post:", [f"{wi:.4f}" for wi in w_post[:-1]], "bias", f"{w_post[-1]:.4f}"
    )

    # Jacobian comparison (which joint’s kinematics disagree?)
    _jacobian_gap_stats(trac_solver, mink_solver, Q)

    # Orientation-only: can constant joint zero-offsets explain any rot gap?
    _fit_joint_zero_offsets_on_rot(trac_solver, mink_solver, Q)


if __name__ == "__main__":
    test_eva_mink_ik()
    # Uncomment to compare solvers:
    # compare_solvers()
