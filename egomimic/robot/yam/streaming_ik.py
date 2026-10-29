"""Streaming differential IK for teleop.

YAM joints 2–4 are parallel, so a fully-converged mink solve (the i2rt default)
can jump between elbow-up/elbow-down each tick while the EE still hits the
target. Teleop needs the *local* Jacobian step from the current q, with
posture regularization, velocity limits, and frozen gripper slides.
"""

import mujoco
import numpy as np
import mink

from i2rt.robots.kinematics import Kinematics


class ArmIK:
    """FK/IK for one YAM arm around ``grasp_site`` (or another named site).

    ``get_joint_pos()`` is 7-DoF (gripper normalized to [0, 1]) while the
    MuJoCo model is often 8-DoF (coupled finger slides). This pads, denormalizes
    slides, and applies equality constraints the same way the viser/mujoco
    control interfaces do. IK only returns the 6 arm joints; the gripper is
    commanded separately.
    """

    def __init__(
        self,
        xml_path,
        ee_site="grasp_site",
        n_arm=6,
        dt=1.0 / 30.0,
        steps=4,
        gain=0.5,
        damping=1e-2,
        posture_cost=0.08,
        max_joint_vel=2.5,
        max_iters=None,
    ):
        self.ee_site = ee_site
        self.n_arm = int(n_arm)
        self.dt = float(dt)
        self.steps = int(steps if max_iters is None else max_iters)
        self.damping = float(damping)
        self.max_joint_vel = float(max_joint_vel)
        self.kin = Kinematics(xml_path, ee_site)
        self._model = self.kin._configuration.model
        self._nq = int(self._model.nq)

        self._ee_task = mink.FrameTask(
            frame_name=ee_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            gain=float(gain),
            lm_damping=1.0,
        )
        nv = int(self._model.nv)
        cost = np.full(nv, float(posture_cost))
        # Bias the parallel-axis chain so the elbow doesn't flip.
        n_hinge = min(self.n_arm, nv)
        if n_hinge >= 4:
            cost[1:4] = max(float(posture_cost) * 3.0, 0.2)
        self._posture = mink.PostureTask(self._model, cost=cost)

        freeze = []
        vel_limits = {}
        for j in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                continue
            dof = int(self._model.jnt_dofadr[j])
            jtype = self._model.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_SLIDE:
                freeze.append(dof)
            elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
                vel_limits[name] = np.array([self.max_joint_vel])
        self._constraints = [mink.DofFreezingTask(self._model, freeze)] if freeze else None
        self._limits = [mink.ConfigurationLimit(self._model)]
        if vel_limits:
            self._limits.append(mink.VelocityLimit(self._model, vel_limits))

    def _to_mink_q(self, q_robot):
        qpos = np.zeros(self._nq)
        n = min(len(q_robot), self._nq)
        qpos[:n] = np.asarray(q_robot, dtype=float)[:n]
        for j in range(self._model.njnt):
            adr = int(self._model.jnt_qposadr[j])
            if adr >= n:
                continue
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                qpos[adr] = lo + qpos[adr] * (hi - lo)
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = int(self._model.jnt_qposadr[self._model.eq_obj1id[i]])
            adr2 = int(self._model.jnt_qposadr[self._model.eq_obj2id[i]])
            coef = self._model.eq_data[i, :5]
            qpos[adr2] = np.polyval(coef[::-1], qpos[adr1])
        return qpos

    def fk(self, q_robot):
        return self.kin.fk(self._to_mink_q(q_robot), self.ee_site)

    def jac(self, q_robot):
        """6×n_arm geometric Jacobian of ``ee_site`` (arm-model world = base)."""
        cfg = self.kin._configuration
        cfg.update(self._to_mink_q(q_robot))
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, cfg.data, jacp, jacr, site_id)
        n = min(self.n_arm, self._model.nv)
        return np.vstack([jacp[:, :n], jacr[:, :n]])

    def ik(self, target_pose, q_robot):
        init = self._to_mink_q(q_robot)
        cfg = self.kin._configuration
        cfg.update(init)
        self._ee_task.set_target(mink.SE3.from_matrix(target_pose))
        self._posture.set_target(init)
        for _ in range(self.steps):
            try:
                vel = mink.solve_ik(
                    cfg,
                    [self._ee_task, self._posture],
                    self.dt,
                    "quadprog",
                    damping=self.damping,
                    limits=self._limits,
                    constraints=self._constraints,
                    safety_break=False,
                )
            except mink.NoSolutionFound:
                return False, np.asarray(init[: self.n_arm], dtype=float).copy()
            cfg.integrate_inplace(vel, self.dt)
        q = np.asarray(cfg.q, dtype=float).copy()
        max_step = self.max_joint_vel * self.dt * self.steps * 1.5
        if np.max(np.abs(q[: self.n_arm] - init[: self.n_arm])) > max_step:
            return False, np.asarray(init[: self.n_arm], dtype=float).copy()
        return True, q[: self.n_arm]
