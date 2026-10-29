"""Hardware-free FK/IK using the configured Yam MuJoCo model and TCP site."""

import numpy as np
from scipy.spatial.transform import Rotation


class MujocoArmKinematics:
    def __init__(
        self,
        xml_path,
        joint_names,
        site_name="tcp_site",
        max_iterations=100,
        position_tolerance=0.002,
        rotation_tolerance=0.02,
        damping=0.001,
        max_iteration_step=0.05,
    ):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(name).id for name in joint_names]
        if len(self.joints) != 6 or len(set(self.joints)) != 6:
            raise ValueError("Specify the six ordered Yam arm joints")
        for joint in self.joints:
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Yam arm joints must be scalar hinge joints")
        self.qidx = self.model.jnt_qposadr[self.joints]
        self.didx = self.model.jnt_dofadr[self.joints]
        self.site = self.model.site(site_name).id
        self.max_iterations = int(max_iterations)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.damping = float(damping)
        self.max_step = float(max_iteration_step)
        if (
            min(
                self.max_iterations,
                self.position_tolerance,
                self.rotation_tolerance,
                self.damping,
                self.max_step,
            )
            <= 0
        ):
            raise ValueError("IK limits and tolerances must be positive")

    def fk(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("FK requires six finite arm positions")
        self.data.qpos[self.qidx] = joints
        self.mj.mj_forward(self.model, self.data)
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.site]
        return pose

    def ik(self, target, seed):
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("IK requires a finite homogeneous target")
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(self.max_iterations):
            current = self.fk(q)
            error = np.r_[
                target[:3, 3] - current[:3, 3],
                Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]
            if (
                np.linalg.norm(error[:3]) <= self.position_tolerance
                and np.linalg.norm(error[3:]) <= self.rotation_tolerance
            ):
                return q
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            self.mj.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack([jp[:, self.didx], jr[:, self.didx]])
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + self.damping**2 * np.eye(6), error
            )
            q += np.clip(delta, -self.max_step, self.max_step)
            for index, joint in enumerate(self.joints):
                if self.model.jnt_limited[joint]:
                    q[index] = np.clip(q[index], *self.model.jnt_range[joint])
        raise ValueError(
            "Predicted Cartesian target is unreachable within IK tolerances"
        )

    def ik_relaxed(self, target, seed, position_tolerance, rotation_tolerance):
        """Try a nearby endpoint, accepting only independently checked FK errors.

        Called only after strict IK fails. The residual is scaled by the two
        explicit acceptance envelopes, so meters and radians are not mixed with
        equal numerical weight. Bounded backtracking avoids worsening iterates.
        This is endpoint IK; the rollout's paired joint-step guard still applies.
        """
        target = np.asarray(target, dtype=float)
        q = np.asarray(seed, dtype=float).copy()
        tolerances = np.array([position_tolerance, rotation_tolerance], dtype=float)
        if (
            target.shape != (4, 4)
            or not np.isfinite(target).all()
            or q.shape != (6,)
            or not np.isfinite(q).all()
            or not np.isfinite(tolerances).all()
            or np.any(tolerances <= 0)
        ):
            raise ValueError(
                "Relaxed IK requires finite poses, joints and positive tolerances"
            )
        lower, upper = np.full(6, -np.inf), np.full(6, np.inf)
        for index, joint in enumerate(self.joints):
            if self.model.jnt_limited[joint]:
                lower[index], upper[index] = self.model.jnt_range[joint]
        q = np.clip(q, lower, upper)
        scale = np.repeat(1.0 / tolerances, 3)

        def residual(joints):
            current = self.fk(joints)
            return np.r_[
                target[:3, 3] - current[:3, 3],
                Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]

        # Include the FK check after the final allowed update.
        for iteration in range(self.max_iterations + 1):
            error = residual(q)
            if (
                np.linalg.norm(error[:3]) <= position_tolerance
                and np.linalg.norm(error[3:]) <= rotation_tolerance
            ):
                return q
            if iteration == self.max_iterations:
                break
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            self.mj.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = scale[:, None] * np.vstack([jp[:, self.didx], jr[:, self.didx]])
            weighted_error = error * scale
            try:
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + self.damping**2 * np.eye(6), weighted_error
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError("Relaxed IK numerical failure") from exc
            if not np.isfinite(delta).all():
                raise ValueError("Relaxed IK produced a nonfinite step")
            delta *= min(1.0, self.max_step / max(np.max(np.abs(delta)), 1e-12))
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = np.clip(q + fraction * delta, lower, upper)
                candidate_error = residual(candidate) * scale
                if np.dot(candidate_error, candidate_error) < np.dot(
                    weighted_error, weighted_error
                ):
                    q = candidate
                    break
            else:
                break
        raise ValueError("No IK endpoint within the configured relaxed pose bounds")
