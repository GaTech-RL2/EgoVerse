"""Optional pytracik endpoint IK using the configured MuJoCo chain exactly.

MuJoCo remains the authority for FK, limits and acceptance. No robot drivers
are imported here. The URDF is generated in memory from the compiled model;
there is no second hand-maintained robot description or TCP transform.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def require_pytracik():
    try:
        from trac_ik import TracIK
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "TRAC-IK requires the optional dependency: pip install pytracik==0.0.3"
        ) from exc
    return TracIK


def chain_urdf(kinematics):
    """Export six serial limited hinges in world-aligned zero-angle frames.

    Each joint frame is at its zero-angle anchor and aligned with world axes.
    Joint axes, intervening fixed transforms, nonzero reference angles, base
    placement and the actual TCP site are therefore all retained.
    """
    k = kinematics
    model, mj = k.model, k.mj
    bodies = []
    body = int(model.site_bodyid[k.site])
    while body:
        bodies.append(body)
        body = int(model.body_parentid[body])
    path_joints = []
    for body in reversed(bodies):
        if model.body_mocapid[body] >= 0:
            raise ValueError("TRAC-IK requires a fixed-base chain")
        start = int(model.body_jntadr[body])
        path_joints.extend(range(start, start + int(model.body_jntnum[body])))
    if path_joints != list(k.joints) or not np.all(model.jnt_limited[k.joints]):
        raise ValueError(
            "TRAC-IK requires the six ordered, limited hinges on the TCP chain"
        )
    data = mj.MjData(model)
    data.qpos[k.qidx] = 0
    mj.mj_forward(model, data)
    robot = ET.Element("robot", name="configured_yam_tcp")
    ET.SubElement(robot, "link", name="world")

    def vector(values):
        return " ".join(format(float(value), ".17g") for value in values)

    previous, anchor = "world", np.zeros(3)
    for index, joint_id in enumerate(k.joints):
        child = f"arm_{index}"
        ET.SubElement(robot, "link", name=child)
        joint = ET.SubElement(
            robot, "joint", name=model.joint(joint_id).name, type="revolute"
        )
        ET.SubElement(joint, "parent", link=previous)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(
            joint, "origin", xyz=vector(data.xanchor[joint_id] - anchor), rpy="0 0 0"
        )
        ET.SubElement(joint, "axis", xyz=vector(data.xaxis[joint_id]))
        lower, upper = model.jnt_range[joint_id]
        ET.SubElement(
            joint, "limit", lower=str(lower), upper=str(upper), effort="1", velocity="1"
        )
        previous, anchor = child, data.xanchor[joint_id].copy()
    ET.SubElement(robot, "link", name="tcp")
    joint = ET.SubElement(robot, "joint", name="configured_tcp", type="fixed")
    ET.SubElement(joint, "parent", link=previous)
    ET.SubElement(joint, "child", link="tcp")
    rotation = Rotation.from_matrix(data.site_xmat[k.site].reshape(3, 3))
    ET.SubElement(
        joint,
        "origin",
        xyz=vector(data.site_xpos[k.site] - anchor),
        rpy=vector(rotation.as_euler("xyz")),
    )
    return ET.tostring(robot, encoding="unicode")


class TracIKEndpoint:
    def __init__(self, kinematics, *, timeout_s=0.005, solver_type="Distance"):
        from egomimic.robot.yam_station.ik_options import validate_ik_solver

        validate_ik_solver(
            dict(backend="pytracik", timeout_s=timeout_s, solver_type=solver_type)
        )
        self.kinematics = kinematics
        factory = require_pytracik()
        with TemporaryDirectory(prefix="yam-tracik-") as directory:
            path = Path(directory) / "arm.urdf"
            path.write_text(chain_urdf(kinematics))
            self.solver = factory(
                "world",
                "tcp",
                str(path),
                timeout=timeout_s,
                epsilon=1e-5,
                solver_type=solver_type,
            )
        self.lower, self.upper = kinematics.model.jnt_range[kinematics.joints].T.copy()
        if self.solver.dof != 6:
            raise ValueError("TRAC-IK chain must have exactly six joints")
        # pytracik 0.0.3 rounds URDF limits to float32 during construction.
        # Its public setter retains doubles; install the authoritative limits
        # explicitly, then verify them without weakening the comparison.
        self.solver.joint_limits = (self.lower, self.upper)
        if not np.allclose(
            self.solver.joint_limits, (self.lower, self.upper), atol=1e-10, rtol=0
        ):
            raise ValueError("TRAC-IK joint limits differ from the configured model")
        # Refuse use if conversion or library conventions disagree. These are
        # model-only calls and never read or command a physical arm.
        rng = np.random.default_rng(20261011)
        samples = [np.clip(np.zeros(6), self.lower, self.upper)]
        samples.extend(rng.uniform(self.lower, self.upper, (12, 6)))
        for q in samples:
            expected = kinematics.fk(q)
            position, rotation = self.solver.fk(q)
            if not np.allclose(
                position, expected[:3, 3], atol=1e-8, rtol=0
            ) or not np.allclose(rotation, expected[:3, :3], atol=1e-8, rtol=0):
                raise ValueError("TRAC-IK FK disagrees with the configured MuJoCo TCP")

    def ik(self, target, seed):
        target, seed = np.asarray(target, dtype=float), np.asarray(seed, dtype=float)
        if (
            target.shape != (4, 4)
            or seed.shape != (6,)
            or not (np.isfinite(target).all() and np.isfinite(seed).all())
        ):
            raise ValueError(
                "TRAC-IK requires a finite target and six finite seed joints"
            )
        result = self.solver.ik(
            target[:3, 3], target[:3, :3], np.clip(seed, self.lower, self.upper)
        )
        if result is None:
            raise ValueError("TRAC-IK did not find an endpoint within its time budget")
        q = np.asarray(result, dtype=float)
        if (
            q.shape != (6,)
            or not np.isfinite(q).all()
            or np.any(q < self.lower)
            or np.any(q > self.upper)
        ):
            raise ValueError("TRAC-IK returned invalid or out-of-limit joints")
        actual = self.kinematics.fk(q)
        position_error = np.linalg.norm(target[:3, 3] - actual[:3, 3])
        rotation_error = Rotation.from_matrix(
            target[:3, :3] @ actual[:3, :3].T
        ).magnitude()
        if (
            position_error > self.kinematics.position_tolerance
            or rotation_error > self.kinematics.rotation_tolerance
        ):
            raise ValueError("TRAC-IK endpoint failed independent FK pose checks")
        return q.copy()
