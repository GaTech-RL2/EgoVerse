"""Observable evidence that each physical mechanism did useful work."""

from __future__ import annotations

import numpy as np

from .articulation_controller import GRASPING, wrap
from .contact_controller import CONTACT_TOOLS, contact_state

ARTICULATED = (*GRASPING, *CONTACT_TOOLS)


def interaction(env):
    if env.pusher_shape in GRASPING:
        return bool(
            getattr(env.agent, "socket_latched", False)
            or env.agent.active_constraints()
        )
    contact = contact_state(env)
    if not contact or env.pusher_shape != "scoop":
        return contact
    # Material inside the concavity, with physical contact on the arc.
    palm = env._pusher_body
    count = sum(
        any(
            shape.point_query(palm.local_to_world((x, y))).distance <= 0.0
            for shape in env._object_shapes
        )
        for x in (-20.0, 0.0, 20.0)
        for y in (-24.0, -12.0, 0.0)
    )
    return count >= 2


class MechanismAudit:
    def __init__(self, env):
        self.previous_object = np.array(env._object_body.position)
        self.previous_swing = float(getattr(env.agent, "swing_deg", 0.0))
        self.steps = 0
        self.contact_angles = []
        self.work = []

    def observe(self, env, engaged):
        obj = np.array(env._object_body.position)
        moved = np.linalg.norm(obj - self.previous_object)
        swing = float(getattr(env.agent, "swing_deg", 0.0))
        if engaged:
            self.contact_angles.append(float(env._pusher_body.angle))
        if env.pusher_shape == "flipper":
            used = abs(swing - self.previous_swing) > 0.05
        elif env.pusher_shape == "spring":
            used = env.agent.compression > 0.25 and env.agent._stiff > 0.5
        else:
            used = True
        useful = bool(engaged and used and moved > 0.05)
        self.steps += int(useful)
        self.work.append(useful)
        self.previous_object = obj
        self.previous_swing = swing

    def metrics(self):
        travel = sum(
            abs(wrap(b - a))
            for a, b in zip(self.contact_angles[:-1], self.contact_angles[1:])
        )
        return dict(mechanism_work_steps=self.steps, contact_angle_travel=travel)

    def passes(self, emb):
        if self.steps < 5:
            return False
        if (
            emb in ("triangle", "pentagon", "scoop")
            and self.metrics()["contact_angle_travel"] < 0.1
        ):
            return False
        return True
