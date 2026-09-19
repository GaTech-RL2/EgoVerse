"""Contact primitives for tools that have no attachment constraint."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .articulation_controller import ArticulationController, SmoothCommand, rot, wrap

CONTACT_TOOLS = ("triangle", "scoop", "flipper", "spring")


class ContactController(ArticulationController):
    def __init__(self, env):
        self.env, self.emb, self.spec = env, env.pusher_shape, env.agent.action_spec
        p = env._pusher_body
        self.command = SmoothCommand(p.position, p.angle, self.spec, speed=4.0)
        self.state = "PLAN"
        self.steps_in_state = 0
        self.reason = None
        self._target_bias = np.zeros(2)
        self._previous_target = None
        self._dither_phase = 0.0
        self.route = []
        self.primitive = None
        self._object_before = None
        self._attempts = 0
        self._stalled = 0
        self._previous_object = None
        self._swing_done = False

    def _object_polygon(self):
        return unary_union(
            [
                Polygon([tuple(sh.body.local_to_world(v)) for v in sh.get_vertices()])
                for sh in self.env._object_shapes
            ]
        ).buffer(1e-7, join_style=2)

    def _rotation_clear(self, p):
        footprint = self._footprint(np.zeros(2), 0.0)
        radius = max(np.linalg.norm(v) for v in footprint.convex_hull.exterior.coords)
        return (
            min(p[0], p[1], 512 - p[0], 512 - p[1]) > radius + 8
            and self._object_polygon().distance(Point(p)) > radius + 7
        )

    def _contact_support(self, angle, normal):
        """Return the contacting point, including asymmetric lateral offset."""
        vertices = np.asarray(
            self._footprint(np.zeros(2), angle).convex_hull.exterior.coords
        )[:-1]
        projection = vertices @ normal
        face = vertices[np.isclose(projection, projection.max(), atol=1e-7, rtol=0)]
        return face.mean(axis=0)

    def _escape(self):
        p, angle, _, _ = self._pose()
        footprint = self._footprint(np.zeros(2), 0.0)
        radius = max(np.linalg.norm(v) for v in footprint.convex_hull.exterior.coords)
        sticky = self.env.agent.control_gap.deadband >= 3.0
        padding = 18.0 if sticky else 10.0
        obstacle = self._object_polygon().buffer(radius + (17.0 if sticky else 9.0))
        candidates = [
            np.array([x, y])
            for x in np.linspace(radius + padding, 512 - padding - radius, 10)
            for y in np.linspace(radius + padding, 512 - padding - radius, 10)
            if not obstacle.intersects(Point(x, y))
        ]
        for target in sorted(candidates, key=lambda v: np.linalg.norm(v - p))[:12]:
            route = self._plan_route(target, angle)
            if route:
                self.route = route
                self._escape_angle = angle
                self.transition("ESCAPE")
                return True
        return False

    def _choose_primitive(self):
        p, angle, c, theta = self._pose()
        goal = np.asarray(self.env._goal_pose)
        polygon = self._object_polygon()
        coords = list(polygon.exterior.coords)
        goal_delta = goal[:2] - c
        poserr = np.linalg.norm(goal_delta)
        angerr = wrap(goal[2] - theta)
        # Use the true centre of mass for contact torque, not the T's origin.
        com = np.array(
            self.env._object_body.local_to_world(
                self.env._object_body.center_of_gravity
            )
        )
        desired = np.r_[goal_delta, angerr * (150.0 if poserr < 60.0 else 45.0)]
        candidates = []
        for a, b in zip(coords[:-1], coords[1:]):
            a, b = np.array(a), np.array(b)
            edge = b - a
            length = np.linalg.norm(edge)
            if length < 15:
                continue
            normal = np.array([-edge[1], edge[0]]) / length
            if not polygon.contains(Point((a + b) / 2 + normal)):
                normal *= -1
            for fraction in (0.2, 0.5, 0.8):
                contact = a + fraction * edge
                arm = contact - com
                torque = arm[0] * normal[1] - arm[1] * normal[0]
                response = np.r_[normal, torque * 0.065]
                gain = float(np.dot(desired, response) / np.linalg.norm(response))
                if gain < 0:
                    continue
                aim = math.atan2(normal[1], normal[0]) + math.pi / 2
                if "angle" not in self.spec:
                    # Translation-only tools cannot execute an orientation command.
                    aim = float(self.env._pusher_body.angle)
                if self.emb == "scoop":
                    aim -= math.pi
                support = self._contact_support(aim, normal)
                ahead = float(support @ normal)
                stage = contact - support - normal * 15.0
                if not self._within_walls(stage, aim):
                    continue
                cost = -gain + np.linalg.norm(p - stage) * 0.045
                candidates.append((cost, stage, aim, normal, contact, ahead))
        for _, stage, aim, normal, contact, ahead in sorted(
            candidates, key=lambda x: x[0]
        )[:24]:
            route = self._plan_route(stage, aim)
            if route:
                self.route = route
                self.primitive = dict(
                    stage=stage,
                    aim=wrap(aim),
                    normal=normal,
                    contact=contact,
                    ahead=ahead,
                    theta=theta,
                    center=c,
                    object_contact=rot(-theta) @ (contact - c),
                    object_normal=rot(-theta) @ normal,
                )
                self._object_before = np.r_[c, theta]
                self._aim_position = p.copy()
                self.transition("AIM")
                return True
        return False

    def __call__(self):
        self.steps_in_state += 1
        p, angle, c, theta = self._pose()
        if self.state == "ESCAPE":
            while len(self.route) > 1 and np.linalg.norm(p - self.route[0]) < 3:
                self.route.pop(0)
            tolerance = 3.5 if self.env.agent.control_gap.deadband >= 3.0 else 1.0
            if np.linalg.norm(p - self.route[0]) < tolerance:
                self.transition("PLAN")
            if self.steps_in_state > 220:
                self.reason = "escape_stalled"
            return self.emit(self.route[0], self._escape_angle, 0.0)
        if self.state == "BACKOFF":
            settled = self.emb != "flipper" or self.env.agent._swing_state < 0.005
            if (
                np.linalg.norm(p - self._backoff) < 3 or self.steps_in_state > 70
            ) and settled:
                self.transition("PLAN")
                self._stalled = 0
            return self.emit(self._backoff, angle, 0.0)
        if self.state == "PLAN":
            self._attempts += 1
            if "angle" in self.spec and not self._rotation_clear(p):
                if self._escape():
                    return self.emit(self.route[0], self._escape_angle, 0.0)
                self.reason = "no_safe_rotation_pose"
                return self.emit(p, angle, 0.0)
            if not self._choose_primitive():
                self.reason = "no_reachable_contact"
                return self.emit(p, angle, 0.0)
        primitive = self.primitive
        if self.state == "AIM":
            if abs(wrap(angle - primitive["aim"])) < 0.005:
                self.transition("ROUTE")
            return self.emit(self._aim_position, primitive["aim"], 0.0)
        if self.state == "ROUTE":
            while len(self.route) > 1 and np.linalg.norm(p - self.route[0]) < 4:
                self.route.pop(0)
            tolerance = max(
                0.5,
                self.env.agent.control_gap.noise_std * 1.8,
                self.env.agent.control_gap.deadband * 0.7,
            )
            if (
                np.linalg.norm(p - primitive["stage"]) < tolerance
                and abs(wrap(angle - primitive["aim"])) < 0.04
            ):
                self.transition("PUSH")
            if self.steps_in_state > 180:
                self.transition("PLAN")
            return self.emit(self.route[0], primitive["aim"], 0.0)
        # Follow a face in object coordinates during each short stroke.
        contact = c + rot(theta) @ primitive["object_contact"]
        normal = rot(theta) @ primitive["object_normal"]
        delta_angle = wrap(theta - primitive["theta"])
        aim = primitive["aim"] + delta_angle
        if "angle" not in self.spec:
            aim = float(self.env._pusher_body.angle)
        goal = np.asarray(self.env._goal_pose)
        poserr = np.linalg.norm(goal[:2] - c)
        angerr = abs(wrap(goal[2] - theta))
        position = (
            contact
            - self._contact_support(aim, normal)
            + normal * (55.0 if self.emb == "scoop" else 22.0)
        )
        grip = 1.0 if self.emb == "spring" else 0.0
        if self.emb == "flipper":
            grip = min(0.55, self.steps_in_state / 90.0)
            # Swing relative to a fixed wrist during contact: this is the
            # mechanism doing work, rather than a cosmetic grip-channel wave.
            aim = primitive["aim"]
            if self.steps_in_state > 35:
                position = p + normal * 3.0
        move = np.linalg.norm(c - self._object_before[:2])
        turn = abs(wrap(theta - self._object_before[2]))
        if self._previous_object is not None:
            progress = np.linalg.norm(c - self._previous_object[:2]) + 40 * abs(
                wrap(theta - self._previous_object[2])
            )
            self._stalled = self._stalled + 1 if progress < 0.03 else 0
        self._previous_object = np.r_[c, theta]
        move_limit = float(np.clip(poserr * 0.4, 1.5, 32.0))
        turn_limit = float(np.clip(angerr * 0.4, 0.015, 0.22))
        if (
            move > move_limit
            or turn > turn_limit
            or self.steps_in_state > 95
            or self._stalled > 35
        ):
            self.transition("BACKOFF")
            self._backoff = np.clip(
                p - normal * (50.0 if self.emb in ("spring", "flipper") else 30.0),
                22,
                490,
            )
        if self.state == "BACKOFF":
            if np.linalg.norm(p - self._backoff) < 3 or self.steps_in_state > 50:
                self.transition("PLAN")
                self._stalled = 0
            return self.emit(self._backoff, angle, 0.0)
        return self.emit(position, aim, grip)


def contact_state(env):
    """True on physical contact; never invent an attachment for these tools."""
    import pymunk

    object_shapes = set(env._object_shapes)
    for shape in env.agent.physics_shapes(env):
        if any(q.shape in object_shapes for q in env._space.shape_query(shape)):
            return True
        # Resting contact may have a small positive solver gap.
        if isinstance(shape, pymunk.Poly):
            points = [shape.body.local_to_world(v) for v in shape.get_vertices()]
        elif isinstance(shape, pymunk.Segment):
            points = [
                shape.body.local_to_world(shape.a),
                shape.body.local_to_world(shape.b),
            ]
        else:
            points = [shape.body.position]
        radius = getattr(shape, "radius", 0.0)
        if any(
            obj.point_query(point).distance - radius < 1.5
            for obj in object_shapes
            for point in points
        ):
            return True
    return False
