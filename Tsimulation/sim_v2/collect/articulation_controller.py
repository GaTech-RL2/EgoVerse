"""Mechanism-aware, smooth controllers for the unchanged Sim V2 physics.

Controllers read simulator state but move bodies only through env.step actions.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from shapely import affinity, prepare
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

GRASPING = ("u_socket", "gripper", "chain_gripper", "suction", "umi")


def wrap(x):
    return float((x + math.pi) % (2 * math.pi) - math.pi)


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class SmoothCommand:
    """Velocity and acceleration bounds, including the first observed pose."""

    def __init__(self, position, angle, spec, speed=4.5):
        self.position = np.array(position, dtype=float)
        self.angle = float(angle)
        self.velocity = np.zeros(2)
        self.omega = 0.0
        self.grip = 0.0
        self.spec = tuple(spec)
        self.speed = speed

    def __call__(self, position, angle, grip=0.0):
        delta = np.asarray(position) - self.position
        distance = np.linalg.norm(delta)
        desired = delta / max(distance, 1e-9) * min(self.speed, distance * 0.3)
        dv = desired - self.velocity
        self.velocity += dv * min(1.0, 0.65 / max(np.linalg.norm(dv), 1e-9))
        self.position += self.velocity
        desired_omega = np.clip(wrap(angle - self.angle) * 0.35, -0.065, 0.065)
        self.omega += float(np.clip(desired_omega - self.omega, -0.012, 0.012))
        self.angle = wrap(self.angle + self.omega)
        self.grip += float(np.clip(grip - self.grip, -0.05, 0.05))
        channels = dict(
            x=self.position[0], y=self.position[1], angle=self.angle, grip=self.grip
        )
        return np.array([channels[key] for key in self.spec], dtype=np.float64)


class ArticulationController:
    """Route with the tool open, insert, verify the grasp, then carry."""

    def __init__(self, env):
        self.env = env
        self.emb = env.pusher_shape
        if self.emb not in GRASPING:
            raise ValueError(
                f"{self.emb} needs a contact controller, not a grasp controller"
            )
        self.spec = env.agent.action_spec
        p = env._pusher_body
        self.command = SmoothCommand(
            p.position,
            p.angle,
            self.spec,
            2.5 if self.emb in ("suction", "umi") else 4.5,
        )
        self.state = "ROUTE"
        self.steps_in_state = 0
        self.feature = None
        self.route = []
        self._route_goal = None
        self._grasp_offset = None
        self._grasp_angle = None
        self._carry_angle = None
        self.feature_candidates = []
        self._target_bias = np.zeros(2)
        self._previous_target = None
        self._dither_phase = 0.0
        self.reason = None
        self.depth = dict(
            u_socket=-11.0, gripper=16.0, chain_gripper=13.0, suction=4.0, umi=27.0
        )[self.emb]
        self._select_feature()

    def transition(self, state):
        if state != self.state:
            self.state = state
            self.steps_in_state = 0
            self._target_bias *= 0.0

    def emit(self, position, angle, grip=0.0):
        position = np.asarray(position)
        p = np.array(self.env._pusher_body.position)
        if self.env.agent.control_gap.deadband >= 3.0:
            steady = (
                self._previous_target is not None
                and np.linalg.norm(position - self._previous_target) < 0.5
            )
            if steady and np.linalg.norm(position - p) < 12.0:
                self._target_bias += 0.13 * (position - p)
                size = np.linalg.norm(self._target_bias)
                self._target_bias *= min(1.0, 9.0 / max(size, 1e-9))
            else:
                self._target_bias *= 0.85
        self._previous_target = position.copy()
        dither = np.zeros(2)
        if (
            self.env.agent.control_gap.deadband >= 3.0
            and np.linalg.norm(position - p) < 12.0
        ):
            self._dither_phase += 0.11
            dither = 2.5 * np.array(
                [math.cos(self._dither_phase), math.sin(self._dither_phase)]
            )
        return self.command(position + self._target_bias + dither, angle, grip)

    def _pose(self):
        p, o = self.env._pusher_body, self.env._object_body
        return (
            np.array(p.position),
            float(p.angle),
            np.array(o.position),
            float(o.angle),
        )

    def _frame(self, center, theta, extra=0.0):
        point, normal = self.feature
        direction = rot(theta) @ normal
        tip = center + rot(theta) @ point
        angle = math.atan2(-direction[1], -direction[0])
        if self.emb != "u_socket":
            angle -= math.pi / 2
        if self.emb == "suction":
            angle += math.pi  # The pad faces local -Y; its stem is behind +Y.
        return tip + (self.depth + extra) * direction, wrap(angle), direction

    def _select_feature(self):
        p, _, center, theta = self._pose()
        goal = np.asarray(self.env._goal_pose)
        candidates = (
            (np.array([0.0, 75.0]), np.array([0.0, 1.0])),
            (np.array([60.0, -30.0]), np.array([1.0, 0.0])),
            (np.array([-60.0, -30.0]), np.array([-1.0, 0.0])),
        )
        if self.emb == "suction":
            candidates = (
                (np.array([15.0, 5.0]), np.array([1.0, 0.0])),
                (np.array([-15.0, 5.0]), np.array([-1.0, 0.0])),
                (np.array([0.0, -45.0]), np.array([0.0, -1.0])),
                *candidates,
            )
        best = None
        ranked = []
        for i, feature in enumerate(candidates):
            self.feature = feature
            seat, angle, direction = self._frame(center, theta)
            final, final_angle, _ = self._frame(goal[:2], goal[2])
            if not self._within_walls(seat, angle) or not self._within_walls(
                final, final_angle
            ):
                continue
            # Prefer the stem unless another grasp substantially shortens routing.
            pre = seat + 60 * direction
            score = np.linalg.norm(p - pre) + i * 45
            ranked.append((score, feature))
            if best is None or score < best[0]:
                best = score, feature
        if best is None:
            self.feature = candidates[0]
            self.reason = "no_grasp_with_start_and_goal_wall_clearance"
        else:
            self.feature = best[1]
        self.feature_candidates = [
            feature for _, feature in sorted(ranked, key=lambda item: item[0])
        ]

    def _footprint(self, position, angle):
        """Rigid transform of the actual open tool's collision geometry."""
        import pymunk

        master = self.env._pusher_body
        delta = angle - float(master.angle)
        shapes = []
        for shape in self.env.agent.physics_shapes(self.env):
            if isinstance(shape, pymunk.Poly):
                vertices = [shape.body.local_to_world(v) for v in shape.get_vertices()]
                polygon = Polygon([(v.x, v.y) for v in vertices])
            elif isinstance(shape, pymunk.Circle):
                v = shape.body.local_to_world(shape.offset)
                polygon = Point(v.x, v.y).buffer(shape.radius)
            elif isinstance(shape, pymunk.Segment):
                a, b = (
                    shape.body.local_to_world(shape.a),
                    shape.body.local_to_world(shape.b),
                )
                polygon = LineString([(a.x, a.y), (b.x, b.y)]).buffer(shape.radius)
            else:
                raise TypeError(type(shape))
            polygon = affinity.translate(
                polygon, -master.position.x, -master.position.y
            )
            polygon = affinity.rotate(polygon, delta, origin=(0, 0), use_radians=True)
            shapes.append(affinity.translate(polygon, position[0], position[1]))
        return unary_union(shapes)

    def _within_walls(self, position, angle):
        # At feature selection the tool is fully open, a conservative test.
        return box(8, 8, 504, 504).covers(self._footprint(position, angle))

    def _plan_route(self, target, angle):
        """Visibility graph in the tool's translation configuration space.

        Convex Minkowski sums for each object/tool polygon keep the narrow
        insertion corridor available, unlike an enclosing collision circle.
        """
        p, _, _, _ = self._pose()
        footprint = self._footprint(np.zeros(2), angle)
        parts = list(footprint.geoms) if hasattr(footprint, "geoms") else [footprint]
        tool_vertices = [list(poly.convex_hull.exterior.coords)[:-1] for poly in parts]
        forbidden = []
        for shape in self.env._object_shapes:
            obj = [shape.body.local_to_world(v) for v in shape.get_vertices()]
            for vertices in tool_vertices:
                sums = [(v.x - x, v.y - y) for v in obj for x, y in vertices]
                forbidden.append(Polygon(sums).convex_hull)
        obstacles = unary_union(forbidden).buffer(3.0, join_style=2)
        minx, miny, maxx, maxy = footprint.bounds
        allowed = box(8 - minx, 8 - miny, 504 - maxx, 504 - maxy)
        prepare(obstacles)
        prepare(allowed)
        if not allowed.covers(Point(target)) or obstacles.contains(Point(target)):
            return []
        polys = list(obstacles.geoms) if hasattr(obstacles, "geoms") else [obstacles]
        nodes = [p, np.asarray(target)]
        for poly in polys:
            outer = poly.buffer(1.0, join_style=2)
            nodes.extend(
                np.array(v)
                for v in list(outer.exterior.coords)[:-1]
                if allowed.covers(Point(v))
            )
        # A rotated footprint can contain the start: let the first edge exit
        # monotonically, then validate actual approach in the rollout.
        start_inside = obstacles.contains(Point(p))

        def clear(i, j):
            segment = LineString([nodes[i], nodes[j]])
            if not allowed.covers(segment):
                return False
            if not obstacles.intersects(segment):
                return True
            if i == 0 and start_inside:
                return segment.intersection(obstacles).geom_type == "LineString"
            return False

        queue = [(0.0, 0)]
        costs, prev = {0: 0.0}, {}
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > costs[i]:
                continue
            if i == 1:
                indices = [1]
                while indices[-1] != 0:
                    indices.append(prev[indices[-1]])
                return [nodes[k] for k in indices[-2::-1]]
            for j in range(1, len(nodes)):
                best = costs.get(j, math.inf)
                if i == j or cost >= best:
                    continue
                new = cost + float(np.linalg.norm(nodes[i] - nodes[j]))
                # Visibility is expensive; an edge that cannot improve the
                # existing path never affects Dijkstra's result.
                if new >= best or not clear(i, j):
                    continue
                costs[j], prev[j] = new, i
                heapq.heappush(queue, (new, j))
        return []

    def __call__(self):
        self.steps_in_state += 1
        p, angle, center, theta = self._pose()
        engaged = bool(
            getattr(self.env.agent, "socket_latched", False)
            or self.env.agent.active_constraints()
        )
        rigid_ready = engaged and (
            self.emb != "umi" or len(self.env.agent.active_constraints()) == 2
        )
        seat, aim, normal = self._frame(center, theta)
        if rigid_ready and self._grasp_offset is None:
            self._grasp_offset = rot(-theta) @ (p - center)
            self._grasp_angle = wrap(angle - theta)
            self._carry_angle = theta
            self.transition("CENTER")
        if self._grasp_offset is not None:
            if not engaged:
                self.reason = "grasp_lost"
            goal = np.asarray(self.env._goal_pose)
            grip = 1.0
            if self.state == "CENTER":
                target_center = np.array([256.0, 256.0])
                target_theta = self._carry_angle
                if np.linalg.norm(center - target_center) < 3.0:
                    self.transition("ROTATE")
            elif self.state == "ROTATE":
                target_center = np.array([256.0, 256.0])
                target_theta = theta + np.clip(wrap(goal[2] - theta), -0.022, 0.022)
                if abs(wrap(goal[2] - theta)) < 0.018:
                    self.transition("DELIVER")
            else:
                target_center = goal[:2]
                target_theta = goal[2]
            if self.emb in ("suction", "umi"):
                target_position = target_center + rot(target_theta - theta) @ (
                    p - center
                )
                target_angle = angle + wrap(target_theta - theta)
            else:
                target_position = target_center + rot(target_theta) @ self._grasp_offset
                target_angle = target_theta + self._grasp_angle
            return self.emit(target_position, target_angle, grip)

        if self.state == "ROUTE":
            if self._route_goal is None:
                # Shorten the approach along its axis, without clamping it off-axis.
                for feature in self.feature_candidates:
                    self.feature = feature
                    seat, aim, normal = self._frame(center, theta)
                    for extra in (65.0, 50.0, 40.0, 30.0, 22.0):
                        pre = seat + extra * normal
                        self.route = self._plan_route(pre, aim)
                        if self.route:
                            self._route_goal = pre
                            break
                    if self._route_goal is not None:
                        break
                if self._route_goal is None:
                    self.reason = "no_collision_free_approach"
                    return self.emit(p, angle, 0.0)
            while len(self.route) > 1 and np.linalg.norm(p - self.route[0]) < 5.0:
                self.route.pop(0)
            waypoint = self.route[0]
            noise = self.env.agent.control_gap.noise_std
            sticky = self.env.agent.control_gap.deadband >= 3.0
            tolerance = max(0.15, noise * 1.5, 6.0 if sticky else 0.0)
            off = p - self._route_goal
            lateral = abs(float(normal[0] * off[1] - normal[1] * off[0]))
            aligned = not sticky or lateral < (0.3 if self.emb == "u_socket" else 2.0)
            if (
                np.linalg.norm(p - self._route_goal) < tolerance
                and aligned
                and abs(wrap(angle - aim)) < 0.002
            ):
                self.transition("INSERT")
            return self.emit(waypoint, aim, 0.0)
        if self.state == "INSERT":
            if np.linalg.norm(p - seat) < 2.5:
                self.transition("CLOSE")
            return self.emit(seat, aim, 0.0)
        return self.emit(seat, aim, 1.0)
