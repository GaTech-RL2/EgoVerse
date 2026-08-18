"""Agents (pushers) for PushShapes.

WHY THIS EXISTS: the u_socket was added by editing the environment. Its latch,
friction and penetration guards -- ~470 lines -- went straight into env.py as
``if pusher_shape == "u_socket"`` branches, and its 3-DOF action became a
hardcoded ``expected_shape = (3,) if ... else (2,)``. That is the single largest
reason sim_v1 and sim_v2 diverged: v1's env has ZERO socket references, v2's has
115.

An agent owns three things the environment should not know about:

  * its ACTION SPACE  -- ``action_dim`` (2 for a free-moving pusher, 3 when the
    agent also controls orientation);
  * its BODY          -- how it is built in the pymunk space;
  * its CONTACT MODEL -- any per-substep latching or penetration guards.

``env.step`` calls ``pre_substep`` / ``post_substep`` around each physics
substep and is otherwise agent-agnostic, so a new agent with an unusual action
space is a new class here rather than another branch in the simulator.

The env handle passed to the hooks exposes what agents legitimately need --
``_space``, ``_pusher_body``, ``_object_body``, ``_pusher_shapes`` and the
geometry queries (penetration depths, arena metrics). Agents keep their own
state (latch constraints, relatch blocking) on themselves.
"""
from __future__ import annotations

import math

import numpy as np
import pymunk

from .shapes import (
    GRIPPER_JAW_HALF_H,
    GRIPPER_JAW_HALF_W,
    GRIPPER_JAW_MAX_GAP,
    GRIPPER_JAW_MIN_GAP,
    OBJECT_FRICTION,
    TWO_POINT_RADIUS,
    U_SOCKET_CROSSBAR_INNER_X,
    U_SOCKET_POCKET_X_MAX,
    U_SOCKET_POCKET_X_MIN,
    U_SOCKET_POCKET_Y_HALF,
    _rect_verts,
    make_pusher,
)

# --- socket geometry constants (moved from env.py: they describe the
# --- agent's pocket/latch tolerances, not the environment) ---
_LATCH_DEPTH_EPSILON = 1e-9
_LATCH_STATIC_MAX_DEPTH = 0.2
_LEGACY_SOCKET_UNLATCH_DEPTH = 0.5
_SOCKET_INNER_CORNER_INSET = 0.5
_SOCKET_INNER_FACE_TOL = 1.0
_SOCKET_LATCH_FACE_TOL = 2.0
_SOCKET_POCKET_MOUTH_INSET = 2.0
_SOCKET_POCKET_PAD = 1.0
_SOCKET_RELATCH_BLOCK = 20
_SOLID_OBJECT_STATIC_MAX_DEPTH = 0.2
_SOLID_PUSHER_OBJECT_MAX_DEPTH = 0.5
_UNLATCHED_EDGE_GUARD_DISTANCE = 1.0
_UNLATCHED_EDGE_MAX_DEPTH = 0.5



class ControlGap:
    """How badly an embodiment executes what it was told.

    The seven contact models make agents differ in WHAT they can do. This
    makes them differ in HOW WELL they do it, which is an independent axis:
    two agents with identical geometry and action space still demand different
    policies if one tracks its command exactly and the other lags, drifts and
    ignores small motions.

    Every term is a distinct failure mode, because they are not
    interchangeable -- a policy compensates for each differently:

      latency_steps  the command that lands is the one from N steps ago, so
                     the policy must predict rather than react;
      lag            first-order servo lag: the target eases toward the
                     command instead of jumping, so fast strokes undershoot;
      deadband       commands moving less than this are ignored, which makes
                     fine positioning impossible and forces overshoot-and-
                     return strategies;
      gain           systematic scale error about the body's current position;
                     a constant, learnable bias rather than noise;
      noise_std      zero-mean jitter -- the only term that cannot be
                     compensated, and so sets an error floor.

    Noise is drawn from a per-episode RNG seeded in ``Agent.on_reset`` from the
    env's own generator, so a replayed episode reproduces the same gap.
    """

    __slots__ = ("latency_steps", "lag", "deadband", "gain", "noise_std")

    def __init__(self, *, latency_steps: int = 0, lag: float = 0.0,
                 deadband: float = 0.0, gain: float = 1.0,
                 noise_std: float = 0.0):
        if not 0.0 <= lag < 1.0:
            raise ValueError(f"lag must be in [0, 1), got {lag}")
        if latency_steps < 0:
            raise ValueError("latency_steps must be >= 0")
        self.latency_steps = int(latency_steps)
        self.lag = float(lag)
        self.deadband = float(deadband)
        self.gain = float(gain)
        self.noise_std = float(noise_std)

    @property
    def is_ideal(self) -> bool:
        return (
            self.latency_steps == 0 and self.lag == 0.0 and self.deadband == 0.0
            and self.gain == 1.0 and self.noise_std == 0.0
        )

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def apply(self, agent, tx, ty, ang):
        if self.is_ideal:
            agent._last_target = (tx, ty)
            return tx, ty, ang

        # latency: FIFO of raw commands, so the servo chases a stale target
        if self.latency_steps:
            agent._cmd_queue.append((tx, ty))
            if len(agent._cmd_queue) > self.latency_steps:
                tx, ty = agent._cmd_queue.pop(0)
            else:
                tx, ty = agent._cmd_queue[0]

        prev = agent._gap_state
        if prev is None:
            prev = (tx, ty)

        # deadband: ignore sub-threshold motion entirely
        if self.deadband > 0.0:
            if math.hypot(tx - prev[0], ty - prev[1]) < self.deadband:
                tx, ty = prev

        # first-order lag toward the command
        if self.lag > 0.0:
            k = 1.0 - self.lag
            tx = prev[0] + (tx - prev[0]) * k
            ty = prev[1] + (ty - prev[1]) * k

        # systematic gain error, measured about the previous target
        if self.gain != 1.0:
            tx = prev[0] + (tx - prev[0]) * self.gain
            ty = prev[1] + (ty - prev[1]) * self.gain

        if self.noise_std > 0.0:
            tx += float(agent._rng.normal(0.0, self.noise_std))
            ty += float(agent._rng.normal(0.0, self.noise_std))

        agent._gap_state = (tx, ty)
        agent._last_target = (tx, ty)
        return tx, ty, ang


#: Named presets, so a dataset can sweep control fidelity as a factor rather
#: than hand-tuning five numbers per run.
CONTROL_GAPS: dict[str, ControlGap] = {
    "ideal": ControlGap(),
    # A good servo: barely perceptible, but enough that exact tracking fails.
    "tight": ControlGap(lag=0.25, noise_std=0.3),
    # A cheap one: visible lag, a little stiction, slight undershoot.
    "loose": ControlGap(latency_steps=2, lag=0.55, deadband=1.5, gain=0.95,
                        noise_std=0.8),
    # Teleop over a bad link: the operator is always behind.
    "laggy": ControlGap(latency_steps=6, lag=0.35, noise_std=0.4),
    # Worn mechanics: large stiction band, consistent undershoot.
    "sticky": ControlGap(deadband=4.0, gain=0.88, lag=0.15),
    # Noisy sensing: no bias to learn, just an irreducible floor.
    "jittery": ControlGap(noise_std=2.5),
}


class Agent:
    """Base 2-DOF pusher with the fixed Sim V2 solid-contact guard."""

    action_dim = 2
    #: True when the agent commands its own orientation (3-DOF and up), so
    #: env._drive_pusher_toward uses target_pose()'s angle verbatim.
    controls_angle = False
    #: True when the body should auto-yaw to face its direction of travel
    #: (the stick's behaviour). Mutually exclusive with controls_angle.
    auto_orients = False
    #: physics knobs an agent contributes to episode_init, so a replay can
    #: reconstruct the exact contact model it was collected under.
    init_fields: tuple[str, ...] = ()

    def __init__(
        self,
        shape: str,
        *,
        solid_pusher: bool = True,
        solid_contact_guard: bool = True,
        control_gap: "ControlGap | str | None" = None,
    ):
        if isinstance(control_gap, str):
            if control_gap not in CONTROL_GAPS:
                raise ValueError(
                    f"unknown control_gap {control_gap!r}; "
                    f"known: {sorted(CONTROL_GAPS)}"
                )
            control_gap = CONTROL_GAPS[control_gap]
        self.control_gap = control_gap or CONTROL_GAPS["ideal"]
        self._gap_state = None
        self._cmd_queue: list[tuple[float, float]] = []
        self._last_target = None
        self._last_command = None
        self._rng = np.random.default_rng(0)
        self.shape = shape
        self.solid_pusher = bool(solid_pusher)
        self.solid_contact_guard = bool(solid_contact_guard)
        # Only the stick auto-yaws; this preserves _ORIENTED_PUSHERS exactly.
        if shape == "stick":
            self.auto_orients = True

    def build(self, space: pymunk.Space, position):
        """Create the pusher body/shapes in ``space``."""
        return make_pusher(self.shape, space, position)

    def _target_pose(self, action):
        """RAW (target_x, target_y, target_angle|None) from an action.

        Subclasses override THIS, not ``target_pose`` -- the base wraps it so
        the control gap cannot be forgotten by a new agent.
        """
        return float(action[0]), float(action[1]), None

    def target_pose(self, action):
        """The target the environment actually servos toward.

        Raw command in, DEGRADED command out. See ControlGap: this is where an
        embodiment stops being a perfect position source.
        """
        tx, ty, ang = self._target_pose(action)
        self._last_command = (tx, ty)
        return self.control_gap.apply(self, tx, ty, ang)

    def tracking_error(self, env) -> float:
        """Distance from the body to the RAW command.

        Measured against the raw command, never the degraded target. Measuring
        against the degraded one inverts the metric: an agent whose lag holds
        the target back sits closer to it, so `laggy` scored 6.29 against
        `ideal`'s 14.00 -- reporting the worst servo as the best. This number
        includes the inherent speed limit (the body cannot teleport), so it is
        non-zero even for an ideal agent; `command_gap` isolates the
        ControlGap on its own.
        """
        if self._last_command is None:
            return 0.0
        p = env._pusher_body.position
        return float(
            math.hypot(p.x - self._last_command[0], p.y - self._last_command[1])
        )

    def command_gap(self) -> float:
        """Distance between the raw command and the target actually servoed to.

        Purely the ControlGap's doing -- zero for an ideal agent regardless of
        how fast the body can move.
        """
        if self._last_command is None or self._last_target is None:
            return 0.0
        return float(math.hypot(
            self._last_target[0] - self._last_command[0],
            self._last_target[1] - self._last_command[1],
        ))

    def reset_control_gap(self, env) -> None:
        """Clear per-episode gap state and reseed the noise RNG.

        Seeded from the env's generator so a replay with the same reset seed
        reproduces the identical noise sequence.
        """
        self._gap_state = None
        self._cmd_queue = []
        self._last_target = None
        self._last_command = None
        seed = int(env._np_random.integers(0, 2**31 - 1))
        self._rng = np.random.default_rng(seed)

    def on_reset(self, env) -> None:
        """Per-episode state reset. No-op for a stateless agent."""

    def pre_substep(self, env):
        """Capture whatever post_substep needs to compare against."""
        return self._capture_solid_contact_guard_pose(env)

    def post_substep(self, env, captured) -> None:
        """Reject a substep that tunnels through the object or static geometry."""
        self._guard_solid_contact_penetration(env, captured)

    def _capture_solid_contact_guard_pose(
        self,
        env,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        float,
    ] | None:
        """Capture the current pair pose and penetration before a substep."""
        if not self.solid_pusher or not self.solid_contact_guard:
            return None
        pusher = env._pusher_body
        obj = env._object_body
        return (
            (float(pusher.position.x), float(pusher.position.y), float(pusher.angle)),
            (float(obj.position.x), float(obj.position.y), float(obj.angle)),
            env._object_static_penetration_depth(),
            env._pusher_object_penetration_depth(),
        )

    def _guard_solid_contact_penetration(
        self,
        env,
        previous_pose: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
        ] | None,
    ) -> None:
        """Restore the last safe pose if a solid contact tunnels deeply.

        The comparison includes the previous penetration so a manually
        restored pose that starts slightly embedded can still move outward.
        """
        if previous_pose is None:
            return
        pusher_pose, object_pose, previous_static_depth, previous_pusher_depth = (
            previous_pose
        )
        static_is_unsafe = env._object_static_penetration_depth() > max(
            _SOLID_OBJECT_STATIC_MAX_DEPTH,
            previous_static_depth,
        ) + _LATCH_DEPTH_EPSILON
        pusher_is_unsafe = env._pusher_object_penetration_depth() > max(
            _SOLID_PUSHER_OBJECT_MAX_DEPTH,
            previous_pusher_depth,
        ) + _LATCH_DEPTH_EPSILON
        if static_is_unsafe or pusher_is_unsafe:
            self._restore_pair_pose(env, pusher_pose, object_pose)

    @staticmethod
    def _restore_pair_pose(
        env,
        pusher_pose: tuple[float, float, float],
        object_pose: tuple[float, float, float],
    ) -> None:
        """Restore both bodies after an unsafe physics substep."""
        pusher = env._pusher_body
        obj = env._object_body
        pusher.position = pusher_pose[:2]
        pusher.angle = pusher_pose[2]
        pusher.velocity = (0.0, 0.0)
        pusher.angular_velocity = 0.0

        # Angle first because the T's non-zero center-of-gravity offset means
        # setting it second would shift the restored world-space position.
        obj.angle = object_pose[2]
        obj.position = object_pose[:2]
        obj.velocity = (0.0, 0.0)
        obj.angular_velocity = 0.0
        env._space.reindex_shapes_for_body(pusher)
        env._space.reindex_shapes_for_body(obj)

    def episode_init(self) -> dict:
        return {name: getattr(self, name) for name in self.init_fields}


class USocketAgent(Agent):
    """U-shaped socket that can LATCH onto the object and rotate it.

    3-DOF: (x, y, angle). Owns the latch constraints and the penetration guards
    that keep a latched pair from tunnelling through the object or the arena.
    """

    action_dim = 3
    controls_angle = True
    init_fields = ("solid_pusher", "socket_inside_friction_only")

    def __init__(self, shape: str = "u_socket", *, solid_pusher: bool = True,
                 socket_inside_friction_only: bool = False,
                 solid_contact_guard: bool = True, control_gap: "ControlGap | str | None" = None):
        super().__init__(
            shape,
            solid_pusher=solid_pusher,
            solid_contact_guard=solid_contact_guard,
            control_gap=control_gap,
        )
        self.socket_inside_friction_only = bool(socket_inside_friction_only)
        self._socket_constraints = None
        self._socket_relatch_block = 0
        self._socket_latch_angle_offset = None
        self._socket_latch_local_object_pos = None

    @property
    def socket_latched(self) -> bool:
        """Latched iff the pivot/gear constraints exist -- derived, never stored,
        so it cannot drift out of sync with the constraints themselves."""
        return self._socket_constraints is not None

    def _target_pose(self, action):
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._socket_constraints = None
        self._socket_relatch_block = 0
        self._socket_latch_angle_offset = None
        self._socket_latch_local_object_pos = None

    def pre_substep(self, env):
        return (
            self._capture_solid_contact_guard_pose(env),
            self._capture_solid_unlatched_edge_pose(env),
            self._capture_latched_pair_pose(env)
            if self.solid_pusher and self.socket_latched
            else None,
        )

    def post_substep(self, env, captured) -> None:
        solid_contact_pose, unlatched_edge_pose, latched_pose = captured
        self._maybe_latch_socket(env)
        self._enforce_solid_socket_latch(env)
        self._guard_socket_penetration(env, latched_pose)
        self._guard_solid_unlatched_object_at_arena_edge(env, unlatched_edge_pose)
        self._guard_solid_contact_penetration(env, solid_contact_pose)

    def _socket_contact_is_on_inner_face(
        self,
        env,
        pusher_shape: pymunk.Shape,
        pusher_local: pymunk.Vec2d,
        object_local: pymunk.Vec2d,
    ) -> bool:
        """Whether a contact lies on the U pocket's sticky bottom face."""
        px, py = float(pusher_local.x), float(pusher_local.y)
        ox, oy = float(object_local.x), float(object_local.y)

        object_is_in_pocket = (
            U_SOCKET_POCKET_X_MIN - _SOCKET_POCKET_PAD
            <= ox
            <= U_SOCKET_POCKET_X_MAX - _SOCKET_POCKET_MOUTH_INSET
            and abs(oy) <= U_SOCKET_POCKET_Y_HALF + _SOCKET_POCKET_PAD
        )
        if not object_is_in_pocket:
            return False

        # Shape order is fixed by make_pusher(): two prongs, then crossbar.
        # The prong side walls are deliberately frictionless, including their
        # inward-facing sides. Only the crossbar face at the closed bottom of
        # the pocket can grip the T.
        crossbar = env._pusher_shapes[-1]
        return (
            pusher_shape is crossbar
            and abs(px - U_SOCKET_POCKET_X_MIN) <= _SOCKET_INNER_FACE_TOL
            and abs(py)
            <= U_SOCKET_POCKET_Y_HALF - _SOCKET_INNER_CORNER_INSET
        )


    def _socket_friction_pre_solve(self, env,  arbiter, space, data) -> None:
        """Enable friction only on the U pocket's inward bottom surface.

        Every V3 socket shape starts frictionless. This callback restores the
        V2 combined friction only for the crossbar's unambiguous inward face.
        The two pocket side walls stay frictionless. Geometry and normal push
        forces are unchanged.
        """
        pusher = env._pusher_body
        pusher_first = arbiter.shapes[0].body is pusher
        pusher_shape = arbiter.shapes[0] if pusher_first else arbiter.shapes[1]
        arbiter.friction = 0.0

        for point in arbiter.contact_point_set.points:
            pusher_world = point.point_a if pusher_first else point.point_b
            object_world = point.point_b if pusher_first else point.point_a
            pusher_local = pusher.world_to_local(pusher_world)
            object_local = pusher.world_to_local(object_world)
            if self._socket_contact_is_on_inner_face(
                env, pusher_shape, pusher_local, object_local
            ):
                # Pymunk combines two 0.6 shape frictions as 0.36. The V3
                # socket shapes start at zero, so restore that value here.
                arbiter.friction = 0.36
                return


    def _maybe_latch_socket(self, env) -> None:
        """Rigidly attach any T face touching the socket's inner crossbar.

        Contact on the crossbar's outer/back face is deliberately ignored.
        A pivot plus a 1:1 gear joint acts as a planar weld while still letting
        the dynamic T collide with walls and obstacles.
        """
        if (
            self.socket_latched
            or env.pusher_shape != "u_socket"
            or env.object_shape != "T"
        ):
            return
        if self._socket_relatch_block > 0:
            self._socket_relatch_block -= 1
            return

        pusher = env._pusher_body
        obj = env._object_body
        crossbar = env._pusher_shapes[-1]
        latch_points: list[pymunk.Vec2d] = []

        c, s = math.cos(float(pusher.angle)), math.sin(float(pusher.angle))
        for query in env._space.shape_query(crossbar):
            if query.shape.body is not obj:
                continue
            normal = query.contact_point_set.normal
            normal_local_x = c * float(normal.x) + s * float(normal.y)
            if normal_local_x <= 0.5:
                continue
            for point in query.contact_point_set.points:
                contact_local = pusher.world_to_local(point.point_a)
                if (
                    abs(float(contact_local.x) - U_SOCKET_CROSSBAR_INNER_X)
                    <= _SOCKET_LATCH_FACE_TOL
                    and abs(float(contact_local.y))
                    <= U_SOCKET_POCKET_Y_HALF - _SOCKET_INNER_CORNER_INSET
                ):
                    latch_points.append(point.point_a)

        if not latch_points:
            return

        contact_world = sum(latch_points, pymunk.Vec2d(0.0, 0.0)) / len(latch_points)
        pivot = pymunk.PivotJoint(pusher, obj, tuple(contact_world))
        gear = pymunk.GearJoint(
            pusher,
            obj,
            phase=float(obj.angle) - float(pusher.angle),
            ratio=1.0,
        )
        # Keep the socket and the T colliding while welded. With collisions
        # off, any slip in the weld let the T sink bodily into the prongs
        # (measured up to 306 units^2 of overlap on real episodes) because
        # nothing was left to push it back out.
        pivot.collide_bodies = True
        gear.collide_bodies = True
        # Correct positional drift in the weld immediately instead of bleeding
        # it off over ~a second, which is what allowed the slip to accumulate.
        pivot.error_bias = 0.0
        gear.error_bias = 0.0
        env._space.add(pivot, gear)
        self._socket_constraints = (pivot, gear)
        local_object_pos = pusher.world_to_local(obj.position)
        self._socket_latch_local_object_pos = (
            float(local_object_pos.x),
            float(local_object_pos.y),
        )
        self._socket_latch_angle_offset = float(obj.angle) - float(pusher.angle)


    def _enforce_solid_socket_latch(self, env) -> None:
        """Keep a solid-physics latch at its exact captured relative pose.

        Pymunk constraints alone can stretch when a kinematic body drives a
        welded dynamic body against an immovable wall. For new solid-pusher
        collection, make the intended rigid attachment explicit after every
        substep. The penetration guard can then stop the complete rigid pair.
        """
        if (
            not self.solid_pusher
            or not self.socket_latched
            or self._socket_latch_local_object_pos is None
            or self._socket_latch_angle_offset is None
        ):
            return

        pusher = env._pusher_body
        obj = env._object_body
        object_position = pusher.local_to_world(self._socket_latch_local_object_pos)
        obj.angle = float(pusher.angle) + self._socket_latch_angle_offset
        obj.position = object_position

        # Match the instantaneous rigid-body velocity at the object's center.
        offset = object_position - pusher.position
        omega = float(pusher.angular_velocity)
        obj.velocity = (
            float(pusher.velocity.x) - omega * float(offset.y),
            float(pusher.velocity.y) + omega * float(offset.x),
        )
        obj.angular_velocity = omega
        env._space.reindex_shapes_for_body(obj)


    def _capture_solid_unlatched_edge_pose(
        self,
        env,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        float,
    ] | None:
        if not self.solid_pusher or self.socket_latched:
            return None
        pusher = env._pusher_body
        obj = env._object_body
        overflow, _clearance = env._object_arena_metrics()
        return (
            (float(pusher.position.x), float(pusher.position.y), float(pusher.angle)),
            (float(obj.position.x), float(obj.position.y), float(obj.angle)),
            overflow,
            env._pusher_object_penetration_depth(),
        )


    def _guard_solid_unlatched_object_at_arena_edge(
        self,
        env,
        previous_pose: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
        ]
        | None,
    ) -> None:
        """Roll back unsafe pusher/object motion at the arena boundary.

        The solid socket is kinematic, so it can otherwise bulldoze an
        unlatched T through a wall or keep advancing inside a T whose position
        was merely projected back in bounds.  Restore both bodies to their
        last valid substep when the silhouette leaves the arena, or when
        pusher/object penetration grows beyond normal resting contact while
        the T is at an edge. Free-space, insertion, and legacy motion remain
        untouched.
        """
        if previous_pose is None:
            return

        pusher_pose, object_pose, previous_overflow, previous_depth = previous_pose
        overflow, clearance = env._object_arena_metrics()
        depth = env._pusher_object_penetration_depth()
        allowed_depth = max(_UNLATCHED_EDGE_MAX_DEPTH, previous_depth)
        escaped_farther = overflow > previous_overflow + _LATCH_DEPTH_EPSILON
        excessive_overlap = (
            clearance <= _UNLATCHED_EDGE_GUARD_DISTANCE
            and depth > allowed_depth + _LATCH_DEPTH_EPSILON
        )
        if not escaped_farther and not excessive_overlap:
            return

        # A latch created during this substep may have captured an already
        # invalid, deeply overlapping pose. Drop it before restoring the last
        # unlatched valid configuration.
        if self.socket_latched:
            self._release_socket_latch(env)
        pusher = env._pusher_body
        obj = env._object_body
        pusher.position = pusher_pose[:2]
        pusher.angle = pusher_pose[2]
        pusher.velocity = (0.0, 0.0)
        pusher.angular_velocity = 0.0

        # Angle first because the T has a non-zero center-of-gravity offset.
        obj.angle = object_pose[2]
        obj.position = object_pose[:2]
        obj.velocity = (0.0, 0.0)
        obj.angular_velocity = 0.0
        env._space.reindex_shapes_for_body(pusher)
        env._space.reindex_shapes_for_body(obj)


    def _capture_latched_pair_pose(
        self,
        env,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
    ]:
        """Capture both poses and their static penetration before a substep."""
        pusher = env._pusher_body
        obj = env._object_body
        return (
            (float(pusher.position.x), float(pusher.position.y), float(pusher.angle)),
            (float(obj.position.x), float(obj.position.y), float(obj.angle)),
            env._object_static_penetration_depth(),
        )


    def _capture_solid_contact_guard_pose(
        self,
        env,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
        float,
        bool,
    ] | None:
        """Capture the last safe pair pose for solid-contact guarding."""
        if not self.solid_pusher or not self.solid_contact_guard:
            return None
        pusher = env._pusher_body
        obj = env._object_body
        return (
            (float(pusher.position.x), float(pusher.position.y), float(pusher.angle)),
            (float(obj.position.x), float(obj.position.y), float(obj.angle)),
            env._object_static_penetration_depth(),
            env._pusher_object_penetration_depth(),
            self.socket_latched,
        )


    def _guard_solid_contact_penetration(
        self,
        env,
        previous_pose: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            bool,
        ] | None,
    ) -> None:
        """Stop unsafe solid contact while preserving normal pushing.

        Pymunk normally resolves dynamic-object/static contacts, but the
        kinematic pusher has effectively infinite mass and can create a deep
        pusher/object or object/static overlap before the solver separates the
        bodies. Roll back only when penetration grows beyond normal contact.
        A pose that begins slightly embedded may still move outward, avoiding
        trapping manually restored states at an obstacle.
        """
        if previous_pose is None:
            return
        (
            pusher_pose,
            object_pose,
            previous_static_depth,
            previous_pusher_depth,
            was_latched,
        ) = previous_pose
        current_static_depth = env._object_static_penetration_depth()
        current_pusher_depth = env._pusher_object_penetration_depth()
        static_is_unsafe = current_static_depth > max(
            _SOLID_OBJECT_STATIC_MAX_DEPTH,
            previous_static_depth,
        ) + _LATCH_DEPTH_EPSILON
        # A latched U-socket intentionally contains the T and can report deep
        # polygon overlap. Only guard direct pusher/object tunnelling while the
        # pair remained unlatched for the whole substep.
        pusher_is_unsafe = (
            not was_latched
            and not self.socket_latched
            and current_pusher_depth
            > max(_SOLID_PUSHER_OBJECT_MAX_DEPTH, previous_pusher_depth)
            + _LATCH_DEPTH_EPSILON
        )
        if not static_is_unsafe and not pusher_is_unsafe:
            return

        # A latch created during the unsafe substep captured an invalid pose.
        # Drop only that new latch; a pre-existing latch remains valid at its
        # restored rigid-pair pose.
        if self.socket_latched and not was_latched:
            self._release_socket_latch(env)
            self._socket_relatch_block = _SOCKET_RELATCH_BLOCK

        self._restore_pair_pose(env, pusher_pose, object_pose)


    def _set_solid_latched_pair_pose(
        self,
        env,
        position: tuple[float, float],
        angle: float,
    ) -> None:
        """Set the socket pose and restore the captured rigid attachment."""
        env._pusher_body.position = position
        env._pusher_body.angle = angle
        env._space.reindex_shapes_for_body(env._pusher_body)
        self._enforce_solid_socket_latch(env)


    def _guard_socket_penetration(
        self,
        env,
        previous_pose: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            float,
        ]
        | None,
    ) -> None:
        """Keep a welded socket/object pair out of static geometry.

        A kinematic pusher has effectively infinite mass, so pymunk contact
        impulses cannot stop its weld from dragging the dynamic T through a
        wall. If a substep creates meaningful static penetration, restore both
        bodies to their last valid poses and stop their motion. The weld stays
        intact: pushing harder cannot tear the T out of the socket.
        """
        if self._socket_constraints is None:
            return

        penetration_depths: list[float] = []
        if previous_pose is None:
            # Legacy replay uses live arbiters to preserve the old breakaway
            # path exactly.
            def _measure(arbiter: pymunk.Arbiter, depths: list[float]) -> None:
                if not any(
                    shape.body.body_type == pymunk.Body.STATIC
                    for shape in arbiter.shapes
                ):
                    return
                for point in arbiter.contact_point_set.points:
                    depths.append(max(0.0, -float(point.distance)))

            env._object_body.each_arbiter(_measure, penetration_depths)
        else:
            # Solid-physics object poses are explicitly locked after the space
            # step, so query their current geometry rather than stale arbiters.
            # object-only (2026-08-05): freeze ONLY when the *inserted T* is
            # driven into static geometry. The socket's own body touching an
            # obstacle from OUTSIDE is handled softly by _clamp_pusher_to_static
            # (slide out), so a socketed pair no longer sticks against a wall
            # unless the T itself would tunnel through it.
            penetration_depths.append(env._object_static_penetration_depth())
        if not penetration_depths:
            return
        max_depth = max(penetration_depths)

        if previous_pose is None:
            if not self.solid_pusher and max_depth > _LEGACY_SOCKET_UNLATCH_DEPTH:
                self._release_socket_latch(env)
                self._socket_relatch_block = _SOCKET_RELATCH_BLOCK
            return
        pusher_pose, object_pose, previous_depth = previous_pose
        allowed_depth = max(_LATCH_STATIC_MAX_DEPTH, previous_depth)
        if max_depth <= allowed_depth + _LATCH_DEPTH_EPSILON:
            # Crucially, a pair that begins slightly embedded may move outward.
            # Only increasing penetration is blocked; otherwise it can become
            # permanently trapped at the wall by its own safety guard.
            return

        candidate_position = (
            float(env._pusher_body.position.x),
            float(env._pusher_body.position.y),
        )
        candidate_angle = float(env._pusher_body.angle)
        candidate_velocity = (
            float(env._pusher_body.velocity.x),
            float(env._pusher_body.velocity.y),
        )
        candidate_angular_velocity = float(env._pusher_body.angular_velocity)

        # If only rotation presses into the wall, preserve the safe outward
        # translation and temporarily block angular motion.
        env._pusher_body.velocity = candidate_velocity
        env._pusher_body.angular_velocity = 0.0
        self._set_solid_latched_pair_pose(env, candidate_position, pusher_pose[2])
        if (
            env._object_static_penetration_depth()
            <= allowed_depth + _LATCH_DEPTH_EPSILON
        ):
            return

        # Conversely, allow a safe rotation in place when translation is the
        # component trying to move deeper into static geometry.
        env._pusher_body.velocity = (0.0, 0.0)
        env._pusher_body.angular_velocity = candidate_angular_velocity
        self._set_solid_latched_pair_pose(env, pusher_pose[:2], candidate_angle)
        if (
            env._object_static_penetration_depth()
            <= allowed_depth + _LATCH_DEPTH_EPSILON
        ):
            return

        env._pusher_body.position = pusher_pose[:2]
        env._pusher_body.angle = pusher_pose[2]
        env._pusher_body.velocity = (0.0, 0.0)
        env._pusher_body.angular_velocity = 0.0

        # Angle first for the T: its non-zero center of gravity means changing
        # angle after position would shift the body's world-space position.
        env._object_body.angle = object_pose[2]
        env._object_body.position = object_pose[:2]
        env._object_body.velocity = (0.0, 0.0)
        env._object_body.angular_velocity = 0.0

        env._space.reindex_shapes_for_body(env._pusher_body)
        env._space.reindex_shapes_for_body(env._object_body)


    def _release_socket_latch(self, env) -> None:
        """Remove the current socket weld, if any."""
        if self._socket_constraints is None:
            return
        if env._space is not None:
            env._space.remove(*self._socket_constraints)
        self._socket_constraints = None
        self._socket_latch_local_object_pos = None
        self._socket_latch_angle_offset = None


# =====================================================================
# Behaviourally-distinct agents
# =====================================================================
#
# These exist to make the DATA diverse, not the geometry. A policy that
# solves `circle` transfers almost unchanged to `stick` or `L` -- all three
# are 2-DOF position-controlled pushers and differ only in contact patch.
# Each agent below breaks a different assumption, so a policy has to change
# strategy rather than just re-fit contact offsets:
#
#   gripper    can CARRY      -- transport decouples from contact geometry
#   suction    attaches, free spin -- position is controllable, angle is not
#   two_point  two contacts   -- caging/pinching, no attachment
#   tether     can only PULL  -- must reposition to change force direction
#   magnet     no contact     -- force at a distance, nothing to push against
#   compliant  force-limited  -- cannot overpower; must use momentum
#   tapper     impulsive      -- ballistic object motion, open-loop between hits
#
# All of them use the same three hooks as the socket, so env.step is
# untouched by any of this.

_SUCTION_GRIP_RADIUS = 26.0      # how close the pad must be to grab
_SUCTION_RELEASE_BLOCK = 10      # substeps before re-grip after a release
_TETHER_MAX_LEN = 150.0          # rope length; beyond this it goes taut
_TETHER_GRAB_RADIUS = 30.0
# Expressed as an ACCELERATION (units/s^2 at r = _MAGNET_MIN_R), not a force:
# the object's mass is 1890, so a force-valued constant silently becomes a
# no-op if the density ever changes. The agent multiplies by mass at use.
_MAGNET_ACCEL = 4.0e5
_MAGNET_MIN_R = 28.0             # clamp so the force cannot blow up at r->0
_MAGNET_MAX_R = 260.0            # beyond this the magnet does nothing
# Measured: a normal circle push runs at penetration mean 0.08 / max 0.31.
# The threshold has to sit INSIDE that band or the agent never yields at all
# (2.0 fired on 0/40 steps and made this a no-op).
# Fraction of PUSHER_SPEED the compliant agent may advance while in contact.
_COMPLIANT_LOADED_SPEED_FRAC = 0.25
# Expressed as a DELTA-V (units/s) imparted at strike=1, for the same reason.
# The old force-valued 900 gave dv = 900/1890 = 0.48, i.e. nothing.
_TAPPER_DELTA_V = 210.0
_TAPPER_REACH = 34.0             # must be this close to connect
_TAPPER_COOLDOWN = 6             # substeps between strikes

# env.DAMPING is 0, i.e. pymunk kills ALL velocity every step: the sim is
# quasi-static and the object moves only while something kinematically pushes
# it. Verified directly -- a 900-unit impulse under damping=0 displaces the
# body by 0.002 and leaves v=0. Force- and impulse-driven agents are therefore
# impossible in that world, so they set their own damping in on_reset (the env
# assigns space.damping BEFORE calling on_reset, so this is a supported
# override, and it is recorded in init_fields so replays reproduce it).
_MAGNET_DAMPING = 0.15
_TAPPER_DAMPING = 0.35


def _object_local_point(env, world_xy):
    """World XY -> the object body's local frame."""
    return env._object_body.world_to_local(pymunk.Vec2d(*world_xy))


class GripperAgent(Agent):
    """Parallel-jaw gripper: 4-DOF (x, y, angle, jaw).

    The only agent that can CARRY. Closing the jaws on the object pins it, so
    transport stops depending on maintaining a push direction -- the policy can
    grasp, move anywhere, rotate, and release. That is a different task
    structure, not a different contact patch.

    jaw in [0, 1]: 0 = closed, 1 = open. The jaws are separate kinematic
    bodies parented to the palm, because env drives exactly one body.
    """

    action_dim = 4
    controls_angle = True
    init_fields = ("grip_force_scale",)

    def __init__(self, shape: str = "gripper", *, grip_force_scale: float = 1.0,
                 solid_pusher: bool = True, solid_contact_guard: bool = True, control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.grip_force_scale = float(grip_force_scale)
        self._jaw_bodies: list[pymunk.Body] = []
        self._jaw_cmd = 1.0
        self._grasp = None

    def build(self, space, position):
        body, shapes = super().build(space, position)
        self._jaw_bodies = []
        for sign in (-1.0, 1.0):
            jaw = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            jaw.position = (position[0] + sign * GRIPPER_JAW_MAX_GAP / 2, position[1])
            poly = pymunk.Poly(
                jaw,
                _rect_verts(0.0, 0.0, 2 * GRIPPER_JAW_HALF_W, 2 * GRIPPER_JAW_HALF_H),
            )
            poly.friction = OBJECT_FRICTION
            space.add(jaw, poly)
            self._jaw_bodies.append(jaw)
        return body, shapes

    def _target_pose(self, action):
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        self._jaw_cmd = min(1.0, max(0.0, float(action[3])))
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._jaw_cmd = 1.0
        self._release(env)

    def _gap(self, env=None) -> float:
        """Commanded gap, floored by the object's width along the jaw axis.

        Without the floor the jaws close to GRIPPER_JAW_MIN_GAP (8 units)
        straight THROUGH an object far wider than that, so the solid-contact
        guard spends every substep fighting the penetration and the grasp
        never transports anything. Real jaws stop where they touch.
        """
        gap = GRIPPER_JAW_MIN_GAP + self._jaw_cmd * (
            GRIPPER_JAW_MAX_GAP - GRIPPER_JAW_MIN_GAP
        )
        if env is not None and self._grasp is not None:
            gap = max(gap, 2.0 * self._object_half_width(env) + 1.0)
        return min(gap, GRIPPER_JAW_MAX_GAP)

    def _object_half_width(self, env) -> float:
        """Object half-extent projected on the palm's jaw axis."""
        palm = env._pusher_body
        ca, sa = math.cos(-palm.angle), math.sin(-palm.angle)
        best = 0.0
        for shape in env._object_shapes:
            for v in shape.get_vertices():
                w = v.rotated(env._object_body.angle) + env._object_body.position
                rel = w - palm.position
                lx = rel.x * ca - rel.y * sa
                best = max(best, abs(lx))
        return best

    def _sync_jaws(self, env) -> None:
        """Keep the jaws rigidly parented to the palm at the commanded gap."""
        palm = env._pusher_body
        half = self._gap(env) / 2.0
        ca, sa = math.cos(palm.angle), math.sin(palm.angle)
        for sign, jaw in zip((-1.0, 1.0), self._jaw_bodies):
            ox = sign * half
            jaw.position = (
                palm.position.x + ox * ca,
                palm.position.y + ox * sa,
            )
            jaw.angle = palm.angle
            jaw.velocity = palm.velocity
            jaw.angular_velocity = palm.angular_velocity

    def _jaws_span_object(self, env) -> bool:
        """True when the object centre lies between the jaws and close to the
        palm axis -- the cheap stand-in for 'the pinch would actually hold'."""
        palm = env._pusher_body
        rel = env._object_body.position - palm.position
        ca, sa = math.cos(-palm.angle), math.sin(-palm.angle)
        lx = rel.x * ca - rel.y * sa
        ly = rel.x * sa + rel.y * ca
        # Capture width is the OPEN span, not the current gap: the object is
        # grasped if it lies between the jaws as they close. Using the closed
        # gap (4 units) demanded the object centre sit on the palm centre, so
        # a grasp essentially never fired.
        return (
            abs(lx) <= GRIPPER_JAW_MAX_GAP / 2.0
            and abs(ly) <= GRIPPER_JAW_HALF_H
        )

    def post_substep(self, env, captured) -> None:
        """Skip the solid-contact guard while grasped.

        The guard restores the last safe pose when pusher-object penetration
        grows, which is the right call for a pusher tunnelling through the
        object. A GRASPED object is legitimately interpenetrating the jaws and
        travelling with them, so the guard reads normal transport as tunnelling
        and pins the palm: measured palm dx = 0.00 with the guard on versus
        -94.00 with it off, i.e. it disabled carrying entirely.
        """
        if self._grasp is not None:
            return
        super().post_substep(env, captured)

    def pre_substep(self, env):
        self._sync_jaws(env)
        closing = self._jaw_cmd <= 0.35
        if closing and self._grasp is None and self._jaws_span_object(env):
            self._attach(env)
        elif not closing and self._grasp is not None:
            self._release(env)
        return super().pre_substep(env)

    def _attach(self, env) -> None:
        palm, obj = env._pusher_body, env._object_body
        pivot = pymunk.PivotJoint(palm, obj, obj.position)
        gear = pymunk.GearJoint(palm, obj, obj.angle - palm.angle, 1.0)
        for c in (pivot, gear):
            c.max_force = 8.0e7 * self.grip_force_scale
            env._space.add(c)
        self._grasp = (pivot, gear)

    def _release(self, env) -> None:
        if self._grasp is None:
            return
        for c in self._grasp:
            if c in env._space.constraints:
                env._space.remove(c)
        self._grasp = None

    @property
    def grasped(self) -> bool:
        return self._grasp is not None


class SuctionAgent(Agent):
    """Suction pad: 3-DOF (x, y, engage).

    Attaches anywhere on the object surface with a PIVOT ONLY -- no gear
    joint, so the object is free to SPIN about the contact point. Position
    becomes directly controllable while orientation does not: the policy has
    to induce rotation by dragging along arcs, or by letting contact with a
    wall spin the object. That is the opposite trade from the u_socket, which
    fixes angle rigidly.

    engage > 0.5 grabs when within reach; dropping below releases.
    """

    action_dim = 3
    init_fields = ("suction_max_force",)

    def __init__(self, shape: str = "suction", *, suction_max_force: float = 4.0e7,
                 solid_pusher: bool = True, solid_contact_guard: bool = True, control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.suction_max_force = float(suction_max_force)
        self._joint = None
        self._engage = 0.0
        self._block = 0

    def _target_pose(self, action):
        self._engage = float(action[2])
        return float(action[0]), float(action[1]), None

    def on_reset(self, env) -> None:
        self._joint = None
        self._engage = 0.0
        self._block = 0

    @property
    def attached(self) -> bool:
        return self._joint is not None

    def pre_substep(self, env):
        if self._block > 0:
            self._block -= 1
        want = self._engage > 0.5
        if want and self._joint is None and self._block == 0:
            pad = env._pusher_body.position
            if (pad - env._object_body.position).length <= _SUCTION_GRIP_RADIUS:
                j = pymunk.PivotJoint(env._pusher_body, env._object_body, pad)
                j.max_force = self.suction_max_force
                env._space.add(j)
                self._joint = j
        elif not want and self._joint is not None:
            if self._joint in env._space.constraints:
                env._space.remove(self._joint)
            self._joint = None
            self._block = _SUCTION_RELEASE_BLOCK
        return super().pre_substep(env)


class TwoPointAgent(Agent):
    """Two independent point contacts: 4-DOF (x1, y1, x2, y2).

    No attachment of any kind. Everything is achieved by CAGING -- closing two
    contacts around the object so it cannot escape, then translating both. It
    can rotate the object by moving the two points differentially, which no
    single-contact pusher can do without relying on friction moments.

    env drives point 1; the agent drives point 2 itself in pre_substep.
    """

    action_dim = 4

    def __init__(self, shape: str = "two_point", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True, control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._second: pymunk.Body | None = None
        self._t2 = (0.0, 0.0)

    def build(self, space, position):
        body, shapes = super().build(space, position)
        second = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        second.position = (position[0] + 3 * TWO_POINT_RADIUS, position[1])
        s = pymunk.Circle(second, TWO_POINT_RADIUS)
        s.friction = OBJECT_FRICTION
        space.add(second, s)
        self._second = second
        self._t2 = (float(second.position.x), float(second.position.y))
        return body, shapes

    def _target_pose(self, action):
        self._t2 = (float(action[2]), float(action[3]))
        return float(action[0]), float(action[1]), None

    def on_reset(self, env) -> None:
        if self._second is not None:
            self._t2 = (float(self._second.position.x), float(self._second.position.y))

    def pre_substep(self, env):
        # Drive the second contact with the same speed cap the env applies to
        # the first, so neither point can outrun the other.
        if self._second is not None:
            pos = self._second.position
            dx, dy = self._t2[0] - pos.x, self._t2[1] - pos.y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                self._second.velocity = (0.0, 0.0)
            else:
                speed = min(env.PUSHER_SPEED, dist / (env.DT / env.SUBSTEPS))
                self._second.velocity = (dx / dist * speed, dy / dist * speed)
        return super().pre_substep(env)

    def post_substep(self, env, captured) -> None:
        super().post_substep(env, captured)
        if self._second is not None:
            self._second.velocity = (0.0, 0.0)

    @property
    def second_pos(self) -> tuple[float, float]:
        if self._second is None:
            return (0.0, 0.0)
        return (float(self._second.position.x), float(self._second.position.y))


class TetherAgent(Agent):
    """Rope tether: 3-DOF (x, y, hook).

    Can only PULL. A SlideJoint with min=0 resists lengthening past the rope
    length but does nothing in compression, so pushing the tether toward the
    object just goes slack. To change the direction of force the policy must
    physically travel around the object -- there is no way to "push from the
    other side" without repositioning. Approach angle stops being a detail and
    becomes the whole plan.
    """

    action_dim = 3
    init_fields = ("tether_length",)

    def __init__(self, shape: str = "tether", *, tether_length: float = _TETHER_MAX_LEN,
                 solid_pusher: bool = True, solid_contact_guard: bool = True, control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.tether_length = float(tether_length)
        self._joint = None
        self._hook = 0.0

    def _target_pose(self, action):
        self._hook = float(action[2])
        return float(action[0]), float(action[1]), None

    def on_reset(self, env) -> None:
        self._joint = None
        self._hook = 0.0

    @property
    def hooked(self) -> bool:
        return self._joint is not None

    def pre_substep(self, env):
        want = self._hook > 0.5
        if want and self._joint is None:
            d = (env._pusher_body.position - env._object_body.position).length
            if d <= _TETHER_GRAB_RADIUS:
                j = pymunk.SlideJoint(
                    env._pusher_body, env._object_body,
                    (0.0, 0.0), (0.0, 0.0),
                    0.0, self.tether_length,
                )
                j.max_force = 3.0e7
                env._space.add(j)
                self._joint = j
        elif not want and self._joint is not None:
            if self._joint in env._space.constraints:
                env._space.remove(self._joint)
            self._joint = None
        return super().pre_substep(env)


class MagnetAgent(Agent):
    """Magnetic dipole: 3-DOF (x, y, strength).

    Applies an inverse-square attractive force at a distance and NEVER needs
    to touch. With no contact there is no normal direction to push along, so
    the object is steered by where the field is placed rather than by where it
    is struck -- and because the force is always toward the magnet, "push it
    left" means "get to its left first". Force is applied at the object's
    centre of gravity, so this agent has essentially no direct torque
    authority.

    strength in [-1, 1]: positive attracts, negative repels.
    """

    action_dim = 3
    init_fields = ("magnet_gain", "space_damping")

    def __init__(self, shape: str = "magnet", *, magnet_gain: float = _MAGNET_ACCEL,
                 solid_pusher: bool = False, solid_contact_guard: bool = False, control_gap: "ControlGap | str | None" = None):
        # solid_* default False: the body is a sensor, so there is no contact
        # for the penetration guard to police and enabling it would restore
        # poses based on a penetration depth that is always zero.
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.magnet_gain = float(magnet_gain)
        self.space_damping = _MAGNET_DAMPING
        self._strength = 0.0

    def _target_pose(self, action):
        self._strength = max(-1.0, min(1.0, float(action[2])))
        return float(action[0]), float(action[1]), None

    def on_reset(self, env) -> None:
        self._strength = 0.0
        # Without this the applied force integrates to nothing. See the
        # _MAGNET_DAMPING note above.
        env._space.damping = self.space_damping

    def pre_substep(self, env):
        if abs(self._strength) > 1e-6:
            obj = env._object_body
            d = env._pusher_body.position - obj.position
            r = d.length
            if 1e-9 < r <= _MAGNET_MAX_R:
                r_eff = max(r, _MAGNET_MIN_R)
                # accel * mass = force; inverse-square falloff normalised so
                # magnet_gain IS the acceleration at r = _MAGNET_MIN_R.
                accel = self.magnet_gain * self._strength * (
                    _MAGNET_MIN_R * _MAGNET_MIN_R
                ) / (r_eff * r_eff)
                mag = accel * obj.mass
                obj.apply_force_at_world_point(
                    (d.x / r * mag, d.y / r * mag), obj.position
                )
        return super().pre_substep(env)


class CompliantAgent(Agent):
    """Force-limited pusher: 2-DOF (x, y), same action space as `circle`.

    Identical commands, different physics. A kinematic pusher has infinite
    authority -- it wins every contact and can shove the object through
    anything. This one YIELDS: once it is more than a shallow depth into the
    object it backs off, so it cannot overpower a jammed or wall-pinned
    object. The policy has to build momentum and approach along directions
    that are actually free, instead of leaning on infinite force.

    This is the control pair for the whole set: it isolates the effect of
    contact AUTHORITY with geometry and action space held constant.
    """

    action_dim = 2
    init_fields = ("loaded_speed_frac",)

    def __init__(self, shape: str = "compliant", *,
                 loaded_speed_frac: float = _COMPLIANT_LOADED_SPEED_FRAC,
                 solid_pusher: bool = True, solid_contact_guard: bool = True, control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.loaded_speed_frac = float(loaded_speed_frac)
        self._pos_before = None

    def pre_substep(self, env):
        self._pos_before = tuple(env._pusher_body.position)
        return super().pre_substep(env)

    def post_substep(self, env, captured) -> None:
        super().post_substep(env, captured)
        if env._pusher_object_penetration_depth() <= 0.0:
            return  # free space: full authority, identical to a plain pusher
        # LOADED. Cap the distance advanced this substep. Nudging the position
        # back by the penetration excess (the first attempt) did nothing: the
        # pusher is position-driven and simply re-advanced next substep, so
        # the net effect was under 0.2 units over a whole episode. Capping the
        # advance is what actually limits authority.
        before = getattr(self, "_pos_before", None)
        if before is None:
            return
        now = env._pusher_body.position
        dx, dy = now.x - before[0], now.y - before[1]
        moved = math.hypot(dx, dy)
        cap = self.loaded_speed_frac * env.PUSHER_SPEED * (env.DT / env.SUBSTEPS)
        if moved > cap > 0.0:
            k = cap / moved
            env._pusher_body.position = (before[0] + dx * k, before[1] + dy * k)


class TapperAgent(Agent):
    """Impulsive tapper: 3-DOF (x, y, strike).

    Does not push continuously. A strike delivers a single impulse and then
    the object travels BALLISTICALLY, decelerating under friction, with no
    contact to correct it mid-flight. Control is open-loop between hits, so
    the policy has to aim and meter force in advance rather than servo
    continuously -- the one agent here whose errors cannot be fixed by
    pressing harder.

    strike in [0, 1]; fires on a rising edge past 0.5, then cools down.
    """

    action_dim = 3
    controls_angle = True
    init_fields = ("impulse_scale", "space_damping")

    def __init__(self, shape: str = "tapper", *, impulse_scale: float = 1.0,
                 solid_pusher: bool = False, solid_contact_guard: bool = False, control_gap: "ControlGap | str | None" = None):
        # solid_* default False: the body is a sensor, so there is no contact
        # for the penetration guard to police and enabling it would restore
        # poses based on a penetration depth that is always zero.
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.impulse_scale = float(impulse_scale)
        self.space_damping = _TAPPER_DAMPING
        self._strike = 0.0
        self._prev_strike = 0.0
        self._cool = 0
        self._angle = 0.0

    def _target_pose(self, action):
        self._strike = min(1.0, max(0.0, float(action[2])))
        # The bar points along its travel direction; angle is derived from the
        # strike vector rather than commanded separately, keeping this 3-DOF.
        return float(action[0]), float(action[1]), self._angle

    def on_reset(self, env) -> None:
        self._strike = self._prev_strike = 0.0
        self._cool = 0
        self._angle = 0.0
        # Ballistic travel between strikes requires momentum to survive the
        # step; env.DAMPING=0 would erase it. See _TAPPER_DAMPING above.
        env._space.damping = self.space_damping

    def pre_substep(self, env):
        if self._cool > 0:
            self._cool -= 1
        rising = self._strike > 0.5 >= self._prev_strike
        self._prev_strike = self._strike
        if rising and self._cool == 0:
            obj = env._object_body
            d = obj.position - env._pusher_body.position
            r = d.length
            if 1e-9 < r <= _TAPPER_REACH:
                self._angle = math.atan2(d.y, d.x)
                # delta-v * mass = impulse.
                mag = _TAPPER_DELTA_V * self.impulse_scale * self._strike * obj.mass
                # Applied at the contact point, not the CoG, so an off-centre
                # tap imparts spin -- that is the agent's only torque authority.
                contact = env._pusher_body.position + d / r * (r - 1.0)
                obj.apply_impulse_at_world_point(
                    (d.x / r * mag, d.y / r * mag), contact
                )
                self._cool = _TAPPER_COOLDOWN
        return super().pre_substep(env)


_SIMPLE = ("circle", "circle_small", "stick", "L")

#: shape name -> agent class, for everything that is not a plain 2-DOF pusher.
_AGENT_CLASSES: dict[str, type[Agent]] = {
    "u_socket": USocketAgent,
    "gripper": GripperAgent,
    "suction": SuctionAgent,
    "two_point": TwoPointAgent,
    "tether": TetherAgent,
    "magnet": MagnetAgent,
    "compliant": CompliantAgent,
    "tapper": TapperAgent,
}

#: Every constructible pusher. env imports this so the two lists cannot drift.
VALID_PUSHERS: tuple[str, ...] = _SIMPLE + tuple(_AGENT_CLASSES)


def make_agent(pusher_shape: str, **kwargs) -> Agent:
    """Build the agent for ``pusher_shape``.

    The environment calls this instead of branching on the shape name, so
    adding an agent means adding a class and one entry here.
    """
    if pusher_shape in _SIMPLE:
        return Agent(pusher_shape, **kwargs)
    cls = _AGENT_CLASSES.get(pusher_shape)
    if cls is not None:
        return cls(pusher_shape, **kwargs)
    raise ValueError(
        "unknown pusher_shape %r; known: %s" % (pusher_shape, list(VALID_PUSHERS))
    )
