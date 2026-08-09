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

import pymunk

from .shapes import (
    U_SOCKET_CROSSBAR_INNER_X,
    U_SOCKET_POCKET_X_MAX,
    U_SOCKET_POCKET_X_MIN,
    U_SOCKET_POCKET_Y_HALF,
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



class Agent:
    """Base 2-DOF pusher with the fixed Sim V2 solid-contact guard."""

    action_dim = 2
    #: physics knobs an agent contributes to episode_init, so a replay can
    #: reconstruct the exact contact model it was collected under.
    init_fields: tuple[str, ...] = ()

    def __init__(
        self,
        shape: str,
        *,
        solid_pusher: bool = True,
        solid_contact_guard: bool = True,
    ):
        self.shape = shape
        self.solid_pusher = bool(solid_pusher)
        self.solid_contact_guard = bool(solid_contact_guard)

    def build(self, space: pymunk.Space, position):
        """Create the pusher body/shapes in ``space``."""
        return make_pusher(self.shape, space, position)

    def target_pose(self, action):
        """(target_x, target_y, target_angle|None) from a raw action."""
        return float(action[0]), float(action[1]), None

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
    init_fields = ("solid_pusher", "socket_inside_friction_only")

    def __init__(self, shape: str = "u_socket", *, solid_pusher: bool = True,
                 socket_inside_friction_only: bool = False,
                 solid_contact_guard: bool = True):
        super().__init__(
            shape,
            solid_pusher=solid_pusher,
            solid_contact_guard=solid_contact_guard,
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

    def target_pose(self, action):
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


_SIMPLE = ("circle", "circle_small", "stick", "L")


def make_agent(pusher_shape: str, **kwargs) -> Agent:
    """Build the agent for ``pusher_shape``.

    The environment calls this instead of branching on the shape name, so
    adding an agent means adding a class and one entry here.
    """
    if pusher_shape == "u_socket":
        return USocketAgent(pusher_shape, **kwargs)
    if pusher_shape in _SIMPLE:
        return Agent(pusher_shape, **kwargs)
    raise ValueError(
        "unknown pusher_shape %r; known: %s"
        % (pusher_shape, ("u_socket",) + _SIMPLE)
    )
