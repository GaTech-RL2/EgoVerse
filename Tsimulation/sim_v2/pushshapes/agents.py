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
    CLUTCH_PIN_HALF_W,
    CLUTCH_PIN_LEN,
    CLUTCH_R,
    FLIPPER_HALF_W,
    FLIPPER_LEN,
    FLIPPER_SWING,
    GRIPPER_FINGER_LEN,
    GRIPPER_RAIL_HALF,
    GRIPPER_JAW_HALF_H,
    UMI_FINGER_HALF_H,
    UMI_FINGER_LEN,
    UMI_HINGE_SPAN,
    UMI_FINGER_HALF_W,
    UMI_MAX_GAP,
    UMI_MIN_GAP,
    UMI_HINGE_OFFSET,
    UMI_WRIST_R,
    TOWBAR_LENGTH,
    SOFT_DAMPING,
    SOFT_NODE_MASS,
    SOFT_NODE_R,
    SOFT_NODES,
    SOFT_SPAN,
    SOFT_STIFFNESS,
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


#: Per-embodiment compliance ranges, sampled per episode when control_gap is
#: "random". These are NOT uniform across agents on purpose: a dataset whose
#: only variation is the contact model teaches that every embodiment executes
#: equally well, which is the one thing no real hardware does. The ranges
#: encode a plausible character per mechanism --
#:
#:   magnet/tapper  act at a distance or in bursts, so their effective command
#:                  is the sloppiest;
#:   gripper        a heavier, geared wrist: lag and backlash, little noise;
#:   two_point      two servos, so noise dominates over any single bias;
#:   compliant      already force-limited, kept near-clean so the two axes
#:                  (authority vs fidelity) stay separable in analysis.
#:
#: Each entry is (low, high) per ControlGap term; latency is integer steps.
_DEFAULT_GAP_RANGE = {
    "latency_steps": (0, 2), "lag": (0.0, 0.45), "deadband": (0.0, 2.0),
    "gain": (0.93, 1.0), "noise_std": (0.0, 1.0),
}
AGENT_GAP_RANGES: dict[str, dict[str, tuple]] = {
    "gripper": {"latency_steps": (1, 4), "lag": (0.2, 0.6), "deadband": (1.0, 4.0),
                "gain": (0.9, 1.0), "noise_std": (0.0, 0.5)},
    "suction": {"latency_steps": (0, 2), "lag": (0.1, 0.4), "deadband": (0.0, 1.5),
                "gain": (0.95, 1.0), "noise_std": (0.0, 0.8)},
    "triangle": {"latency_steps": (0, 3), "lag": (0.1, 0.45), "deadband": (0.0, 2.0),
                 "gain": (0.93, 1.0), "noise_std": (0.0, 1.0)},
    "scoop": {"latency_steps": (0, 3), "lag": (0.15, 0.5), "deadband": (0.0, 2.5),
              "gain": (0.92, 1.0), "noise_std": (0.0, 1.2)},
    "two_point": {"latency_steps": (0, 3), "lag": (0.1, 0.5), "deadband": (0.0, 2.0),
                  "gain": (0.93, 1.0), "noise_std": (0.5, 2.5)},
    "tether": {"latency_steps": (1, 4), "lag": (0.2, 0.6), "deadband": (0.0, 3.0),
               "gain": (0.9, 1.0), "noise_std": (0.0, 1.2)},
}


def sample_control_gap(rng, shape: str | None = None) -> ControlGap:
    """Draw a ControlGap for one episode from ``shape``'s range.

    ``rng`` must be the agent's per-episode generator (seeded from the env's),
    so the draw is reproducible from the reset seed alone -- otherwise a
    recorded episode could never be replayed under the compliance it was
    collected with.
    """
    r = dict(_DEFAULT_GAP_RANGE)
    r.update(AGENT_GAP_RANGES.get(shape or "", {}))
    return ControlGap(
        latency_steps=int(rng.integers(r["latency_steps"][0],
                                       r["latency_steps"][1] + 1)),
        lag=float(rng.uniform(*r["lag"])),
        deadband=float(rng.uniform(*r["deadband"])),
        gain=float(rng.uniform(*r["gain"])),
        noise_std=float(rng.uniform(*r["noise_std"])),
    )

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

    #: CANONICAL LAYOUT: ("x", "y", "angle") or ("x", "y", "angle", "grip").
    #: GRIP SEMANTICS, one meaning for every agent: 1.0 = holding on (jaws
    #: closed, suction on, hitch made), 0.0 = released. The grippers used to
    #: read 0 as "closed" while suction read 1 as "on", so a single UI mapping
    #: engaged half the roster and released the other half.
    #: Slot 2 is ALWAYS orientation and slot 3 is ALWAYS the grip/engage
    #: scalar, for every embodiment. Previously each agent chose its own
    #: order -- suction read slot 2 as engage while everything else read it as
    #: angle -- which meant a policy or UI could not address them uniformly
    #: and mis-wiring was silent.
    #:
    #: Named channels, one per action slot. The UI and any scripted policy
    #: build actions from THIS, never from action_dim -- encoding by dimension
    #: silently mis-wired three agents at once: suction read slot 2 as
    #: `engage` while the UI sent an angle there (so it never suctioned), and
    #: wrench and scoop read slot 2 as `angle` while the UI sent 0/1 engage
    #: (so their orientation was pinned and they were 3-DOF in name only).
    action_spec: tuple[str, ...] = ("x", "y")

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
        self.randomize_gap = control_gap == "random"
        if isinstance(control_gap, str) and not self.randomize_gap:
            if control_gap not in CONTROL_GAPS:
                raise ValueError(
                    f"unknown control_gap {control_gap!r}; "
                    f"known: {sorted(CONTROL_GAPS)} or 'random'"
                )
            control_gap = CONTROL_GAPS[control_gap]
        if self.randomize_gap:
            control_gap = None  # drawn per episode in reset_control_gap
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
        if self.randomize_gap:
            # Drawn from the SAME episode RNG, so the reset seed alone
            # determines both the compliance and the noise realised under it.
            self.control_gap = sample_control_gap(self._rng, self.shape)

    def on_reset(self, env) -> None:
        """Per-episode state reset. No-op for a stateless agent."""

    @property
    def dim(self) -> int:
        return len(self.action_spec)

    def active_constraints(self) -> tuple:
        """Constraints this agent currently holds against the object.

        The solid-contact guard MUST be skipped while any of these exist --
        see post_substep.
        """
        return ()

    def pre_substep(self, env):
        """Capture whatever post_substep needs to compare against."""
        if self.active_constraints():
            return None
        return self._capture_solid_contact_guard_pose(env)

    def post_substep(self, env, captured) -> None:
        """Reject a substep that tunnels through the object or static geometry.

        SKIPPED while the agent holds a constraint. The guard restores the
        pusher's pose when penetration grows, which is right for a pusher
        tunnelling through the object and catastrophically wrong for one that
        is BOUND to it: constraint-driven motion reads as tunnelling and the
        agent is pinned in place. That silently disabled three agents before
        this rule existed -- the gripper travelled 0.00 instead of -94, and the
        wrench sat at its spawn for 600 steps while reporting coupled=True.
        """
        if captured is None or self.active_constraints():
            return
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

    action_spec = ("x", "y", "angle")
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
# DESIGNED AROUND WHAT THIS ENGINE CAN ACTUALLY DO. space.damping is 0, i.e.
# pymunk erases ALL velocity every step, so the sim is quasi-static: the
# object moves only while something is actively resolving against it inside
# the solver. Measured, pusher held 90 units away so it never touches:
#
#   PivotJoint, pusher recedes ....... object follows 118.1   <- works
#   SlideJoint, pusher recedes ....... object follows 118.3   <- works
#   GearJoint only, pusher rotates ... rotates 156.7 deg, moves 8.4  <- works
#   apply_force  (accel 4e5) ......... 67.3, and only by drifting into contact
#   apply_impulse (dv 210) ........... 1.8  -- i.e. nothing
#
# So forces and impulses are the WRONG PRIMITIVE here: a magnet or a ballistic
# tapper cannot work without changing world physics, and changing space.damping
# per agent (the earlier approach) both made the object bounce and turned the
# embodiment into a confound, since it alters the OBJECT rather than the
# end effector. Every agent below is built from contact geometry plus hard
# constraints, which the solver handles exactly and without jitter.
#
# The set is chosen so each one holds an affordance no other has:
#
#   agent      translate  rotate  pull  remote  holds   the thing only it does
#   suction    yes        NO      yes   no      yes     position without orientation
#   gripper    yes        yes     yes   no      yes     full rigid control, if grasped
#   wrench     NO         yes     no    YES     no      orient at a distance
#   tether     pull only  weak    yes   no      yes     cannot push at all
#   two_point  yes        yes     no    no      no      rotate with nothing attached
#   scoop      yes        yes     no    no      loose   carry, droppable by tilting

# Measured: with the pad flush against the object the BODY CENTRE sits 15.0
# from the surface, because the pad is 26 wide. A radius tighter than that can
# never be satisfied no matter how the operator drives.
# Measured: with the stem moved behind it, the pad face rests 3.75 from the
# surface. It attaches on real contact now, not across a 21.6-unit gap.
_SUCTION_GRIP_RADIUS = 6.0
_SUCTION_RELEASE_BLOCK = 10
#: Two orders below _CONSTRAINT_FORCE: enough to reorient a settled object,
#: not enough to stop a fast drag twisting it. This is the pad slipping.
_SUCTION_TWIST_FORCE = 6.0e5
_WRENCH_RANGE = 150.0
_TETHER_LINKS = 10
_TETHER_LINK_LEN = 14.0
_TETHER_GRAB_RADIUS = 16.0    # agent body -> object SURFACE
_CONSTRAINT_FORCE = 5.0e7
#: GearJoints are LEFT UNLIMITED. _CONSTRAINT_FORCE is sized for a LINEAR
#: pivot against the object's mass (1890); the same number on an ANGULAR
#: constraint is fought by its moment of inertia (3.65e6) and does essentially
#: nothing -- measured 1.0 deg of object rotation for 172 deg of wrist
#: rotation, i.e. a "rigid" clamp transmitting no angle at all. 5e9 recovers
#: 152 deg; unlimited tracks exactly.
_GEAR_FORCE = None


def _add_gear(space, a, b, phase):
    """GearJoint with no force cap. See _GEAR_FORCE."""
    g = pymunk.GearJoint(a, b, phase, 1.0)
    if _GEAR_FORCE is not None:
        g.max_force = _GEAR_FORCE
    space.add(g)
    return g


def _surface_distance(env, point) -> float:
    """Distance from `point` to the object's SURFACE (negative inside).

    Grip conditions must use this, not the distance to the centroid. The T
    spans 120x120, so a pad resting against its surface still sits ~45 units
    from the centre -- a centroid-based radius of 30 can never be satisfied
    and the mechanism simply never fires. That bug silently disabled suction
    and the tether, and had already been fixed once for the gripper without
    being carried across.
    """
    best = float("inf")
    for sh in env._object_shapes:
        best = min(best, float(sh.point_query(point).distance))
    return best
#: Any non-zero group; pymunk skips collisions between shapes sharing one, so
#: the rope cannot snag on itself or on the pusher that is dragging it.
_ROPE_GROUP = 0x50F7


class GripperAgent(Agent):
    """Parallel-jaw gripper: 4-DOF (x, y, angle, jaw).

    LINEAR parallel jaws sliding on a back plate -- the counterpart to umi's
    revolute pincer. Because the jaws stay parallel, they contact along their
    whole face at any opening, where a pincer touches only at the tips as it
    splays. Pivot + Gear once closed, so the object becomes rigidly attached:
    full translation AND orientation authority, but only while grasped. Everything
    else here gives up one of the two.

    jaw in [0, 1]: 0 closed, 1 open.
    """

    action_spec = ("x", "y", "angle", "grip")
    action_dim = 4
    controls_angle = True

    def __init__(self, shape: str = "gripper", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._jaws: list[pymunk.Body] = []
        self._jaw_cmd = 1.0
        self._grasp = None
        self._held_gap = GRIPPER_JAW_MIN_GAP

    def build(self, space, position):
        body, shapes = super().build(space, position)
        self._jaws = []
        for sign in (-1.0, 1.0):
            jaw = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            jaw.position = (position[0] + sign * GRIPPER_JAW_MAX_GAP / 2, position[1])
            # Runs FORWARD from the hinge, same as umi. Centred jaws sliding
            # in parallel rendered as two blocks either side of the palm --
            # a bolt, not a gripper.
            poly = pymunk.Poly(jaw, _rect_verts(
                0.0, GRIPPER_FINGER_LEN / 2,
                2 * GRIPPER_JAW_HALF_W, GRIPPER_FINGER_LEN))
            poly.friction = OBJECT_FRICTION
            space.add(jaw, poly)
            self._jaws.append(jaw)
        return body, shapes

    def _target_pose(self, action):
        # grip 1 = closed, so jaw_cmd (0 closed .. 1 open) is its complement.
        self._jaw_cmd = 1.0 - min(1.0, max(0.0, float(action[3])))
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._jaw_cmd, self._grasp = 1.0, None

    @property
    def grasped(self) -> bool:
        return self._grasp is not None

    @property
    def dim(self) -> int:
        return len(self.action_spec)

    def active_constraints(self) -> tuple:
        return tuple(self._grasp) if self._grasp else ()

    def _gap(self, env) -> float:
        gap = GRIPPER_JAW_MIN_GAP + self._jaw_cmd * (
            GRIPPER_JAW_MAX_GAP - GRIPPER_JAW_MIN_GAP)
        # Floor whenever the object is in the pocket, not only once grasped.
        # Gating on _grasp was circular -- the jaws closed to their 26-unit
        # minimum against a 30-wide stem, so they never straddled it, so the
        # grasp never formed, so the floor never applied. Same bug as umi's,
        # fixed there and not carried across.
        if self._grasp is not None:
            # FROZEN at grasp time. Recomputing while held made the fingers
            # drift open and shut as the object shifted, so a holding gripper
            # visibly let go and re-gripped every few frames.
            return self._held_gap
        if self._spans(env):
            gap = max(gap, 2.0 * self._obj_half_width(env) + 1.0)
        return min(gap, GRIPPER_JAW_MAX_GAP)

    def _obj_half_width(self, env) -> float:
        """Half-width of the object material INSIDE THE JAW POCKET.

        Restricted to vertices lying within the fingers' reach. Taking the max
        over every vertex returned the whole T's half-extent -- 60.0 when the
        jaws were closing on a 15-wide limb -- so the gap floored at ~121 and
        the jaws never came near the object they were supposedly holding.
        """
        palm = env._pusher_body
        ca, sa = math.cos(-palm.angle), math.sin(-palm.angle)
        best = 0.0
        for sh in env._object_shapes:
            for v in sh.get_vertices():
                w = v.rotated(env._object_body.angle) + env._object_body.position
                rel = w - palm.position
                lx = rel.x * ca - rel.y * sa
                ly = rel.x * sa + rel.y * ca
                if -GRIPPER_FINGER_LEN * 0.25 <= ly <= GRIPPER_FINGER_LEN:
                    best = max(best, abs(lx))
        return best if best > 0.0 else GRIPPER_JAW_MIN_GAP / 2

    def _sync(self, env):
        """Slide the jaws along the back plate, always PARALLEL.

        No hinge and no splay: the gap IS the jaw separation, so contact is a
        flat face at every opening. umi is the revolute one; keeping this
        linear is what makes the two mechanically different rather than two
        drawings of the same pincer.
        """
        palm = env._pusher_body
        half = self._gap(env) / 2.0
        ca, sa = math.cos(palm.angle), math.sin(palm.angle)
        for sign, jaw in zip((-1.0, 1.0), self._jaws):
            hx = sign * half
            jaw.position = (palm.position.x + hx * ca, palm.position.y + hx * sa)
            jaw.angle = palm.angle          # parallel, always
            jaw.velocity = palm.velocity
            jaw.angular_velocity = palm.angular_velocity

    def _spans(self, env) -> bool:
        """Is there object MATERIAL between the jaws?

        Not "is the centroid between the jaws" -- that was the first version
        and it made grasping the T impossible. The T spans 120x120 but its
        limbs are only 30 wide, so a 46-unit gripper grasps a LIMB; the
        centroid sits at the junction, where approaching drives the jaws into
        the crossbar and shoves the object away (measured: it flew to x=451
        and was never grasped).
        """
        palm = env._pusher_body
        half = GRIPPER_JAW_MAX_GAP / 2 * 0.8
        # Sample the FULL finger length. Using JAW_HALF_H (20) checked a
        # pocket shorter than the fingers actually are (34), so a limb sitting
        # between them registered as absent.
        # Sample the pocket BETWEEN and IN FRONT OF the jaws. Sampling at the
        # palm itself never fires: the jaws stick out 23 units and stop the
        # palm ~21 from the surface, so nothing at the palm is ever inside the
        # object even when a limb is squarely between the jaws.
        for lx in (-half, -half / 2, 0.0, half / 2, half):
            for ly in (-GRIPPER_FINGER_LEN * 0.2, 0.0, GRIPPER_FINGER_LEN * 0.35,
                       GRIPPER_FINGER_LEN * 0.7, GRIPPER_FINGER_LEN * 0.95):
                w = palm.local_to_world((lx, ly))
                if _surface_distance(env, w) <= 0.0:
                    return True
        return False

    def pre_substep(self, env):
        self._sync(env)
        # HOLD once grasped: only a command to OPEN releases it. Re-testing
        # _spans every substep would drop the object the moment it shifted in
        # the jaws.
        closing = self._jaw_cmd <= 0.35
        # A gripper cannot hold what it cannot close around. Without this the
        # jaws "grasped" the T's 120-wide crossbar, floored fully open, and
        # sat there visibly not touching it.
        graspable = 2.0 * self._obj_half_width(env) <= GRIPPER_JAW_MAX_GAP * 0.95
        if closing and self._grasp is None and graspable and self._spans(env):
            palm, obj = env._pusher_body, env._object_body
            # Anchor WHERE THE JAWS ARE, not at the object's centre of mass.
            # Anchoring at obj.position pinned a point ~60 units away from the
            # contact, so the T hung off the gripper at that radius and swung
            # about it -- on screen the gripper was plainly not touching the
            # object it was "holding".
            pj = pymunk.PivotJoint(palm, obj, palm.position)
            # UNCAPPED, like the gear. Capping the pivot at _CONSTRAINT_FORCE
            # while the gear ran unlimited let the gear win: the object rotated
            # about its own centre and dragged the anchor off the jaws, so the
            # palm-to-surface distance grew from 6 at grasp to 45 mid-carry --
            # a grasp that visibly stopped touching the thing it held.
            env._space.add(pj)
            gj = _add_gear(env._space, palm, obj, obj.angle - palm.angle)
            self._held_gap = max(GRIPPER_JAW_MIN_GAP,
                                 2.0 * self._obj_half_width(env) + 1.0)
            self._grasp = (pj, gj)
        elif not closing and self._grasp is not None:
            for c in self._grasp:
                if c in env._space.constraints:
                    env._space.remove(c)
            self._grasp = None
        return super().pre_substep(env)




class SuctionAgent(Agent):
    """Suction pad: 3-DOF (x, y, engage).

    Pivot plus a DELIBERATELY WEAK gear: position is locked to the pad, but
    orientation is only held by pad friction, so it slips under load. You can
    reorient, slowly, and a fast drag will twist the object out of alignment.

    A bare pivot (the first version) has zero rotational stiffness: the object
    free-spun to 89.9 degrees while being dragged and IoU capped at 0.286 no
    matter how exactly position was solved. And the pad cannot reach the
    centroid to avoid the torque -- it collides with the object's surface --
    so some rotational stiffness is required for this to be an embodiment
    rather than a partial one.

    Distinct from the gripper, which holds angle rigidly and cannot slip.
    """

    action_spec = ("x", "y", "angle", "grip")
    action_dim = 4
    controls_angle = True

    def __init__(self, shape: str = "suction", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._joint = None
        self._engage = 0.0
        self._block = 0

    def _target_pose(self, action):
        # Canonical order: angle in slot 2, grip in slot 3. Angle is commanded
        # but only weakly enforced (see the gear's max_force) -- the pad twists
        # the object rather than locking it.
        self._engage = float(action[3])
        return float(action[0]), float(action[1]), float(action[2])

    def on_reset(self, env) -> None:
        self._joint, self._engage, self._block = None, 0.0, 0

    @property
    def attached(self) -> bool:
        return self._joint is not None

    @property
    def dim(self) -> int:
        return len(self.action_spec)

    def active_constraints(self) -> tuple:
        return tuple(self._joint) if self._joint else ()

    def pre_substep(self, env):
        if self._block:
            self._block -= 1
        want = self._engage > 0.5
        if want and self._joint is None and not self._block:
            pad = env._pusher_body.position
            if _surface_distance(env, pad) <= _SUCTION_GRIP_RADIUS:
                pu, obj = env._pusher_body, env._object_body
                j = pymunk.PivotJoint(pu, obj, pad)
                j.max_force = _CONSTRAINT_FORCE
                g = pymunk.GearJoint(pu, obj, obj.angle - pu.angle, 1.0)
                g.max_force = _SUCTION_TWIST_FORCE   # slips well before the pivot does
                env._space.add(j, g)
                self._joint = (j, g)
        elif not want and self._joint is not None:
            for c in self._joint:
                if c in env._space.constraints:
                    env._space.remove(c)
            self._joint, self._block = None, _SUCTION_RELEASE_BLOCK
        return super().pre_substep(env)


class WrenchAgent(Agent):
    """Remote orienter: 3-DOF (x, y, angle).

    A GearJoint and nothing else, engaged whenever the agent is within range --
    NO CONTACT REQUIRED. The object's orientation follows the agent's, while
    its position is left completely free.

    Measured in isolation: 156.7 degrees of rotation for 8.4 units of drift.
    That inverts every other embodiment here, which buy position and struggle
    for angle. Since the goal pose has an orientation, an agent that can only
    supply angle is genuinely useful and genuinely incomplete -- it has to be
    paired with something that translates, which is the point of a diverse set.

    The head is SOLID, so translation comes from ordinary pushing while
    orientation comes from the remote gear. That split -- push for position,
    reach for angle -- is its own strategy: it can fix the angle from across
    the arena before it ever arrives.

    This replaces the old force-based magnet, which could not work: applying
    force in a world that erases velocity every step moved the object 1.8-67
    units where a constraint moves it 118.
    """

    action_spec = ("x", "y", "angle")
    action_dim = 3
    controls_angle = True

    def __init__(self, shape: str = "wrench", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._joint = None
        self._engage = 0.0

    def _target_pose(self, action):
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        self._engage = 1.0
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._joint = None

    @property
    def coupled(self) -> bool:
        return self._joint is not None

    @property
    def dim(self) -> int:
        return len(self.action_spec)

    def active_constraints(self) -> tuple:
        return (self._joint,) if self._joint is not None else ()

    def in_range(self, env) -> bool:
        return (env._pusher_body.position
                - env._object_body.position).length <= _WRENCH_RANGE

    def pre_substep(self, env):
        want = self.in_range(env)
        if want and self._joint is None:
            pu, obj = env._pusher_body, env._object_body
            j = pymunk.GearJoint(pu, obj, obj.angle - pu.angle, 1.0)
            j.max_force = _CONSTRAINT_FORCE
            env._space.add(j)
            self._joint = j
        elif not want and self._joint is not None:
            if self._joint in env._space.constraints:
                env._space.remove(self._joint)
            self._joint = None
        return super().pre_substep(env)


class TetherAgent(Agent):
    """Rope tether: 3-DOF (x, y, hook).

    An ACTUAL ROPE: ten dynamic links pinned end to end, the last of which
    pins to the object when hooked. Not a distance constraint pretending to be
    one -- the chain exists in the space, drapes, and is drawn from real body
    positions.

    Pull only. Slack rope transmits nothing, so pushing the agent at the object
    does nothing at all; to change the direction of force you must physically
    travel around it. Approach angle stops being a detail and becomes the plan.
    """

    action_spec = ("x", "y", "engage")
    action_dim = 3

    def __init__(self, shape: str = "tether", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._links: list[pymunk.Body] = []
        self._link_joints: list[pymunk.Constraint] = []
        self._hook_joint = None
        self._hook = 0.0

    def build(self, space, position):
        body, shapes = super().build(space, position)
        self._links, self._link_joints, self._hook_joint = [], [], None
        prev = body
        for i in range(_TETHER_LINKS):
            lb = pymunk.Body(1.0, pymunk.moment_for_segment(
                1.0, (0, 0), (_TETHER_LINK_LEN, 0), 2.0))
            lb.position = (position[0] + (i + 1) * _TETHER_LINK_LEN, position[1])
            sh = pymunk.Segment(lb, (0, 0), (_TETHER_LINK_LEN, 0), 2.0)
            sh.friction = OBJECT_FRICTION
            # Rope must not snag on itself or on the pusher; only the object
            # and the walls should ever stop it.
            sh.filter = pymunk.ShapeFilter(group=_ROPE_GROUP)
            space.add(lb, sh)
            anchor_prev = (0.0, 0.0) if prev is body else (_TETHER_LINK_LEN, 0.0)
            j = pymunk.PivotJoint(prev, lb, anchor_prev, (0.0, 0.0))
            space.add(j)
            self._link_joints.append(j)
            self._links.append(lb)
            prev = lb
        return body, shapes

    def _target_pose(self, action):
        self._hook = float(action[2])
        return float(action[0]), float(action[1]), None

    def on_reset(self, env) -> None:
        self._links, self._link_joints, self._hook_joint = [], [], None
        self._hook = 0.0

    @property
    def hooked(self) -> bool:
        return self._hook_joint is not None

    @property
    def dim(self) -> int:
        return len(self.action_spec)

    def active_constraints(self) -> tuple:
        return (self._hook_joint,) if self._hook_joint is not None else ()

    def rope_points(self) -> list[tuple[float, float]]:
        """World polyline of the rope, for rendering."""
        pts = []
        for lb in self._links:
            a = lb.local_to_world((0.0, 0.0))
            pts.append((float(a.x), float(a.y)))
        if self._links:
            b = self._links[-1].local_to_world((_TETHER_LINK_LEN, 0.0))
            pts.append((float(b.x), float(b.y)))
        return pts

    def pre_substep(self, env):
        want = self._hook > 0.5
        if want and self._hook_joint is None and self._links:
            # Hook when the AGENT reaches the object, not when the rope TIP
            # does: the chain trails BEHIND the agent, so its tip points away
            # from whatever you walk towards -- measured 58.2 units off while
            # the agent itself was already 8.4 from the surface. Distance is
            # to the SURFACE, since the T spans 120 and a centroid radius can
            # never be met. Once pinned, the links straighten into the gap.
            tip = env._pusher_body.position
            if _surface_distance(env, tip) <= _TETHER_GRAB_RADIUS:
                # Anchor where the rope actually TOUCHES, not at the centroid.
                # An off-centre hook turns a pull into a pull plus a torque, so
                # WHERE you attach chooses how the object turns -- the tether's
                # only means of controlling angle.
                local = env._object_body.world_to_local(tip)
                j = pymunk.PivotJoint(
                    self._links[-1], env._object_body,
                    (_TETHER_LINK_LEN, 0.0), (float(local.x), float(local.y)))
                j.max_force = _CONSTRAINT_FORCE
                env._space.add(j)
                self._hook_joint = j
        elif not want and self._hook_joint is not None:
            if self._hook_joint in env._space.constraints:
                env._space.remove(self._hook_joint)
            self._hook_joint = None
        return super().pre_substep(env)


class TwoPointAgent(Agent):
    """Two independent contacts: 4-DOF (x1, y1, x2, y2).

    Nothing attaches. Everything is caging: close two contacts around the
    object and translate, or move them differentially to spin it. The only
    agent that can rotate the object with NO constraint binding it, which also
    means it can drop it at any instant.
    """

    action_spec = ("x", "y", "x2", "y2")
    action_dim = 4

    def __init__(self, shape: str = "two_point", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._second = None
        self._t2 = (0.0, 0.0)

    def build(self, space, position):
        body, shapes = super().build(space, position)
        b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        b2.position = (position[0] + 3 * TWO_POINT_RADIUS, position[1])
        sh = pymunk.Circle(b2, TWO_POINT_RADIUS)
        sh.friction = OBJECT_FRICTION
        space.add(b2, sh)
        self._second = b2
        self._t2 = (float(b2.position.x), float(b2.position.y))
        return body, shapes

    def _target_pose(self, action):
        self._t2 = (float(action[2]), float(action[3]))
        return float(action[0]), float(action[1]), None

    def on_reset(self, env) -> None:
        if self._second is not None:
            self._t2 = (float(self._second.position.x),
                        float(self._second.position.y))

    @property
    def second_pos(self):
        if self._second is None:
            return (0.0, 0.0)
        return (float(self._second.position.x), float(self._second.position.y))

    def pre_substep(self, env):
        if self._second is not None:
            pos = self._second.position
            dx, dy = self._t2[0] - pos.x, self._t2[1] - pos.y
            d = math.hypot(dx, dy)
            if d < 1e-9:
                self._second.velocity = (0.0, 0.0)
            else:
                sp = min(env.PUSHER_SPEED, d / (env.DT / env.SUBSTEPS))
                self._second.velocity = (dx / d * sp, dy / d * sp)
        return super().pre_substep(env)

    def post_substep(self, env, captured) -> None:
        super().post_substep(env, captured)
        if self._second is not None:
            self._second.velocity = (0.0, 0.0)


class ScoopAgent(Agent):
    """Concave scoop: 3-DOF (x, y, angle).

    Carries the object inside its arc with NOTHING attached -- transport held
    by geometry and friction alone. Rotate the opening away from travel and
    the object leaves. It sits between the pushers, which cannot transport,
    and the gripper, which cannot drop by accident.
    """

    action_spec = ("x", "y", "angle")
    action_dim = 3
    controls_angle = True

    def _target_pose(self, action):
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle



class TowbarAgent(Agent):
    """Rigid tow bar: 3-DOF (x, y, hitch).

    The object is pinned to a TOW POINT the agent carries at distance L on a
    commanded BEARING, so `angle` steers where the load sits relative to you --
    swing it around yourself, or hold it out to one side while you travel.

    Angle used to spin a near-symmetric hitch ball, which did nothing:
    measured -14.6 degrees of object rotation for 172 degrees of command,
    against 172.0 for the gripper. Same action slot, no effect -- so the
    embodiment silently had one fewer usable DOF than its action space claimed.

    The link is rigid rather than a rope, so it transmits PUSH as well as pull. Measured: agent advances, object moves
    +178.3, which a tether physically cannot do; and steering makes the object
    swing about the hitch (25 degrees over a straight haul, more when weaving).

    Its character is indirect control. You never place the object; you place
    yourself, and the object arrives on the far end of a bar that swings. That
    is a different skill from carrying (fixed offset) and from pushing (contact
    face), and it is the only agent here that must anticipate a trailing load.
    """

    action_spec = ("x", "y", "angle", "grip")
    action_dim = 4
    controls_angle = True

    def __init__(self, shape: str = "towbar", *, length: float = TOWBAR_LENGTH,
                 solid_pusher: bool = True, solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.length = float(length)
        self._joint = None
        self._hitch = 0.0
        self._bearing = 0.0
        self._tow = None

    def _target_pose(self, action):
        self._hitch = float(action[3])
        # Angle is the BEARING of the tow point, not the hitch body's spin.
        self._bearing = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), self._bearing

    def on_reset(self, env) -> None:
        self._joint, self._hitch = None, 0.0
        self._bearing, self._tow = 0.0, None

    @property
    def hitched(self) -> bool:
        return self._joint is not None

    @property
    def dim(self) -> int:
        return len(self.action_spec)

    def active_constraints(self) -> tuple:
        return (self._joint,) if self._joint is not None else ()

    def _tow_point(self, env):
        p = env._pusher_body.position
        return (p.x + math.cos(self._bearing) * self.length,
                p.y + math.sin(self._bearing) * self.length)

    def pre_substep(self, env):
        want = self._hitch > 0.5
        if self._tow is None:
            self._tow = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            self._tow.position = self._tow_point(env)
            env._space.add(self._tow)
        # The tow point rides at the commanded bearing every substep; the
        # object is pinned to IT, so steering the bearing swings the load.
        self._tow.position = self._tow_point(env)
        if want and self._joint is None:
            if _surface_distance(env, env._object_body.position) <= self.length:
                j = pymunk.PivotJoint(self._tow, env._object_body,
                                      env._object_body.position)
                env._space.add(j)
                self._joint = j
        elif not want and self._joint is not None:
            if self._joint in env._space.constraints:
                env._space.remove(self._joint)
            self._joint = None
        return super().pre_substep(env)


class CompliantAgent(Agent):
    """Force-limited pusher: 2-DOF (x, y), same action space as `circle`.

    Compliance here is LIMITED AUTHORITY, not a spring. Springs and
    force-capped constraints both fail in this engine: velocity is erased every
    step so a DampedSpring integrates to nothing, and a PivotJoint's max_force
    against a KINEMATIC pusher (infinite mass) changes nothing at all --
    measured carried-fraction was identical (97/97/95/71/46 %) from max_force
    5e4 through 1e7, because the solver satisfies the constraint by moving only
    the object.

    What IS expressible: cap how far the pusher may advance while in contact.
    A rigid pusher has infinite authority and wins every contact; this one
    cannot overpower a jammed or wall-pinned object and must approach along
    directions that are actually free. Measured 3.2x weaker than `circle` on
    the same commands.
    """

    action_spec = ("x", "y")
    action_dim = 2

    def __init__(self, shape: str = "compliant", *,
                 loaded_speed_frac: float = 0.25,
                 solid_pusher: bool = True, solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self.loaded_speed_frac = float(loaded_speed_frac)
        self._before = None

    def pre_substep(self, env):
        self._before = (float(env._pusher_body.position.x),
                        float(env._pusher_body.position.y))
        return super().pre_substep(env)

    def post_substep(self, env, captured) -> None:
        super().post_substep(env, captured)
        if env._pusher_object_penetration_depth() <= 0.0 or self._before is None:
            return          # free space: full authority, identical to a pusher
        now = env._pusher_body.position
        dx, dy = now.x - self._before[0], now.y - self._before[1]
        moved = math.hypot(dx, dy)
        cap = self.loaded_speed_frac * env.PUSHER_SPEED * (env.DT / env.SUBSTEPS)
        if moved > cap > 0.0:
            k = cap / moved
            env._pusher_body.position = (self._before[0] + dx * k,
                                         self._before[1] + dy * k)


class UmiAgent(Agent):
    """UMI-style rotary gripper: 4-DOF (x, y, wrist, grip).

    Continuous jaw width and a free-spinning wrist, in the spirit of the
    handheld UMI gripper. What makes it distinct from `gripper` is that grip is
    GRADED, with three physical regimes rather than open/closed:

        grip > 0.66   RELEASED  -- fingers clear, object free
        0.33-0.66     PINCHED   -- pivot only: the object is held in position
                                   but free to SPIN between the fingers, so it
                                   can settle or be turned by contact
        grip < 0.33   CLAMPED   -- pivot + gear: rigid, wrist rotation drives
                                   the object's orientation directly

    Grading it this way rather than by constraint strength is deliberate:
    max_force cannot grade anything against a kinematic body -- measured
    identical carried-fraction (97/97/95/71/46 %) from max_force 5e4 through
    1e7 -- so "how hard you squeeze" has to be expressed as WHICH DEGREES OF
    FREEDOM are constrained, which the solver does handle exactly.

    The pinched regime is the affordance nothing else here has: hold position
    while deliberately surrendering orientation, then clamp to lock it.
    """

    action_spec = ("x", "y", "angle", "grip")
    action_dim = 4
    controls_angle = True

    def __init__(self, shape: str = "umi", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._fingers: list[pymunk.Body] = []
        self._grip = 1.0
        self._cs = None
        self._held_gap = UMI_MIN_GAP
        self._mode = "released"

    def build(self, space, position):
        body, shapes = super().build(space, position)
        self._fingers = []
        for sign in (-1.0, 1.0):
            f = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            f.position = (position[0] + sign * UMI_HINGE_SPAN, position[1])
            # Rectangle offset so it runs FORWARD from the hinge at the body
            # origin. Centring it on the hinge (the first version) put half
            # the finger behind the pivot and made swinging it invisible --
            # open, half and closed all rendered identically.
            poly = pymunk.Poly(f, _rect_verts(
                0.0, UMI_FINGER_LEN / 2,
                2 * UMI_FINGER_HALF_W, UMI_FINGER_LEN))
            poly.friction = OBJECT_FRICTION
            space.add(f, poly)
            self._fingers.append(f)
        return body, shapes

    def _target_pose(self, action):
        # grip 1 = clamped; _grip stays "how open" internally.
        self._grip = 1.0 - min(1.0, max(0.0, float(action[3])))
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._grip, self._cs, self._mode = 1.0, None, "released"

    @property
    def grasped(self) -> bool:
        return self._cs is not None

    @property
    def mode(self) -> str:
        return self._mode

    def active_constraints(self) -> tuple:
        return tuple(self._cs) if self._cs else ()

    def _gap(self, env) -> float:
        gap = UMI_MIN_GAP + self._grip * (UMI_MAX_GAP - UMI_MIN_GAP)
        # Floor at the object's width whenever it is BETWEEN the fingers, not
        # only once already grasped. Gating on _cs was circular: the fingers
        # closed straight through the limb and shoved it away, so the grasp
        # could never form, so the floor never applied. Fingers stop where
        # they touch.
        if self._cs is not None:
            return self._held_gap          # frozen at grasp -- see GripperAgent
        if self._between(env):
            gap = max(gap, 2.0 * self._half_width(env) + 1.0)
        return min(gap, UMI_MAX_GAP)

    def _half_width(self, env) -> float:
        """Half-width of the object material INSIDE THE FINGER POCKET.

        See GripperAgent._obj_half_width -- taking the max over all vertices
        measured the entire object (60.0 against a 15-wide limb) and floored
        the gap wide open.
        """
        wr = env._pusher_body
        ca, sa = math.cos(-wr.angle), math.sin(-wr.angle)
        best = 0.0
        for sh in env._object_shapes:
            for v in sh.get_vertices():
                w = v.rotated(env._object_body.angle) + env._object_body.position
                rel = w - wr.position
                lx = rel.x * ca - rel.y * sa
                ly = rel.x * sa + rel.y * ca
                if -UMI_FINGER_LEN * 0.25 <= ly <= UMI_FINGER_LEN:
                    best = max(best, abs(lx))
        return best if best > 0.0 else UMI_MIN_GAP / 2

    def _finger_angle(self, env) -> float:
        """Splay angle per finger: 90 deg fully open, 0 deg fully closed.

        REVOLUTE closing -- the fingers hinge at the wrist and swing together
        like a pincer. Straight from the commanded grip so the visual and the
        physical gap cannot disagree.
        """
        # Invert the real geometry: tip separation = 2*(hinge + len*sin(phi)).
        # The previous linear map treated gap as a fraction of travel, so a
        # commanded gap of 31 (to hold a 30-wide stem) put the tips 61 apart
        # -- the fingers sat splayed in a V around an object they were
        # supposedly gripping.
        half = self._gap(env) / 2.0
        return math.asin(min(1.0, max(0.0, (half - UMI_HINGE_SPAN) / UMI_FINGER_LEN)))

    def _sync(self, env):
        wr = env._pusher_body
        phi = self._finger_angle(env)
        ca, sa = math.cos(wr.angle), math.sin(wr.angle)
        for sign, f in zip((-1.0, 1.0), self._fingers):
            hx = sign * UMI_HINGE_SPAN
            # Hinge rides with the wrist; the finger runs forward from it,
            # rotated outward by phi. phi = 0 leaves both fingers parallel and
            # pointing ahead (closed); phi = 90 deg splays them flat (open).
            f.position = (wr.position.x + hx * ca, wr.position.y + hx * sa)
            # MINUS sign: the left finger (sign -1) must swing toward -X and
            # the right toward +X, i.e. outward. Using +sign rotated each one
            # across the centreline, so at half-closed they crossed into an X
            # instead of opening.
            f.angle = wr.angle - sign * phi
            f.velocity = wr.velocity
            f.angular_velocity = wr.angular_velocity

    def _between(self, env) -> bool:
        wr = env._pusher_body
        half = UMI_MAX_GAP / 2 * 0.8
        # Sample BOTH sides of the wrist: the fingers extend +/-26 in local Y,
        # so an object entering from the negative side is just as grasped.
        # Sampling only positive ly missed it entirely and the grip never
        # formed regardless of how the operator closed the fingers.
        for lx in (-half, -half / 2, 0.0, half / 2, half):
            for ly in (-UMI_FINGER_HALF_H * 0.9, -UMI_FINGER_HALF_H * 0.45, 0.0,
                       UMI_FINGER_HALF_H * 0.45, UMI_FINGER_HALF_H * 0.9):
                if _surface_distance(env, wr.local_to_world((lx, ly))) <= 0.0:
                    return True
        return False

    def _detach(self, env):
        if self._cs:
            for c in self._cs:
                if c in env._space.constraints:
                    env._space.remove(c)
        self._cs = None

    def pre_substep(self, env):
        self._sync(env)
        want = ("released" if self._grip > 0.66
                else "pinched" if self._grip > 0.33 else "clamped")
        if want != self._mode:
            self._detach(env)
            graspable = 2.0 * self._half_width(env) <= UMI_MAX_GAP * 0.95
            if want != "released" and graspable and self._between(env):
                wr, obj = env._pusher_body, env._object_body
                # Anchor at the wrist (between the fingers), not the object's
                # centre of mass -- see GripperAgent.
                pj = pymunk.PivotJoint(wr, obj, wr.position)
                env._space.add(pj)   # uncapped -- see GripperAgent
                self._held_gap = max(UMI_MIN_GAP,
                                     2.0 * self._half_width(env) + 1.0)
                cs = [pj]
                if want == "clamped":
                    cs.append(_add_gear(env._space, wr, obj, obj.angle - wr.angle))
                self._cs = tuple(cs)
                self._mode = want
            else:
                self._mode = "released" if want == "released" else self._mode
        return super().pre_substep(env)


class TriangleAgent(Agent):
    """Equilateral triangle pusher: 3-DOF (x, y, angle).

    Pure contact, no constraints, no action at a distance. Its affordance is
    that YOU CHOOSE THE CONTACT PATCH by rotating it, and that patch sets both
    the SIGN and the SIZE of the spin you induce. Measured on an identical
    straight push:

        flat face into the object  ->  moved 123.9, rotated -33.8 deg
        vertex into the object     ->  moved 136.5, rotated +30.0 deg
        edge at 90 deg             ->  moved 139.7, rotated +54.9 deg

    So the SAME command yields ~90 degrees of spread in induced rotation
    depending only on how the pusher is held. Note a flat face is NOT
    spin-free -- the torque comes from where contact lands relative to the
    object's centre of mass, not from the pusher's face alone -- so the skill
    is picking an orientation for the turn you want, not avoiding turning.

    A circle offers no such choice, and stick/L auto-orient to their heading
    rather than letting you pick.

    This replaces the wrench, which coupled a GearJoint on PROXIMITY with no
    engage channel: it was already coupled at spawn from 136 units away, so
    the object's orientation was silently locked to the agent across most of
    the arena with no way to release it.
    """

    action_spec = ("x", "y", "angle")
    action_dim = 3
    controls_angle = True

    def _target_pose(self, action):
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle


class FlipperAgent(Agent):
    """Hinged flipper: 4-DOF (x, y, angle, grip).

    A bar pivoting at the wrist, like a pinball flipper. `grip` drives it
    through its arc, and because the bar swings about the wrist rather than
    riding on it, THE TIP MOVES WHILE THE BASE STAYS STILL. Every other
    embodiment here has to travel to do work; this one can strike and sweep
    from a standstill, and the far end of the bar moves several times faster
    than the wrist ever does.

    The consequence is that reach and force are traded against control: a
    swing delivers a fast tangential hit that both slides and spins the
    object, but you aim it before it lands rather than steering it during.
    """

    action_spec = ("x", "y", "angle", "grip")
    action_dim = 4
    controls_angle = True

    def __init__(self, shape: str = "flipper", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._bar = None
        self._swing = 0.0

    def build(self, space, position):
        body, shapes = super().build(space, position)
        bar = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        bar.position = position
        poly = pymunk.Poly(bar, _rect_verts(
            0.0, -FLIPPER_LEN / 2, 2 * FLIPPER_HALF_W, FLIPPER_LEN))
        poly.friction = OBJECT_FRICTION
        space.add(bar, poly)
        self._bar = bar
        return body, shapes

    def _target_pose(self, action):
        self._swing = min(1.0, max(0.0, float(action[3])))
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._swing = 0.0

    @property
    def swing_deg(self) -> float:
        return math.degrees(self._swing * FLIPPER_SWING)

    @property
    def mode(self) -> str:
        return ("retracted" if self._swing < 0.2
                else "swinging" if self._swing < 0.85 else "extended")

    def pre_substep(self, env):
        wr = env._pusher_body
        if self._bar is not None:
            self._bar.position = wr.position
            self._bar.angle = wr.angle + self._swing * FLIPPER_SWING
            self._bar.velocity = wr.velocity
            # Give the bar real angular velocity so a swing transfers a
            # tangential impulse instead of teleporting through the object.
            self._bar.angular_velocity = wr.angular_velocity
        return super().pre_substep(env)


class ClutchAgent(Agent):
    """Clutch head: 4-DOF (x, y, angle, grip). Articulation is MANDATORY.

    The head has two mutually exclusive modes and neither can finish the task:

        grip 0  SLIDE -- a near-frictionless roller face. It pushes the object
                         around freely but transmits almost no torque, so it
                         can place the object anywhere and cannot aim it.
        grip 1  TURN  -- a pin drops and bites. Pivot + gear pin the object to
                         the head, so the wrist drives its ORIENTATION exactly,
                         and translation now moves the pair rigidly rather than
                         sliding the object over the face.

    Since the goal is a POSE, position and orientation both have to be solved,
    and no single mode does both. You slide to place, clutch in to aim, and
    repeat. That is the difference from the poker this replaces, whose
    extension was optional -- you could ignore the DOF and use it as a long
    stick. Here ignoring the DOF means never fixing the angle.
    """

    action_spec = ("x", "y", "angle", "grip")
    action_dim = 4
    controls_angle = True

    def __init__(self, shape: str = "clutch", *, solid_pusher: bool = True,
                 solid_contact_guard: bool = True,
                 control_gap: "ControlGap | str | None" = None):
        super().__init__(shape, solid_pusher=solid_pusher,
                         solid_contact_guard=solid_contact_guard,
                         control_gap=control_gap)
        self._pin = None
        self._cs = None
        self._grip = 0.0

    def build(self, space, position):
        body, shapes = super().build(space, position)
        pin = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        pin.position = position
        poly = pymunk.Poly(pin, _rect_verts(
            0.0, -CLUTCH_PIN_LEN / 2, 2 * CLUTCH_PIN_HALF_W, CLUTCH_PIN_LEN))
        poly.friction = OBJECT_FRICTION
        poly.sensor = True          # visual + engagement cue; the constraint holds
        space.add(pin, poly)
        self._pin = pin
        return body, shapes

    def _target_pose(self, action):
        self._grip = min(1.0, max(0.0, float(action[3])))
        angle = (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
        return float(action[0]), float(action[1]), angle

    def on_reset(self, env) -> None:
        self._grip, self._cs = 0.0, None

    @property
    def mode(self) -> str:
        return "turn" if self._grip > 0.5 else "slide"

    @property
    def engaged(self) -> bool:
        return self._cs is not None

    def active_constraints(self) -> tuple:
        return tuple(self._cs) if self._cs else ()

    def _detach(self, env):
        if self._cs:
            for c in self._cs:
                if c in env._space.constraints:
                    env._space.remove(c)
        self._cs = None

    def pre_substep(self, env):
        wr = env._pusher_body
        if self._pin is not None:
            # The pin extends while clutched, retracts while sliding.
            d = CLUTCH_PIN_LEN * (0.15 + 0.85 * self._grip)
            ca, sa = math.cos(wr.angle), math.sin(wr.angle)
            self._pin.position = (wr.position.x + sa * d, wr.position.y - ca * d)
            self._pin.angle = wr.angle
            self._pin.velocity = wr.velocity
            self._pin.angular_velocity = wr.angular_velocity
        want = self._grip > 0.5
        if want and self._cs is None:
            if _surface_distance(env, wr.position) <= CLUTCH_R + 6.0:
                obj = env._object_body
                # Pin to the STATIC body, not to the wrist. Pinning to the
                # wrist made turn mode a rigid grasp that both translated and
                # rotated (moved 218, rotated 160), so the clutch was
                # optional -- one mode did the whole task. Anchored to the
                # world the object cannot translate at all while clutched;
                # the gear still drives its angle. Position and orientation
                # now genuinely require different modes.
                pj = pymunk.PivotJoint(env._space.static_body, obj, obj.position)
                env._space.add(pj)
                gj = _add_gear(env._space, wr, obj, obj.angle - wr.angle)
                self._cs = (pj, gj)
        elif not want and self._cs is not None:
            self._detach(env)
        return super().pre_substep(env)


_SIMPLE = ("circle", "circle_small", "stick", "L")

#: shape name -> agent class, for everything that is not a plain 2-DOF pusher.
_AGENT_CLASSES: dict[str, type[Agent]] = {
    "u_socket": USocketAgent,
    "gripper": GripperAgent,
    "suction": SuctionAgent,
    "triangle": TriangleAgent,
    "umi": UmiAgent,
    "scoop": ScoopAgent,
    "flipper": FlipperAgent,
    "clutch": ClutchAgent,
}

#: Every constructible pusher. env imports this so the two lists cannot drift.
VALID_PUSHERS: tuple[str, ...] = _SIMPLE + tuple(_AGENT_CLASSES)

#: The behaviourally-distinct agents added on top of the original five
#: position-controlled pushers (circle/circle_small/stick/L) and u_socket.
#: Derived, not hand-listed, so adding an agent class cannot leave it out.
NEW_AGENTS: tuple[str, ...] = tuple(
    a for a in _AGENT_CLASSES if a != "u_socket"
)


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
