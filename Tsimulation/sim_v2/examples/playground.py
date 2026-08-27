"""Interactive playground: fly every agent by hand, one window, no recording.

This is a FEEL tool, not a collection tool -- `collect/mouse_collect.py` is
what writes episodes. The point here is that each agent's distinctive
mechanism is something you have to *do* to believe: latching, carrying,
pulling, coasting. A number in a test table does not tell you whether an
embodiment is actually controllable by a person, which is the precondition
for teleoperating demonstrations with it.

Bodies are drawn straight from the pymunk space rather than through
render._draw_pusher, for two reasons: that function falls through to the
stick branch for any shape it does not recognise, and it draws only the
single pusher body, so the gripper's jaws and two_point's second contact --
separate bodies owned by their agents -- would be invisible.

Run::

    python -m Tsimulation.sim_v2.examples.playground
    python -m Tsimulation.sim_v2.examples.playground --agent gripper --object U

Controls::

    mouse          primary contact XY
    SPACE (hold)   engage: grip / suck / hook / magnetise / strike
    chain gripper  hold SPACE to close; hold S to open; release to hold gap
    A / D          rotate (agents that command their own angle)
    W / S          second-contact spread            (two_point)
    Q / E          second-contact orbit angle       (two_point)
    G              cycle control gap (ideal/tight/loose/laggy/sticky/jittery)
    [ / ]          previous / next agent
    1..9, 0, -, =  jump straight to an agent
    TAB            cycle object shape  T -> U -> Z
    ENTER          start / stop recording (with --output)
                   (--auto re-arms automatically after each save and moves to
                    the next embodiment once --per-agent is reached)
    left click     unfreeze simulation and start recording when
                   --left-click-to-start is enabled
    BACKSPACE      discard the take in progress
    R              reset episode (new random layout, or retry current curated init)
    ESC            quit
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pygame
import pymunk

from Tsimulation.sim_v2.collect.obstacle_init import (
    level_entries,
    load_manifest,
    reset_to_manifest_entry,
)
from Tsimulation.sim_v2.collect.replay_init import (
    ObstacleInitKey,
    collected_obstacle_init_keys,
    collected_seed_keys,
)
from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.sim_v2.pushshapes.agents import (
    CONTROL_GAPS,
    NEW_AGENTS,
    VALID_PUSHERS,
)
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v2.pushshapes.render import (
    BG_COLOR,
    GOAL_COLOR,
    OBJECT_COLOR,
    to_image_obs,
)
from Tsimulation.sim_v2.pushshapes.shapes import SHAPES

WORLD = 512
# 1.6 made the window 935 px tall, so the HUD fell off the bottom of a laptop
# screen. 1.25 keeps the whole thing (640 + 116) inside a 13" display.
SCALE = 1.25
WIN = int(WORLD * SCALE)
HUD_H = 116
# Match the canonical U-socket collector: both the socket and chain gripper
# change their commanded orientation at 45 degrees/second under A/D.
SOCKET_KEY_TURN_SPEED = math.radians(45.0)

# The arena background is (240, 240, 240) -- near white. These must be DARK
# to read against it; the first version used (235, 235, 235) for the pusher,
# one step off the background, which made it effectively invisible.
COL_PUSHER = (200, 40, 40)  # matches render.PUSHER_COLOR
COL_EXTRA = (225, 110, 20)  # agent-owned extra bodies (jaws, 2nd point)
COL_ENGAGED = (0, 150, 60)  # mechanism currently latched/attached
COL_SENSOR = (170, 90, 0)  # non-contact body (magnet, tapper)
COL_TEXT = (228, 228, 228)
COL_HUD_OK = (110, 230, 140)  # green on the DARK hud panel, not the arena
COL_DIM = (140, 140, 140)

# What SPACE means, per agent -- shown in the HUD so the control is discoverable.
ENGAGE_LABEL = {
    "gripper": "SPACE catch/hold; S release",
    "chain_gripper": "SPACE curl/hold; S release",
    "suction": "SPACE suction on",
    "tether": "SPACE hook rope",
    "magnet": "SPACE magnetise",
    "tapper": "SPACE strike",
    "u_socket": "SPACE (n/a)",
}


def _agent_state(env) -> tuple[str, bool]:
    """Human-readable mechanism state, and whether it is currently engaged."""
    a = env.agent
    if env.pusher_shape == "gripper":
        if bool(getattr(a, "grasped", False)):
            return "GRASPED — A/D ROTATES; S RELEASES", True
        if getattr(a, "_jaw_cmd", 1.0) <= 0.35 and a._spans(env):
            return "PINCHED ONLY — ALIGN BOTH JAWS", False
        return "OPEN — SPACE TO GRASP STEM", False
    for attr, name in (
        ("grasped", "GRASPED"),
        ("attached", "ATTACHED"),
        ("hooked", "HOOKED"),
        ("socket_latched", "LATCHED"),
    ):
        if hasattr(a, attr):
            on = bool(getattr(a, attr))
            # The U-socket's latch is passive contact mechanics, not an
            # operator-commanded mechanism.  Keep its body and status text in
            # their normal colours after latching so collection does not gain
            # an artificial green success cue unavailable to the policy.
            if attr == "socket_latched":
                return (name if on else name.lower()), False
            return (name if on else name.lower()), on
    return "", False


def _write_aligned_step(
    writer: ZarrDemoWriter,
    *,
    pre_obs: dict[str, np.ndarray],
    action: np.ndarray,
    reward: float,
) -> None:
    """Write ``state[t]`` before ``action[t]`` and its post-step reward."""
    act = np.asarray(action, dtype=np.float64).reshape(-1)
    writer.add_step(
        image=pre_obs["image"],
        pusher_obs_pose=np.concatenate(
            [pre_obs["agent_pos"], pre_obs["agent_angle"]]
        ),
        object_obs_pose=pre_obs["object_pose"],
        pusher_cmd_pose=np.array(
            [act[0], act[1], act[2] if act.size >= 3 else 0.0],
            dtype=np.float64,
        ),
        action=act,
        reward=reward,
        goal_pose=pre_obs["goal_pose"],
    )


def _draw_space(surf, env, engaged: bool) -> None:
    """Draw every body in the space, so agent-owned extras are visible."""
    pusher_bodies = {id(env._pusher_body)}
    for shape in env._space.shapes:
        body = shape.body
        if body is env._object_body:
            continue  # object drawn separately, filled
        is_pusher = id(body) in pusher_bodies
        if getattr(shape, "sensor", False):
            colour = COL_SENSOR
        elif is_pusher:
            colour = COL_ENGAGED if engaged else COL_PUSHER
        else:
            colour = COL_EXTRA
        if isinstance(shape, pymunk.Circle):
            c = body.local_to_world(shape.offset)
            pygame.draw.circle(
                surf,
                colour,
                (int(c.x * SCALE), int(c.y * SCALE)),
                max(2, int(shape.radius * SCALE)),
                0 if not getattr(shape, "sensor", False) else 3,
            )
        elif isinstance(shape, pymunk.Poly):
            pts = [body.local_to_world(v) for v in shape.get_vertices()]
            pts = [(p.x * SCALE, p.y * SCALE) for p in pts]
            if len(pts) >= 3:
                pygame.draw.polygon(
                    surf,
                    colour,
                    pts,
                    0 if not getattr(shape, "sensor", False) else 3,
                )
        elif isinstance(shape, pymunk.Segment):
            # MUST go through local_to_world. shape.a/.b are BODY-LOCAL, so
            # drawing them raw put the tether hook, the compliant ring and the
            # scoop arc at coordinates like (6, 6) -- the top-left corner --
            # while the body was at (300, 300). They were invisible, not
            # missing: the physics worked the whole time. It looked correct
            # only for the arena walls, whose static body sits at the origin
            # with zero rotation, where local and world happen to coincide.
            wa = body.local_to_world(shape.a)
            wb = body.local_to_world(shape.b)
            wall = body is env._space.static_body
            pygame.draw.line(
                surf,
                (110, 110, 120) if wall else colour,
                (wa.x * SCALE, wa.y * SCALE),
                (wb.x * SCALE, wb.y * SCALE),
                max(3, int(shape.radius * 2 * SCALE)),
            )


def _draw_shape_polys(surf, shape_name, pose, colour, width=0):
    cx0, cy0, th = pose
    ct, st = math.cos(th), math.sin(th)
    for cx, cy, w, h in SHAPES[shape_name]:
        verts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, ly = cx + sx * w / 2, cy + sy * h / 2
            verts.append(
                (
                    (cx0 + lx * ct - ly * st) * SCALE,
                    (cy0 + lx * st + ly * ct) * SCALE,
                )
            )
        pygame.draw.polygon(surf, colour, verts, width)


def main() -> int:
    # Declared up front: `global` must precede every use of the name in this
    # scope, and argparse's default=SCALE reads it near the top. ast.parse()
    # accepts the wrong order -- it is a compile error, not a parse error --
    # so validate with compile(), not ast.parse().
    global SCALE, WIN

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=None, choices=list(VALID_PUSHERS))
    ap.add_argument("--object", default=None, choices=list(SHAPES))
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument(
        "--gap",
        default="ideal",
        choices=list(CONTROL_GAPS) + ["random"],
        help="'random' draws a fresh compliance every episode",
    )
    ap.add_argument(
        "--gap-matrix",
        action="store_true",
        help=(
            "collect every fixed control-gap preset separately; output is "
            "<output>/<gap>/<agent>/<object> and --auto advances through "
            "every (gap, agent) cell"
        ),
    )
    ap.add_argument(
        "--output",
        default=None,
        help="record episodes to <output>/<agent>/<object>/*.zarr",
    )
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument(
        "--min-frames",
        type=int,
        default=60,
        help="takes shorter than this are discarded, not committed",
    )
    ap.add_argument(
        "--per-agent",
        type=int,
        default=10,
        help="episodes to collect per embodiment before advancing",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=SCALE,
        help="window scale; lower it if the HUD is off-screen",
    )
    ap.add_argument(
        "--agents",
        default=None,
        help="'all', 'new' (the 7 behaviourally-distinct ones), or a "
        "comma-separated list. Restricts both the [ ] cycle and "
        "--auto advancement.",
    )
    ap.add_argument(
        "--seeds-file",
        "--init-manifest",
        dest="seeds_file",
        type=Path,
        default=None,
        help=(
            "versioned obstacle-init manifest; constrains collection to its "
            "level-specific T + ChainGripper seed bank"
        ),
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="re-arm recording after every save and advance to the "
        "next embodiment once --per-agent is reached",
    )
    ap.add_argument(
        "--left-click-to-start",
        action="store_true",
        help=(
            "after every reset, freeze the simulation until a left mouse "
            "click starts both stepping and recording; --auto still saves "
            "successes and advances layouts"
        ),
    )
    ap.add_argument(
        "--reset-seed-bank-seed",
        type=int,
        default=0,
        help=(
            "seed for the reproducible random reset-seed stream used without "
            "a manifest; saved and attempted reset seeds are skipped"
        ),
    )
    args = ap.parse_args()
    if args.reset_seed_bank_seed < 0:
        ap.error("--reset-seed-bank-seed must be non-negative")
    if args.left_click_to_start and args.output is None:
        ap.error("--left-click-to-start requires --output")

    init_manifest = None
    init_entries = None
    init_manifest_sha = None
    init_level_bank_sha = None
    if args.seeds_file is not None:
        try:
            init_manifest = load_manifest(args.seeds_file)
            init_entries = level_entries(init_manifest, args.obstacles)
        except (OSError, TypeError, ValueError) as exc:
            ap.error(f"invalid --seeds-file: {exc}")
        init_manifest_sha = hashlib.sha256(args.seeds_file.read_bytes()).hexdigest()
        init_level_bank_sha = str(
            init_manifest["level_bank_sha256"][str(args.obstacles)]
        )
        required_agent = str(init_manifest["pusher_shape"])
        required_object = str(init_manifest["object_shape"])
        if args.agent not in (None, required_agent):
            ap.error(f"--seeds-file requires --agent {required_agent}")
        if args.object not in (None, required_object):
            ap.error(f"--seeds-file requires --object {required_object}")
        if args.agents not in (None, "all", required_agent):
            ap.error(f"--seeds-file requires --agents {required_agent}")
        if args.per_agent > len(init_entries):
            ap.error(
                f"--per-agent {args.per_agent} exceeds the level "
                f"bank size {len(init_entries)}"
            )
        args.agent = required_agent
        args.agents = required_agent
        args.object = required_object
    else:
        args.agent = args.agent or "circle"
        args.agents = args.agents or "all"
        args.object = args.object or "T"

    SCALE = float(args.scale)
    WIN = int(WORLD * SCALE)

    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN + HUD_H))
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 14)
    big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 19, bold=True)
    clock = pygame.time.Clock()

    # ``random`` samples a new continuous gap every episode.  It is useful as
    # augmentation, but is not one of the six named control-gap conditions in
    # the MR and therefore is deliberately excluded from the fixed matrix.
    gaps = list(CONTROL_GAPS) if args.gap_matrix else list(CONTROL_GAPS) + ["random"]
    if args.gap_matrix and args.gap == "random":
        ap.error(
            "--gap-matrix uses the six fixed presets; --gap random is not a matrix mode"
        )
    gi = gaps.index(args.gap)
    if args.agents == "all":
        agents = list(VALID_PUSHERS)
    elif args.agents == "new":
        agents = list(NEW_AGENTS)
    else:
        agents = [a.strip() for a in args.agents.split(",") if a.strip()]
        unknown = [a for a in agents if a not in VALID_PUSHERS]
        if unknown:
            ap.error(f"unknown agent(s) {unknown}; known: {list(VALID_PUSHERS)}")
    if not agents:
        ap.error("--agents selected nothing")
    ai = agents.index(args.agent) if args.agent in agents else 0
    objects = [args.object] if init_entries is not None else list(SHAPES)
    oi = objects.index(args.object)

    def apply_gap(e):
        name = gaps[gi]
        e.agent.randomize_gap = name == "random"
        if not e.agent.randomize_gap:
            e.agent.control_gap = CONTROL_GAPS[name]

    out_root = Path(args.output) if args.output else None
    writer = None
    writer_key = [None]
    current_entry = None
    recording = False
    saved = 0
    steps_rec = 0
    attempted_unmanifested_seeds: set[int] = set()
    unmanifested_seed_rng = np.random.default_rng(args.reset_seed_bank_seed)
    current_reset_seed_draw_index: int | None = None
    unmanifested_seed_draw_count = 0

    # ONE writer per output dir, not per episode: the writer owns episode
    # naming and index resumption (episode_<obj>_<pusher>_obs<N>_NNNNNN.zarr).
    # Handing it a file path instead produced a directory zarr could not open.
    def output_dir(
        gap_index: int | None = None,
        agent_index: int | None = None,
        object_index: int | None = None,
    ) -> Path:
        """Directory for one independently-counted collection cell."""
        assert out_root is not None
        gap_index = gi if gap_index is None else gap_index
        agent_index = ai if agent_index is None else agent_index
        object_index = oi if object_index is None else object_index
        root = out_root / gaps[gap_index] if args.gap_matrix else out_root
        return root / agents[agent_index] / objects[object_index]

    def saved_here() -> int:
        """Episodes already on disk for this (gap, agent, object).

        Counted from disk rather than a session counter so relaunching resumes
        where you stopped instead of collecting a second set of ten.
        """
        if out_root is None:
            return 0
        return get_writer().existing_episode_count()

    def get_writer():
        nonlocal writer
        key = (gaps[gi], agents[ai], objects[oi])
        if writer is not None and writer_key[0] == key:
            return writer
        if writer is not None:
            writer.close()
        d = output_dir()
        d.mkdir(parents=True, exist_ok=True)
        writer = ZarrDemoWriter(
            path=d,
            env_args={
                "object_shape": objects[oi],
                "pusher_shape": agents[ai],
                "obstacle_level": args.obstacles,
                "control_gap_mode": gaps[gi],
            },
            image_size=args.image_size,
            metadata_override={
                "transition_schema_version": 1,
                "observation_alignment": "pre_step",
                "reward_alignment": "post_step_result_of_action_same_index",
            },
        )
        writer_key[0] = key
        return writer

    target_entries = (
        list(init_entries[: args.per_agent]) if init_entries is not None else None
    )
    entry_index_by_seed = (
        {int(entry["seed"]): index for index, entry in enumerate(target_entries)}
        if target_entries is not None
        else {}
    )

    def pending_entries(
        gap_index: int | None = None,
        agent_index: int | None = None,
        object_index: int | None = None,
    ) -> list[dict]:
        """Curated entries not yet committed for one collection cell."""
        if target_entries is None:
            return []
        if out_root is None:
            return list(target_entries)
        assert init_manifest is not None
        assert init_level_bank_sha is not None
        completed = collected_obstacle_init_keys(
            output_dir(gap_index, agent_index, object_index)
        )
        level = int(args.obstacles)
        return [
            entry
            for entry in target_entries
            if ObstacleInitKey(
                level_bank_sha256=init_level_bank_sha,
                sampler_revision=str(init_manifest["sampler_revision"]),
                geometry_hash=str(entry["geometry_hash"]),
                obstacle_level=level,
                reset_seed=int(entry["seed"]),
                entry_index=entry_index_by_seed[int(entry["seed"])],
                control_gap_mode=gaps[gi if gap_index is None else gap_index],
            )
            not in completed
        ]

    def completed_here(
        gap_index: int | None = None,
        agent_index: int | None = None,
        object_index: int | None = None,
    ) -> int:
        if target_entries is not None:
            return len(target_entries) - len(
                pending_entries(gap_index, agent_index, object_index)
            )
        if out_root is None:
            return 0
        if any(index is not None for index in (gap_index, agent_index, object_index)):
            d = output_dir(gap_index, agent_index, object_index)
            return sum(1 for entry in d.glob("*.zarr") if entry.is_dir())
        return saved_here()

    def cell_complete(
        gap_index: int | None = None,
        agent_index: int | None = None,
        object_index: int | None = None,
    ) -> bool:
        return completed_here(gap_index, agent_index, object_index) >= args.per_agent

    def build():
        nonlocal angle, current_entry, current_reset_seed_draw_index
        nonlocal unmanifested_seed_draw_count
        e = PushShapesEnv(
            object_shape=objects[oi],
            pusher_shape=agents[ai],
            obstacle_level=args.obstacles,
            image_size=args.image_size,
        )
        if not e.solid_pusher or not e.solid_contact_guard:
            raise RuntimeError(
                "Sim V2 collector requires solid_pusher=True and "
                "solid_contact_guard=True"
            )
        apply_gap(e)
        if target_entries is None:
            # Draw from a reproducible 32-bit random stream, rejecting every
            # reset seed already saved in this output cell or attempted in the
            # current process. The exact seed remains in episode_init.
            saved_seeds = (
                {
                    seed
                    for level, seed in collected_seed_keys(output_dir())
                    if level == int(args.obstacles)
                }
                if out_root is not None
                else set()
            )
            while True:
                reset_seed = int(
                    unmanifested_seed_rng.integers(
                        0,
                        np.iinfo(np.uint32).max + 1,
                        dtype=np.uint32,
                    )
                )
                current_reset_seed_draw_index = unmanifested_seed_draw_count
                unmanifested_seed_draw_count += 1
                if (
                    reset_seed not in saved_seeds
                    and reset_seed not in attempted_unmanifested_seeds
                ):
                    break
            attempted_unmanifested_seeds.add(reset_seed)
            e.reset(seed=reset_seed)
            current_entry = None
        else:
            pending = pending_entries()
            # A completed cell still needs an environment for its final HUD.
            current_entry = pending[0] if pending else target_entries[-1]
            reset_to_manifest_entry(e, current_entry, verify=True)
        apply_gap(e)
        # Every reset creates a fresh pusher orientation. Carrying the prior
        # take's UI target across that boundary makes the first commanded step
        # snap toward a stale angle, especially for unmanifested level 0.
        angle = e.pusher_angle
        return e

    def advance_cell() -> bool:
        """Move to the next collection cell that still needs episodes.

        Normal collection advances only across embodiments for the selected
        gap. Matrix collection advances across embodiments first and then all
        six fixed gaps. Returns False when every required cell met the quota.
        """
        nonlocal ai, gi
        cell_count = len(agents) * (len(gaps) if args.gap_matrix else 1)
        for _ in range(cell_count):
            ai = (ai + 1) % len(agents)
            if ai == 0 and args.gap_matrix:
                gi = (gi + 1) % len(gaps)
            if not cell_complete():
                return True
        return False

    def start_recording():
        nonlocal recording, steps_rec
        if out_root is None:
            return
        stop_recording(discard=True)
        w = get_writer()
        # episode_init carries the sampled compliance, so a recorded demo can
        # be replayed under the exact gap it was collected with.
        episode_init = env.get_episode_init()
        # Record the human-readable preset as well as its numeric parameters.
        # This makes the matrix self-describing even after directories are
        # merged for training.
        episode_init["control_gap_mode"] = gaps[gi]
        if current_entry is None:
            episode_init["reset_seed_allocator"] = {
                "algorithm": "numpy.PCG64.uint32",
                "stream_seed": int(args.reset_seed_bank_seed),
                "draw_index": current_reset_seed_draw_index,
            }
        if current_entry is not None:
            assert init_manifest is not None
            assert init_manifest_sha is not None
            assert init_level_bank_sha is not None
            assert target_entries is not None
            entry_index = entry_index_by_seed[int(current_entry["seed"])]
            episode_init["obstacle_init"] = {
                "schema_version": int(init_manifest["schema_version"]),
                "sampler_revision": str(init_manifest["sampler_revision"]),
                "manifest_path": str(args.seeds_file.expanduser().resolve()),
                "manifest_sha256": init_manifest_sha,
                "level_bank_sha256": init_level_bank_sha,
                "entry_index": entry_index,
                "entry_count": len(target_entries),
                "geometry_hash": str(current_entry["geometry_hash"]),
                "chain_joint_angle": float(current_entry["chain_joint_angle"]),
            }
        w.start_episode(init_state=episode_init)
        recording, steps_rec = True, 0

    def stop_recording(discard=False, successful=False) -> bool:
        """Stop the current take and report whether it was committed."""
        nonlocal recording, saved, steps_rec
        if writer is None or not writer.is_recording:
            recording, steps_rec = False, 0
            return False
        committed = False
        if discard or (steps_rec < args.min_frames and not successful):
            # Too short to be a demonstration -- a stray ENTER produced a
            # 1-frame manual episode that would pollute training as surely as
            # a runaway does.  A real simulator success is different: once
            # coverage reaches the fixed 0.95 threshold, save it regardless
            # of length rather than silently throwing away a solved take.
            writer.abort_episode()
        else:
            # commit_episode, NOT close: close() calls abort_episode() and
            # silently discards the take.
            episode_index = writer.commit_episode()
            committed = episode_index >= 0
            if committed:
                saved += 1
            if successful and committed:
                print(
                    f"[playground] auto-saved success {episode_index:06d} "
                    f"({steps_rec} frames, coverage >= {env.SUCCESS_THRESHOLD:.2f})"
                )
        recording, steps_rec = False, 0
        return committed

    all_done = False
    angle = 0.0
    spread = 34.0
    grip = 0.0
    orbit = math.pi / 2
    running = True

    env = build()
    if args.auto and out_root is not None:
        if cell_complete():
            if not advance_cell():
                all_done = True
            else:
                env = build()
        if not all_done and not args.left_click_to_start:
            start_recording()

    while running:
        # In automatic collection, being idle before the quota is a bug, not
        # a user-visible mode.  Re-arm defensively if any previous transition
        # left the writer inactive.
        if (
            args.auto
            and out_root is not None
            and not args.left_click_to_start
            and not recording
            and not all_done
        ):
            start_recording()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and args.left_click_to_start
                and not recording
                and not all_done
            ):
                # The click starts from the visible frozen pose, never from a
                # stale command left by the preceding take.
                angle = env.pusher_angle
                start_recording()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    stop_recording(discard=True)
                    env = build()
                elif ev.key == pygame.K_RETURN:
                    if recording:
                        committed = stop_recording()
                        if target_entries is not None:
                            if committed:
                                if cell_complete() and not advance_cell():
                                    all_done = True
                                if not all_done:
                                    env = build()
                            else:
                                # An under-length take was discarded. Replay
                                # the same pending entry before recording again.
                                env = build()
                    elif not args.left_click_to_start:
                        start_recording()
                elif ev.key == pygame.K_BACKSPACE:
                    stop_recording(discard=True)
                    env = build()
                elif ev.key == pygame.K_g:
                    stop_recording(discard=True)
                    gi = (gi + 1) % len(gaps)
                    if args.auto and out_root is not None:
                        if cell_complete() and not advance_cell():
                            all_done = True
                    env = build()
                elif ev.key == pygame.K_TAB:
                    stop_recording(discard=True)
                    oi = (oi + 1) % len(objects)
                    env = build()
                elif ev.key == pygame.K_LEFTBRACKET:
                    stop_recording(discard=True)
                    ai = (ai - 1) % len(agents)
                    env = build()
                elif ev.key == pygame.K_RIGHTBRACKET:
                    stop_recording(discard=True)
                    ai = (ai + 1) % len(agents)
                    env = build()
                else:
                    keys = "1234567890-="
                    ch = pygame.key.name(ev.key)
                    if ch in keys and keys.index(ch) < len(agents):
                        stop_recording(discard=True)
                        ai = keys.index(ch)
                        env = build()

        simulation_frozen = (
            args.left_click_to_start
            and out_root is not None
            and not recording
            and not all_done
        )
        held = pygame.key.get_pressed()
        turn_step = (
            SOCKET_KEY_TURN_SPEED * env.DT
            if env.pusher_shape in {"u_socket", "chain_gripper"}
            else 0.06
        )
        if not simulation_frozen:
            if held[pygame.K_a]:
                angle -= turn_step
            if held[pygame.K_d]:
                angle += turn_step
            if held[pygame.K_w]:
                spread = min(160.0, spread + 1.5)
                grip = min(1.0, grip + 0.02)
            if held[pygame.K_s]:
                spread = max(12.0, spread - 1.5)
                grip = max(0.0, grip - 0.02)
            if held[pygame.K_q]:
                orbit -= 0.05
            if held[pygame.K_e]:
                orbit += 0.05
        engage = 1.0 if held[pygame.K_SPACE] else 0.0

        mx, my = pygame.mouse.get_pos()
        wx = float(np.clip(mx / SCALE, 0, WORLD))
        wy = float(np.clip(my / SCALE, 0, WORLD))

        # Build the action FROM THE AGENT'S OWN SPEC. Encoding by dimension
        # (the previous approach) mis-wired three agents at once: suction
        # never suctioned because the angle landed in its engage slot, and
        # wrench/scoop had their orientation pinned to 0-or-1 radians because
        # engage landed in their angle slot -- 3-DOF in name only.
        spec = env.agent.action_spec
        # Once the parallel gripper has actually formed a grasp, keep sending
        # the closed command after SPACE is released.  Otherwise the very next
        # frame opened the jaws, so A/D appeared unable to rotate a caught T.
        # S remains an explicit, immediate release command.
        retain_gripper_grasp = (
            env.pusher_shape == "gripper"
            and bool(getattr(env.agent, "grasped", False))
            and not held[pygame.K_s]
        )
        if env.pusher_shape == "chain_gripper":
            if held[pygame.K_SPACE]:
                commanded_grip = 1.0
            elif held[pygame.K_s]:
                commanded_grip = 0.0
            else:
                # Neither key means HOLD THE CURRENT OPENING.  Sending zero
                # here used to release the grasp as soon as SPACE was let go,
                # making A/D appear unable to rotate a caught object.
                commanded_grip = env.agent.grip_fraction
        else:
            commanded_grip = 1.0 if engage or retain_gripper_grasp else grip
        chan = {
            "x": wx,
            "y": wy,
            "angle": angle,
            # ONE meaning everywhere: SPACE = grip/engage/close/hitch, and
            # W/S grades it for the agents that accept a continuous value.
            "grip": commanded_grip,
        }
        act = np.array([chan[c] for c in spec], dtype=np.float64)
        pre_obs = env._get_obs()

        if simulation_frozen:
            # Waiting is a true causal boundary: mouse input may update the
            # prospective XY command, but neither physics nor the pusher
            # advances until the same left click starts recording.
            obs = pre_obs
            reward = 0.0
            info = {}
            coverage = 0.0
            success = False
        else:
            obs, reward, _terminated, _trunc, info = env.step(act)
            coverage = float(info.get("coverage", reward))
            # Use the unrounded coverage itself as the source of truth.  This
            # avoids depending on a stale/misrouted Gym termination flag while
            # preserving the exact Sim V2 >= 0.95 success contract.
            success = math.isfinite(coverage) and coverage >= float(
                env.SUCCESS_THRESHOLD
            )
        if recording and writer is not None:
            _write_aligned_step(
                writer,
                pre_obs=pre_obs,
                action=act,
                reward=reward,
            )
            steps_rec += 1
        # Auto-stop on success so a solved demo is never lost by forgetting to
        # press ENTER, and immediately reset for the next one.
        if recording and success:
            print(
                f"[playground] success detected: coverage={coverage:.9f}; saving",
                flush=True,
            )
            committed = stop_recording(successful=True)
            if target_entries is not None and out_root is not None and committed:
                if cell_complete() and not advance_cell():
                    all_done = True
                if not all_done:
                    env = build()
                    if args.auto and not args.left_click_to_start:
                        start_recording()
            elif args.auto and out_root is not None:
                if cell_complete() and not advance_cell():
                    all_done = True
                if not all_done:
                    env = build()
                    if not args.left_click_to_start:
                        start_recording()
        terr = info.get("tracking_error", 0.0)
        cgap = info.get("command_gap", 0.0)
        label, engaged = _agent_state(env)

        screen.fill((22, 22, 26))
        arena = pygame.Surface((WIN, WIN))
        arena.fill(BG_COLOR)
        _draw_shape_polys(arena, objects[oi], env.goal_pose, GOAL_COLOR, 3)
        _draw_shape_polys(arena, objects[oi], env.object_pose, OBJECT_COLOR, 0)
        _draw_space(arena, env, engaged)
        # The tether's rope is a constraint, not a body, so nothing in the
        # space draws it -- without this the agent looks like a hook that
        # happens to drag things.
        # The rope is real bodies now, so it draws itself in _draw_space; only
        # the final hook link -> object attachment needs a line.
        if getattr(env.agent, "hitched", False):
            px, py = env.agent_pos
            ox, oy, _ = env.object_pose
            pygame.draw.line(
                arena, COL_EXTRA, (px * SCALE, py * SCALE), (ox * SCALE, oy * SCALE), 5
            )
        if getattr(env.agent, "hooked", False):
            pts = env.agent.rope_points()
            ox, oy, _ = env.object_pose
            if pts:
                pygame.draw.line(
                    arena,
                    COL_ENGAGED,
                    (pts[-1][0] * SCALE, pts[-1][1] * SCALE),
                    (ox * SCALE, oy * SCALE),
                    3,
                )
        screen.blit(arena, (0, 0))

        y = WIN + 8
        name = agents[ai]
        screen.blit(
            big.render(
                f"[{ai + 1}] {name}   {'+'.join(spec)}   {objects[oi]}", True, COL_TEXT
            ),
            (10, y),
        )
        cov = f"coverage {coverage:5.3f}"
        if success:
            cov += "   SOLVED"
        screen.blit(
            big.render(cov, True, COL_HUD_OK if success else COL_TEXT), (WIN - 240, y)
        )
        y += 26
        if label:
            screen.blit(
                font.render(label, True, COL_HUD_OK if engaged else COL_DIM), (10, y)
            )
        bits = []
        if "engage" in spec or "jaw" in spec:
            bits.append(ENGAGE_LABEL.get(name, "SPACE engage"))
        if "angle" in spec:
            bits.append(f"A/D angle {math.degrees(angle):4.0f}deg")
        if "grip" in spec:
            m = getattr(env.agent, "mode", "")
            if name == "chain_gripper":
                bits.append("hold SPACE close   hold S open   release = hold gap")
            else:
                bits.append(
                    f"SPACE/WS grip {commanded_grip:.2f}"
                    + (" [held; S releases]" if retain_gripper_grasp else "")
                    + (f" [{m}]" if m else "")
                )
        hint = "   ".join(bits)
        screen.blit(font.render(hint, True, COL_DIM), (170, y))
        y += 20
        if out_root is not None:
            here = completed_here()
            if recording:
                rec = f"REC {steps_rec:4d}"
            elif args.left_click_to_start and not all_done:
                rec = "LEFT CLICK TO START"
            else:
                rec = "idle    "
            rcol = (255, 110, 110) if recording else COL_DIM
            screen.blit(
                font.render(
                    f"{rec}   {here}/{args.per_agent} this agent   {saved} this run",
                    True,
                    rcol,
                ),
                (WIN - 400, y - 26),
            )
            gap_indices = range(len(gaps)) if args.gap_matrix else (gi,)
            remaining = sum(
                1
                for gap_index in gap_indices
                for agent_index in range(len(agents))
                if not cell_complete(gap_index, agent_index, oi)
            )
            screen.blit(
                font.render(
                    f"{remaining} collection cell(s) still short of {args.per_agent}",
                    True,
                    COL_DIM,
                ),
                (WIN - 400, y - 6),
            )
        gname = gaps[gi]
        gcol = COL_TEXT if gname == "ideal" else COL_SENSOR
        init_status = ""
        if current_entry is not None and target_entries is not None:
            entry_number = entry_index_by_seed[int(current_entry["seed"])] + 1
            init_status = (
                f"   L{args.obstacles:02d} init {entry_number:02d}/"
                f"{len(target_entries):02d} seed {int(current_entry['seed'])}"
            )
        screen.blit(
            font.render(
                f"control gap: {gname:<8} track_err {terr:6.2f}"
                f"   cmd_gap {cgap:5.2f}{init_status}",
                True,
                gcol,
            ),
            (10, y),
        )
        y += 20
        screen.blit(
            font.render(
                "[ ] agent   1-9,0,-,= jump   G gap   TAB object   R reset   ESC quit",
                True,
                COL_DIM,
            ),
            (10, y),
        )

        if all_done:
            banner = big.render(
                f"ALL {len(agents) * (len(gaps) if args.gap_matrix else 1)} "
                f"CELLS x {args.per_agent} COLLECTED",
                True,
                COL_HUD_OK,
            )
            screen.blit(banner, (WIN // 2 - banner.get_width() // 2, WIN // 2))

        pygame.display.flip()
        clock.tick(60)

    stop_recording(discard=True)
    if writer is not None:
        writer.close()
    if out_root is not None:
        print(f"[playground] saved {saved} episode(s) under {out_root}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
