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
    BACKSPACE      discard the take in progress
    R              reset episode (new layout)
    ESC            quit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pygame
import pymunk

from Tsimulation.sim_v2.pushshapes.agents import (
    CONTROL_GAPS,
    NEW_AGENTS,
    VALID_PUSHERS,
)
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v2.pushshapes.render import BG_COLOR, GOAL_COLOR, OBJECT_COLOR
from Tsimulation.sim_v2.pushshapes.render import to_image_obs
from Tsimulation.sim_v2.pushshapes.shapes import SHAPES
from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter

WORLD = 512
# 1.6 made the window 935 px tall, so the HUD fell off the bottom of a laptop
# screen. 1.25 keeps the whole thing (640 + 116) inside a 13" display.
SCALE = 1.25
WIN = int(WORLD * SCALE)
HUD_H = 116

# The arena background is (240, 240, 240) -- near white. These must be DARK
# to read against it; the first version used (235, 235, 235) for the pusher,
# one step off the background, which made it effectively invisible.
COL_PUSHER = (200, 40, 40)       # matches render.PUSHER_COLOR
COL_EXTRA = (225, 110, 20)       # agent-owned extra bodies (jaws, 2nd point)
COL_ENGAGED = (0, 150, 60)       # mechanism currently latched/attached
COL_SENSOR = (170, 90, 0)        # non-contact body (magnet, tapper)
COL_TEXT = (228, 228, 228)
COL_HUD_OK = (110, 230, 140)     # green on the DARK hud panel, not the arena
COL_DIM = (140, 140, 140)

# What SPACE means, per agent -- shown in the HUD so the control is discoverable.
ENGAGE_LABEL = {
    "gripper": "SPACE close jaws",
    "suction": "SPACE suction on",
    "tether": "SPACE hook rope",
    "magnet": "SPACE magnetise",
    "tapper": "SPACE strike",
    "wrench": "auto-couples in range",
    "towbar": "SPACE hitch / unhitch",
    "u_socket": "SPACE (n/a)",
}


def _agent_state(env) -> tuple[str, bool]:
    """Human-readable mechanism state, and whether it is currently engaged."""
    a = env.agent
    for attr, name in (
        ("grasped", "GRASPED"),
        ("attached", "ATTACHED"),
        ("hooked", "HOOKED"),
        ("socket_latched", "LATCHED"),
    ):
        if hasattr(a, attr):
            on = bool(getattr(a, attr))
            return (name if on else name.lower()), on
    return "", False


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
                surf, colour, (int(c.x * SCALE), int(c.y * SCALE)),
                max(2, int(shape.radius * SCALE)),
                0 if not getattr(shape, "sensor", False) else 3,
            )
        elif isinstance(shape, pymunk.Poly):
            pts = [body.local_to_world(v) for v in shape.get_vertices()]
            pts = [(p.x * SCALE, p.y * SCALE) for p in pts]
            if len(pts) >= 3:
                pygame.draw.polygon(
                    surf, colour, pts,
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
                surf, (110, 110, 120) if wall else colour,
                (wa.x * SCALE, wa.y * SCALE), (wb.x * SCALE, wb.y * SCALE),
                max(3, int(shape.radius * 2 * SCALE)),
            )


def _draw_shape_polys(surf, shape_name, pose, colour, width=0):
    cx0, cy0, th = pose
    ct, st = math.cos(th), math.sin(th)
    for cx, cy, w, h in SHAPES[shape_name]:
        verts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, ly = cx + sx * w / 2, cy + sy * h / 2
            verts.append((
                (cx0 + lx * ct - ly * st) * SCALE,
                (cy0 + lx * st + ly * ct) * SCALE,
            ))
        pygame.draw.polygon(surf, colour, verts, width)


def main() -> int:
    # Declared up front: `global` must precede every use of the name in this
    # scope, and argparse's default=SCALE reads it near the top. ast.parse()
    # accepts the wrong order -- it is a compile error, not a parse error --
    # so validate with compile(), not ast.parse().
    global SCALE, WIN

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="circle", choices=list(VALID_PUSHERS))
    ap.add_argument("--object", default="T", choices=list(SHAPES))
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--gap", default="ideal",
                    choices=list(CONTROL_GAPS) + ["random"],
                    help="'random' draws a fresh compliance every episode")
    ap.add_argument("--output", default=None,
                    help="record episodes to <output>/<agent>/<object>/*.zarr")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--min-frames", type=int, default=60,
                    help="takes shorter than this are discarded, not committed")
    ap.add_argument("--per-agent", type=int, default=10,
                    help="episodes to collect per embodiment before advancing")
    ap.add_argument("--scale", type=float, default=SCALE,
                    help="window scale; lower it if the HUD is off-screen")
    ap.add_argument("--agents", default="all",
                    help="'all', 'new' (the 7 behaviourally-distinct ones), or a "
                         "comma-separated list. Restricts both the [ ] cycle and "
                         "--auto advancement.")
    ap.add_argument("--auto", action="store_true",
                    help="re-arm recording after every save and advance to the "
                         "next embodiment once --per-agent is reached")
    args = ap.parse_args()

    SCALE = float(args.scale)
    WIN = int(WORLD * SCALE)

    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN + HUD_H))
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 14)
    big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 19, bold=True)
    clock = pygame.time.Clock()

    gaps = list(CONTROL_GAPS) + ["random"]
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
    objects = list(SHAPES)
    oi = objects.index(args.object)

    def apply_gap(e):
        name = gaps[gi]
        e.agent.randomize_gap = name == "random"
        if not e.agent.randomize_gap:
            e.agent.control_gap = CONTROL_GAPS[name]

    def build():
        e = PushShapesEnv(
            object_shape=objects[oi], pusher_shape=agents[ai],
            obstacle_level=args.obstacles, image_size=args.image_size,
        )
        apply_gap(e)
        # Seed explicitly: it is the only thing needed to reproduce BOTH the
        # layout and the sampled compliance, and it goes into episode_init.
        e.reset(seed=int(np.random.randint(0, 10_000)))
        apply_gap(e)
        return e

    out_root = Path(args.output) if args.output else None
    writer = None
    writer_key = [None]
    recording = False
    saved = 0
    steps_rec = 0

    # ONE writer per output dir, not per episode: the writer owns episode
    # naming and index resumption (episode_<obj>_<pusher>_obs<N>_NNNNNN.zarr).
    # Handing it a file path instead produced a directory zarr could not open.
    def saved_here() -> int:
        """Episodes already on disk for this (agent, object).

        Counted from disk rather than a session counter so relaunching resumes
        where you stopped instead of collecting a second set of ten.
        """
        if out_root is None:
            return 0
        return get_writer().existing_episode_count()

    def get_writer():
        nonlocal writer
        key = (agents[ai], objects[oi])
        if writer is not None and writer_key[0] == key:
            return writer
        if writer is not None:
            writer.close()
        d = out_root / agents[ai] / objects[oi]
        d.mkdir(parents=True, exist_ok=True)
        writer = ZarrDemoWriter(
            path=d,
            env_args={
                "object_shape": objects[oi], "pusher_shape": agents[ai],
                "obstacle_level": args.obstacles,
            },
            image_size=args.image_size,
        )
        writer_key[0] = key
        return writer

    def advance_agent() -> bool:
        """Move to the next embodiment that still needs episodes.

        Returns False when every embodiment has met the quota.
        """
        nonlocal ai
        for _ in range(len(agents)):
            ai = (ai + 1) % len(agents)
            if saved_here() < args.per_agent:
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
        w.start_episode(init_state=env.get_episode_init())
        recording, steps_rec = True, 0

    def stop_recording(discard=False):
        nonlocal recording, saved, steps_rec
        if writer is None or not writer.is_recording:
            recording, steps_rec = False, 0
            return
        if discard or steps_rec < args.min_frames:
            # Too short to be a demonstration -- a stray ENTER produced a
            # 1-frame episode that would pollute training as surely as a
            # runaway does.
            writer.abort_episode()
        else:
            # commit_episode, NOT close: close() calls abort_episode() and
            # silently discards the take.
            writer.commit_episode()
            saved += 1
        recording, steps_rec = False, 0

    all_done = False
    angle = 0.0
    spread = 34.0
    orbit = math.pi / 2
    running = True

    env = build()
    if args.auto and out_root is not None:
        if saved_here() >= args.per_agent and not advance_agent():
            all_done = True
        env = build()
        if not all_done:
            start_recording()

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    stop_recording(discard=True); env = build()
                elif ev.key == pygame.K_RETURN:
                    if recording:
                        stop_recording()
                    else:
                        start_recording()
                elif ev.key == pygame.K_BACKSPACE:
                    stop_recording(discard=True)
                elif ev.key == pygame.K_g:
                    stop_recording(discard=True); gi = (gi + 1) % len(gaps); env = build()
                elif ev.key == pygame.K_TAB:
                    stop_recording(discard=True); oi = (oi + 1) % len(objects); env = build()
                elif ev.key == pygame.K_LEFTBRACKET:
                    stop_recording(discard=True); ai = (ai - 1) % len(agents); env = build()
                elif ev.key == pygame.K_RIGHTBRACKET:
                    stop_recording(discard=True); ai = (ai + 1) % len(agents); env = build()
                else:
                    keys = "1234567890-="
                    ch = pygame.key.name(ev.key)
                    if ch in keys and keys.index(ch) < len(agents):
                        stop_recording(discard=True)
                        ai = keys.index(ch); env = build()

        held = pygame.key.get_pressed()
        if held[pygame.K_a]:
            angle -= 0.06
        if held[pygame.K_d]:
            angle += 0.06
        if held[pygame.K_w]:
            spread = min(160.0, spread + 1.5)
        if held[pygame.K_s]:
            spread = max(12.0, spread - 1.5)
        if held[pygame.K_q]:
            orbit -= 0.05
        if held[pygame.K_e]:
            orbit += 0.05
        engage = 1.0 if held[pygame.K_SPACE] else 0.0

        mx, my = pygame.mouse.get_pos()
        wx = float(np.clip(mx / SCALE, 0, WORLD))
        wy = float(np.clip(my / SCALE, 0, WORLD))

        dim = env.agent.action_dim
        act = np.zeros(dim, dtype=np.float64)
        act[0], act[1] = wx, wy
        if agents[ai] == "two_point":
            act[2] = wx + math.cos(orbit) * spread
            act[3] = wy + math.sin(orbit) * spread
        elif dim == 4:                      # gripper: x, y, angle, jaw
            act[2] = angle
            act[3] = 0.0 if engage else 1.0  # SPACE closes
        elif dim == 3:
            act[2] = angle if agents[ai] == "u_socket" else engage

        obs, reward, terminated, _trunc, info = env.step(act)
        if recording and writer is not None:
            px, py = env.agent_pos
            ox, oy, oth = env.object_pose
            writer.add_step(
                image=obs["image"] if "image" in obs else to_image_obs(
                    pygame.display.get_surface(), args.image_size),
                pusher_obs_pose=np.array([px, py, env.pusher_angle]),
                object_obs_pose=np.array([ox, oy, oth]),
                pusher_cmd_pose=np.array([act[0], act[1],
                                          act[2] if dim >= 3 else 0.0]),
                action=act, reward=reward, goal_pose=np.array(env.goal_pose),
            )
            steps_rec += 1
        # Auto-stop on success so a solved demo is never lost by forgetting to
        # press ENTER, and immediately reset for the next one.
        if recording and terminated:
            stop_recording()
            if args.auto and out_root is not None:
                if saved_here() >= args.per_agent and not advance_agent():
                    all_done = True
                if not all_done:
                    env = build()
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
            pygame.draw.line(arena, COL_EXTRA,
                             (px * SCALE, py * SCALE), (ox * SCALE, oy * SCALE), 5)
        if getattr(env.agent, "hooked", False):
            pts = env.agent.rope_points()
            ox, oy, _ = env.object_pose
            if pts:
                pygame.draw.line(arena, COL_ENGAGED,
                                 (pts[-1][0] * SCALE, pts[-1][1] * SCALE),
                                 (ox * SCALE, oy * SCALE), 3)
        screen.blit(arena, (0, 0))

        y = WIN + 8
        name = agents[ai]
        screen.blit(big.render(
            f"[{ai + 1}] {name}   {dim}-DOF   {objects[oi]}", True, COL_TEXT), (10, y))
        cov = f"coverage {reward:5.3f}"
        if terminated:
            cov += "   SOLVED"
        screen.blit(big.render(cov, True,
                    COL_HUD_OK if terminated else COL_TEXT), (WIN - 240, y))
        y += 26
        if label:
            screen.blit(font.render(label, True,
                        COL_HUD_OK if engaged else COL_DIM), (10, y))
        hint = ENGAGE_LABEL.get(name, "")
        if name == "two_point":
            hint = f"W/S spread {spread:.0f}   Q/E orbit {math.degrees(orbit):3.0f}deg"
        elif name in ("u_socket", "gripper"):
            hint = f"{hint}   A/D angle {math.degrees(angle):4.0f}deg"
        screen.blit(font.render(hint, True, COL_DIM), (170, y))
        y += 20
        if out_root is not None:
            here = saved_here()
            rec = f"REC {steps_rec:4d}" if recording else "idle    "
            rcol = (255, 110, 110) if recording else COL_DIM
            screen.blit(font.render(
                f"{rec}   {here}/{args.per_agent} this agent   {saved} this run",
                True, rcol), (WIN - 400, y - 26))
            remaining = sum(
                1 for a in agents
                if len(list((out_root / a / objects[oi]).glob("*.zarr")))
                < args.per_agent
            )
            screen.blit(font.render(
                f"{remaining} embodiment(s) still short of {args.per_agent}",
                True, COL_DIM), (WIN - 400, y - 6))
        gname = gaps[gi]
        gcol = COL_TEXT if gname == "ideal" else COL_SENSOR
        screen.blit(font.render(
            f"control gap: {gname:<8} track_err {terr:6.2f}   cmd_gap {cgap:5.2f}",
            True, gcol), (10, y))
        y += 20
        screen.blit(font.render(
            "[ ] agent   1-9,0,-,= jump   G gap   TAB object   R reset   ESC quit",
            True, COL_DIM), (10, y))

        if all_done:
            banner = big.render(
                f"ALL {len(agents)} EMBODIMENTS x {args.per_agent} COLLECTED",
                True, COL_HUD_OK)
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
