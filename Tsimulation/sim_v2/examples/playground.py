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
    R              reset episode (new layout)
    ESC            quit
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pygame
import pymunk

from Tsimulation.sim_v2.pushshapes.agents import CONTROL_GAPS, VALID_PUSHERS
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v2.pushshapes.render import BG_COLOR, GOAL_COLOR, OBJECT_COLOR
from Tsimulation.sim_v2.pushshapes.shapes import SHAPES

WORLD = 512
SCALE = 1.6
WIN = int(WORLD * SCALE)
HUD_H = 116

COL_PUSHER = (235, 235, 235)
COL_EXTRA = (150, 205, 255)      # agent-owned extra bodies (jaws, 2nd point)
COL_ENGAGED = (120, 240, 150)    # mechanism currently latched/attached
COL_SENSOR = (255, 190, 90)      # non-contact body (magnet, tapper)
COL_TEXT = (228, 228, 228)
COL_DIM = (140, 140, 140)

# What SPACE means, per agent -- shown in the HUD so the control is discoverable.
ENGAGE_LABEL = {
    "gripper": "SPACE close jaws",
    "suction": "SPACE suction on",
    "tether": "SPACE hook rope",
    "magnet": "SPACE magnetise",
    "tapper": "SPACE strike",
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
                0 if not getattr(shape, "sensor", False) else 2,
            )
        elif isinstance(shape, pymunk.Poly):
            pts = [body.local_to_world(v) for v in shape.get_vertices()]
            pts = [(p.x * SCALE, p.y * SCALE) for p in pts]
            if len(pts) >= 3:
                pygame.draw.polygon(
                    surf, colour, pts,
                    0 if not getattr(shape, "sensor", False) else 2,
                )
        elif isinstance(shape, pymunk.Segment):
            pygame.draw.line(
                surf, (110, 110, 120),
                (shape.a.x * SCALE, shape.a.y * SCALE),
                (shape.b.x * SCALE, shape.b.y * SCALE),
                max(2, int(shape.radius * 2 * SCALE)),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="circle", choices=list(VALID_PUSHERS))
    ap.add_argument("--object", default="T", choices=list(SHAPES))
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--gap", default="ideal", choices=list(CONTROL_GAPS))
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN + HUD_H))
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 14)
    big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 19, bold=True)
    clock = pygame.time.Clock()

    gaps = list(CONTROL_GAPS)
    gi = gaps.index(args.gap)
    agents = list(VALID_PUSHERS)
    ai = agents.index(args.agent)
    objects = list(SHAPES)
    oi = objects.index(args.object)

    def build():
        e = PushShapesEnv(
            object_shape=objects[oi], pusher_shape=agents[ai],
            obstacle_level=args.obstacles,
        )
        e.agent.control_gap = CONTROL_GAPS[gaps[gi]]
        e.reset(seed=np.random.randint(0, 10_000))
        e.agent.control_gap = CONTROL_GAPS[gaps[gi]]
        return e

    env = build()
    angle = 0.0
    spread = 34.0
    orbit = math.pi / 2
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    env = build()
                elif ev.key == pygame.K_g:
                    gi = (gi + 1) % len(gaps); env = build()
                elif ev.key == pygame.K_TAB:
                    oi = (oi + 1) % len(objects); env = build()
                elif ev.key == pygame.K_LEFTBRACKET:
                    ai = (ai - 1) % len(agents); env = build()
                elif ev.key == pygame.K_RIGHTBRACKET:
                    ai = (ai + 1) % len(agents); env = build()
                else:
                    keys = "1234567890-="
                    ch = pygame.key.name(ev.key)
                    if ch in keys and keys.index(ch) < len(agents):
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

        _obs, reward, terminated, _trunc, info = env.step(act)
        terr = info.get("tracking_error", 0.0)
        cgap = info.get("command_gap", 0.0)
        label, engaged = _agent_state(env)

        screen.fill((22, 22, 26))
        arena = pygame.Surface((WIN, WIN))
        arena.fill(BG_COLOR)
        _draw_shape_polys(arena, objects[oi], env.goal_pose, GOAL_COLOR, 3)
        _draw_shape_polys(arena, objects[oi], env.object_pose, OBJECT_COLOR, 0)
        _draw_space(arena, env, engaged)
        screen.blit(arena, (0, 0))

        y = WIN + 8
        name = agents[ai]
        screen.blit(big.render(
            f"[{ai + 1}] {name}   {dim}-DOF   {objects[oi]}", True, COL_TEXT), (10, y))
        cov = f"coverage {reward:5.3f}"
        if terminated:
            cov += "   SOLVED"
        screen.blit(big.render(cov, True,
                    COL_ENGAGED if terminated else COL_TEXT), (WIN - 240, y))
        y += 26
        if label:
            screen.blit(font.render(label, True,
                        COL_ENGAGED if engaged else COL_DIM), (10, y))
        hint = ENGAGE_LABEL.get(name, "")
        if name == "two_point":
            hint = f"W/S spread {spread:.0f}   Q/E orbit {math.degrees(orbit):3.0f}deg"
        elif name in ("u_socket", "gripper"):
            hint = f"{hint}   A/D angle {math.degrees(angle):4.0f}deg"
        screen.blit(font.render(hint, True, COL_DIM), (170, y))
        y += 20
        gname = gaps[gi]
        gcol = COL_TEXT if gname == "ideal" else COL_SENSOR
        screen.blit(font.render(
            f"control gap: {gname:<8} track_err {terr:6.2f}   cmd_gap {cgap:5.2f}",
            True, gcol), (10, y))
        y += 20
        screen.blit(font.render(
            "[ ] agent   1-9,0,-,= jump   G gap   TAB object   R reset   ESC quit",
            True, COL_DIM), (10, y))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
