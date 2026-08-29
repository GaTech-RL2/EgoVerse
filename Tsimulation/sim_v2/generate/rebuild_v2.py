"""Dedupe a collected control-gap cell and regenerate it with real variety.

Why this exists
---------------
The 6x13x1000 control-gap dataset is only nominally 1000 episodes per cell.
Measured on it, ~70% of every cell has a near-duplicate within 20 units of
path shape, and a cell collapses to 228-339 distinct behaviours at a 60-unit
threshold. Scaling that dataset does not scale the information in it.

This module rebuilds a cell as: its DISTINCT core, plus new demos generated
from that core and admitted only if they are behaviourally new.

Runtime note
------------
This must run against the sim revision that COLLECTED the data (def38a75,
"match collected embodiment physics and control gaps"). Replayed against a
diverged agent implementation, 4 of 13 embodiments desynchronise -- object
paths drift 80-177 units and essentially nothing re-validates -- while the
other 8 stay bit-exact. That asymmetry looks exactly like a dedupe result,
which is why it has to be asserted rather than assumed: see
``check_replay_fidelity``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.sim_v2.generate.diversity import (
    NoveltyFilter, perturb_actions, trajectory_signature,
)
from Tsimulation.sim_v2.generate.mimicgen import (
    GenResult, SourceDemo, apply_source_control_gap, replay,
    sample_equivariant_layout, wrap, _frame_delta,
)
from Tsimulation.sim_v2.generate.reseed import load_cell
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv

_LAYOUT_FAIL = "could not fit a rigidly augmented demonstration inside the arena"


def behaviour_signature(object_xy, pusher_xy) -> np.ndarray:
    """Concatenated object-path + effector-path descriptor.

    The object path alone is a poor diversity measure here: it is the task
    OUTCOME, and many different approaches produce the same push. Measured on
    ideal/umi/T it reports PC95=3 and 49% near-duplicates, while the same
    episodes under object+pusher report PC95=5 and 24.5%. Two episodes that
    move the object identically but approach it differently are different
    training data for a policy, and this signature can tell them apart.

    It is not merely permissive: the degenerate cells still read as
    degenerate under it (ideal/stick 14 distinct of 400, jittery/triangle 82).
    """
    return np.concatenate([trajectory_signature(np.asarray(object_xy)),
                           trajectory_signature(np.asarray(pusher_xy))])


def distinct_subset(records: list[dict], min_distance: float) -> list[dict]:
    """Greedy dedupe of raw records on the behaviour signature."""
    kept: list[dict] = []
    sigs: list[np.ndarray] = []
    for r in records:
        s = behaviour_signature(r["object_xy"], r["pusher_xy"])
        if sigs and np.linalg.norm(np.asarray(sigs) - s[None, :], axis=1).min() < min_distance:
            continue
        kept.append(r)
        sigs.append(s)
    return kept


def to_source(rec: dict) -> SourceDemo:
    """Raw record -> SourceDemo, carrying the episode's own controller mode.

    ``control_gap`` MUST survive this hop. Replaying a demo recorded under a
    laggy or jittery controller inside a clean one desynchronises it.
    """
    ini = rec["init"]
    return SourceDemo(
        agent=str(ini["pusher_shape"]),
        actions=np.asarray(rec["actions"], dtype=np.float64),
        object_pose=tuple(ini["object_pose"]),
        goal_pose=tuple(ini["goal_pose"]),
        agent_pos=tuple(ini["agent_pos"]),
        agent_angle=float(ini.get("agent_angle", 0.0)),
        object_shape=str(ini["object_shape"]),
        obstacle_level=int(ini.get("obstacle_level", 0)),
        control_gap=(dict(ini["control_gap"]) if "control_gap" in ini else None),
        control_gap_mode=(str(ini["control_gap_mode"])
                          if "control_gap_mode" in ini else None),
    )


def generate_diverse(sources: list[SourceDemo], n_attempts: int, *, seed: int,
                     perturb_frac: float = 0.65) -> GenResult:
    """Upstream SE(2) augmentation, plus off-manifold action perturbation.

    Pure SE(2) retargeting cannot raise the intrinsic dimensionality of the
    set: every generated demo is a rigid image of a source, so a model that
    has seen the source has, in effect, seen it. Perturbing the action
    sequence first is what actually adds new behaviour; the rigid transform
    then places it somewhere new.
    """
    rng = np.random.default_rng(seed)
    res = GenResult(attempts=0)
    for i in range(n_attempts):
        src = sources[i % len(sources)]
        res.attempts += 1
        if rng.random() < perturb_frac:
            src = SourceDemo(
                agent=src.agent,
                actions=perturb_actions(src.actions, rng),
                object_pose=src.object_pose, goal_pose=src.goal_pose,
                agent_pos=src.agent_pos, agent_angle=src.agent_angle,
                object_shape=src.object_shape,
                obstacle_level=src.obstacle_level,
                control_gap=src.control_gap,
                control_gap_mode=src.control_gap_mode,
            )
        try:
            obj, goal, agent = sample_equivariant_layout(src, rng)
        except RuntimeError as exc:
            if not str(exc).startswith(_LAYOUT_FAIL):
                raise
            res.layout_failures += 1
            continue
        ok, _cov, played, start = replay(src, obj, goal, new_agent=agent)
        if not ok:
            continue
        _f, rot_obj = _frame_delta(src.object_pose, obj)
        res.demos.append(SourceDemo(
            agent=src.agent, actions=played, object_pose=obj, goal_pose=goal,
            agent_pos=(float(start[0]), float(start[1])),
            agent_angle=wrap(float(src.agent_angle) + rot_obj),
            object_shape=src.object_shape, obstacle_level=src.obstacle_level,
            control_gap=(dict(src.control_gap)
                         if src.control_gap is not None else None),
            control_gap_mode=src.control_gap_mode))
    return res


def _open_writer(out_root: Path, agent: str, shape: str, image_size: int):
    d = Path(out_root) / agent / shape
    d.mkdir(parents=True, exist_ok=True)
    return ZarrDemoWriter(path=d,
                          env_args={"object_shape": shape, "pusher_shape": agent,
                                    "obstacle_level": 0},
                          image_size=image_size)


def _rollout(dm: SourceDemo, agent: str, image_size: int, render: bool):
    """Replay one demo. Returns (reached_goal, object_path, pusher_path, frames).

    ``frames`` is None when render=False. A headless rollout is bit-identical
    to a rendered one -- both reset(seed=0) and reseed the control gap the same
    way -- so a cheap headless pass can decide admission and only survivors pay
    for rendering. Verified on ideal and jittery umi: signature delta 0.
    """
    env = PushShapesEnv(object_shape=dm.object_shape, pusher_shape=agent,
                        obstacle_level=0, image_size=image_size)
    env.reset(seed=0)
    apply_source_control_gap(env, dm)
    if not render:
        env._skip_obs_render = True
    env.set_state(object_pose=dm.object_pose, goal_pose=dm.goal_pose,
                  agent_pos=(float(dm.agent_pos[0]), float(dm.agent_pos[1])),
                  agent_angle=float(dm.agent_angle))
    O, P, F = [], [], ([] if render else None)
    ok = False
    for a in dm.actions:
        a = np.asarray(a, dtype=np.float64)
        obs, r, term, _t, _i = env.step(a)
        px, py = env.agent_pos
        ox, oy, oth = env.object_pose
        O.append((ox, oy)); P.append((px, py))
        if render:
            F.append((obs["image"], np.array([px, py, env.pusher_angle]),
                      np.array([ox, oy, oth]),
                      np.array([a[0], a[1], a[2] if len(a) > 2 else 0.0]),
                      a, r, np.array(env.goal_pose)))
        if term:
            ok = True
            break
    return ok, np.asarray(O), np.asarray(P), F


def _init_for(dm: SourceDemo, agent: str, image_size: int) -> dict:
    env = PushShapesEnv(object_shape=dm.object_shape, pusher_shape=agent,
                        obstacle_level=0, image_size=image_size)
    env.reset(seed=0)
    apply_source_control_gap(env, dm)
    env.set_state(object_pose=dm.object_pose, goal_pose=dm.goal_pose,
                  agent_pos=(float(dm.agent_pos[0]), float(dm.agent_pos[1])),
                  agent_angle=float(dm.agent_angle))
    init_state = env.get_episode_init()
    if dm.control_gap_mode is not None:
        init_state["control_gap_mode"] = dm.control_gap_mode
    return init_state


def render_admit(w, nf: NoveltyFilter, demos: list[SourceDemo], agent: str,
                 image_size: int, stop_at: int | None = None,
                 prescreen: bool = True) -> dict:
    """Admit only episodes that re-validate AND are behaviourally new.

    Most GENERATED candidates are duplicates -- on ideal/L, 1946 of 2467 --
    so admission is decided on a cheap headless rollout (~0.4s) and only
    survivors pay to be rendered (~1.7s). Rendering first cost ~55 minutes of
    pure waste on that cell alone. SOURCES skip the screen (prescreen=False):
    they were already deduped at this radius, so nearly all are admitted and
    screening them is pure overhead -- measured 372s for 120 source episodes.

    Novelty is judged on the object path AND the effector path. The object
    path alone is the task OUTCOME, and many different approaches produce the
    same push: it reports PC95=3 where object+pusher reports 5.

    Re-validation is not a formality for the noisy gaps: ``noise_std > 0``
    controllers were never seeded reproducibly in the collected data, so a
    replay draws a fresh noise sequence. Measured at scale this still keeps
    99.8% of sources, but the ones it drops would otherwise ship with stored
    images that do not match their own actions.
    """
    st = {"offered": len(demos), "failed_replay": 0, "rejected_dup": 0,
          "written": 0}
    for dm in demos:
        if stop_at is not None and len(nf) >= stop_at:
            break
        if prescreen:
            ok, O, P, _ = _rollout(dm, agent, image_size, render=False)
            if not ok or len(O) == 0:
                st["failed_replay"] += 1
                continue
            if not nf.offer_signature(behaviour_signature(O, P)):
                st["rejected_dup"] += 1
                continue
            ok2, _O2, _P2, frames = _rollout(dm, agent, image_size, render=True)
            if not ok2 or not frames:
                # Cannot happen given headless/rendered determinism; guard
                # rather than write a half-episode if that ever breaks.
                st["failed_replay"] += 1
                continue
        else:
            ok2, O, P, frames = _rollout(dm, agent, image_size, render=True)
            if not ok2 or not frames:
                st["failed_replay"] += 1
                continue
            if not nf.offer_signature(behaviour_signature(O, P)):
                st["rejected_dup"] += 1
                continue
        w.start_episode(init_state=_init_for(dm, agent, image_size))
        for img, pobs, oobs, cmd, act, rew, goal in frames:
            w.add_step(image=img, pusher_obs_pose=pobs, object_obs_pose=oobs,
                       pusher_cmd_pose=cmd, action=act, reward=rew,
                       goal_pose=goal)
        if w.steps_in_episode > 0:
            w.commit_episode(); st["written"] += 1
        else:
            w.abort_episode(); st["failed_replay"] += 1
    return st


def check_replay_fidelity(cell: str, n: int = 4, tol: float = 1.0) -> float:
    """Median max object-path deviation when re-running recorded actions.

    Near 0 means this runtime is the one that collected the cell. Anything
    large means the agent implementation has drifted and NOTHING downstream
    of here is meaningful.
    """
    import glob
    import zarr
    devs = []
    for p in sorted(glob.glob(str(Path(cell) / "*.zarr")))[:n]:
        g = zarr.open(p, mode="r")
        f = int(g.attrs["total_frames"])
        if f < 5:
            continue
        ini = json.loads(str(g.attrs["episode_init"]))
        st = np.asarray(g["observations.state"])[:f]
        acts = np.asarray(g["actions"])[:f]
        env = PushShapesEnv(object_shape=ini["object_shape"],
                            pusher_shape=ini["pusher_shape"],
                            obstacle_level=int(ini.get("obstacle_level", 0)),
                            image_size=96)
        env.reset(seed=int(ini["reset_seed"]))
        apply_source_control_gap(
            env, SourceDemo(agent=ini["pusher_shape"], actions=acts,
                            object_pose=tuple(ini["object_pose"]),
                            goal_pose=tuple(ini["goal_pose"]),
                            agent_pos=tuple(ini["agent_pos"]),
                            agent_angle=float(ini.get("agent_angle", 0.0)),
                            object_shape=ini["object_shape"],
                            obstacle_level=int(ini.get("obstacle_level", 0)),
                            control_gap=ini.get("control_gap"),
                            control_gap_mode=ini.get("control_gap_mode")))
        env.set_state(object_pose=tuple(ini["object_pose"]),
                      goal_pose=tuple(ini["goal_pose"]),
                      agent_pos=tuple(ini["agent_pos"]),
                      agent_angle=float(ini.get("agent_angle", 0.0)))
        O = []
        for a in acts:
            env.step(np.asarray(a, dtype=np.float64))
            ox, oy, _ = env.object_pose
            O.append((ox, oy))
        O = np.asarray(O)
        devs.append(float(np.max(np.linalg.norm(O - st[:len(O), 3:5], axis=1))))
    return float(np.median(devs)) if devs else float("nan")


def process_cell(cell: Path, src_root: Path, out_root: Path, *,
                 min_distance: float, target: int, image_size: int,
                 novelty: float, seed: int = 17, gate_tol: float = 1.0,
                 max_rounds: int = 40) -> dict:
    """Rebuild one <gap>/<agent>/<shape> cell to ``target`` distinct episodes."""
    cell = Path(cell)
    gap_dir, agent, shape = Path(cell).relative_to(src_root).parts[:3]
    rel = f"{gap_dir}/{agent}/{shape}"

    # Runtime gate. Probe the DETERMINISTIC sibling (ideal has noise_std=0),
    # because a noisy cell cannot be replayed exactly even on the correct
    # runtime -- so a deviation check there measures the controller's RNG,
    # not whether this code collected the data. On the wrong runtime the
    # drifting embodiments read 80-177 units here; the right one reads 0.
    gate = Path(src_root) / "ideal" / agent / shape
    fid = check_replay_fidelity(str(gate)) if gate.is_dir() else float("nan")
    if not (fid == fid and fid <= gate_tol):
        return {"cell": rel, "read": 0, "distinct": 0, "written": 0,
                "error": f"runtime mismatch for {agent!r}: ideal-cell "
                         f"replay deviates {fid:.1f} > {gate_tol}"}

    recs = load_cell(str(cell))
    if not recs:
        return {"cell": rel, "read": 0, "distinct": 0, "written": 0}
    dis = distinct_subset(recs, min_distance)
    sources = [to_source(r) for r in dis]

    w = _open_writer(Path(out_root) / gap_dir, agent, shape, image_size)
    nf = NoveltyFilter(min_distance=novelty)
    st = render_admit(w, nf, sources, agent, image_size, stop_at=target,
                      prescreen=False)
    st["from_source"] = st["written"]

    # Adaptive exploration schedule.
    #
    # Rigid SE(2) augmentation is the cheap, high-yield generator, but it can
    # only emit rigid images of its sources, so it exhausts: measured on
    # ideal/umi over four rounds of 150 attempts it returned 101, 83, 70, 47
    # new demos -- decaying ~0.75x per round toward a finite total. Action
    # perturbation costs success rate but leaves the rigid-transform manifold,
    # and its yield did NOT decay over the same rounds (50, 48, 49, 53).
    #
    # Success rate is the wrong thing to optimise here: failed attempts are
    # discarded headlessly and cost ~0.4s, while every ACCEPTED demo costs
    # ~1.7s to re-render, so a lower hit rate is a modest constant factor
    # rather than a ceiling. What matters is which generator still produces
    # anything at round 20. So: start cheap, and escalate perturbation as the
    # rigid generator dries up.
    rounds = gen_attempts = gen_ok = 0
    best_round = 0
    frac = 0.15
    yields: list[int] = []
    while len(nf) < target and rounds < max_rounds:
        rounds += 1
        g = generate_diverse(sources, max(200, 3 * (target - len(nf))),
                             seed=seed + 1000 * rounds, perturb_frac=frac)
        gen_attempts += g.attempts
        gen_ok += len(g.demos)
        if not g.demos:
            break
        s2 = render_admit(w, nf, g.demos, agent, image_size, stop_at=target)
        for k in ("offered", "failed_replay", "rejected_dup", "written"):
            st[k] += s2[k]
        yields.append(s2["written"])
        best_round = max(best_round, s2["written"])
        if s2["written"] == 0:
            # Nothing new even at full exploration: this cell's source set is
            # genuinely exhausted at this radius. Stop rather than burn
            # attempts rediscovering manoeuvres already written.
            if frac >= 0.85:
                break
            frac = 0.9
        elif s2["written"] < 0.6 * best_round:
            # Yield is decaying -> shift budget from rigid augmentation to
            # off-manifold perturbation.
            frac = min(0.9, frac + 0.2)
    st["yield_per_round"] = yields
    st["final_perturb_frac"] = round(frac, 2)

    w.close()
    return {"cell": rel, "read": len(recs), "distinct": len(dis),
            "fidelity": fid, "rounds": rounds, "gen_attempts": gen_attempts,
            "gen_ok": gen_ok, "intrinsic_dim": nf.intrinsic_dim(), **st}
