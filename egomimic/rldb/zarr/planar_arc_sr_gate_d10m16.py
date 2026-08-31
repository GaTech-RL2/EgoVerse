"""SR gate for the EXACT tokenizer settings this study trains on.

planar_arc_sr_gate.py covers D50/M50, D200/M20 and a hybrid, but not
D=10 M=16 rotation_radius=0 velocity_layout=append — which is what the control
mode configs use. A tokenizer that fails this trains perfectly well and
produces meaningless SR, so the settings that are actually shipped are the ones
worth gating.

Tokenize at every timestep, execute exactly ONE action, and success rate must
equal the untokenized baseline.
"""
import glob
import json
import sys

import numpy as np
import zarr

from egomimic.rldb.zarr.arc_length_tokenizer import TokenizePlanarArcLength
from Tsimulation.pushshapes.agents import CONTROL_GAPS
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv

BASE = "/Users/rpunamiya/Desktop/GEAR/sim_run"


def mk(ini, acts, agent):
    """Build the env and put it under the episode's own controller.

    planar_arc_sr_gate.py reaches for mimicgen.apply_source_control_gap, which
    lives only in the untracked sim_run/runtime copy — so that gate cannot run
    from a clean checkout. The gap is set directly here instead: it is one
    assignment, and `reset_control_gap` preserves it because randomize_gap is
    False.
    """
    env = PushShapesEnv(object_shape=ini["object_shape"], pusher_shape=agent,
                        obstacle_level=0, image_size=96)
    env.reset(seed=0)
    mode = ini.get("control_gap_mode") or "ideal"
    if mode not in CONTROL_GAPS:
        raise ValueError(f"episode declares unknown control_gap_mode {mode!r}")
    env.agent.control_gap = CONTROL_GAPS[mode]
    env.agent.randomize_gap = False
    env.agent.reset_control_gap(env)
    env._skip_obs_render = True
    env.set_state(object_pose=tuple(ini["object_pose"]),
                  goal_pose=tuple(ini["goal_pose"]),
                  agent_pos=tuple(ini["agent_pos"]),
                  agent_angle=float(ini.get("agent_angle", 0.0)))
    return env


def run(ini, acts, agent, tok=None):
    env = mk(ini, acts, agent)
    C = acts.shape[1]
    for i in range(len(acts)):
        a = (acts[i] if tok is None
             else tok.decode_first_action(tok.tokenize_at(acts, i), C))
        _o, _r, term, _t, _i = env.step(np.asarray(a, dtype=np.float64))
        if term:
            return True
    return False


EMB = sys.argv[1] if len(sys.argv) > 1 else "gripper"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
# The gate reads ds_src (verified clean); ds_gen/ideal is the corrupt cell.
MODE = sys.argv[3] if len(sys.argv) > 3 else "ideal"

eps = []
for p in sorted(glob.glob(f"{BASE}/ds_src/{MODE}/{EMB}/T/*.zarr"))[:N]:
    g = zarr.open(p, mode="r")
    n = int(g.attrs["total_frames"])
    if n >= 5:
        eps.append((json.loads(str(g.attrs["episode_init"])),
                    np.asarray(g["actions"])[:n]))
print(f"{EMB} / {MODE}: {len(eps)} episodes")

baseline = 100 * np.mean([run(i, a, EMB) for i, a in eps])
print(f"{'BASELINE raw':<34}{baseline:>7.0f}%", flush=True)

CFGS = [
    ("SHIPPED D10 M16 r0 append",
     dict(min_distance_unit=10.0, resampled_vector_length=16,
          rotation_radius=0.0, velocity_mode="mean_scalar",
          velocity_layout="append")),
    ("D10 M16 r0 concat",
     dict(min_distance_unit=10.0, resampled_vector_length=16,
          rotation_radius=0.0, velocity_mode="mean_scalar",
          velocity_layout="concat")),
]
ok = True
for name, kw in CFGS:
    tok = TokenizePlanarArcLength(**kw)
    sr = 100 * np.mean([run(i, a, EMB, tok) for i, a in eps])
    flag = "" if abs(sr - baseline) < 1e-9 else "   <-- MISMATCH"
    ok &= not flag
    print(f"{name:<34}{sr:>7.0f}%{flag}", flush=True)

print("\nGATE PASS" if ok else "\nGATE FAIL")
sys.exit(0 if ok else 1)
