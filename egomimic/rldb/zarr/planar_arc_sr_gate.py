"""SR gate: token at every timestep, execute exactly 1 action. Must equal baseline."""
import glob, json, sys, numpy as np, zarr
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v2.generate.mimicgen import SourceDemo, apply_source_control_gap
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizePlanarArcLength
BASE='/Users/rpunamiya/Desktop/GEAR/sim_run'

def mk(ini, acts, agent):
    env = PushShapesEnv(object_shape=ini['object_shape'], pusher_shape=agent,
                        obstacle_level=0, image_size=96)
    env.reset(seed=0)
    apply_source_control_gap(env, SourceDemo(agent=agent, actions=acts,
        object_pose=tuple(ini['object_pose']), goal_pose=tuple(ini['goal_pose']),
        agent_pos=tuple(ini['agent_pos']), agent_angle=float(ini.get('agent_angle',0.)),
        object_shape=ini['object_shape'], obstacle_level=0,
        control_gap=ini.get('control_gap'), control_gap_mode=ini.get('control_gap_mode')))
    env._skip_obs_render = True
    env.set_state(object_pose=tuple(ini['object_pose']), goal_pose=tuple(ini['goal_pose']),
                  agent_pos=tuple(ini['agent_pos']), agent_angle=float(ini.get('agent_angle',0.)))
    return env

def run(ini, acts, agent, tok=None):
    env = mk(ini, acts, agent); C = acts.shape[1]
    for i in range(len(acts)):
        a = acts[i] if tok is None else tok.decode_first_action(tok.tokenize_at(acts, i), C)
        _o,_r,term,_t,_i = env.step(np.asarray(a, dtype=np.float64))
        if term: return True
    return False

EMBS = sys.argv[1].split(','); N = int(sys.argv[2])
cfgs = [('r=0  D50 M50', dict(min_distance_unit=50., resampled_vector_length=50, rotation_radius=0.)),
        ('r=30 D50 M50', dict(min_distance_unit=50., resampled_vector_length=50, rotation_radius=30.)),
        ('r=0  D200 M20', dict(min_distance_unit=200., resampled_vector_length=20, rotation_radius=0.)),
        ('hybrid Dr=0.5', dict(min_distance_unit=50., resampled_vector_length=50, rotation_radius=0., hybrid_rotation_unit=0.5))]
print(f"{'config':<16}" + "".join(f"{e:>11}" for e in EMBS))
base = {}
for e in EMBS:
    eps = []
    for p in sorted(glob.glob(f'{BASE}/ds_src/ideal/{e}/T/*.zarr'))[:N]:
        g=zarr.open(p,mode='r'); n=int(g.attrs['total_frames'])
        if n>=5: eps.append((json.loads(str(g.attrs['episode_init'])), np.asarray(g['actions'])[:n]))
    base[e]=eps
row = f"{'BASELINE raw':<16}"
for e in EMBS: row += f"{100*np.mean([run(i,a,e) for i,a in base[e]]):>10.0f}%"
print(row, flush=True)
for name, kw in cfgs:
    row = f'{name:<16}'
    for e in EMBS:
        t = TokenizePlanarArcLength(**kw)
        row += f"{100*np.mean([run(i,a,e,t) for i,a in base[e]]):>10.0f}%"
    print(row, flush=True)
