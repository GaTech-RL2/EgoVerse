"""Is u_socket replay repeatable WITHIN a process and ACROSS processes?"""
import sys, json, os
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")
import numpy as np, zarr
from Tsimulation.collect.zarr_writer import ACTION_KEY, STATE_KEY
from Tsimulation.pushshapes.env import PushShapesEnv

D = "/coc/flash7/paphiwetsa3/datasets/u_socket_3000"
def once(idx):
    p = os.path.join(D, f"episode_T_u_socket_obs0_{idx}.zarr")
    s = zarr.open_group(p, mode="r"); a = dict(s.attrs); tf = a.get("total_frames")
    acts = np.asarray(s[ACTION_KEY][:])[:tf]; st = np.asarray(s[STATE_KEY][:])[:tf]
    ei = json.loads(a["episode_init"]) if isinstance(a["episode_init"], str) else a["episode_init"]
    ea = (json.loads(a["task_description"]) if isinstance(a["task_description"], str) else a["task_description"])["env_args"]
    env = PushShapesEnv(object_shape=ea["object_shape"], pusher_shape=ea["pusher_shape"],
                        obstacle_level=ea.get("obstacle_level",0), image_size=ea.get("image_size",96))
    env._skip_obs_render = True
    env.reset(seed=ei.get("reset_seed"))
    env.set_state(agent_pos=tuple(ei["agent_pos"]), agent_angle=float(ei.get("agent_angle",0.0)),
                  object_pose=tuple(ei["object_pose"]), goal_pose=tuple(ei["goal_pose"]))
    dmax=0.0; latch=None; cov=0.0
    for i in range(len(acts)):
        obs,_,term,_,info = env.step(acts[i])
        cov=max(cov,info["coverage"])
        if env.socket_latched and latch is None: latch=i
        if i+1<len(st):
            live=np.concatenate([obs["agent_pos"],obs["agent_angle"],obs["object_pose"]])
            dmax=max(dmax,float(np.linalg.norm(st[i+1]-live)))
        if term: break
    env.close()
    return round(dmax,4), latch, round(cov,4)

EPS = ["001228","000433","002688","000860","000000"]
print("PYTHONHASHSEED =", os.environ.get("PYTHONHASHSEED"))
for idx in EPS:
    runs = [once(idx) for _ in range(3)]
    same = len(set(runs)) == 1
    print(f"  ep{idx}: runs={runs}  within_process_repeatable={same}")
