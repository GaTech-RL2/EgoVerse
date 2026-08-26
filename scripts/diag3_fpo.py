import numpy as np, torch, sys
from egomimic.eval.core.ckpt_loading import load_algo_from_ckpt
from egomimic.eval.core.eval_sim import _ENV_TO_ZARR
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
ckpt=sys.argv[1]; dev=torch.device("cuda")
algo,cfg=load_algo_from_ckpt(ckpt); algo.nets=algo.nets.to(dev); algo.device=dev; algo.nets.eval(); algo.replan_every=8
emb="pushshapes_sim"; eid=get_embodiment_id(emb); conv=_ENV_TO_ZARR[emb]
from Tsimulation.pushshapes import PushShapesEnv
env=PushShapesEnv(object_shape="T",pusher_shape="circle",obstacle_level=0,image_size=96)
GOAL=np.array([256.0,256.0,0.7853981633974483],dtype=np.float32)
obs,info=env.reset(seed=0)
env.set_state(agent_pos=obs["agent_pos"],object_pose=obs["object_pose"],goal_pose=GOAL)
print("after pin: goal_pose=",env._goal_pose, "obj=",obs["object_pose"][:3])
for t in range(500):
    obs=env._get_obs(); oz=conv(obs,dev)
    a=algo.inference_step(oz,t,eid)
    obs,r,term,trunc,info=env.step(np.asarray(a,dtype=np.float32).reshape(-1)[:2])
    if t in (0,50,100,200,300,499):
        print(f"t={t} obj={np.round(obs['object_pose'],1)} cov={info['coverage']:.3f} goal={np.round(env._goal_pose,1)}")
    if term: print("TERM t",t); break
# also try: does the env keep the pinned goal across steps?
print("final goal_pose", env._goal_pose)
