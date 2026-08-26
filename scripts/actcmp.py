import numpy as np, torch, sys
sys.path.insert(0,".")
from scripts.train_fpo import load_algo_from_ckpt
from egomimic.algo.fpo.fpo_policy import FPOPolicy
from egomimic.algo.fpo.rollout import _reset_fixed_goal
from egomimic.eval.eval_sim import _OBS_FORMATTERS
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
ckpt=sys.argv[1]; dev=torch.device("cuda")
algo,cfg=load_algo_from_ckpt(ckpt); algo.nets=algo.nets.to(dev); algo.device=dev; algo.nets.eval()
fpo=FPOPolicy(algo, emb_name="pushshapes_sim", device=dev)
# action norm stats shape
ns=algo.norm_stats; eid=get_embodiment_id("pushshapes_sim"); ackey=algo.ac_keys["pushshapes_sim"]
import numpy as np
# probe norm stats via unnormalize of zeros and ones
z=ns.unnormalize({ackey: torch.zeros(8,2,device=dev)}, eid)[ackey]
o=ns.unnormalize({ackey: torch.ones(8,2,device=dev)}, eid)[ackey]
print("unnorm(0) per-pos[:3]=", z[:3].cpu().numpy().round(1).tolist())
print("unnorm(1) per-pos[:3]=", o[:3].cpu().numpy().round(1).tolist(), "-> per-position?", not np.allclose(z[0].cpu(),z[1].cpu()))
from Tsimulation.pushshapes import PushShapesEnv
env=PushShapesEnv(object_shape="T",pusher_shape="circle",obstacle_level=0,image_size=96)
obs,info=_reset_fixed_goal(env,0,[256.,256.,0.7853981634],"pushshapes_sim")
print("seed0 init: agent",np.round(obs["agent_pos"],1),"obj",np.round(obs["object_pose"],1),"cov",round(float(info["coverage"]),3))
data=fpo.build_data(_OBS_FORMATTERS["pushshapes_sim"](obs,dev))
torch.manual_seed(0)
a_norm,val,chunk_world=fpo.sample(data)
print("fpo.sample world chunk (8 actions):")
print(np.round(chunk_world,1).tolist())
print("EV2 reference seed0 first6: [310.4,285.4],[314.6,279.8],[314.7,274.8],[325.1,268.9],[334.7,263.4],[333.4,259.9]")
