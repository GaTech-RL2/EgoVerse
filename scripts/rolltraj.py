import numpy as np, torch, sys
sys.path.insert(0,".")
from scripts.train_fpo import load_algo_from_ckpt
from egomimic.algo.fpo.fpo_policy import FPOPolicy
from egomimic.algo.fpo.rollout import _reset_fixed_goal
from egomimic.eval.eval_sim import _OBS_FORMATTERS
ckpt=sys.argv[1]; dev=torch.device("cuda")
algo,cfg=load_algo_from_ckpt(ckpt); algo.nets=algo.nets.to(dev); algo.device=dev; algo.nets.eval()
fpo=FPOPolicy(algo,emb_name="pushshapes_sim",device=dev)
from Tsimulation.pushshapes import PushShapesEnv
print("SUCCESS_THRESHOLD=", getattr(PushShapesEnv,"SUCCESS_THRESHOLD","?"))
env=PushShapesEnv(object_shape="T",pusher_shape="circle",obstacle_level=0,image_size=96)
GOAL=[256.,256.,0.7853981634]
for s in [0,1]:
    obs,info=_reset_fixed_goal(env,s,GOAL,"pushshapes_sim")
    maxc=float(info["coverage"]); finalc=maxc; t=0; term=False
    while t<500:
        oz=_OBS_FORMATTERS["pushshapes_sim"](obs,dev); data=fpo.build_data(oz)
        _,_,cw=fpo.sample(data)
        for j in range(8):
            obs,r,term,trunc,info=env.step(cw[j]); c=float(info["coverage"]); maxc=max(maxc,c); finalc=c
            if term or trunc: break
        t+=8
        if t%80==0 or term: print(f"  s{s} t{t} cov={c:.3f} max={maxc:.3f} obj={np.round(obs['object_pose'],0)}")
        if term: print(f"  s{s} TERMINATED t{t}"); break
    print(f"seed {s}: FINAL={finalc:.3f} MAX={maxc:.3f}")
