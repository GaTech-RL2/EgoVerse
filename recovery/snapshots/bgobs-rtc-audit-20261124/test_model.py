"""Actual FT10k model, CPU, recorded observations only; no robot connection."""
import sys,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
B=Path('/coc/flash7/cxin34/egoa180-cart-fix-FT10k-robotpc-20261124')
sys.path.insert(0,str(B))
from serve_robotpc import load
policy=load('cpu')
z=np.load(B/'testdata/reference.npz',allow_pickle=False)
keys=['front_pcd_1','front_pcd_2','front_pcd_3','robot0_joint_pos','eef_pose_glass','hand_left_qpos','hand_right_qpos']
results=[]
for i in range(3):
    obs={k:z[str(i)+'_'+k] for k in keys};obs['_simulation_seed']=123+i
    start=time.time();a=np.asarray(policy.infer(obs)['actions'])
    assert a.shape==(64,49) and np.isfinite(a).all()
    obs.update(rtc_prev_actions=a[13:],rtc_freeze_steps=12,rtc_soft_horizon=21,rtc_soft_power=1.,_simulation_seed=223+i)
    b=np.asarray(policy.infer(obs)['actions'])
    error=float(abs(b[:12]-a[13:25]).max())
    assert b.shape==(64,49) and np.isfinite(b).all() and error<1e-5,error
    results.append(dict(fixture=i,freeze_max_error=error,seconds=time.time()-start))
    print(results[-1],flush=True)
(R/'model-results.json').write_text(json.dumps(dict(passed=True,device='cpu',freeze=12,period_tail_offset=13,soft=21,results=results),indent=2))
