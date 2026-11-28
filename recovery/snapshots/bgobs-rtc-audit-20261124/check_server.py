import argparse,json,socket,time,hashlib
from pathlib import Path
import numpy as np,msgpack_numpy
from websockets.sync.client import connect
ROOT=Path(__file__).resolve().parent
def check(port=8176,report=None):
 z=np.load(ROOT/'testdata/reference.npz',allow_pickle=False)
 keys=['front_pcd_1','front_pcd_2','front_pcd_3','robot0_joint_pos','eef_pose_glass','hand_left_qpos','hand_right_qpos']
 errors=[]
 with connect('ws://127.0.0.1:'+str(port),max_size=None,compression=None) as ws:
  meta=msgpack_numpy.unpackb(ws.recv())
  assert meta['checkpoint_sha256']==hashlib.sha256((ROOT/'checkpoint.pt').read_bytes()).hexdigest()
  assert meta['normalizer_sha256']==hashlib.sha256((ROOT/'normalizer.json').read_bytes()).hexdigest()
  assert meta['abl']=='TGZ10HD2' and meta['virtual_view']
  assert meta['action_dim']==49 and meta['action_horizon']==64 and meta['robot0_joint_pos_dim']==22 and meta['pin_constant_proprio']
  # CPU fixtures are portable; compare on the serving device to avoid CPU/CUDA RNG differences.
  gpu_reference=None
  if meta.get('inference_device','cuda')!='cpu':
   from serve_robotpc import load
   from campaign.cart_sim import CampaignSimPolicy,PROPRIO
   direct=load(device=meta['inference_device'])
   gpu_reference=CampaignSimPolicy(direct.model,direct.normalizer,'B')
  def reference(obs,i):
   if gpu_reference is None:return z[str(i)+'_plain'],z[str(i)+'_rtc']
   pinned=dict(obs)
   for raw,key in PROPRIO.items():
    norm=gpu_reference.normalizer[key];lo=np.asarray(norm['quantile_1']);hi=np.asarray(norm['quantile_99'])
    x=np.asarray(obs[raw],np.float32).copy();x[hi-lo<1e-6]=lo[hi-lo<1e-6];pinned[raw]=x
   pinned['_simulation_seed']=123+i
   plain=gpu_reference.infer(pinned)['actions']
   pinned.update(rtc_prev_actions=plain,rtc_freeze_steps=8,rtc_soft_horizon=13,rtc_soft_power=1.,_simulation_seed=223+i)
   return plain,gpu_reference.infer(pinned)['actions']
  for i in range(3):
   obs={k:z[str(i)+'_'+k] for k in keys};obs['_simulation_seed']=123+i
   ref,rtc_ref=reference(obs,i)
   ws.send(msgpack_numpy.packb(obs));data=ws.recv();assert not isinstance(data,str),data
   act=msgpack_numpy.unpackb(data)['actions']
   assert act.shape==(64,49) and np.isfinite(act).all()
   err=np.abs(act-ref);assert err.mean()<.001 and err.max()<.01,(i,float(err.mean()),float(err.max()))
   obs.update(rtc_prev_actions=ref,rtc_freeze_steps=8,rtc_soft_horizon=13,rtc_soft_power=1.,_simulation_seed=223+i)
   ws.send(msgpack_numpy.packb(obs));data=ws.recv();assert not isinstance(data,str),data
   rtc=msgpack_numpy.unpackb(data)['actions'];diff=np.abs(rtc-rtc_ref)
   assert rtc.shape==(64,49) and np.isfinite(rtc).all()
   assert diff.mean()<.001 and diff.max()<.01
   np.testing.assert_allclose(rtc[:8],ref[:8],rtol=1e-5,atol=1e-6)
   errors.append(dict(frame=i,plain_mean=float(err.mean()),plain_max=float(err.max()),rtc_max=float(diff.max())))
  obs.pop('_simulation_seed');ws.send(msgpack_numpy.packb(obs));data=ws.recv();assert not isinstance(data,str),data
  assert np.isfinite(msgpack_numpy.unpackb(data)['actions']).all()
 result=dict(passed=True,host=socket.gethostname(),no_hardware_access=True,protocol='hardware obs without simulation seed supported',cases=errors,metadata=meta)
 if report:Path(report).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8176);a=p.parse_args()
 check(a.port,ROOT/'reports/robotpc-selftest.json')
