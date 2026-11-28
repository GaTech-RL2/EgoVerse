import argparse,json,secrets
from pathlib import Path
import numpy as np,torch
from campaign.cart_sim import CampaignSimPolicy,PROPRIO
from campaign.policy import CampaignPolicy
from campaign.data import sha
from egomimic.serving.websocket_policy_server import WebsocketPolicyServer
ROOT=Path(__file__).resolve().parent
class RobotPolicy(CampaignSimPolicy):
 def observations(self,obs):
  obs=dict(obs)
  for raw,key in PROPRIO.items():
   x=np.asarray(obs[raw],np.float32).copy()
   norm=self.normalizer[key];lo=np.asarray(norm['quantile_1']);hi=np.asarray(norm['quantile_99'])
   if x.shape!=lo.shape or not np.isfinite(x).all():raise ValueError('Invalid '+raw)
   x[hi-lo<1e-6]=lo[hi-lo<1e-6];obs[raw]=x
  return super().observations(obs)
 def infer(self,obs):
  if '_sim_observe' in obs:raise ValueError('Hardware server expects client-built clouds')
  request=dict(obs)
  request.setdefault('_simulation_seed',secrets.randbits(32))
  result=super().infer(request);result['embodiment']='rby1';return result
 def infer_batch(self,observations):return [self.infer(x) for x in observations]
def load(device='cuda'):
 cfgpath=ROOT/'configs/training.json';normpath=ROOT/'normalizer.json'
 cfg=json.loads(cfgpath.read_text());norm=json.loads(normpath.read_text())
 ck=torch.load(ROOT/'checkpoint.pt',map_location='cpu',weights_only=False)
 assert ck['step']==10000 and ck['identity']['config_sha256']==sha(cfgpath) and ck['identity']['normalizer_sha256']==sha(normpath)
 assert ck['original_checkpoint_sha256']=='44406cd6dc07e57bf73cc35509fb73be7cc806c774c217bea06fccf4f07f6bf9'
 robot=json.loads((ROOT/'assets/robot.json').read_text());robot['urdf']=str(ROOT/'assets/robot.urdf')
 model=CampaignPolicy(ROOT/'configs/hd2-reference.yaml','B',norm,robot)
 model.load_state_dict(ck['model'],strict=True);del ck
 model.policy.heads['rby1'].num_inference_steps=16
 model.to(device).eval();torch.set_num_threads(2)
 return RobotPolicy(model,norm,'B')
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8270);p.add_argument('--device',default='cuda');a=p.parse_args()
 policy=load(a.device)
 meta=dict(embodiment='rby1',action_horizon=64,action_dim=49,robot0_joint_pos_dim=22,
  camera_keys=['front_pcd_1','front_pcd_2','front_pcd_3'],proprio_keys=list(PROPRIO),rtc=True,
  inference_device=a.device,num_inference_steps=16,pin_constant_proprio=True,abl='TGZ10HD2',virtual_view=True,
  checkpoint_sha256=sha(ROOT/'checkpoint.pt'),normalizer_sha256=sha(ROOT/'normalizer.json'))
 print(json.dumps(meta),flush=True)
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=meta).serve_forever()
if __name__=='__main__':main()
