"""Simulation-only bridge for campaign checkpoints and the native Cart twin protocol."""
import argparse,importlib.util,json,random,sys,types
from pathlib import Path
import numpy as np
import torch
from .policy import CampaignPolicy
from .gaze_policy import GazePolicy
from .gaze_preprocessing import lift_sensor,uniform_clean_cloud
from .data import sha
from .contracts import check_normalizer

LEGACY=Path('/coc/flash7/czhang883/sim_twins/cart_round3_policies/code/serve_r3_hd2.py')
VV_PATH=Path(__file__).resolve().parents[1]/'assets/virtual_cam_basket.npz'
TRANSPORT=Path('/coc/flash7/czhang883/warp-data/pipelines/cart_1102_ptfwam/code/EgoVerse_cart1102_hd2/egomimic/serving/websocket_policy_server.py')
PROPRIO={'robot0_joint_pos':'state_robot0_joint_pos','eef_pose_glass':'state_eef_pose_glass',
         'hand_left_qpos':'state_hand_left_qpos','hand_right_qpos':'state_hand_right_qpos'}

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def rtc_inference(head,noise,features,previous,freeze,soft,power):
    """Original split-head Euler loop with explicit normalized RTC targets."""
    length=min(len(previous),noise.shape[1]);freeze=min(int(freeze),length);soft=min(int(soft),length)
    if not 0<=freeze<=soft or power<=0:raise ValueError('Invalid RTC window')
    target=torch.zeros_like(noise);target[:,:length]=previous[:length]
    weights=torch.zeros((1,noise.shape[1],1),device=noise.device,dtype=noise.dtype)
    weights[:,:freeze]=1
    if soft>freeze:
        weights[:,freeze:soft,0]=(1-torch.arange(soft-freeze,device=noise.device)/(soft-freeze)).pow(power)
    x=noise;dt=-1./head.num_inference_steps;t=torch.ones(len(noise),device=noise.device)
    while t[0]>=-dt/2:
        x=weights*(t[:,None,None]*noise+(1-t[:,None,None])*target)+(1-weights)*x
        parents=[None]*head.n_blocks;velocity=[]
        for i,sl in enumerate(head.block_slices):
            v=head.block_models[i](x[...,sl],t,head._build_cond(features,parents,i))
            parents[i]=x[...,sl]-t[:,None,None]*v;velocity.append(v)
        x=x+dt*torch.cat(velocity,-1);t=t+dt
    x[:,:freeze]=target[:,:freeze]
    return x

class CampaignSimPolicy:
    def __init__(self,model,normalizer,arm):
        self.model=model.eval();self.normalizer=normalizer;self.arm=arm
        self.device=next(model.parameters()).device

    def observations(self,obs):
        forbidden={'gaze_uv','gaze_valid','gaze_timestamp'}&obs.keys()
        if forbidden:raise ValueError('Ground-truth gaze is prohibited at inference: '+str(forbidden))
        batch={}
        for i,n in enumerate((2048,1024,1024),1):
            k=f'front_pcd_{i}';x=np.asarray(obs[k],np.float32)
            if x.shape!=(n,6) or not np.isfinite(x).all():raise ValueError('Invalid '+k)
            batch[k]=torch.as_tensor(x.reshape(1,1,1,-1),device=self.device)
        for raw,key in PROPRIO.items():
            x=np.asarray(obs[raw],np.float32);norm=self.normalizer[key]
            if x.shape!=(len(norm['center']),) or not np.isfinite(x).all():raise ValueError('Invalid '+raw)
            x=(x-np.asarray(norm['center'],np.float32))/np.asarray(norm['scale'],np.float32)
            batch[key]=torch.as_tensor(x[None,None],device=self.device)
        if self.arm=='E':
            x=np.asarray(obs['clean_cloud'],np.float32)
            if x.shape!=(2048,6) or not np.isfinite(x).all():raise ValueError('Invalid clean sensor view')
            batch['clean_cloud']=torch.as_tensor(x[None],device=self.device)
        return batch

    @torch.inference_mode()
    def infer(self,obs):
        batch=self.observations(obs);seed=int(obs['_simulation_seed'])
        if not 0<=seed<2**32:raise ValueError('Invalid simulation seed')
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            m=self.model
            if self.arm=='E':features,gaze=m.predicted_features(m.encode(batch))
            else:
                m.policy.device=self.device;m.policy.train_mode=False
                features,_=m.policy.forward_features('rby1',batch);gaze=None
            # Dedicated noise generator makes paired flow noise independent of extra E work.
            gen=torch.Generator(device=self.device).manual_seed(seed)
            noise=torch.randn((1,64,49),generator=gen,device=self.device,dtype=features.dtype)
            head=m.policy.heads['rby1'];rtc=None
            if 'rtc_prev_actions' in obs:
                previous=torch.as_tensor(np.asarray(obs['rtc_prev_actions'],np.float32),device=self.device)
                if previous.ndim!=2 or previous.shape[1]!=49 or not torch.isfinite(previous).all():raise ValueError('Invalid previous actions')
                previous=(previous-m.action_center)/m.action_scale
                freeze=int(obs['rtc_freeze_steps']);soft=int(obs['rtc_soft_horizon']);power=float(obs.get('rtc_soft_power',1.))
                action=rtc_inference(head,noise,features,previous,freeze,soft,power)
                rtc={'freeze_steps':freeze,'soft_horizon':soft,'soft_power':power}
            else:action=head.inference(noise,features)
            action=action.float()*m.action_scale+m.action_center
            if not torch.isfinite(action).all():raise RuntimeError('Nonfinite policy action')
            out={'actions':action[0].cpu().numpy(),'rtc':rtc}
            if gaze is not None:out['predicted_gaze_uv']=((gaze.float()+.5)/224)[0].cpu().numpy()
            return out

class CartPerceptionServer:
    def __init__(self,policy,legacy_path=LEGACY,vv_path=VV_PATH):
        self.policy=policy;self.legacy=module('campaign_legacy_cart',legacy_path)
        self.vv=self.legacy.vv.load_model(str(vv_path))
        self.cfg=self.legacy.fp.resolve_dual_rgb_config(types.SimpleNamespace(abl_variant='TGZ10HD2'),2048)
        self.gaze=None if policy.arm=='E' else self.legacy.gz.GazeCropper(','.join(self.legacy.GAZE),median_window=3)

    def observe(self,r):
        import cv2
        lib=self.legacy;depth=lib.stray.remove_stray_points_224(np.asarray(r['depth'],np.float32))
        rgb=np.asarray(r['rgb01'],np.float32);geo=r['geometry'];seed=int(r['seed'])
        cfg=dict(self.cfg);n=cfg.pop('num_points')
        if self.gaze is None:
            # The native plain-slab branch lacks the training <64-point fallback.
            # An infinite, fixed ball is an identity mask and retains that fallback
            # without reading a gaze prediction, label, or robot EEF centre.
            cfg.update(global_mode='slab_gaze_ball',gaze_radius=float('inf'));center=np.zeros(3)
            gaze_info={'mode':'internal_policy_prediction','external_gaze_model':False,'global_crop':'fixed_scene_no_gaze'}
        else:
            if r['reset']:self.gaze._hist.clear();self.gaze.n_hit=0;self.gaze.n_fallback=0
            center,gaze_info=self.gaze.center_from_frame(rgb,depth,r['R_device_rect'],(geo['p_eef_glass_raw']+geo['p_eefL_glass'])/2)
            if not np.isfinite(gaze_info['depth']):gaze_info['depth']=None
        vt=lib.vv.predict_virtual_T(r['q22'],self.vv)
        points,info=lib.fp.dual_rgb_streams_from_depth(depth,rgb,r['R_device_rect'],geo,n,gaze_center=center,
          virtual_T=vt,virtual_kh=self.vv['KH'],rng=np.random.default_rng(seed),**cfg)
        if points is None:raise RuntimeError('Empty scene cloud')
        if self.policy.arm=='E':
            # This branch never uses geometry/q22/virtual camera or gaze predictions.
            d=cv2.resize(depth,(224,224),interpolation=cv2.INTER_NEAREST)
            sensor=lift_sensor(d,rgb*255.,(lib.fp.DP3C_F224,lib.fp.DP3C_F224,lib.fp.DP3C_C224,lib.fp.DP3C_C224))
            points['clean_cloud']=uniform_clean_cloud(sensor,np.random.default_rng(seed))
        points['eef_pose_glass']=np.asarray(geo['eef_pose_glass'],np.float32)
        info={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}
        return {'observation':points,'info':{'gaze':gaze_info,'cloud':info,'virtual_camera':vt.tolist()}}

    def infer(self,obs):return self.observe(obs['_sim_observe']) if '_sim_observe' in obs else self.policy.infer(obs)
    def infer_batch(self,observations):return [self.infer(o) for o in observations]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--normalizer',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--virtual-camera',default=str(VV_PATH))
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];cfg=json.loads(Path(a.config).read_text());norm=json.loads(Path(a.normalizer).read_text())
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['identity']['normalizer_sha256']==sha(a.normalizer) and ck['identity']['config_sha256']==sha(a.config)
    robot=json.loads((root/'assets/robot.json').read_text());robot['urdf']=str(root/'assets/robot.urdf')
    model=GazePolicy(root/'configs/hd2-reference.yaml',norm,robot,cfg['gaze_weight']) if cfg['arm']=='E' else CampaignPolicy(root/'configs/hd2-reference.yaml','B',norm,robot)
    model.load_state_dict(ck['model'],strict=True);model.to(a.device).eval()
    assert cfg['arm'] in ('B','E')
    if a.diagnostic:
        assert ck.get('diagnostic_only') and norm['fit_partition']=='synthetic_unit'
    else:
        assert not ck.get('diagnostic_only');check_normalizer(cfg,norm)
    del ck
    policy=CampaignSimPolicy(model,norm,cfg['arm']);server=CartPerceptionServer(policy,vv_path=Path(a.virtual_camera))
    metadata={'checkpoint_sha256':sha(a.checkpoint),'normalizer_sha256':sha(a.normalizer),'config_sha256':sha(a.config),
      'action_horizon':64,'action_dim':49,'num_inference_steps':64,'rtc':True,
      'abl':'H2G' if cfg['arm']=='E' else 'TGZ10HD2','camera_keys':['front_pcd_1','front_pcd_2','front_pcd_3']+(['clean_cloud'] if cfg['arm']=='E' else []),
      'proprio_keys':list(PROPRIO),'external_gaze_model':cfg['arm']!='E','source_normalizer':norm['fit_partition']=='source_train','normalizer_fit_partition':norm['fit_partition'],'diagnostic_only':a.diagnostic,
      'legacy_perception_sha256':sha(LEGACY),'virtual_camera_sha256':sha(a.virtual_camera)}
    if cfg['arm']!='E':metadata['gaze_members']=[{'file':f,'sha256':sha(f)} for f in server.legacy.GAZE]
    transport=module('campaign_cart_websocket',TRANSPORT)
    transport.WebsocketPolicyServer(server,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()

if __name__=='__main__':main()
