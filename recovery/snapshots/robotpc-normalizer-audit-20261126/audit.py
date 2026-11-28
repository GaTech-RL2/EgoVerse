import json,hashlib,numpy as np,torch
from pathlib import Path
root=Path('/home/aloha/RB_Y1_workspace');out={'robot_access':False,'policy_inference_run':False,'bundles':[]}
for name in ['cart_B180_msc_cone_bs_FT10k','cart_B180_msc_ball_bs_FT10k']:
 p=root/'policies'/name;n=json.loads((p/'normalizer.json').read_text());ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
 c=np.array(n['action']['center'],np.float32);s=np.array(n['action']['scale'],np.float32);ns=hashlib.sha256((p/'normalizer.json').read_bytes()).hexdigest()
 assert ck['identity']['normalizer_sha256']==ns
 assert np.array_equal(ck['model']['action_center'].numpy(),c) and np.array_equal(ck['model']['action_scale'].numpy(),s)
 assert np.isfinite(s).all() and (s>0).all()
 y=np.array([-0.01,-0.001,0,0.001,0.01],np.float32);z=(y-c[2])/s[2];back=z*s[2]+c[2]
 ref=np.load(p/'testdata/reference.npz');refs=[]
 for i in range(3):
  a=ref[str(i)+'_plain'];rt=ref[str(i)+'_rtc'];refs.append(dict(case=i,mean_yaw_rad=float(a[:,2].mean()),yaw_sum_deg=float(np.rad2deg(a[:,2].sum())),rtc_frozen_prefix_max_error=float(abs(a[:8]-rt[:8]).max())))
 out['bundles'].append(dict(name=name,normalizer_sha256=ns,checkpoint_identity_and_buffers_match=True,yaw_center_rad=float(c[2]),yaw_scale_rad=float(s[2]),yaw_sparse_fallback=n['action']['sparse_fallback'][2],zero_yaw_normalized=float(-c[2]/s[2]),normalization_roundtrip_max_error=float(abs(back-y).max()),normalized_error_001_physical_deg=float(np.rad2deg(s[2]*.01)),fixture_outputs_not_rollout_outputs=refs))
p=root/'rollout_logs/cart_B180_msc_cone_bs_FT10k_bgobs_20261126_001423';stats=[]
n=json.loads((root/'policies/cart_B180_msc_cone_bs_FT10k/normalizer.json').read_text())
for raw,key in [('position','state_robot0_joint_pos'),('obs_eef_pose_glass','state_eef_pose_glass')]:
 vals=[]
 for f in sorted(p.glob('frame_*.npz')):
  z=np.load(f);v=z[raw];v=v[4:26] if raw=='position' else v;st=n[key];lo=np.asarray(st['quantile_1']);hi=np.asarray(st['quantile_99']);v=v.copy();v[hi-lo<1e-6]=lo[hi-lo<1e-6];vals.append((v-np.array(st['center']))/np.array(st['scale']))
 a=np.array(vals);stats.append(dict(key=key,max_abs_normalized=float(abs(a).max()),dims_outside_unit=np.flatnonzero((abs(a)>1).any(0)).tolist(),finite=bool(np.isfinite(a).all())))
out.update(latest_rollout=str(p),saved_observations=12,recorded_action_or_odometry=False,observed_proprio=stats)
print(json.dumps(out,indent=2))
