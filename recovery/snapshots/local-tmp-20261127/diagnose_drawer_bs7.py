from pathlib import Path
import os,json,hashlib
import numpy as np,h5py,mujoco
os.umask(0o022)
R=Path("/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125")
O=R/"reports/bs-example7-20261125";O.mkdir(exist_ok=True)
e=json.loads((R/"drawer/segment-manifest.json").read_text())[7];P=Path(e["segment"])
model=mujoco.MjModel.from_xml_path("/coc/flash7/czhang883/Documents/SEW-aria-pipeline/projects/xhand_teleop/rby1_with_xhand/model_v1.3_xhand_act.xml");d=mujoco.MjData(model)
names=[*[f"torso_{i}" for i in range(6)],*[f"right_arm_{i}" for i in range(7)],*[f"left_arm_{i}" for i in range(7)],*[f"head_{i}" for i in range(2)]]
ads=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,k)] for k in names]
def fk(q,bp):
 out=[]
 for qi,b in zip(q,bp):
  d.qpos[:]=0;d.qpos[ads]=qi[4:26];mujoco.mj_forward(model,d)
  x,y,a=b;c,s=np.cos(a),np.sin(a);S=np.eye(4);S[:2,:2]=[[c,-s],[s,c]];S[:2,3]=[x,y]
  rr=[]
  for name in ["right_eef","left_eef","link_head_2"]:
   z=d.body(name);T=np.eye(4);T[:3,:3]=z.xmat.reshape(3,3);T[:3,3]=z.xpos;rr.append(S@T)
  out.append(rr)
 return np.array(out)
TCP=np.array([[.0265,.0065,-.1077,1],[-.0265,.0065,-.1077,1]])
def hands(T):return np.einsum("nsij,sj->nsi",T[:,:2],TCP)[:,:,:3]
def stats(x):return {k:float(v) for k,v in zip(["min","median","p95","p999","max"],np.percentile(x,[0,50,95,99.9,100]))}
a={};hashes={}
for key,file in [("old","fix.hdf5"),("new","fix_bs.hdf5")]:
 path=P/"retarget"/file;hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
 with h5py.File(path) as f:
  g=f["data/demo_0"];q=g["obs/robot0_joint_pos"][:];base=g["obs/robot_base_pose"][:];T=np.stack([g["obs/eef_right_proprio"][:].reshape(-1,4,4),g["obs/eef_left_proprio"][:].reshape(-1,4,4)],1)
  a[key]=dict(q=q,base=base,T=T,ts=g["obs/aria_ts"][:],fk=fk(q,base),rgb=g["obs/aria_image"][:])
n=len(a["old"]["q"]);np.testing.assert_array_equal(a["old"]["ts"],a["new"]["ts"]);np.testing.assert_array_equal(a["old"]["rgb"],a["new"]["rgb"])
# Constant rigid world anchor; robust median translation and nearest rotation.
anchors=a["old"]["T"]@np.linalg.inv(a["old"]["fk"][:,:2]);cand=anchors.reshape(-1,4,4)
u,sv,vt=np.linalg.svd(np.median(cand[:,:3,:3],axis=0));A=np.eye(4);A[:3,:3]=u@vt;A[:3,3]=np.median(cand[:,:3,3],axis=0)
old=a["old"];new=a["new"];h0=hands(old["T"]);h1=hands(new["T"]);f0=hands(A@old["fk"]);f1=hands(A@new["fk"])
cb=hands(A@fk(old["q"],new["base"]));qt=old["q"].copy();qt[:,4:10]=new["q"][:,4:10];ct=hands(A@fk(qt,new["base"]))
err=np.linalg.norm(h1-h0,axis=2)*100;rf=np.linalg.norm(f1-f0,axis=2)*100;peak=np.unravel_index(np.argmax(err),err.shape);i,s=map(int,peak)
yaw=np.degrees(np.arctan2(np.sin(new["base"][:,2]-old["base"][:,2]),np.cos(new["base"][:,2]-old["base"][:,2])))
wrapped=np.degrees(np.abs(np.arctan2(np.sin(new["q"][:,10:24]-old["q"][:,10:24]),np.cos(new["q"][:,10:24]-old["q"][:,10:24]))))
result=dict(segment=e,frames=n,hashes=hashes,gate_hand_cm=stats(err),independent_smoothed_FK_change_cm=stats(rf),stored_vs_smoothed_FK_cm={k:stats(np.linalg.norm(h-f,axis=2)*100) for k,h,f in [("old",h0,f0),("new",h1,f1)]},base_xy_change_cm=stats(np.linalg.norm(new["base"][:,:2]-old["base"][:,:2],axis=1)*100),base_yaw_change_deg=stats(yaw),wrapped_arm_change_deg=stats(wrapped),peak=dict(frame=i,time_s=float((old["ts"][i]-old["ts"][0])/1e9),side=["right","left"][s],gate_cm=float(err[i,s]),independent_fk_cm=float(rf[i,s]),base_xy_change_cm=float(np.linalg.norm(new["base"][i,:2]-old["base"][i,:2])*100),base_yaw_change_deg=float(yaw[i]),old_hand_world=h0[i,s].tolist(),new_hand_world=h1[i,s].tolist(),fk_original=f0[i,s].tolist(),fk_base_only=cb[i,s].tolist(),fk_base_torso=ct[i,s].tolist(),fk_all_new=f1[i,s].tolist(),old_q=old["q"][i].tolist(),new_q=new["q"][i].tolist()),exceed_rows={str(t):int((err.max(1)>t).sum()) for t in [1,2,10,50]},first_gt2cm=int(np.where(err.max(1)>2)[0][0]))
(O/"summary.json").write_text(json.dumps(result,indent=2));np.savez(O/"trajectories.npz",time=(old["ts"]-old["ts"][0])/1e9,h_old=h0,h_new=h1,fk_old=f0,fk_new=f1,base_only=cb,base_torso=ct,old_base=old["base"],new_base=new["base"],old_q=old["q"],new_q=new["q"],error_cm=err)
from PIL import Image
for j in sorted({0,i,n//2,n-1}):Image.fromarray(old["rgb"][j]).save(O/f"rgb-{j}.png")
print(json.dumps(result,indent=2))
