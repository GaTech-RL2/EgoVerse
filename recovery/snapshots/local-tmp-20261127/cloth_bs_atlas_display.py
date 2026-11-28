import os,sys,json,hashlib,shutil,fcntl
from pathlib import Path
import numpy as np,h5py,pyarrow as pa,pyarrow.parquet as pq
os.umask(0o022)
sys.path.insert(0,"/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/scripts")
from retarget_tensor_state import StateConverter
W=Path("/coc/flash7/czhang883/warp-data");P=W/".cxin34-storage/20261125-four-tasks/registry";D=W/"datasets/cxin34_cloth_new50_fix1124_bs"
V="r2_bs_display";SRC=D/"releases/r1_hd2";ST=D/(".staging-"+V);OUT=D/"releases"/V
lock=(P/(V+".lock")).open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not ST.exists() and not OUT.exists()
ST.mkdir();shutil.copytree(SRC/"meta",ST/"meta");(ST/"data/chunk-000").mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mat(p9):
 p9=np.asarray(p9,float);x=p9[:,3:6];y=p9[:,6:9];out=np.repeat(np.eye(4)[None],len(p9),axis=0);out[:,:3,3]=p9[:,:3];out[:,:3,:3]=np.stack([x,y,np.cross(x,y)],axis=2);return out
def transform(T,x):return np.einsum("nij,nkj->nki",T[:,:3,:3],x)+T[:,None,:3,3]
def replace(t,k,v):return t.set_column(t.schema.get_field_index(k),k,pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)),int(np.prod(v.shape[1:]))))
conv=StateConverter();maps=[json.loads(l) for l in (SRC/"source_map.jsonl").read_text().splitlines()];proof=[]
for m in maps:
 idx=m["episode_index"];rel=Path("data/chunk-000")/f"episode_{idx:06d}.parquet";f=SRC/rel;before=sha(f);t=pq.read_table(f);n=len(t)
 bs=Path(m["source_hdf5"]);fix=bs.with_name("old.hdf5");hs=[sha(fix),sha(bs)]
 with h5py.File(fix) as a,h5py.File(bs) as b:
  g=a["data/demo_0"];h=b["data/demo_0"]
  oldp=np.stack(t["obs.eef_pose_glass"].to_numpy())
  oldR=g["obs/eef_right_proprio"][:].reshape(n,4,4);newR=h["obs/eef_right_proprio"][:].reshape(n,4,4)
  oldL=g["obs/eef_left_proprio"][:].reshape(n,4,4);newL=h["obs/eef_left_proprio"][:].reshape(n,4,4)
  oldG=oldR@np.linalg.inv(mat(oldp))
  def head_world(group):
   q=group["obs/robot0_joint_pos"][:];bp=group["obs/robot_base_pose"][:];out=[]
   for row,base in zip(q,bp):
    x,y,yaw=base;c=np.cos(yaw);s=np.sin(yaw);S=np.eye(4);S[:2,:2]=[[c,-s],[s,c]];S[:2,3]=[x,y];out.append(S@conv.fk(row)[0]@conv.mount)
   return np.asarray(out)
  anchors=oldG@np.linalg.inv(head_world(g));anchor_error=float(np.max(np.abs(anchors-anchors[0])))
  assert anchor_error<2e-5,(idx,anchor_error)
  newG=anchors[0]@head_world(h);relpose=np.linalg.inv(newG)@newR
  newp=np.concatenate([relpose[:,:3,3],relpose[:,:3,0],relpose[:,:3,1]],axis=1).astype("float32")
  np.testing.assert_allclose(newR@np.linalg.inv(mat(newp)),newG,atol=2e-5,rtol=0)
  err={}
  for k,oldT,newT in [("obs.aria_pcdc",oldG,newG),("obs.aria_pcdc_local",oldR,newR),("obs.aria_pcdc_local_left",oldL,newL)]:
   x=np.stack(t[k].to_numpy()).reshape(n,-1,6);y=x.copy();y[:,:,:3]=transform(np.linalg.inv(newT)@oldT,x[:,:,:3]).astype("float32")
   e=float(np.max(np.abs(transform(oldT,x[:,:,:3])-transform(newT,y[:,:,:3]))));assert e<2e-5,(idx,k,e);assert np.array_equal(x[:,:,3:],y[:,:,3:]);err[k]=e
   t=replace(t,k,y.reshape(n,-1))
  t=replace(t,"obs.eef_pose_glass",newp)
 dest=ST/rel;pq.write_table(t,dest,compression="zstd",row_group_size=128);back=pq.read_table(dest);assert back.equals(t,check_metadata=False)
 original=pq.read_table(f)
 changed={"obs.aria_pcdc","obs.aria_pcdc_local","obs.aria_pcdc_local_left","obs.eef_pose_glass"}
 for k in original.column_names:
  if k not in changed:assert original[k].equals(back[k]),(idx,k)
 assert sha(f)==before and hs==[sha(fix),sha(bs)]
 proof.append(dict(episode=idx,segment=m["uuid"],rows=n,source_sha256=before,output_sha256=sha(dest),max_world_error_m=err,source_hdf5_sha256=hs,constant_world_anchor_error=anchor_error))
 print("REFRAMED",idx,n,max(err.values()),flush=True)
for name in ("source_map.jsonl","qc.json"):
 if (SRC/name).exists():shutil.copy2(SRC/name,ST/name)
recipe=dict(visualization_only=True,training_authorized=False,source=str(SRC),purpose="Re-express unchanged physical cloud points in post-BS robot frames for Atlas. Not exact model-input coordinates.",unchanged="all RGB, action, joints, hand q, base pose, times and cloud colors",transform="inv(T_world_cloud_postBS) @ T_world_cloud_preBS; global and each hand independently",source_tensors_preserved=True)
(ST/"recipe.json").write_text(json.dumps(recipe,indent=2))
(ST/"manifest.json").write_text(json.dumps(dict(dataset="cxin34_cloth_new50_fix1124_bs",release=V,episodes=len(proof),frames=sum(x["rows"] for x in proof),checksums={f"data/chunk-000/episode_{x['episode']:06d}.parquet":x["output_sha256"] for x in proof}),indent=2))
(P/(V+"-audit.json")).write_text(json.dumps(dict(passed=True,training_authorized=False,episodes=proof,recipe=recipe),indent=2))
(ST/"READY.json").write_text(json.dumps(dict(passed=True,purpose="Display export integrity ONLY; not training readiness")))
ST.rename(OUT);print("DISPLAY_EXPORT_PASS",OUT,flush=True)
