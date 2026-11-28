from pathlib import Path
import numpy as np,h5py,json,hashlib,os
C=Path('/coc/flash7/czhang883/warp-data');R=Path('/coc/flash7/cxin34/warp-data/registry/migration-four-tasks-20261125')
ret=C/'collections/cxin34_cloth_new50_fix1124_bs/processed/r1_hd2/retargeted';work=ret/'work';work.mkdir(exist_ok=True)
plan=json.loads(Path('/coc/flash7/scratch/cxin34/cloth50-recovered-bs-20261125/plan.json').read_text());parents={p['parent']['uuid']:p['parent'] for p in plan}
meta=json.loads((ret/'metadata.json').read_text());demos=meta['hdf5_files']['cxin34_cloth_new50_fix1124_bs.hdf5']['demos'];results=[]
with h5py.File(ret/'cxin34_cloth_new50_fix1124_bs.hdf5','r') as f:
 for m in demos:
  par=parents[m['recording_id']];src=Path(par['episode'])/'retarget/work'/(par['retarget_tag']+'_aria60.npz');z=np.load(src);ts=z['ts'];obs=f['data/'+m['demo']+'/obs'];target=obs['aria_ts'][:].reshape(-1)/1000
  idx=np.searchsorted(ts,target);idx=np.clip(idx,0,len(ts)-1);prev=np.maximum(idx-1,0);idx=np.where(abs(ts[prev]-target)<abs(ts[idx]-target),prev,idx)
  err=float(np.max(abs(ts[idx]-target)));assert err<0.01,(m['demo'],err);assert np.all(np.diff(idx)==1)
  data={k:(z[k][idx] if z[k].ndim and len(z[k])==len(ts) else z[k]) for k in z.files};data['n']=np.array(len(idx));assert len(idx)==m['frames'];out=work/(m['source_tag']+'_aria60.npz')
  if not out.exists():np.savez_compressed(out,**data)
  rgb=work/(m['source_tag']+'_rgb224.npz')
  if not rgb.exists():np.savez_compressed(rgb,frames=obs['aria_image'][:],sel=np.arange(len(idx)),grid_us=target,res=np.array(224))
  with np.load(out) as q:assert np.array_equal(q['ts'],data['ts'])
  results.append(dict(demo=m['demo'],tag=m['source_tag'],source=str(src),rows=len(idx),timestamp_error_us=err,tracking_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest()))
(R/'cloth-atlas-work-repair.json').write_text(json.dumps(dict(passed=True,segments=results,training_data_unchanged=True),indent=2));print('CLOTH_CACHE_PASS',len(results),sum(x['rows'] for x in results))
