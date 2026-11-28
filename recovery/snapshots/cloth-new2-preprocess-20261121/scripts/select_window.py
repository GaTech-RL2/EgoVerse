import sys,json,os
from pathlib import Path
import numpy as np,pandas as pd,h5py
from finish import save
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');item=json.loads((R/'plan.json').read_text())[int(sys.argv[1])];e=Path(item['episode'])
src=e/'retarget/episode.hdf5';raw=e/'retarget/episode-pipeline-trimmed.hdf5'
assert not raw.exists();src.rename(raw)
h=pd.read_csv(Path(item['mps'])/'hand_tracking/hand_tracking_results.csv');ht=h.tracking_timestamp_us.to_numpy(float)
with h5py.File(raw) as f:
 g=f['data/demo_0'];ts=g['obs/aria_ts'][:]/1000;n=len(ts);assert n>0
 j=np.clip(np.searchsorted(ht,ts),1,len(ht)-1);j=np.where(abs(ht[j]-ts)<abs(ht[j-1]-ts),j,j-1)
 valid=abs(ht[j]-ts)<=34000
 for side in ['left','right']:
  valid &= (h[side+'_tracking_confidence'].to_numpy()[j]>=0)&np.isfinite(h[[f't{k}_{side}_device_wrist' for k in 'xyz']].to_numpy()[j]).all(1)
 edges=np.diff(np.r_[False,valid,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
 if not len(starts) or max(ends-starts)<64:
  save(e/'quality-quarantine.json',{'reason':'No contiguous bilateral-valid window of 64 frames','original_rows':n,'take':item['take']});raise SystemExit(0)
 z=int(np.argmax(ends-starts));a,b=int(starts[z]),int(ends[z])
 def copy(x,y):
  for k,v in x.attrs.items():y.attrs[k]=v
  for k,v in x.items():
   if isinstance(v,h5py.Group):copy(v,y.create_group(k))
   else:
    arr=v[()]
    if getattr(arr,'ndim',0)>0 and arr.shape[0]==n:arr=arr[a:b]
    y.create_dataset(k,data=arr,compression='gzip' if k=='aria_image' else None)
 with h5py.File(src,'w') as out:
  copy(f,out);out['data/demo_0'].attrs['num_samples']=b-a
 save(e/'valid-window.json',{'take':item['take'],'original_rows':n,'start_row':a,'end_row':b,'rows':b-a,'policy':'Longest contiguous bilateral MPS-valid window after existing pipeline tail trim; frozen before train/valid split outcomes','start_ns':int(ts[a]*1000),'end_ns':int(ts[b-1]*1000),'base_pose':'Original world origin preserved; no coordinate reset'})
print('VALID_WINDOW',item['take'],a,b,flush=True)
