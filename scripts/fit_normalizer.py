"""Fit task source-train statistics only. Constant dimensions require reviewed physical floors."""
import argparse,json
from pathlib import Path
import numpy as np
from campaign.data import sha

def fit(x,floors):
    assert np.isfinite(x).all()
    low,high=np.quantile(x,[.01,.99],axis=0);lo=x.min(0);hi=x.max(0)
    sparse=(high-low)<.05*(hi-lo)
    low=np.where(sparse,lo,low);high=np.where(sparse,hi,high)
    center=(low+high)/2;scale=(high-low)/2
    floors=np.broadcast_to(np.asarray(floors),scale.shape)
    assert np.isfinite(floors).all() and (floors>=0).all()
    scale=np.maximum(scale,floors)
    assert (scale>1e-8).all(),'Constant/tiny dimensions need a reviewed positive physical scale floor'
    return dict(center=center.tolist(),scale=scale.tolist(),sparse_fallback=sparse.tolist(),min=lo.tolist(),max=hi.tolist(),floor=floors.tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--floors',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.manifest);meta=json.loads(path.read_text());floors=json.loads(Path(a.floors).read_text())
    output=Path(a.output);assert not output.exists(),'Never overwrite frozen statistics'
    train=[e for e in meta['episodes'] if e['split']=='train'];assert train
    stats=dict(fit_partition='source_train',manifest_sha256=sha(path),train_uuids=[e['uuid'] for e in train],floors_sha256=sha(a.floors))
    keys=['action','state_robot0_joint_pos','state_eef_pose_glass','state_hand_left_qpos','state_hand_right_qpos','current_q']
    for key in keys:
        values=[];masks=[]
        for e in train:
            rec=e['arrays'][key];f=path.parent/rec['file'];assert sha(f)==rec['sha256']
            x=np.load(f,allow_pickle=False)
            if key=='current_q':
                r=e['arrays']['current_q_valid'];f=path.parent/r['file'];assert sha(f)==r['sha256']
                masks.append(np.load(f,allow_pickle=False).astype(bool))
            values.append(x)
        x=np.concatenate(values)
        if key=='current_q':
            mask=np.concatenate(masks);assert mask.shape==x.shape
            assert np.isfinite(x[mask]).all() and mask.any(0).all()
            stats[key]=dict(mean=[float(x[mask[:,i],i].mean()) for i in range(46)],std=[max(.05,float(x[mask[:,i],i].std())) for i in range(46)])
        else:stats[key]=fit(x,floors.get(key,0.))
    with output.open('x') as f:json.dump(stats,f,indent=2)
if __name__=='__main__':main()
