"""CPU independent smoke gate and full-frame ensemble evaluation."""
import argparse, json, os, hashlib
from pathlib import Path
import numpy as np, torch
from common import Heat7, load_rows, ang_err, stats, ep_file
from prepare import save, sha
R=Path(os.environ['GAZE_ROOT'])

def verify(run,epochs):
    s=json.loads((run/'summary.json').read_text());assert s['args']['epochs']==epochs
    records=[json.loads(x) for x in (run/'log.jsonl').read_text().splitlines()]
    assert [x['ep'] for x in records]==list(range(epochs))
    assert all(np.isfinite([x['train_loss'],x['sel_median'],x['sel_p90']]).all() for x in records)
    assert records[-1]['train_loss']<5,'Collapsed or failed member: do not deliver'
    state=torch.load(run/'ckpt.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==epochs-1 and state['sched']['last_epoch']==epochs
    assert state['opt']['state'] and all(k in state for k in ['torch_rng','cuda_rng','np_rng','py_rng'])
    for value in state['opt']['state'].values():
        for x in value.values():
            if torch.is_tensor(x):assert torch.isfinite(x).all()
    net=Heat7(pretrained=False)
    weights=torch.load(run/'best.pt',map_location='cpu',weights_only=True);net.load_state_dict(weights,strict=True);net.eval()
    assert all(torch.isfinite(t).all() for t in weights.values())
    initial=Heat7(pretrained=True).state_dict()
    changed=sum(not torch.equal(weights[k],v) for k,v in initial.items() if k.startswith('body.') and v.is_floating_point())
    assert changed>0,'Backbone unchanged from ImageNet'
    return net,dict(changed_backbone_tensors=changed,seed=s['seed'],best_epoch=s['best_epoch'],train_loss=records[-1]['train_loss'],sha256=sha(run/'best.pt'))

def main():
    torch.set_num_threads(4)
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['smoke','first','final']);mode=a.parse_args().mode
    data=json.loads((R/'reports/data-ready.json').read_text());assert data['passed']
    split=data['split'];imgs,uv,epi,rows=load_rows(split['selection'],stride=1,inimg=False)
    assert len(imgs)>0
    members=[];preds=[]
    if mode=='final':
        assert sha(Path(data['source_manifest']))==data['source_manifest_sha256']
        assert sha(R/'splits/split_final.json')==data['split_sha256']
        for e in data['episodes']:
            assert sha(ep_file(e['key']))==e['cache_sha256']
            for path,digest in e['label_sha'].items():assert sha(Path(path))==digest
    runs=[R/'smoke'] if mode=='smoke' else ([R/'runs/s42'] if mode=='first' else [R/f'runs/s{seed}' for seed in range(42,48)])
    for run in runs:
        net,rec=verify(run,2 if mode=='smoke' else 30)
        with torch.inference_mode():
            p=torch.cat([net(torch.from_numpy(imgs[i:i+64].copy()).permute(0,3,1,2).float()/255) for i in range(0,len(imgs),64)]).numpy()
        assert np.isfinite(p).all() and np.std(p,axis=0).max()>.05,'Degenerate constant predictor'
        rec['validation']=stats(ang_err(p,uv));members.append(rec);preds.append(p)
    if mode=='first':
        assert json.loads((R/'runs/s42/resume-verified.json').read_text())['from_epoch']==0
    if mode=='smoke':
        assert (R/'smoke/resume-verified.json').exists()
        proof=json.loads((R/'smoke/resume-verified.json').read_text());assert proof['from_epoch']==0
    avg=np.mean(preds,axis=0)
    result=dict(passed=True,members=members,validation=stats(ang_err(avg,uv)),
        center_baseline=stats(ang_err(np.full_like(uv,112),uv)),
        validation_scope=split['scope'],split_sha256=sha(R/'splits/split_final.json'))
    assert result['validation']['median']<result['center_baseline']['median'],'Must beat center baseline'
    result['per_recording']=[dict(key=key,validation=stats(ang_err(avg[epi==i],uv[epi==i])),center_baseline=stats(ang_err(np.full_like(uv[epi==i],112),uv[epi==i]))) for i,key in enumerate(split['selection'])]
    if mode=='final':
        import shutil
        for i,run in enumerate(runs):
            out=R/f'deploy/gaze_heat7_cloth_new2_m{i}.pt'
            if out.exists():assert sha(out)==sha(run/'best.pt')
            else:shutil.copyfile(run/'best.pt',out)
        result['deployment_name']='cloth_new2';result['robot_camera_accuracy']='not measured; human validation only'
        np.savez(R/'reports/validation-predictions.npz',predicted=avg,label=uv,members=np.stack(preds))
    if mode=='first':
        result['acceptance_scope']='Full30epoch first member with same recipe and separate-process resume; reuse as ensemble seed42. Earlier2epoch smoke preserved, health threshold unchanged.'
        save(R/'reports/smoke-independent-audit.json',result)
    save(R/f'reports/{mode}-independent-audit.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
