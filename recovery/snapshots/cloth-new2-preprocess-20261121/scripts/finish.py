"""Per-recording downstream build. Runs only on an allocated GPU; never submits MPS."""
import os, sys, json, subprocess, hashlib, time
from pathlib import Path
import numpy as np
import h5py

ROOT = Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121')
PY = '/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python'
DEPTH_PY = '/coc/flash7/scratch/czhang883/envs/aria_fs/bin/python'

def save(p, obj):
    t = p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj, indent=2)); t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def run(e, name, cmd, env):
    status = e/(name+'-stage.json')
    if status.exists():
        previous=json.loads(status.read_text())
        assert previous['status']=='COMPLETE', f'{name}: existing unsuccessful intent needs audit'
        return
    record=dict(status='RUNNING', command=cmd, started=time.time(), job_id=os.getenv('SLURM_JOB_ID'))
    save(status,record)
    with (e/'logs'/(name+'.log')).open('x') as out:
        result=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if result.returncode==0 else 'FAILED',returncode=result.returncode,finished=time.time())
    save(status,record)
    assert result.returncode==0, f'{name} failed; inspect retained log'

def main():
    e=ROOT/'episodes'/sys.argv[1]
    assert e.is_dir()
    f=e/'retarget/episode.hdf5'
    # Reject the upstream script's silent RGB fallback and partial-success exit.
    rows=(e/'retarget/tag_manifest.tsv').read_text().strip().splitlines()
    assert len(rows)==1 and rows[0].split('\t')[-1]=='ok'
    rec=rows[0].split('\t')[2][:-4]
    with h5py.File(f,'r') as h:
        assert list(h['data'])==['demo_0']
        g=h['data/demo_0']; ts=g['obs/aria_ts'][:]; n=len(ts)
        assert n>=64 and np.all(np.diff(ts)>0)
        assert abs(float(np.median(np.diff(ts)))-1e9/60)<2000, 'not 60 Hz rows'
        action=g['actions/joint'][:]
        assert action.shape==(n,49) and np.isfinite(action).all()
        for k in ['obs/robot0_joint_pos','obs/eef_right_proprio','obs/eef_left_proprio']:
            assert len(g[k])==n and np.isfinite(g[k][:]).all(),k
        rgb=g['obs/aria_image'][:]
        assert rgb.shape==(n,224,224,3) and rgb.std()>3 and (rgb>0).mean()>.1,'empty RGB'
        dt_report=dict(rows=n,median_dt_ns=float(np.median(np.diff(ts))),max_dt_ns=int(np.diff(ts).max()))
    save(e/'cuts.json',{'cuts':{'0':n},'meaning':'retain all original retarget-trimmed rows; task-tail review pending'})
    env=os.environ.copy(); env['INPUT_MPS']=json.loads((e/'input.json').read_text())['mps']
    env.update(TOY_SRC=str(f),TOY_H5=str(f),TOY_MAN=str(e/'retarget/tag_manifest.tsv'),
      TOY_VRSD=str(e/'input/episode'),DEPD=str(e/'depth'),OUTD=str(e/'rgbd/shards'),
      NSHARD='1',SHARD='0',TOY_MRG=str(e/'rgbd/shards/shard_0.hdf5'),
      TOY_SHARDS=str(e/'rgbd/shards'),TOY_OUTD=str(e/'clouds'),MPS_ROOT=str(e/'mps'),
      GAZE_CSV=str(e/'gaze/mps-general.csv'),TOY_CUTS=str(e/'cuts.json'),TOY_VIS='1',TOY_LAT='1.0',TOY_GZR='1.0',TOY_NG='2048',
      TOY_ZMAX='2.5',TOY_NEUT='.30',TOY_HR='.30',TOY_HC='tcp',TOY_LR='1.0',
      TOY_LEROBOT=str(e),TOY_DST=str(e/'unused'),
      PYTHONPATH='/coc/flash7/cxin34/human-pretraining-hd2-20261107/source',
      HF_HOME='/coc/flash7/czhang883/.cache/huggingface',HF_HUB_OFFLINE='1')
    depths=sorted((e/'depth'/rec).glob('*.npy'))
    assert len(depths)>10, 'depth missing'
    for p in depths:
        a=np.load(p); assert a.shape==(512,512) and np.isfinite(a).all(),p
    run(e,'rgbd',[DEPTH_PY,str(ROOT/'source/rgbd.py')],env)
    with h5py.File(env['TOY_MRG'],'r') as h:
        g=h['data/demo_0']; assert len(g['obs/aria_depth'])==n
        assert float(g.attrs['depth_staleness_ms_max'])<200,'stale depth'
        d=g['obs/aria_depth'][:]; assert np.isfinite(d).all() and (d>.05).mean()>.05
    run(e,'cloud',[PY,str(ROOT/'source/real_dp3c_tri_gz_vis_hd2.py')],env)
    run(e,'dino',[PY,str(ROOT/'source/dino_hdf5.py'),str(f),str(e/'dino')],env)
    metrics={}
    run(e,'retarget-qc',[PY,str(ROOT/'source/glitch_scan.py'),str(f)],env)
    qc=(e/'logs/retarget-qc.log').read_text()
    if '<--' in qc:
        save(e/'quality-quarantine.json',dict(uuid=e.name,job_id=os.getenv('SLURM_JOB_ID'),
            disposition='QUALITY_QUARANTINED',time=time.time(),
            reason='Original retarget QC rejected this episode; excluded, no retry, thresholds unchanged',
            qc_log_sha256=sha(e/'logs/retarget-qc.log')))
        print('QUALITY_QUARANTINED: original retarget QC flag; not a valid preprocessing release',flush=True)
        return
    metrics['retarget_qc']=qc.strip()
    for name,count in [('g_hd2',2048),('l_hd2',1024),('l3_hd2',1024)]:
        a=np.load(e/'clouds'/f'{name}_000000.npy').reshape(n,count,6)
        assert np.isfinite(a).all() and a[:,:,:3].std()>.001,name
        assert a[:,:,3:].min()>=0 and a[:,:,3:].max()<=1,name
        metrics[name]={'shape':list(a.shape),'xyz_std':float(a[:,:,:3].std())}
    d=np.load(e/'dino/dino_000000.npy').reshape(n,16,768)
    assert np.isfinite(d).all() and np.max(abs(d.mean(-1)))<.01
    assert .9<float(d.std(-1).mean())<1.1
    metrics['dino']={'shape':list(d.shape),'mean_token_std':float(d.std(-1).mean())}
    gh=np.load(e/'clouds/gaze_hit_000000.npy')
    metrics['gaze_hit_fraction']=float(gh.mean())
    files=[f,e/'rgbd/shards/shard_0.hdf5',*sorted((e/'clouds').glob('*.npy')),*sorted((e/'dino').glob('*.npy'))]
    result=dict(passed=True,kind='preprocessing_artifacts_not_training_release',uuid=e.name,
      created=time.time(),job_id=os.getenv('SLURM_JOB_ID'),timestamps=dt_report,metrics=metrics,
      input=json.loads((e/'input.json').read_text()),artifacts={str(p):sha(p) for p in files},
      raw_vrs_sha256=sha(json.loads((e/'input.json').read_text())['vrs']),
      pending=['task-tail trimming/visual QC','session split','train-only normalization','A-D release assembly'])
    save(e/'preprocess-proof.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
