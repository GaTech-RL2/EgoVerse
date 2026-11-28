"""All-recording non-LA input preflight; deliberately not a READY gate."""
import hashlib,json
from pathlib import Path
import numpy as np,h5py
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
p=json.load(open(R/'cart/input-manifest.json'));a=Path(p['observation_root']);b=Path(p['retarget_root'])
old=json.load(open(p['source_manifest']));meta={e['uuid']:e for e in old['episodes']}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
gate=json.load(open(b/'retargeted/gate_basefix.json'))
assert gate['demos']==107 and gate['checks']['limit_rows']==[0,0]
out=[]
with h5py.File(a/'retargeted/chuye_cart_1102.hdf5','r')as f,h5py.File(b/'retargeted/chuye_cart_1102.hdf5','r')as g:
 for e in p['records']:
    i=e['episode_index'];n=e['rows'];x=f['data'][e['demo']];y=g['data'][e['demo']]
    np.testing.assert_array_equal(x['obs/aria_ts'][:],y['obs/aria_ts'][:])
    ts=y['obs/aria_ts'][:n]/1e9;oldts=np.load(meta[e['uuid']]['arrays']['timestamp']['file'])
    np.testing.assert_allclose(ts,oldts,atol=1e-6,rtol=0)
    q=y['obs/robot0_joint_pos'][:n];assert np.isfinite(q).all()
    np.testing.assert_array_equal(x['obs/robot0_joint_pos'][:n,24:26],q[:,24:26])
    np.testing.assert_array_equal(x['obs/aria_image'][:n],y['obs/aria_image'][:n])
    files={};stats={}
    for key,sub,name,shape in [('global','clouds_hd2','g_hd2',(n,2048*6)),('right','clouds_hd2','l_hd2',(n,1024*6)),('left','clouds_hd2','l3_hd2',(n,1024*6)),('dino','dino_wm_r2','dino',(n,16*768)),('camera','clouds_hd2','tg',(n,4,4)),('gaze','clouds_hd2','gp',(n,3))]:
        path=a/sub/(name+f'_{i:06d}.npy');v=np.load(path,mmap_mode='r');assert v.shape==shape and np.isfinite(v).all(),(e['uuid'],key,v.shape,shape)
        if key in ['global','left','right']:
            z=v.reshape(n,-1,6);assert z[...,3:].min()>=0 and z[...,3:].max()<=1
            stats[key]=dict(points=z.shape[1],xyz_min=z[...,:3].min(axis=(0,1)).tolist(),xyz_max=z[...,:3].max(axis=(0,1)).tolist())
        files[key]=dict(path=str(path),sha256=sha(path),shape=list(v.shape))
    reg=json.load(open(a/'clouds_hd2'/f'gz_{i:06d}.json'));assert reg['demo']==e['demo'] and reg['rows']==n
    out.append(dict(uuid=e['uuid'],split=e['split'],rows=n,passed=True,files=files,registration=reg,cloud_stats=stats))
    print('INPUT_PASS',e['uuid'],n,flush=True)
report=dict(passed=True,training_ready=False,phase='input_alignment_preflight',records=out,rows=sum(e['rows']for e in out),base_gate_sha256=sha(b/'retargeted/gate_basefix.json'),look_at=False,pending=['canonical tensor export','source isolation','full real loader','normalization','independent geometry and visual review','per-run smoke'])
(R/'cart/reports/input-audit.json').write_text(json.dumps(report,indent=2))
print('ALL_CART_INPUTS_PASS',report['rows'],flush=True)
