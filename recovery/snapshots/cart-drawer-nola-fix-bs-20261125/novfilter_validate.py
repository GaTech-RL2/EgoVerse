import json,sys,hashlib
from pathlib import Path
import numpy as np
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');i=int(sys.argv[1]);e=next(e for e in json.load(open(R/'cart/input-manifest.json'))['records']if e['episode_index']==i);n=e['rows'];out=R/'cart-novfilter/clouds_hd2';files={}
for name,width in [('g_hd2',12288),('l_hd2',6144),('l3_hd2',6144),('p',9),('gp',3)]:
 p=out/(name+f'_{i:06d}.npy');a=np.load(p);assert a.shape==(n,width)and np.isfinite(a).all();files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
r=json.load(open(out/f'gz_{i:06d}.json'));assert r['rows']==n and r['demo']==e['demo']
(out/f'proof_{i:06d}.json').write_text(json.dumps(dict(passed=True,virtual_camera_filter=False,rows=n,files=files),indent=2));print('NOVFILTER_NUMERIC_PASS',i,n)
