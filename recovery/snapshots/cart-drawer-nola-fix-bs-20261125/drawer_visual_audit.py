"""Read-only independent tensor checks and sampled RGB / three-cloud sheets."""
import os
os.environ['MPLCONFIGDIR']='/tmp/cxin34-drawer-mpl'
import json,sys,hashlib
from pathlib import Path
import numpy as np,h5py,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
manifest=json.load(open(R/'drawer/segment-manifest.json'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for v in iter(lambda:f.read(8388608),b''):h.update(v)
 return h.hexdigest()
for arg in sys.argv[1:]:
 i=int(arg);e=manifest[i];p=Path(e['segment']);o=p/'downstream';ep=json.load(open(o/'episode.json'));proof=json.load(open(o/'proof.json'))
 assert proof['passed'] and sha(o/'episode.json')==proof['episode_sha256']
 a={}
 for k,v in ep['arrays'].items():
  assert sha(v['file'])==v['sha256'];a[k]=np.load(v['file'],mmap_mode='r');assert list(a[k].shape)==v['shape'] and np.isfinite(a[k]).all()
 n=ep['rows'];t=a['timestamp'];assert t[-1]-t[0]>1 and np.all(np.diff(t)>0)
 with h5py.File(p/'retarget/fix_bs.hdf5')as f:
  g=f['data/demo_0'];raw=g['actions/joint'][:]
  np.testing.assert_array_equal(a['action'],np.concatenate([raw[:,22:25],raw[:,14:22],raw[:,:14],raw[:,25:]],-1).astype('float32'))
  np.testing.assert_array_equal(a['state_robot0_joint_pos'],g['obs/robot0_joint_pos'][:,4:26].astype('float32'))
  rows=[0,n//2,n-1];rgb=g['obs/aria_image'][rows]
 fig=plt.figure(figsize=(15,10))
 for row,(j,img)in enumerate(zip(rows,rgb)):
  ax=fig.add_subplot(3,4,4*row+1);ax.imshow(img);ax.axis('off');ax.set_title(f'RGB frame {j} / {n}')
  for col,(key,label)in enumerate(zip(['front_pcd_1','front_pcd_2','front_pcd_3'],['Global 2048','Right 1024','Left 1024']),2):
   v=a[key][j];ax=fig.add_subplot(3,4,4*row+col,projection='3d');ax.scatter(v[:,0],v[:,1],v[:,2],s=1,c=v[:,3:]);ax.set_title(label);ax.view_init(elev=18,azim=-65)
   center=np.median(v[:,:3],axis=0);span=max(.2,float(np.max(np.ptp(v[:,:3],axis=0)))/2)
   ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span);ax.set_box_aspect((1,1,1));ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
 fig.suptitle(f'Drawer {i}: {e["uuid"]}; {e["split"]}; CPF + wrists + BS, no LA; 1.57m')
 fig.tight_layout();dst=R/'drawer/reports'/f'visual-{i:03d}.jpg';fig.savefig(dst,dpi=130);plt.close(fig)
 result=dict(passed=True,index=i,uuid=e['uuid'],episode_sha256=sha(o/'episode.json'),rows=n,span_s=float(t[-1]-t[0]),arrays_checked=len(a),image=str(dst),image_sha256=sha(dst),visual_review_pending=True)
 report=R/'drawer/reports'/f'independent-{i:03d}.json'
 with report.open('x')as f:json.dump(result,f,indent=2)
 print(json.dumps(result),flush=True)
