from pathlib import Path
import json,sys
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');plan=json.loads((R/'plan.json').read_text());out=R/'reports/visual';out.mkdir(exist_ok=True)
selected=[r for r in plan if (R/'tensors/episodes'/r['uuid']/'episode.json').exists()]
for offset in range(0,len(selected),12):
 rows=selected[offset:offset+12];fig,axes=plt.subplots(len(rows),3,figsize=(9,2.1*len(rows)),squeeze=False)
 for ri,r in enumerate(rows):
  with h5py.File(Path(r['episode'])/'retarget/episode.hdf5') as h:
   rgb=h['data/demo_0/obs/aria_image'];n=len(rgb)
   for col,t in enumerate([0,n//2,n-1]):axes[ri,col].imshow(rgb[t]);axes[ri,col].set_title(f"take {r['take']} {r['split']} | frame {t}/{n}",fontsize=8);axes[ri,col].axis('off')
 fig.tight_layout();fig.savefig(out/f'rgb_{offset//12}.png',dpi=110);plt.close(fig)
for r in selected:
 if r['take'] not in [1,3,5,16,39,43,60]:continue
 e=Path(r['episode']);td=R/'tensors/episodes'/r['uuid'];n=json.loads((td/'episode.json').read_text())['rows'];t=n//2
 fig=plt.figure(figsize=(12,7))
 for j,(key,title) in enumerate([('front_pcd_1','Global'),('front_pcd_2','Right local'),('front_pcd_3','Left local')]):
  ax=fig.add_subplot(2,3,j+1,projection='3d');a=np.load(td/(key+'.npy'),mmap_mode='r')[t];ax.scatter(a[:,0],a[:,1],a[:,2],c=a[:,3:].clip(0,1),s=2);ax.set_title(title);ax.set_box_aspect((1,1,1));ax.set_xlabel('x');ax.set_ylabel('y')
 with h5py.File(e/'rgbd/shards/shard_0.hdf5') as h:
  ax=fig.add_subplot(2,3,4);ax.imshow(h['data/demo_0/obs/aria_image'][t]);ax.set_title('Stereo-aligned RGB');ax.axis('off')
  ax=fig.add_subplot(2,3,5);im=ax.imshow(h['data/demo_0/obs/aria_depth'][t].reshape(224,224),vmin=0,vmax=3,cmap='viridis');ax.set_title('Depth m');fig.colorbar(im,ax=ax)
 ax=fig.add_subplot(2,3,6);q=np.load(td/'state_robot0_joint_pos.npy');ax.plot(np.arange(n)/60,q[:,6:20],lw=.5);ax.set_title('Arm joints radians');ax.set_xlabel('seconds')
 fig.suptitle(f"new2 take{r['take']} {n} valid frames");fig.tight_layout();fig.savefig(out/f'geometry_take{r["take"]}.png',dpi=120);plt.close(fig)
print('VISUAL_QC_FILES',len(list(out.glob('*.png'))))
