"""Explicit row-index adapter for frozen per-episode tensor releases; no fps rounding."""
import hashlib,json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

class _EpisodeArrays:
    """Lazy mapping; returned tensors copy their rows before the cache can evict."""
    def __init__(self,dataset,index):self.dataset=dataset;self.index=index
    def __getitem__(self,key):return self.dataset._arrays(self.index)[key]

class EpisodeDataset(Dataset):
    def __init__(self,manifest,split,arm,normalizer):
        self.path=Path(manifest);self.meta=json.loads(self.path.read_text());self.arm=arm;self.norm=normalizer
        assert self.meta['recording_hz']==60 and self.meta['horizon_rows']==64 and self.meta['dino_offset_rows']==10
        assert self.meta['quality_passed'] and self.meta['split_frozen']
        ids=[x['uuid'] for x in self.meta['episodes']];assert len(ids)==len(set(ids))
        self.episodes=[];self.index=[];self._paths=[];self._cache=OrderedDict();has_training_gaze=False
        for e in self.meta['episodes']:
            if e['split']!=split:continue
            arrays={};paths={}
            for key,record in e['arrays'].items():
                p=self.path.parent/record['file'];assert sha(p)==record['sha256']
                arrays[key]=np.load(p,mmap_mode='r',allow_pickle=False)
                paths[key]=p
            n=len(arrays['action']);assert arrays['action'].shape==(n,49)
            assert all(len(v)==n for v in arrays.values())
            assert np.isfinite(arrays['action']).all()
            shapes={'front_pcd_1':(2048,6),'front_pcd_2':(1024,6),'front_pcd_3':(1024,6),'state_robot0_joint_pos':(22,), 'state_eef_pose_glass':(9,), 'state_hand_left_qpos':(12,), 'state_hand_right_qpos':(12,)}
            if arm in ('B','E'):shapes['dino']=(16,768)
            if arm=='C':shapes.update(clean_cloud=(2048,6),current_q=(46,),current_q_valid=(46,))
            if arm=='E':shapes.update(clean_cloud=(2048,6),e_front_pcd_1=(2048,6),gaze_uv=(2,),gaze_valid=(),gaze_timestamp=())
            for key,shape in shapes.items():
                assert arrays[key].shape==(n,*shape),(key,arrays[key].shape)
                if key not in ('current_q','gaze_uv','gaze_timestamp'):assert np.isfinite(arrays[key]).all(),key
            assert arrays['action_valid'].shape==(n,) and arrays['action_valid'].any()
            assert np.isfinite(arrays['timestamp']).all()
            assert np.allclose(np.diff(arrays['timestamp']),1/60,rtol=.05,atol=.001)
            if arm=='C':assert self.meta['clean_cloud_provenance']=='sensor_uniform_no_pose_no_gaze'
            if arm=='E':
                assert self.meta['clean_cloud_provenance']=='sensor_uniform_no_pose_no_gaze'
                assert self.meta['e_global_provenance']=='hd2_hand_dense_fixed_scene_no_gaze'
                gaze=self.meta['gaze_contract']
                assert gaze['source']=='mps_general' and gaze['coordinates']=='normalized_uv'
                assert gaze['causal_alignment'] and gaze['projection_verified']
                assert gaze['camera_frame'] and gaze['calibration_sha256'] and gaze['projection_recipe_sha256']
                observed=arrays['gaze_valid'].astype(bool);uv=arrays['gaze_uv'][observed]
                assert np.isfinite(uv).all() and ((uv>=0)&(uv<=1)).all()
                age=arrays['timestamp'][observed]-arrays['gaze_timestamp'][observed]
                assert np.isfinite(age).all() and (age>=0).all() and (age<=gaze['max_age_seconds']).all()
                has_training_gaze |= bool(observed.any())
            ei=len(self.episodes);self._paths.append(paths);self.episodes.append(_EpisodeArrays(self,ei))
            self.index.extend((ei,t) for t in range(n));del arrays
        assert self.index,'Empty split'
        if arm=='E' and split=='train':
            assert has_training_gaze,'E scratch requires measured training gaze'

    def _arrays(self,index):
        # Hundreds of source recordings otherwise hold thousands of descriptors;
        # bound mmap ownership without changing validation, row order, or RNG.
        if index in self._cache:
            self._cache.move_to_end(index);return self._cache[index]
        if len(self._cache)>=4:self._cache.popitem(last=False)
        arrays={k:np.load(p,mmap_mode='r',allow_pickle=False) for k,p in self._paths[index].items()}
        self._cache[index]=arrays;return arrays

    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        ei,t=self.index[i];e=self.episodes[ei];n=len(e['action']);idx=np.minimum(t+np.arange(64),n-1)
        valid=t+np.arange(64)<n
        out={}
        for k in ('front_pcd_1','front_pcd_2','front_pcd_3','state_robot0_joint_pos','state_eef_pose_glass','state_hand_left_qpos','state_hand_right_qpos'):
            source='e_front_pcd_1' if self.arm=='E' and k=='front_pcd_1' else k
            x=e[source][t].astype(np.float32)
            if k.startswith('state_'):
                st=self.norm[k];x=(x-np.array(st['center']))/np.array(st['scale']);x=x[None]
            else:x=x.reshape(1,1,-1)
            out[k]=torch.tensor(x,dtype=torch.float32)
        st=self.norm['action'];out['action']=torch.tensor((e['action'][idx]-np.array(st['center']))/np.array(st['scale']),dtype=torch.float32)
        out['action_valid']=torch.tensor(valid & e['action_valid'][idx].astype(bool))
        if self.arm in ('B','E'):
            out['dino']=torch.tensor(e['dino'][min(t+10,n-1)],dtype=torch.float32)
            out['dino_valid']=torch.tensor(t+10<n and bool(e['dino_valid'][min(t+10,n-1)]))
        if self.arm=='C':
            for k in ('clean_cloud','current_q','current_q_valid'):out[k]=torch.tensor(e[k][t])
            out['current_q_valid']=out['current_q_valid'].bool()
        if self.arm=='E':
            for k in ('clean_cloud','gaze_uv'):out[k]=torch.tensor(e[k][t])
            out['gaze_valid']=torch.tensor(bool(e['gaze_valid'][t]),dtype=torch.bool)
        return out
