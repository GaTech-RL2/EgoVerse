"""Mmap RGB captures indexed by supplied sel; never interpolate or cross episodes."""
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import sha

STATE_KEYS = ('state_robot0_joint_pos','state_eef_pose_glass','state_hand_left_qpos','state_hand_right_qpos')

class RGBDataset(Dataset):
    def __init__(self,manifest,split,arm,normalizer):
        assert arm == 'B'
        self.path=Path(manifest);self.meta=json.loads(self.path.read_text());self.norm=normalizer
        assert self.meta['rgb_compatibility_passed'] and self.meta['split_frozen']
        assert (self.meta['recording_hz'],self.meta['horizon_rows'],self.meta['dino_offset_rows'])==(60,64,10)
        self.entries=[e for e in self.meta['episodes'] if e['split']==split]
        self.index=[];self._cache=OrderedDict()
        for ei,e in enumerate(self.entries):
            for rec in list(e['arrays'].values())+list(e['rgb_arrays'].values()):
                assert sha(rec['file'])==rec['sha256'],rec['file']
            self.index.extend((ei,t) for t in range(e['rows']))
        assert self.index

    def __len__(self): return len(self.index)

    def arrays(self,ei):
        if ei in self._cache:
            self._cache.move_to_end(ei);return self._cache[ei]
        if len(self._cache)>=4:self._cache.popitem(last=False)
        e=self.entries[ei]
        data={k:np.load(v['file'],mmap_mode='r',allow_pickle=False) for k,v in {**e['arrays'],**e['rgb_arrays']}.items()}
        self._cache[ei]=data
        return data

    def __getitem__(self,i):
        ei,t=self.index[i];e=self.arrays(ei);n=len(e['action'])
        out={'front_rgb':torch.from_numpy(e['rgb_frames'][e['rgb_sel'][t]].copy().transpose(2,0,1)).unsqueeze(0)}
        for key in STATE_KEYS:
            st=self.norm[key]
            out[key]=torch.tensor((e[key][t]-np.array(st['center']))/np.array(st['scale']),dtype=torch.float32).unsqueeze(0)
        idx=np.minimum(t+np.arange(64),n-1);st=self.norm['action']
        out['action']=torch.tensor((e['action'][idx]-np.array(st['center']))/np.array(st['scale']),dtype=torch.float32)
        out['action_valid']=torch.tensor((t+np.arange(64)<n)&e['action_valid'][idx])
        future=min(t+10,n-1)
        out['dino']=torch.tensor(e['dino'][future].copy())
        out['dino_valid']=torch.tensor(t+10<n and bool(e['dino_valid'][future]))
        return out
