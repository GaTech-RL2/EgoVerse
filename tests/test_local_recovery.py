"""Behavior checks for recovered augmentation, row alignment, and FT contracts."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from campaign import dino_lora
from campaign.contracts import check_normalizer, release_path
from campaign.data import EpisodeDataset, sha
from campaign.geoaug import augment as geoaugment
from campaign.photometric import augment, rng_state, restore_rng
from campaign.pointcept_color import ChromaticJitter, RandomColorJitter

STRONG = dict(brightness=.6, contrast=.6, saturation=.3, hue=.03,
              p=.8, color_noise_p=.95, color_noise_std=.075, consistency_weight=0.)
GEO = dict(point_dropout=dict(p=1., global_ratio=[.1,.2], hand_ratio=[.1,.2]),
           xyz_jitter=dict(p=1., std_m=.003, clip_m=.009), grayscale=dict(p=1.))


def test_strong_msc_matches_pinned_transform_and_preserves_labels():
    b = {f'front_pcd_{i}': torch.linspace(.1,.9,36).reshape(1,1,1,36)
         for i in range(1,4)}
    b['action'] = torch.randn(1,64,49)
    original = {k:v.clone() for k,v in b.items()}
    state = rng_state()
    out = augment(b, STRONG)
    restore_rng(state)
    colors = np.concatenate([b[k].numpy().reshape(-1,6)[:,3:]*255
                             for k in b if k.startswith('front_pcd_')])
    ref = ChromaticJitter(p=.95,std=.075)(
        RandomColorJitter(.6,.6,.3,.03,p=.8)({'color':colors}))['color']/255
    actual = np.concatenate([out[k].numpy().reshape(-1,6)[:,3:]
                             for k in b if k.startswith('front_pcd_')])
    np.testing.assert_array_equal(actual, ref)
    for k in b:
        assert torch.equal(b[k],original[k])
        if k.startswith('front_pcd_'):
            assert torch.equal(b[k].reshape(-1,6)[:,:3],out[k].reshape(-1,6)[:,:3])
        else:
            assert torch.equal(b[k],out[k])


def test_geoaug_wiring_rng_padding_and_metric_bounds():
    b = {'front_pcd_1':torch.zeros(2,1,1,48), 'action':torch.randn(2,64,49)}
    b['front_pcd_1'][0,0,0,6:12]=torch.tensor([.2,.3,.4,.6,.7,.8])
    original = {k:v.clone() for k,v in b.items()}
    config = {**STRONG, 'geoaug':GEO}
    state = rng_state(); first = augment(b,config)
    restore_rng(state); second = augment(b,config)
    assert all(torch.equal(first[k],second[k]) for k in first)
    assert all(torch.equal(b[k],original[k]) for k in b)
    assert first['front_pcd_1'][1].eq(0).all()
    assert first['front_pcd_1'][0,0,0,:6].eq(0).all()
    assert torch.equal(b['action'],first['action'])
    g = copy.deepcopy(GEO);g['point_dropout']['p']=0
    result = geoaugment(b,g)['front_pcd_1'].reshape(2,-1,6)
    assert (result[...,:3]-b['front_pcd_1'].reshape(2,-1,6)[...,:3]).abs().max()<=.009001
    assert torch.equal(result[...,3],result[...,4]) and torch.equal(result[...,4],result[...,5])


@pytest.fixture
def release(tmp_path):
    n=12
    shapes={'action':(49,), 'front_pcd_1':(2048,6), 'front_pcd_2':(1024,6),
            'front_pcd_3':(1024,6), 'state_robot0_joint_pos':(22,),
            'state_eef_pose_glass':(9,), 'state_hand_left_qpos':(12,),
            'state_hand_right_qpos':(12,), 'dino':(16,768)}
    arrays={k:np.broadcast_to(np.arange(n,dtype=np.float32).reshape(n,*([1]*len(s))),
                             (n,*s)).copy() for k,s in shapes.items()}
    arrays.update(timestamp=np.arange(n)/60,action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))
    records={}
    for k,a in arrays.items():
        p=tmp_path/(k+'.npy');np.save(p,a)
        records[k]={'file':p.name,'sha256':sha(p)}
    manifest=dict(recording_hz=60,horizon_rows=64,dino_offset_rows=10,
                  quality_passed=True,split_frozen=True,
                  episodes=[dict(uuid='synthetic-only',split='train',arrays=records)])
    path=tmp_path/'target.json';path.write_text(json.dumps(manifest))
    norm={k:dict(center=[1.]*s[0],scale=[2.]*s[0]) for k,s in shapes.items()
          if k=='action' or k.startswith('state_')}
    return path,norm


def test_loader_horizon_future_offset_and_tail_mask(release):
    path,norm=release;ds=EpisodeDataset(path,'train','B',norm)
    first=ds[0];last=ds[11]
    assert len(ds)==12 and first['action'].shape==(64,49)
    assert first['action_valid'].sum()==12 and last['action_valid'].sum()==1
    assert first['dino'].eq(10).all() and first['dino_valid']
    assert last['dino'].eq(11).all() and not last['dino_valid']
    assert last['action'].eq(5).all()  # repeat final row, mask padded future
    assert last['state_robot0_joint_pos'].eq(5).all()


def test_loader_rejects_mutated_tensor(release):
    path,norm=release
    with (path.parent/'action.npy').open('ab') as f:f.write(b'changed')
    with pytest.raises(AssertionError):EpisodeDataset(path,'train','B',norm)


def test_loader_rejects_duplicate_recording_identity(release):
    path,norm=release;meta=json.loads(path.read_text())
    meta['episodes'].append(meta['episodes'][0]);path.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):EpisodeDataset(path,'train','B',norm)


def ft_contract():
    dimensions={'action':[3,8],'state_robot0_joint_pos':[0,5]}
    cfg=dict(stage='finetune',split_role='target',arm='B',release_id='drawer',
             formal_training_authorized=True,train_recordings=1,valid_recordings=1,
             source_identity=dict(manifest_sha256='source',normalizer_sha256='norm'),
             normalizer_adaptation=dict(passed=True,valid_used=False,source_sha256='norm',changes=dimensions))
    norm=dict(fit_partition='source_train',manifest_sha256='source',
              adaptation=dict(valid_used=False,source_normalizer_sha256='norm',
                              dimensions=dimensions,train_uuids=['train']))
    manifest=dict(episodes=[dict(uuid='train',split='train'),dict(uuid='valid',split='valid')])
    return cfg,norm,manifest


def test_fixed_ft_contract_accepts_recovered_drawer_name():
    cfg,norm,manifest=ft_contract()
    assert release_path(Path('/local'),cfg)==Path('/local/releases/drawer')
    check_normalizer(cfg,norm,manifest)


@pytest.mark.parametrize('failure',['validation_fit','overlap','unauthorized','wrong_dimensions'])
def test_fixed_ft_rejects_contract_violation(failure):
    cfg,norm,manifest=ft_contract()
    if failure=='validation_fit':norm['adaptation']['train_uuids'].append('valid')
    if failure=='overlap':manifest['episodes'][1]['uuid']='train'
    if failure=='unauthorized':cfg['formal_training_authorized']=False
    if failure=='wrong_dimensions':norm['adaptation']['dimensions']={'action':[0]}
    with pytest.raises(AssertionError):
        release_path(Path('/local'),cfg);check_normalizer(cfg,norm,manifest)


def test_lora_inherits_dino_freezes_base_and_updates_adapters():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.policy=torch.nn.Module()
            self.policy.heads=torch.nn.ModuleDict({'rby1_dino_wm_future':torch.nn.Sequential(torch.nn.Linear(4,4)),
                                                   'rby1':torch.nn.Linear(4,4)})
    source=Tiny();target=Tiny()
    dino_lora.install(target,dict(rank=8,alpha=16,dropout=0,base_frozen=True,inherit=True))
    dino_lora.transfer(target,source.state_dict())
    initial={k:p.detach().clone() for k,p in target.named_parameters()}
    x=torch.ones(2,4)
    torch.testing.assert_close(target.policy.heads['rby1_dino_wm_future'](x),source.policy.heads['rby1_dino_wm_future'](x))
    optimizer=torch.optim.AdamW(target.parameters(),lr=.001)
    sum(h(x).square().sum() for h in target.policy.heads.values()).backward();optimizer.step()
    assert dino_lora.smoke_checks(target,initial)['inherited_head_base_unchanged']
