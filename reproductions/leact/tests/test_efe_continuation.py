import copy
import hashlib
import io
import json

import pytest

from reproductions.leact.cluster.efe_train_vl import training_plan
from reproductions.leact.cluster.efe_training_resume import validate_reuse
from reproductions.leact.cluster.efe_training_state import (
    FORMAT,HashingWriter,choose_resume,continuation_lr,discard_uncommitted,validate_continuation,verify_files,
)


def fixture(tmp_path):
    rows=[{'id':f'{i}-{d}','state_id':str(i),'direction':d,'suite':'suite',
           'task':f'task-{i%7}','episode':f'episode-{i%21}'}
          for i in range(256) for d in ['forward','backward']]
    original={'base_model':'base','base_revision':'frozen','context':32768,'seed':17,
              'effective_batch':32,'scene_audit_sha256':'audit','lr':1e-6}
    _,parent,_,_=training_plan(rows,{**original,'balanced_passes':3})
    spec={**original,'balanced_passes':6,'expected_optimizer_steps':96,'lr':3.12696703292044e-5,
          'checkpoint_fractions':[.5,2/3,5/6,1.],
          'continuation':{'origin_step':32,'target_step':96,'initial_learning_rate':3.12696703292044e-5,
             'initial_checkpoint':'source-step32','state_directory':str(tmp_path),'lineage_sha256':'lineage',
             'restart_interval':2,'keep_restart_states':1,
             'source_prefix_sha256':hashlib.sha256(json.dumps([r['id'] for r in parent[:1024]],separators=(',',':')).encode()).hexdigest()}}
    spec['schedule_comparison']={'study_id':'continuation','declared_changes':{
        k:{'previous':original.get(k,1.0),'current':spec[k]} for k in ('lr','balanced_passes')}}
    return original,rows,spec,parent


def test_continuation_has_64_new_updates_and_preserves_trained_prefix(tmp_path):
    original,rows,spec,parent=fixture(tmp_path)
    assert validate_reuse(spec,original)
    steps,schedule,saves,_=training_plan(rows,spec)
    validate_continuation(spec,schedule)
    assert steps-spec['continuation']['origin_step']==64
    assert schedule[:len(parent)]==parent and saves==[48,64,80,96]
    remaining=schedule[32*32:]
    assert len(remaining)==64*32
    assert sum(r['direction']=='forward' for r in remaining)==64*16
    assert continuation_lr(spec,32)==spec['lr']
    values=[continuation_lr(spec,s) for s in range(32,96)]
    assert all(a>b>0 for a,b in zip(values,values[1:]))
    with pytest.raises(ValueError):continuation_lr(spec,31)
    with pytest.raises(ValueError):continuation_lr(spec,96)
    altered=copy.deepcopy(schedule);altered[0]['id']='wrong'
    with pytest.raises(ValueError,match='already-trained'):validate_continuation(spec,altered)


def committed(root,step,lineage='lineage'):
    folder=root/f'checkpoint-{step}';folder.mkdir()
    (folder/'optimizer.pt').write_bytes(b'fixture')
    (folder/'training-state.json').write_text(json.dumps({'format':FORMAT,'lineage_sha256':lineage,
        'step':step,'files':{'optimizer.pt':{'bytes':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}}}))
    return folder


def test_only_complete_matching_restart_states_are_selected(tmp_path):
    _,_,spec,_=fixture(tmp_path)
    assert choose_resume(spec)['step']==32
    committed(tmp_path,34);committed(tmp_path,36)
    pending=tmp_path/'.pending-worker-38';pending.mkdir()
    (pending/'pending.json').write_text(json.dumps({'lineage_sha256':'lineage'}))
    assert choose_resume(spec)['step']==36
    discard_uncommitted(spec)
    assert not pending.exists() and (tmp_path/'checkpoint-34').exists()
    (tmp_path/'checkpoint-36/optimizer.pt').write_bytes(b'corrupt')
    # Size alone is checked in selection; the worker checks full hashes before loading.
    restore=choose_resume(spec)
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify_files(restore['path'],restore['receipt']['files'])
    (tmp_path/'checkpoint-36/optimizer.pt').write_bytes(b'truncated')
    with pytest.raises(ValueError,match='size mismatch'):choose_resume(spec)


def test_foreign_state_and_uncommitted_namespaces_are_not_deleted(tmp_path):
    _,_,spec,_=fixture(tmp_path);committed(tmp_path,34,lineage='other')
    with pytest.raises(ValueError,match='Incompatible'):choose_resume(spec)
    pending=tmp_path/'.pending-other-38';pending.mkdir()
    (pending/'pending.json').write_text(json.dumps({'lineage_sha256':'other'}))
    with pytest.raises(ValueError,match='Unexpected pending'):discard_uncommitted(spec)
    assert pending.exists()


def test_streaming_checkpoint_hash_matches_written_bytes():
    stream=io.BytesIO();writer=HashingWriter(stream)
    writer.write(b'first');writer.write(memoryview(b'second'));writer.flush()
    assert writer.bytes==11 and writer.tell()==11
    assert writer.digest.hexdigest()==hashlib.sha256(stream.getvalue()).hexdigest()
