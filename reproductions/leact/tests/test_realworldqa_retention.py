import json

import pytest

from reproductions.leact.cluster.efe_realworldqa_retention import (
    cached_rows, compare, grade, protocol_digest, question_indices, save_row,
)


def test_reasoning_cannot_leak_an_answer_into_the_final_score():
    assert not grade('ANSWER: yes', 'yes')['correct']
    assert not grade('ANSWER: yes</think>Still considering.', 'yes')['correct']
    assert grade('Thinking. </think>ANSWER: YES', 'yes')['correct']
    assert not grade('</think>ANSWER: yes.', 'yes')['correct']
    assert grade('ANSWER: yes', 'yes', thinking_required=False)['correct']
    assert not grade('<think>ANSWER: yes', 'yes', thinking_required=False)['correct']


def test_preliminary_panel_is_explicit_and_rejects_duplicates_or_invalid_indices():
    assert question_indices({})==list(range(765))
    assert question_indices({'question_ids':[764,3,0]})==[764,3,0]
    for ids in [[],[0,0],[-1],[765],[True],['3']]:
        with pytest.raises(ValueError):question_indices({'question_ids':ids})


def test_paired_comparison_requires_identical_inputs_and_counts_failures():
    row={'input_sha256':'same','seed':7,'correct':False,'format_ok':False,
         'finish_reason':'length','output_tokens':32}
    base={'a':row,'b':{**row,'correct':True,'format_ok':True,'finish_reason':'stop'}}
    warm={'a':{**row,'correct':True,'format_ok':True,'finish_reason':'stop'},'b':base['b']}
    result=compare(base,warm,resamples=200)
    assert result['warm_minus_base']==.5
    assert result['base']['format_failures']==result['base']['truncated']==1
    assert result['paired_95ci']==[0.,1.]
    with pytest.raises(ValueError,match='identical questions'):compare(base,{'a':warm['a']})
    with pytest.raises(ValueError,match='model inputs'):
        compare(base,{'a':{**warm['a'],'seed':8},'b':warm['b']})


def test_resume_ignores_partial_writes_and_rejects_protocol_drift(tmp_path):
    digest=protocol_digest({'seed':7,'model':'fixed'})
    row={'id':'0000','model_revision':'fixed','protocol_sha256':digest}
    save_row(tmp_path/'0000.json.gz',row)
    (tmp_path/'0001.json.partial').write_text('interrupted')
    assert cached_rows(tmp_path,protocol_sha256=digest,model_revision='fixed')=={'0000':row}
    with pytest.raises(ValueError,match='protocol'):
        cached_rows(tmp_path,protocol_sha256=protocol_digest({'seed':8}))
    with pytest.raises(ValueError,match='checkpoint'):
        cached_rows(tmp_path,model_revision='other')
    assert protocol_digest({'a':1,'b':2})==protocol_digest(json.loads('{"b":2,"a":1}'))


def test_evaluation_monitor_does_not_retry_a_code_or_scoring_failure(tmp_path,monkeypatch):
    from reproductions.leact.cluster import launch_efe_realworldqa_retention as launcher,scale_efe
    (tmp_path/'launch.json').write_text(json.dumps({'name':'eval','build':'frozen-build'}))
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(scale_efe,'command',lambda args:{'status':'FAILED','failure_message':'Scoring input mismatch'})
    def unexpected(*args,**kwargs):raise AssertionError('A source failure must not launch another GPU job')
    monkeypatch.setattr(launcher,'submit',unexpected)
    launcher.monitor()
    assert json.loads((tmp_path/'monitor.json').read_text())['phase']=='needs_source_review'


def test_throughput_probe_cannot_shorten_the_full_evaluation(tmp_path,monkeypatch):
    import hashlib
    import yaml
    from reproductions.leact.cluster import launch_efe_realworldqa_probe as launcher
    source=tmp_path/'source';source.mkdir()
    root=tmp_path/'receipts';root.mkdir()
    builds=tmp_path/'builds';builds.mkdir()
    base=tmp_path/'project';(base/'cluster').mkdir(parents=True)
    (base/'cluster/efe_realworldqa_serving_probe.py').write_text('# probe fixture\n')
    spec={'id':'fixed','study_version':2,'shared_root':'/shared/study_v2','batch_size':1,
          'max_output_tokens':32768,'models':[{'id':'base','revision':'fixed-base'},
                                           {'id':'warmstart496','revision':'fixed-warm'}],
          'dataset':{'id':'RealWorldQA','revision':'fixed-data'},'seed':7,
          'sampling':{'temperature':1.,'top_p':.95,'top_k':20},
          'serving_profile':{'async_scheduling':False},'optimizer_updates':0}
    (source/'spec.json').write_text(json.dumps(spec))
    (source/'workflow.yaml').write_text(yaml.safe_dump({'workflow':{'tasks':[{'args':[]} ]}}))
    (source/'runner.py').write_text('# immutable evaluation\n')
    (source/'efe_model_cache.py').write_text('# immutable cache\n')
    (source/'source-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in source.iterdir()}))
    (root/'launch.json').write_text(json.dumps({'build':str(source),'name':'original-serial-evaluation'}))
    monkeypatch.setattr(launcher,'ROOT',root);monkeypatch.setattr(launcher,'BUILD_ROOT',builds)
    monkeypatch.setattr(launcher,'BASE',base)
    build=launcher.prepare(8,evaluate=True)
    full=json.loads((build/'evaluation-spec.json').read_text());probe=json.loads((build/'spec.json').read_text())
    assert probe['probe_output_tokens']==512 and probe['accuracy_evaluation']is False
    assert full['max_output_tokens']==32768 and full['optimizer_updates']==0
    for field in ['models','dataset','seed','sampling']:
        assert full[field]==probe[field]==spec[field]
    assert full['batch_size']==8 and probe['shared_root']!=full['shared_root']
    workflow=yaml.safe_load((build/'workflow.yaml').read_text())
    assert workflow['workflow']['tasks'][0]['args'][-2:]==['--full-spec','/tmp/rp-retention/evaluation-spec.json']
    assert (build/'retention_runner.py').read_bytes()==(source/'runner.py').read_bytes()
    with pytest.raises(ValueError,match='frozen batch size'):launcher.prepare(4,evaluate=True)


def test_preliminary_builder_freezes_a_paired_random_panel_and_keeps_inference_budget(tmp_path,monkeypatch):
    import hashlib
    import random
    import yaml
    from reproductions.leact.cluster import launch_efe_realworldqa_preliminary as launcher
    reference=tmp_path/'reference';reference.mkdir()
    builds=tmp_path/'builds';builds.mkdir()
    profile={'async_scheduling':False,'enable_prefix_caching':False}
    original={'batch_size':8,'max_output_tokens':32768,'serving_profile':profile,
              'models':[{'id':'base','revision':'base-pin'},{'id':'warmstart496','revision':'warm-pin'}],
              'dataset':{'id':'RealWorldQA','revision':'data-pin'},'sampling':{'temperature':1.,'top_p':.95,'top_k':20},
              'seed':20261118,'optimizer_updates':0}
    (reference/'evaluation-spec.json').write_text(json.dumps(original))
    (reference/'workflow.yaml').write_text(yaml.safe_dump({'workflow':{'tasks':[{}]}}))
    (reference/'efe_model_cache.py').write_text('# immutable cache\n')
    (reference/'source-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                            for p in reference.iterdir()}))
    monkeypatch.setattr(launcher,'BUILD_ROOT',builds)
    qualification={'success':True,'batch_size':8,'batching_profile':profile,
                   'batches':[{'aggregate_output_tokens_per_second':60}]*2}
    build=launcher.prepare(reference,qualification)
    spec=json.loads((build/'spec.json').read_text())
    assert spec['question_ids']==random.Random(20261118).sample(range(765),40)
    assert len(set(spec['question_ids']))==40 and spec['preliminary']is True
    assert spec['execution_order']==['warmstart496','base']
    for key in ['models','dataset','sampling','seed','max_output_tokens','serving_profile','optimizer_updates']:
        assert spec[key]==original[key]
    with pytest.raises(ValueError,match='frozen panel'):launcher.prepare(reference,qualification,questions=30)
    with pytest.raises(ValueError,match='profile'):
        launcher.prepare(reference,{**qualification,'batching_profile':{'async_scheduling':True}})
