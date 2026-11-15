from collections import Counter
import json

import pytest

from reproductions.leact.cluster import efe_training_pilot as pilot
from reproductions.leact.cluster.efe_leact_vl import reasoning_from_output
from reproductions.leact.cluster.efe_train_vl import presample
from reproductions.leact.cluster.efe_training_data import sha


def test_unfinished_or_copied_executable_candidates_are_not_admitted():
    action=[{'function':{'arguments':{'cmd':'python /app/output/controller.py'}}}]
    explanation='The visible obstacle requires a smaller movement.'
    assert reasoning_from_output(explanation,'length',action)[0] is None
    assert reasoning_from_output(explanation,'stop',action)[0] is None
    assert reasoning_from_output('Use python /app/output/controller.py</think>DONE','stop',action)[0] is None
    assert reasoning_from_output(explanation+'</think>DONE','stop',action)==(explanation,None)


def test_candidate_multiplicity_does_not_double_decision_or_replay_weight():
    rows=[]
    for source in ['warmstart','leact']:
        for decision,candidates in [('one',1),('two',2 if source=='leact' else 1)]:
            for candidate in range(candidates):
                for direction in ['forward','backward']:
                    rows.append({'source':source,'suite':'suite','task':'task','episode':'episode',
                                 'decision_id':decision,'state_id':f'{source}/{decision}/{candidate}',
                                 'direction':direction})
    sampled=presample(rows,1000,20261027)
    assert Counter(r['source'] for r in sampled)=={'warmstart':16000,'leact':16000}
    assert Counter(r['direction'] for r in sampled)=={'forward':16000,'backward':16000}
    leact=Counter(r['decision_id'] for r in sampled if r['source']=='leact')
    assert abs(leact['one']-leact['two'])<800
    assert sampled==presample(rows,1000,20261027)


def test_prepared_corpus_recovery_keeps_membership_and_rejects_modified_data(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();destination=tmp_path/'destination';destination.mkdir()
    spec={key:key for key in ['base_model','base_revision','scene_audit_sha256','student_prompt_requests','pilot_decisions','context','seed']}
    files={'spec.json':spec,'pilot-membership.json':{'state_ids':['b','a'],'context_exclusions':[]},
           'trace-admission.json':{'decisions':3},'student-interface.json':{}}
    for name,value in files.items():(source/name).write_text(json.dumps(value))
    (source/'decisions.jsonl').write_text(''.join(json.dumps({'id':ident})+'\n' for ident in ['a','b','c']))
    spec['prepared_corpus']={'archive':str(source),'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in source.iterdir()}}
    monkeypatch.setattr(pilot,'ROOT',destination)
    states,_=pilot.prepared_states(spec)
    assert [state['id'] for state in states]==['b','a']
    (source/'decisions.jsonl').write_text('{"id":"z"}\n')
    with pytest.raises(ValueError,match='Prepared corpus changed'):pilot.prepared_states(spec)
