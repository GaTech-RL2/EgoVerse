import json
from pathlib import Path

import pytest

from reproductions.leact.cluster.efe_pear_curriculum import INITIAL_QUOTAS
from reproductions.leact.cluster.efe_pear_curriculum_state import CurriculumState


def manifest():
    return {'partition':'train','qualification':'passed','configs':[
        {'config_id':f'{family}/{reset}','family':family,'suite':'fixture','task':family,
         'stratum':'full' if reset<2 else 'near_goal','partition':'train','admitted':True,
         'already_successful':False,'observation_complete':True,'scene_sha256':f'scene-{family}-{reset}',
         'admission_receipt_sha256':'unit-test-fixture','env_class':'FixtureEnv',
         'env_args':{'family':family,'reset':reset}}
        for family in INITIAL_QUOTAS for reset in range(3)]}


def test_canonical_families_match_the_authorized_proposal():
    proposal=json.loads((Path(__file__).parents[1]/'proposals/efe_pearplusplus_multitask_v2.json').read_text())
    mixture={x['family']:x['groups'] for x in proposal['mixture']}
    # The proposal compresses the eight equal reasoning allocations into one row.
    if 'eight_reasoning_families' in mixture:
        total=mixture.pop('eight_reasoning_families')
        assert total%len(proposal['reasoning_training']['families'])==0
        n=total//len(proposal['reasoning_training']['families'])
        mixture.update({x['id']:n for x in proposal['reasoning_training']['families']})
    assert mixture==INITIAL_QUOTAS


def test_resume_preserves_dispatches_sgs_history_and_next_draws():
    m=manifest();a=CurriculumState(m)
    for i in range(300):
        key=a.draw(step=40);a.register(str(i),key,40)
        a.observe(str(i),[0,1]*4 if i%2 else [0]*8)
    saved=json.loads(json.dumps(a.state_dict()))
    b=CurriculumState(m);b.restore(saved)
    assert [a.draw(step=40) for _ in range(50)]==[b.draw(step=40) for _ in range(50)]
    assert b.dispatches==a.dispatches and b.history.state_dict()==a.history.state_dict()
    before=len(b.family_outcomes[next(iter(INITIAL_QUOTAS))])
    # The complete group stored in both TQ and the curriculum cannot be counted twice on resume.
    uid=next(iter(b.groups));b.observe(uid,b.groups[uid]['outcomes'])
    assert len(b.family_outcomes[next(iter(INITIAL_QUOTAS))])==before
    m['configs'][0]['scene_sha256']='changed-scene'
    with pytest.raises(ValueError,match='changed'):CurriculumState(m).restore(saved)


def test_no_partial_or_evaluation_curriculum_can_start_training():
    for mutate in [lambda m:m.update(partition='test'),
                   lambda m:m.update(configs=m['configs'][3:]),
                   lambda m:m['configs'][0].update(already_successful=True),
                   lambda m:m['configs'][0].update(observation_complete=False)]:
        m=manifest();mutate(m)
        with pytest.raises(ValueError):CurriculumState(m)


def test_control_bootstrap_is_explicit_and_cannot_silently_replace_the_full_mixture():
    m=manifest();m['configs']=[c for c in m['configs']if c['family']=='general_manipulation']
    with pytest.raises(ValueError,match='Every authorized'):CurriculumState(m)
    m['curriculum_stage']='control_bootstrap'
    for c in m['configs']:c['suite']='metaworldplus'
    state=CurriculumState(m)
    assert state.quotas=={'general_manipulation':128}
    assert state.loss_weights=={'general_manipulation':1.}
    restored=CurriculumState(m);restored.restore(state.state_dict())
    assert restored.quotas==state.quotas
    m['configs'][0]['suite']='unqualified-other-suite'
    with pytest.raises(ValueError,match='restricted'):CurriculumState(m)


def test_refill_targets_missing_families_without_losing_full_starts():
    state=CurriculumState(manifest());family='physical_prediction'
    ready={f:n for f,n in INITIAL_QUOTAS.items() if f!=family}
    for step,minimum in [(0,.5),(300,.75)]:
        counts={'full':0,'near_goal':0}
        for i in range(120):
            key=state.draw(ready,step=step);c=state.configs[key]
            assert c['family']==family
            counts[c['stratum']]+=1;state.register(f'{step}-{i}',key,step)
        # The consolidation phase may need to make up for earlier assisted starts.
        total=sum(state.dispatches.values())
        full=sum(n for k,n in state.dispatches.items() if state.configs[k]['stratum']=='full')
        assert full/total>=minimum


def test_adaptation_rejects_heldout_or_missing_progress_signals():
    state=CurriculumState(manifest())
    with pytest.raises(ValueError,match='training-only'):
        state.adapt(INITIAL_QUOTAS,step=32,evidence={'partition':'test'})
    with pytest.raises(ValueError,match='progress evidence'):
        state.adapt(INITIAL_QUOTAS,step=32,evidence={'partition':'train','step':32,'families':{}})
