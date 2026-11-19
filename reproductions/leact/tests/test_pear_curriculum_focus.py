import pytest

from reproductions.leact.cluster.efe_pear_curriculum_focus import focus_tasks,focus_families
from reproductions.leact.cluster.efe_pear_curriculum import INITIAL_QUOTAS


def test_known_training_tasks_receive_extra_draws_with_a_full_exploration_floor_and_decay():
    base={str(i):.01 for i in range(100)};rates={k:.5 for k in base}
    p=focus_tasks(base,base,{'0','1'},rates,step=1)
    assert sum(p.values())==pytest.approx(1.)
    assert p['0']+p['1']>.8 and all(p[k]>=.2*base[k]for k in p)
    assert focus_tasks(base,base,{'0','1'},rates,step=9)==pytest.approx(base)
    mastered={k:(1. if k in {'0','1'}else.5)for k in base}
    assert focus_tasks(base,base,{'0','1'},mastered,step=1)==pytest.approx(base)


def test_focus_increases_known_family_effort_without_dropping_unseen_families():
    base={f:n/128 for f,n in INITIAL_QUOTAS.items()}
    configs={'known':{'task':'known-control','family':'general_manipulation'}}
    p=focus_families(base,INITIAL_QUOTAS,configs,{'known-control'},{},step=1)
    assert sum(p.values())==pytest.approx(1.)
    assert p['general_manipulation']>base['general_manipulation']
    assert all(p[f]>=.2*base[f]for f in base)
    assert focus_families(base,INITIAL_QUOTAS,configs,{'known-control'},{},step=9)==pytest.approx(base)


def test_initial_probes_are_admitted_on_policy_draws_and_resume_at_the_next_probe():
    from test_efe_pear_curriculum_state import manifest
    from reproductions.leact.cluster.efe_pear_curriculum_state import CurriculumState
    m=manifest();m['sampling_policy']='task-signal-window-v2'
    probes=[c['config_id']for c in m['configs']if c['stratum']=='full'][:2]
    m['warmup_sampling']={'partition':'train','confirmed_training_tasks':[],
        'probe_config_ids':probes,'decay_updates':8}
    a=CurriculumState(m);assert a.draw(step=1)==probes[0]
    b=CurriculumState(m);b.restore(a.state_dict())
    assert b.draw(step=1)==probes[1]and b.last_draw['planned_training_probe']
    with pytest.raises(ValueError,match='admitted training'):
        b.set_sampling_control({'partition':'train','confirmed_training_tasks':['held-out-evaluation-task']})
