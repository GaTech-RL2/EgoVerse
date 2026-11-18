import json
from pathlib import Path

import pytest

from reproductions.leact.cluster.efe_pear_telemetry import Telemetry, WorkerDiagnostics, reward_metrics
from reproductions.leact.cluster.efe_pear_yam_split import build, task_group
from reproductions.leact.tracking.server import ingest,ingest_pear,reasoning_workspace
from reproductions.leact.tracking.reasoning_live import compact_telemetry,telemetry_readers,telemetry_roots


def test_collection_never_execs_into_primary_startup_or_a_different_filesystem():
    primary={'name':'leact-rp-jhh-pear-primary-current-1','pool':'groot-h100-01'}
    labelled={'name':'renamed-pear-cluster','pool':'groot-h100-02',
              'labels':{'pear_phase':'primary_training'}}
    lfs={'name':'leact-rp-jhh-pear-worker-lfs-1','pool':'groot-l40s-01'}
    reader={'name':'leact-rp-jhh-pear-worker-reader-1','pool':'groot-h100-01'}
    assert telemetry_readers([primary,labelled,lfs,reader])==[reader]
    assert telemetry_readers([primary,labelled,lfs])==[]


def test_live_collection_tracks_the_admitted_run_instead_of_the_plan_placeholder():
    arms=[{'id':'warm','run_directory':'/runs/warm'},{'id':'base','run_directory':'/runs/base'}]
    primary={'run_id':'warm','run':'/runs/rp_jhh_curriculum_hash'}
    assert telemetry_roots(arms,primary)=={'warm':'/runs/rp_jhh_curriculum_hash','base':'/runs/base'}
    assert telemetry_roots(arms,{'run_id':'unregistered','run':'/other'})==telemetry_roots(arms,{})


def test_unfiltered_macro_reward_includes_rejected_groups_but_not_infrastructure():
    configs={k:{'family':'control','suite':'sim','task':task,'stratum':start}
             for k,task,start in [('a','easy','full'),('b','hard','near_goal')]}
    groups={str(i):{'config_id':key,'infrastructure_ok':True,'outcomes':values}
            for i,(key,values)in enumerate([('a',[1]*8),('a',[1]*8),('b',[0]*8)])}
    groups['broken']={'config_id':'b','infrastructure_ok':False,'outcomes':[]}
    values=reward_metrics(groups,configs)
    assert values['reward_unfiltered_cumulative/all']==pytest.approx(2/3)
    assert values['reward_unfiltered_cumulative/macro_task']==.5
    assert values['groups_cumulative/judged']==3
    assert values['groups_cumulative/constant_reward']==3
    assert values['reward_unfiltered_cumulative/start/near_goal']==0


def test_resume_deduplication_does_not_create_zero_kl_or_phantom_steps(tmp_path):
    t=Telemetry(tmp_path,'run')
    t.metrics(7,{'actor/pg_loss':.2,'actor/grad_norm':1.,'actor/ppo_kl':.03})
    # Simulate a killed append. The previous complete step must survive.
    with t.events.open('a')as stream:stream.write('{"partial":')
    restored=Telemetry(tmp_path,'run');restored.metrics(7,{'actor/pg_loss':.9})
    restored.metrics(8,{'actor/pg_loss':.1})
    rows=[]
    for line in t.events.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    compact=compact_telemetry(rows)
    assert [r['step']for r in compact['history']]==[7,8]
    assert compact['history'][0]['policy_loss']==.2
    assert compact['history'][0]['rollout_policy_kl']==.03
    assert compact['history'][1]['rollout_policy_kl']is None
    assert all(r['reference_policy_kl']is None for r in compact['history'])
    assert json.loads((tmp_path/'steps/00000007.json').read_text())['headline']['policy_loss']==.2
    with pytest.raises(ValueError,match='another run'):Telemetry(tmp_path,'different')


def test_nonfinite_updates_are_recorded_and_stop_the_driver(tmp_path):
    t=Telemetry(tmp_path,'run')
    with pytest.raises(FloatingPointError):t.metrics(1,{'actor/grad_norm':float('nan')})
    row=json.loads(t.events.read_text())
    assert row['nonfinite_metrics']==['actor/grad_norm']
    assert row['headline']['grad_norm']is None


def test_rollout_processes_keep_separate_cumulative_diagnostics(tmp_path):
    a,b=WorkerDiagnostics(tmp_path),WorkerDiagnostics(tmp_path)
    a.add({'completed_episodes':2,'end_reason/parse':1});a.add({'completed_episodes':1})
    b.add({'completed_episodes':1,'end_reason/budget':1})
    rows=[json.loads(p.read_text())for p in (tmp_path/'rollout-workers').glob('*.json')]
    assert sum(r['totals']['completed_episodes']for r in rows)==4
    assert len(rows)==2


def test_yam_aliases_and_existing_layouts_cannot_cross_the_split(tmp_path):
    assignments={'pour_cup':'pour','pour_cup_contact_a4':'pour','pour_bowl':'pour','pick':'pick','uncap':'uncap'}
    assert task_group('pour_cup_contact_a4',assignments)==task_group('pour_bowl',assignments)=='pour'
    with pytest.raises(ValueError,match='Unreviewed'):task_group('unknown',assignments)
    graphs=tmp_path/'gr00t_sim/tasks/env_graphs/yam';graphs.mkdir(parents=True)
    for name in assignments:(graphs/(name+'.yaml')).write_text('env_name: '+name+'\ntask: {}\n')
    policy={'source_revision':'fixture','group_assignments':assignments,'test_groups':['pour'],'development_groups':['uncap']}
    frozen=build(tmp_path,'fixture',policy)
    train={r['semantic_group']for r in frozen['entries']if r['split']=='train'}
    test={r['semantic_group']for r in frozen['entries']if r['split']=='test'}
    assert not train&test and test=={'pour'}
    assert frozen['scene_policy']['all_existing_saved_layouts'].startswith('evaluation_only')
    assert all(not r['rl_admitted']for r in frozen['entries'])
    with pytest.raises(ValueError,match='changed native YAM revision'):build(tmp_path,'new revision',policy)
    assignments['pour_cup_contact_a4']='pick'
    with pytest.raises(ValueError,match='alias crosses'):build(tmp_path,'fixture',policy)


def test_direct_ingestion_is_scoped_and_survives_catalog_refresh(tmp_path):
    db=tmp_path/'db.sqlite';workspace={'pear':{'arms':[{'id':'registered-arm'}]},'inventory':{'version':1}}
    def snapshot():return {'schema_version':1,'runs':[],'training':[],
                           'status':{'reasoning_workspace':workspace}}
    ingest(db,snapshot())
    def upload(step):return {'schema_version':1,'run_id':'registered-arm',
        'data':{'history':[{'step':step,'policy_loss':1/step}],'optimizer_steps_observed':step}}
    assert ingest_pear(db,{**upload(1),'dry_run':True})['dry_run']
    assert not reasoning_workspace(db).get('live_pear')
    ingest_pear(db,upload(1))
    workspace['inventory']['version']=2;ingest(db,snapshot())
    ingest_pear(db,upload(2))
    result=reasoning_workspace(db)
    assert result['inventory']['version']==2
    assert [p['step']for p in result['live_pear']['runs']['registered-arm']['history']]==[1,2]
    with pytest.raises(ValueError,match='registered'):ingest_pear(db,{**upload(1),'run_id':'unregistered'})
    with pytest.raises(ValueError,match='cannot modify'):ingest_pear(db,{**upload(1),'data':{'inventory':{}}})
