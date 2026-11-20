import json
from types import SimpleNamespace

import pytest

from reproductions.leact.cluster import launch_efe_pear_primary as primary


def test_capacity_requests_are_tracked_separately_from_the_original_backup(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    directory=tmp_path/'model-capacity-requests';directory.mkdir()
    for name in ['pool02-low.json','pool01-high.json']:(directory/name).write_text('{}')
    assert primary.standby_allocation_paths()==[tmp_path/'backup-allocation.json',
        directory/'pool01-high.json',directory/'pool02-low.json']


def test_ready_extra_capacity_promotes_once_and_preserves_other_requests(tmp_path,monkeypatch):
    from reproductions.leact.cluster import efe_pear_allocation_jobs as jobs
    monkeypatch.setattr(primary,'DEST',tmp_path)
    spec={'allocation_lifecycle':'persistent_cluster','rollout_nodes':4,
        'model_revision':'pinned','run':'same-run','run_id':'same-run-id','profile':6}
    paths=[]
    for name in ['queued-build','ready-build','job-build']:
        p=tmp_path/name;p.mkdir();(p/'spec.json').write_text(json.dumps(spec));paths.append(p)
    state={'workflow':'queued02','build':str(paths[0]),'standby':True,'profile':6}
    extra=tmp_path/'extra.json';extra.write_text(json.dumps({'workflow':'ready01','build':str(paths[1]),
        'standby':True,'priority':'HIGH','pool':'groot-h100-01','gpu':40}))
    default=tmp_path/'backup-allocation.json';default.write_text('{"workflow":"queued56"}')
    def command(args):
        if args[3]=='queued02':return{'status':'PENDING'}
        assert args[3]=='ready01'
        return{'status':'RUNNING','groups':[{'tasks':[
            {'name':r,'status':'RUNNING'}for r in ['trainer','rollout-0','rollout-1','rollout-2','rollout-3']]}]}
    monkeypatch.setattr(primary,'command',command)
    monkeypatch.setattr(primary,'prepare',lambda *a,**k:paths[2])
    monkeypatch.setattr(jobs,'previous_model_writer',lambda wf,run:'actual-ended-owner')
    submitted=[]
    monkeypatch.setattr(jobs,'submit',lambda state,build,**kw:submitted.append((state['workflow'],kw))or{'job_id':'job-1'})
    value=primary.promote_allocated_backup(state,extra)
    assert value['workflow']=='ready01'and value['standby']is False
    assert submitted==[('ready01',{'previous_workflow':'actual-ended-owner'})]
    assert json.loads(extra.read_text())['workflow']=='queued02'
    assert json.loads(default.read_text())=={'workflow':'queued56'}
    assert json.loads((tmp_path/'primary-launch.json').read_text())['job_id']=='job-1'
    # A second pass sees the actual primary running; it cannot start queued02.
    monkeypatch.setattr(primary,'command',lambda *_:{'status':'RUNNING'})
    assert primary.promote_allocated_backup(value,extra)is None
    assert len(submitted)==1


def test_only_explicit_release_cancels_workers_owned_by_that_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    rows={
        'owned':{'workflow':'leact-rp-jhh-pear-worker-owned-1','owner_model_workflow':'model-a'},
        'other':{'workflow':'independent-study-1','owner_model_workflow':'model-b'},
        'done':{'workflow':'leact-rp-jhh-pear-worker-done-1','owner_model_workflow':'model-a','scheduler_status':'COMPLETED'}}
    (tmp_path/'primary-native-queue.json').write_text(json.dumps(rows))
    calls=[]
    def run(args,**kwargs):calls.append(args);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(primary.subprocess,'run',run)
    primary.cancel_native_workers('model-a');primary.cancel_native_workers('model-a')
    assert calls==[]  # Failure/retry cleanup must preserve allocations.
    primary.cancel_native_workers('model-a',explicit_user_release=True)
    primary.cancel_native_workers('model-a',explicit_user_release=True)
    assert len(calls)==1 and calls[0][3]=='leact-rp-jhh-pear-worker-owned-1'
    assert json.loads((tmp_path/'primary-native-queue.json').read_text())['owned']['cancel_requested']


def test_cancellation_rejects_an_unrelated_workflow_even_with_a_bad_owner_tag(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-native-queue.json').write_text(json.dumps({
        'bad':{'workflow':'independent-study-1','owner_model_workflow':'model-a'}}))
    monkeypatch.setattr(primary.subprocess,'run',lambda *a,**k:pytest.fail('Unrelated workflow was touched'))
    with pytest.raises(ValueError,match='outside this primary'):
        primary.cancel_native_workers('model-a',explicit_user_release=True)


def test_gpu_device_plugin_rejection_is_retried_but_cuda_oom_is_not():
    from reproductions.leact.cluster.efe_pear_submission import infrastructure_failure
    assert infrastructure_failure({'status':'FAILED','failure_message':
        'Pod was rejected: Allocate failed due to requested number of devices unavailable for nvidia.com/gpu'})
    assert not infrastructure_failure({'status':'FAILED','failure_message':'CUDA out of memory while allocating model tensors'})


@pytest.mark.parametrize('status',['FAILED_CANCELED','RUNNING'])
def test_model_without_a_live_child_never_retargets_native_workers(tmp_path,monkeypatch,status):
    from reproductions.leact.cluster import efe_pear_backup as backup
    from reproductions.leact.cluster import efe_pear_recipe_transition as transition
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-launch.json').write_text(json.dumps({'workflow':'model-without-job',
        'profile':16,'native_workers_launched':False,'standby':False}))
    monkeypatch.setattr(backup,'maintain',lambda *_:None)
    monkeypatch.setattr(primary,'promote_allocated_backup',lambda *_:None)
    monkeypatch.setattr(transition,'tick',lambda *_:None)
    monkeypatch.setattr(primary,'command',lambda *a,**kw:{'status':status,
        'start_time':'2026-11-19T20:00:00','failure_message':'Manual user cancellation'})
    monkeypatch.setattr(primary,'native_builds',lambda *_:pytest.fail(
        'A scheduler-ended allocation or a model without a child cannot take native workers'))
    result=primary.tick()
    assert result['scheduler_status']==status
    assert not result['native_workers_launched']


def test_checkpoint_recovery_keeps_capacity_without_activating_an_unresumable_run(tmp_path,monkeypatch):
    from reproductions.leact.cluster import efe_pear_backup as backup
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-launch.json').write_text(json.dumps({'workflow':'idle-capacity','standby':True}))
    (tmp_path/'checkpoint-recovery-required.json').write_text(json.dumps({
        'required':True,'reason':'Optimizer shards incomplete; awaiting explicit recovery choice'}))
    maintained=[]
    monkeypatch.setattr(backup,'maintain',lambda path:maintained.append(path))
    monkeypatch.setattr(primary,'command',lambda *a,**kw:{'status':'RUNNING'})
    monkeypatch.setattr(primary,'promote_allocated_backup',lambda *_:pytest.fail('Must not activate an unresumable writer'))
    result=primary.tick()
    assert result['phase']=='awaiting_explicit_checkpoint_recovery'
    assert result['scheduler_status']=='RUNNING'and result['standby']is True
    assert maintained==[tmp_path/'backup-allocation.json']


def test_verified_source_repair_still_waits_for_all_prelaunch_gates(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-launch.json').write_text(json.dumps({'workflow':'terminated-attempt',
        'terminal_confirmed':True,'source_repair_ready':True,'profile':0,'pool':'groot-h100-01'}))
    attempted=[]
    def launch(profile,*,pool):
        attempted.append((profile,pool))
        raise primary.PendingAdmission('Exact primary command remains pending')
    monkeypatch.setattr(primary,'launch',launch)
    monkeypatch.setattr(primary,'command',lambda *a,**k:pytest.fail('Old terminated job was queried instead of checking new admission'))
    value=primary.tick()
    assert attempted==[(0,'groot-h100-01')]and value['primary_rl_updates']==0
    assert value['phase']=='awaiting_primary_qualification'


def test_trainer_cpu_reservations_fit_queue_controller_driver_and_all_gpu_ranks():
    with pytest.raises(ValueError,match='cannot fit'):
        primary.validate_trainer_cpu_capacity(56)
    assert primary.validate_trainer_cpu_capacity(64)==58


def test_primary_model_rejects_a_pool_without_the_qualified_shared_volume():
    with pytest.raises(ValueError,match='qualified AMLFS07 H100 pool'):
        primary.launch(pool='groot-l40s-01')


def test_allocation_retries_keep_the_smaller_layout_after_the_initial_sweep():
    assert[primary.allocation_profile(i)for i in range(4)]==primary.PROFILES
    retries=[primary.allocation_profile(i)for i in range(4,24)]
    assert{nodes for nodes,_ in retries}=={4}
    assert{priority for _,priority in retries}=={'HIGH','LOW'}


@pytest.mark.parametrize('pool',['groot-l40-02','groot-l40-03','groot-l40s-01','groot-l40s-03'])
def test_explicit_l40_high_priority_overrides_the_previous_low_fallback(pool):
    from reproductions.leact.cluster.efe_pear_submission import scheduling_priority
    assert scheduling_priority(pool,'LOW',{'l40_priority':'HIGH'})=='HIGH'
    assert scheduling_priority('groot-h100-01','LOW',{'l40_priority':'HIGH'})=='LOW'
    assert scheduling_priority(pool,'LOW',{})=='LOW'


def test_four_rollout_nodes_have_room_for_the_full_proxy_pools_and_inference():
    with pytest.raises(ValueError,match='cannot fit inference and simulator proxies'):
        primary.validate_rollout_cpu_capacity(4,56)
    assert primary.validate_rollout_cpu_capacity(4,64)==248
    assert primary.validate_rollout_cpu_capacity(6,56)==288


def test_queue_failure_rechecks_admission_and_preserves_the_next_allocation_profile(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-launch.json').write_text(json.dumps({'workflow':'timed-out-model',
        'profile':6,'pool':'groot-h100-01','native_workers_launched':False}))
    monkeypatch.setattr(primary,'command',lambda *a,**k:{'status':'FAILED_QUEUE_TIMEOUT'})
    monkeypatch.setattr(primary,'cancel_native_workers',lambda workflow:None)
    attempted=[]
    def launch(profile,*,pool):
        attempted.append((profile,pool));raise primary.PendingAdmission('Pool migration remains pending')
    monkeypatch.setattr(primary,'launch',launch)
    assert primary.tick()['phase']=='awaiting_primary_qualification'
    assert primary.tick()['phase']=='awaiting_primary_qualification'
    assert attempted==[(7,'groot-h100-01'),(7,'groot-h100-01')]


def test_simulator_pool_follows_the_explicit_l40s01_override():
    from reproductions.leact.cluster.efe_pear_submission import simulator_pool
    assert simulator_pool('groot-l40-02',{'l40_pool':'groot-l40s-01'})=='groot-l40s-01'
    with pytest.raises(ValueError):simulator_pool('groot-l40-02',{'l40_pool':'arbitrary-pool'})


def test_renderer_migration_check_cannot_inherit_the_primary_training_flag(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setattr(primary,'BUILD_ROOT',tmp_path)
    source=tmp_path/'efe_pear_rtx_check.py';source.write_text('pass\n')
    build=primary.worker_build({'qualification_only':True},{source.name:source},source.name,
        'pinned-renderer-image','python3',gpu=1,cpu=8,memory='64Gi',storage='150Gi',platform='ovx-l40s')
    task=yaml.safe_load((build/'workflow.yaml').read_text())['workflow']['tasks'][0]
    assert task['environment']['PEAR_PRIMARY_TRAINING']=='0'
    assert 'credentials'not in task


@pytest.mark.parametrize('pool',['groot-h100-01','groot-h100-02'])
def test_explicit_h100_priority_retains_quota_eligibility_after_a_low_capacity_fallback(pool):
    from reproductions.leact.cluster.efe_pear_submission import scheduling_priority
    assert scheduling_priority(pool,'LOW',{'h100_priority':'HIGH'})=='HIGH'
    assert scheduling_priority(pool,'LOW',{'l40_priority':'HIGH'})=='LOW'


def test_compatible_graphics_pools_can_use_low_while_the_allocated_pool_and_h100_stay_high():
    from reproductions.leact.cluster.efe_pear_submission import scheduling_priority
    control={'h100_priority':'HIGH','l40_priority':'HIGH','l40_priority_by_pool':{
        'groot-l40-02':'LOW','groot-l40-03':'LOW','groot-l40s-01':'HIGH'}}
    assert scheduling_priority('groot-l40-02','HIGH',control)=='LOW'
    assert scheduling_priority('groot-l40-03','HIGH',control)=='LOW'
    assert scheduling_priority('groot-l40s-01','LOW',control)=='HIGH'
    assert scheduling_priority('groot-h100-01','LOW',control)=='HIGH'
