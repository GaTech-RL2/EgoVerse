"""Validate the complete frozen EFE control inventory, independently of RL.

Each GPU runs one native task at a time. These jobs supply environment receipts,
never teacher demonstrations or primary optimizer steps.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .launch_efe_pear_scaled import DEST,REMOTE,TEMPLATE
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,read,write,submit,command

ARM='moe-warmstart496-pearplusplus'
SUITES=('abc-sim','dexjoco','libero-max','metaworldplus','robocasa','robodojo','robolab','vlabench','yamlab')
QUEUE=DEST/'expanded-validation-queue.json'


def allowed():
    c=read(DEST/'control.json',{})
    if c.get('paused')or ARM not in c.get('enabled_training_arms',[]):
        raise RuntimeError('Warmstart preparation is not authorized')


def prepare(suite,*,resume=False):
    allowed()
    if suite not in SUITES:raise ValueError('Unapproved simulator family')
    members=read(DEST/'proposed-candidate-membership-v2.json')
    audit=read(DEST/'source-scene-audit.json')
    if audit['protected_state_errors']:raise ValueError('Protected source-state audit is incomplete')
    rows=[r for r in members['rows']if r['suite']==suite]
    if not rows:raise ValueError('An approved family has no task inventory')
    original=read(BUILD_ROOT/'efe-pear-native-qualification-20261114-v1/spec.json')
    spec={k:original[k]for k in ['repository','revision','podman','compose','fixed_action_config']}
    spec.update(tasks=rows,tracking_run_id=ARM,max_workers=4 if suite not in {'robodojo','robolab','yamlab'}else 1,enable_gpu_cdi=True,
        podman_storage_driver='overlay',archive_root=REMOTE+'/expanded-control-validation-v1',
        expected_scenes={r['task']:audit['tasks'][r['task']]['scene']['scene_sha256']for r in rows},
        source_scene_checks={r['task']:audit['tasks'][r['task']]['source_scene_check_passed']for r in rows},
        primary_rl_updates=0,suite=suite,scope='All proposed task variants; primary RL is separate.')
    spec['fixed_action_config']={**spec['fixed_action_config'],'model':'external-pear-policy',
        'allowed_action_types_by_robot':{r:['joint_position']for r in
            ['franka-allegro-bimanual','franka-robotiq-2f85','i2rt-yam-bimanual']}}
    if resume:
        previous=read(QUEUE)[suite]
        scheduler=command(['osmo','workflow','query',previous['workflow'],'--format-type','json'])
        if scheduler['status']not in {'COMPLETED','FAILED'}:
            raise ValueError('Do not resubmit a control suite while its previous validation is active')
        receipt=DEST/'validation-receipts'/suite/'checks.json'
        checks=read(receipt,{})
        if len(checks)!=len(rows):raise ValueError('Download the complete prior checks before resuming failures')
        if all(v.get('success')or v.get('qualification')=='protected_scene_check_failed'for v in checks.values()):
            raise ValueError('No failed native controls remain in this suite')
        spec['resume_checks']={'path':REMOTE+'/expanded-control-validation-v1/'+previous['workflow']+'/checks.json',
                               'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
    if suite in {'robodojo','robolab','yamlab'}:
        spec['prebuild_native_base']={'cache_root':REMOTE+'/native-image-cache',
                                     'clean_locked_venv':suite in {'robodojo','yamlab'}}
    files=['efe_pear_bridge.py','efe_pear_context.py','efe_action_loop.py','efe_action_agent.py',
           'efe_native_pilot.py','efe_pear_gpu_cdi.py','efe_pear_catalog_probe.py',
           'efe_pear_native_image.py','efe_pear_editable_restore.py','efe_pear_checkpoint_archive.py',
           'efe_pear_observation_check.py']
    digests={f:hashlib.sha256((BASE/'cluster'/f).read_bytes()).hexdigest()for f in
             files+['efe_pear_control_admission.py','efe_pear_scene_audit.py']}
    key=hashlib.sha256(json.dumps([spec,digests],sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/f'rp-jhh-pear-envcheck-{suite}-{key}'
    if (build/'source-manifest.json').exists():return build
    build.mkdir(parents=True,exist_ok=True);write(build/'spec.json',spec)
    for f in files:shutil.copyfile(BASE/'cluster'/f,build/f)
    shutil.copyfile(BASE/'cluster/efe_pear_scene_audit.py',build/'scene_identity.py')
    shutil.copyfile(BASE/'cluster/efe_pear_control_admission.py',build/'worker.py')
    w=yaml.safe_load(TEMPLATE.read_text());w['workflow']['name']='leact-'+build.name
    w['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'24h'}
    w['workflow']['labels']={'pear_run_id':ARM,'pear_phase':'all_tasks_'+suite}
    w['workflow']['resources']['default'].update(gpu=8,cpu=48,memory='512Gi',storage='300Gi')
    task=w['workflow']['tasks'][0];task.pop('credentials',None);task['privileged']=True
    task['args']=['-u','/tmp/efe-pear-model-check/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-model-check/'+p.name}
                   for p in sorted(build.iterdir())if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(build)
    return build


def launch(suites,*,resume=False):
    allowed()
    with (DEST/'expanded-validation-submit.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(QUEUE,{})
        for suite in suites:
            build=prepare(suite,resume=resume)
            if queue.get(suite,{}).get('build')==str(build):continue
            receipt=submit(build,priority='HIGH',pool='groot-h100-01',nodes=1)
            previous=queue.get(suite,{})
            history=previous.get('history',[])+([{k:v for k,v in previous.items()if k!='history'}]if previous else[])
            queue[suite]={'build':str(build),'workflow':receipt['name'],'attempt':0,'priority':'HIGH',
                          'submitted_utc':datetime.now(timezone.utc).isoformat(),'gpu':8,'history':history}
            write(QUEUE,queue);print(json.dumps({'suite':suite,**queue[suite]}),flush=True)


def tick():
    from .launch_efe_coverage import infra_retryable
    allowed();queue=read(QUEUE,{});states={}
    for suite,launch in queue.items():
        result=command(['osmo','workflow','query',launch['workflow'],'--format-type','json'])
        value={**launch,'scheduler_status':result['status'],'primary_rl_updates':0}
        if result.get('start_time'):
            try:
                logs=subprocess.run(['osmo','workflow','logs',launch['workflow'],'-n','25'],
                                    capture_output=True,text=True,timeout=12)
                lines=logs.stdout.splitlines()
            except subprocess.TimeoutExpired as exc:
                previous=read(DEST/'expanded-validation-status.json',{}).get('suites',{}).get(suite,{})
                if previous.get('workflow')==launch['workflow']and previous.get('worker'):value['worker']=previous['worker']
                value['log_error']='OSMO log request timed out; scheduler state remains authoritative.'
                partial=exc.stdout or''
                lines=(partial.decode(errors='replace')if isinstance(partial,bytes)else partial).splitlines()
            for line in lines:
                if 'PEAR_ADMISSION_CHECK 'in line:
                    try:value['worker']=json.JSONDecoder().raw_decode(line.split('PEAR_ADMISSION_CHECK ',1)[1])[0]
                    except ValueError:pass
        if (suite in {'robodojo','robolab','yamlab'}and result['status']=='RUNNING'
                and value.get('worker',{}).get('phase')=='native_simulator_base_ready'
                and not launch.get('rtx_execution_required')):
            # The CPU-heavy image build is useful on this allocated builder.
            # Its published/cache-verified OCI archive is retained, but an H100
            # cannot qualify Isaac's RGB sensors. Stop only this named builder.
            if (not launch['workflow'].startswith('leact-rp-jhh-pear-envcheck-'+suite+'-')
                    or result.get('labels',{}).get('pear_phase')!='all_tasks_'+suite):
                raise ValueError('Refusing to stop a workflow outside this validation suite')
            cancellation=subprocess.run(['osmo','workflow','cancel',launch['workflow']],
                capture_output=True,text=True,timeout=30,check=True)
            queue[suite]={**launch,'image_cache_ready':True,'rtx_execution_required':True}
            write(QUEUE,queue);value.update(image_cache_ready=True,rtx_execution_required=True)
        if launch.get('rtx_execution_required')or value.get('rtx_execution_required'):
            value['phase']='native_image_cached_rtx_execution_pending'
        if result['status'].startswith('FAILED')and infra_retryable(result)and launch['attempt']<12:
            original=Path(launch['build']);target=BUILD_ROOT/(original.name.split('-retry-')[0]+'-retry-'+str(launch['attempt']+1))
            if not(target/'source-manifest.json').exists():
                target.mkdir(exist_ok=True)
                for name,digest in read(original/'source-manifest.json').items():
                    if hashlib.sha256((original/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen validation source changed')
                    shutil.copyfile(original/name,target/name)
                w=yaml.safe_load((target/'workflow.yaml').read_text());w['workflow']['name']='leact-'+target.name
                (target/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(target)
            receipt=submit(target,priority='LOW',pool='groot-h100-01',nodes=1)
            queue[suite]={**launch,'build':str(target),'workflow':receipt['name'],'attempt':launch['attempt']+1,'priority':'LOW'}
            write(QUEUE,queue)
        states[suite]=value
        write(DEST/'expanded-validation-status.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
            'primary_rl_updates':0,'phase':'expanded_environment_validation',
            'suites':{**read(DEST/'expanded-validation-status.json',{}).get('suites',{}),**states},
            'required_suites':list(SUITES),'full_curriculum_qualified':False})
    output={'updated_utc':datetime.now(timezone.utc).isoformat(),'primary_rl_updates':0,
            'phase':'expanded_environment_validation','suites':states,
            'required_suites':list(SUITES),'full_curriculum_qualified':False}
    write(DEST/'expanded-validation-status.json',output)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suites',nargs='+',choices=SUITES,default=list(SUITES))
    p.add_argument('--submit',action='store_true');p.add_argument('--watch',action='store_true')
    p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if a.submit:launch(a.suites,resume=a.retry_failed)
    elif not a.watch:
        for suite in a.suites:print(prepare(suite))
    if a.watch:
        with (DEST/'expanded-validation-watch.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            deadline=time.monotonic()+48*3600
            while time.monotonic()<deadline:
                try:tick()
                except Exception as exc:write(DEST/'expanded-validation-watch-error.json',{'error':str(exc)[-800:]})
                time.sleep(30)


if __name__=='__main__':main()
