"""Obtain a bounded native reward batch for the PEAR optimizer qualification.

The first base-policy group passed transport/provenance but was all failure.
Use two already-admitted TRAINING tasks with observed base-policy competence;
these draws are not held-out evaluation results or primary trained-RL results.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import yaml

from .scale_efe import read, write, submit, BUILD_ROOT
from .launch_efe_pear import DEST, COVERAGE
from .launch_efe_pear_native import prepare as first_group
from .launch_efe_training import freeze_sources

GROUPS = {'door-lock':'metaworldplus-door-lock-i00-standard',
          'coffee-button':'metaworldplus-coffee-button-i00-standard'}


def training_successes(evidence, membership):
    result = {}
    for workflow, value in evidence.get('evaluations', {}).items():
        if not workflow.startswith('leact-efe-coverage-eval-base-'):
            continue
        for trial in value.get('results', {}).get('trials', []):
            if (trial.get('partition')=='training_evaluation' and trial.get('status')=='graded'
                and trial.get('score')==1 and trial['task'] in membership['training_tasks']):
                result[trial['task']] = {'workflow':workflow, 'trial':trial,
                    'trial_receipt_sha256':hashlib.sha256(json.dumps(trial,sort_keys=True).encode()).hexdigest()}
    return result


def prepare():
    original = first_group()
    result = read(DEST/'native-qualification-result.json')
    if not result.get('success') or result['group']['filter_reason']!='zero_reward_variance':
        raise ValueError('This bounded follow-up is for the reviewed constant-reward native group')
    membership = read(DEST/'training-membership.json')
    eligible = training_successes(read(COVERAGE/'evidence.json'), membership)
    tasks = {r['task']:r for r in read(COVERAGE/'plan.json')['tasks']}
    builds = []
    for tag, task_name in GROUPS.items():
        if task_name not in eligible or task_name not in tasks:
            raise ValueError('Qualification task lacks a successful admitted training-scene control')
        folder = BUILD_ROOT/('efe-pear-update-qualification-20261115-'+tag+'-v1')
        if not (folder/'source-manifest.json').exists():
            folder.mkdir(exist_ok=True)
            for name in read(original/'source-manifest.json'):
                shutil.copyfile(original/name, folder/name)
            spec = read(folder/'spec.json')
            spec['tasks'] = [dict(tasks[task_name],partition='train_candidates',collection_attempt=i+1,
                                 pear_episode_index=i) for i in range(8)]
            spec.update(agent_slug='pear-update-'+tag, agent_label='PEAR optimizer qualification · '+tag,
                purpose='Collect one eight-episode training group with exact probabilities for a discarded real-update qualification.',
                scope='Adaptive training-only qualification after an all-failure group. Selected using previously observed competence on admitted training scenes; no held-out outcome is used.',
                local_results_directory='pear-v1/optimizer-qualification/'+tag)
            spec['pear'].update(role='native_optimizer_qualification',
                selection_evidence=eligible[task_name], primary_comparison_data=False,
                qualification_only=True)
            write(folder/'spec.json', spec)
            workflow = yaml.safe_load((folder/'workflow.yaml').read_text())
            workflow['workflow']['name'] = 'leact-efe-pear-update-qual-'+tag+'-v1'
            (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
            freeze_sources(folder)
        destination = DEST/'optimizer-qualification'/tag
        destination.mkdir(parents=True,exist_ok=True)
        builds.append({'id':tag,'task':task_name,'build':str(folder),'directory':str(destination)})
    path = DEST/'optimizer-qualification-plan.json'
    value = {'groups':builds, 'planned_native_episodes':16, 'max_parallel_nodes':2,
        'basis':'Base Qwen had native successes on these exact already-admitted training tasks. No development/core/held-out outcome was used.',
        'reason':'First push group had eight zero rewards: native integration passed but no GRPO learning signal was available.',
        'optimizer_updates':0, 'retain_all_outcomes':True,
        'purpose':'Obtain an informative native batch for a discarded optimizer/restart qualification before the final common warmstart is available.',
        'primary_comparison':'Both RL arms still start from the same final 737-update warmstart, with the frozen 36-task evaluation panel.'}
    if path.exists() and read(path)!=value:
        raise ValueError('Optimizer-qualification task plan changed')
    write(path,value)
    publish_progress()
    return builds


def publish_progress():
    path = DEST/'optimizer-qualification-plan.json'
    if not path.exists():return
    scope = read(path)
    rows = []
    for group in scope['groups']:
        state = read(Path(group['directory'])/'status.json', {})
        result = read(Path(group['directory'])/'native-qualification-result.json', {})
        rows.append({**group, **state, 'group_admitted':result.get('group',{}).get('admitted')})
    with (DEST/'optimizer-qualification-plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan = read(DEST/'plan.json')
        plan['optimizer_qualification'] = {**scope,'groups':rows,
            'updated_utc':datetime.now(timezone.utc).isoformat(),
            'summary':f"Optimizer qualification: {sum(r.get('native_rollouts',0) for r in rows)}/16 episodes verified across door-lock and coffee-button. These tasks were selected using prior performance on admitted training scenes only. All outcomes are retained; primary RL training has not started."}
        write(DEST/'plan.json',plan)


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');args=p.parse_args()
    for group in prepare():
        build, destination = Path(group['build']), Path(group['directory'])
        if not args.submit:
            print(build);continue
        queue_path = destination/'native-qualification-queue.json'
        if queue_path.exists():
            print(json.dumps({'group':group['id'],'existing_queue':read(queue_path)}),flush=True)
            continue
        receipt = submit(build,priority='HIGH')
        write(queue_path,{'attempts':[{'workflow':receipt['name'],'build':str(build),
            'priority':'HIGH','submitted_utc':datetime.now(timezone.utc).isoformat()}]})
        write(destination/'status.json',{'workflow':receipt['name'],'scheduler_status':'PENDING',
            'phase':'native_qualification_submitted','native_rollouts':0,'optimizer_updates':0,
            'updated_utc':datetime.now(timezone.utc).isoformat()})
        with (destination/'native-monitor.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-S','-u','-m','reproductions.leact.cluster.monitor_efe_pear_native',
                '--group',group['id']],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(destination/'native-monitor-launch.json',{'pid':child.pid})
        publish_progress()
        print(json.dumps({'group':group['id'],'workflow':receipt['name'],'monitor_pid':child.pid}),flush=True)
    if args.submit:
        from ..embodied_first_exam import report
        report.publish()


if __name__=='__main__':main()
