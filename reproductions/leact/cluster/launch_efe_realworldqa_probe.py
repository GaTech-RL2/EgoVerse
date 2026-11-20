"""Qualify batched synchronous inference without launching either PEAR arm."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml

from .launch_efe_realworldqa_retention import ROOT
from .launch_efe_training import freeze_sources
from .scale_efe import BASE, BUILD_ROOT, read, submit, write


def prepare(batch_size=8, *, evaluate=False):
    if batch_size not in {2, 4, 8, 16}:
        raise ValueError('Choose a bounded serving batch')
    if evaluate and batch_size != 8:
        raise ValueError('The v3 evaluation has a frozen batch size of eight')
    source = Path(read(ROOT/'launch.json')['build'])
    build = BUILD_ROOT/('rp-jhh-warm496-realworldqa-v3-r10' if evaluate else
                       'rp-jhh-vqa-serving-probe-b'+str(batch_size)+'-v1')
    if (build/'source-manifest.json').exists():
        return build
    for name, digest in read(source/'source-manifest.json').items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source evaluation payload changed')
    build.mkdir(exist_ok=True)
    spec = read(source/'spec.json')
    if evaluate:
        spec.update(id='rp_jhh_warm496_realworldqa_v3', study_version=3,
                    shared_root=spec['shared_root'].replace('_v2', '_v3'), batch_size=batch_size)
        spec['serving_profile']['reason'] = 'Batched synchronous serving, admitted only after two throughput-probe batches'
        write(build/'evaluation-spec.json', spec)
        write(build/'serial-fallback-launch.json', read(ROOT/'launch.json'))
    spec.update(id=build.name, purpose='serving_throughput_probe', batch_size=batch_size,
                shared_root=spec['shared_root']+'-serving-probe-b'+str(batch_size),
                probe_output_tokens=512, accuracy_evaluation=False)
    if not evaluate:
        spec['serving_profile']['reason'] = 'Batched synchronous serving qualification only'
    write(build/'spec.json', spec)
    shutil.copyfile(source/'runner.py', build/'retention_runner.py')
    shutil.copyfile(source/'efe_model_cache.py', build/'efe_model_cache.py')
    shutil.copyfile(BASE/'cluster/efe_realworldqa_serving_probe.py', build/'runner.py')
    workflow = yaml.safe_load((source/'workflow.yaml').read_text())
    workflow['workflow']['name'] = 'leact-'+build.name
    workflow['workflow']['timeout'] = {'queue_timeout': '20m', 'exec_timeout': '24h' if evaluate else '90m'}
    names = ['runner.py', 'retention_runner.py', 'efe_model_cache.py', 'spec.json']
    if evaluate:
        names.append('evaluation-spec.json')
        workflow['workflow']['tasks'][0]['args'].extend(['--full-spec', '/tmp/rp-retention/evaluation-spec.json'])
    workflow['workflow']['tasks'][0]['files'] = [
        {'localpath': name, 'path': '/tmp/rp-retention/'+name}
        for name in names]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    freeze_sources(build)
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--evaluate-after-qualification', action='store_true')
    parser.add_argument('--pool', choices=['groot-h100-01', 'groot-h100-02'], default='groot-h100-02')
    parser.add_argument('--priority', choices=['HIGH', 'LOW'], default='HIGH')
    parser.add_argument('--monitor-evaluation', action='store_true')
    args = parser.parse_args()
    if args.monitor_evaluation:
        return monitor_evaluation()
    build = prepare(args.batch_size, evaluate=args.evaluate_after_qualification)
    result = {'build': str(build), 'scope': 'retention evaluation only' if args.evaluate_after_qualification else
              'serving qualification only; no accuracy score or training', 'pool': args.pool, 'priority': args.priority}
    if args.submit:
        result.update(submit(build, priority=args.priority, pool=args.pool, nodes=1))
    write(ROOT/('batched-evaluation-launch.json' if args.evaluate_after_qualification else 'serving-probe-launch.json'), result)
    print(json.dumps(result))


def monitor_evaluation():
    """Release only the redundant serial eval, after batched answers persist."""
    from .scale_efe import command
    from .monitor_efe_preflight import remote
    from .launch_efe_coverage import infra_retryable
    original=read(ROOT/'batched-evaluation-launch.json')
    build=Path(original['build']);fallback=read(build/'serial-fallback-launch.json')
    full=read(build/'evaluation-spec.json')
    if not fallback['name'].startswith('leact-rp-jhh-warm496-realworldqa-v2-'):
        raise ValueError('Refusing to release a workflow outside this serial evaluation')
    with (ROOT/'batched-monitor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+4*3600
        while time.monotonic()<deadline:
            try:
                launch=read(ROOT/'launch.json')
                active_spec=read(Path(launch['build'])/'spec.json')
                if active_spec['shared_root']!=read(build/'spec.json')['shared_root']:
                    write(ROOT/'batched-monitor.json',{'phase':'stopped_after_evaluation_changed'});return
                state=command(['osmo','workflow','query',launch['name'],'--format-type','json'])
                value={'updated_utc':datetime.now(timezone.utc).isoformat(),
                       'scheduler_status':state['status'],'scope':'retention evaluation only'}
                if state['status'].startswith('FAILED')and not infra_retryable(state):
                    write(ROOT/'batched-monitor.json',{**value,'phase':'needs_source_review_serial_fallback_retained'});return
                progress={}
                if state['status']=='RUNNING':
                    script=('from pathlib import Path;import json;p=Path('+repr(full['shared_root'])+
                            ')/"progress.json";print("EFE_CAPTURE "+json.dumps(json.loads(p.read_text())if p.exists()else{}),flush=True)')
                    progress=remote(launch['name'],script)
                if state['status']=='COMPLETED' or progress.get('completed',0)>=8:
                    previous=command(['osmo','workflow','query',fallback['name'],'--format-type','json'])
                    if previous['status']in {'RUNNING','PENDING','WAITING'}:
                        receipt=command(['osmo','workflow','cancel',fallback['name'],'--format-type','json',
                                         '--message','Batched retention evaluation has persisted answers; release redundant serial evaluation.'])
                        write(ROOT/'serial-evaluation-release.json',receipt)
                    write(ROOT/'batched-monitor.json',{**value,'phase':'batched_answers_verified_serial_released',
                                                        'completed_questions':progress.get('completed')});return
                write(ROOT/'batched-monitor.json',{**value,'phase':'waiting_for_batched_answers','progress':progress})
            except Exception as exc:
                write(ROOT/'batched-monitor.json',{'phase':'monitor_retry','error':str(exc)[-500:],
                                                   'updated_utc':datetime.now(timezone.utc).isoformat()})
            time.sleep(30)


if __name__ == '__main__':
    main()
