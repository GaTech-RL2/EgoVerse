"""Run isolated Harbor oracle/nop controls on frozen EFE development tasks.

This workflow never calls a teacher, trains a model, or exports oracle solutions
as demonstration data. It preserves task instructions, limits and grading.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT=Path('/tmp/efe-native')
SPEC=Path('/tmp/efe-native-spec.json')
WORKFLOW=os.environ.get('EFE_WORKFLOW_ID','local')
ARCHIVE=Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-native-controls-20261026-v1')/WORKFLOW


def now():return datetime.now(timezone.utc).isoformat()


def write(name,value):
    data=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    for parent in (ROOT,ARCHIVE):
        parent.mkdir(parents=True,exist_ok=True)
        path=parent/name;path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(data);temp.replace(path)


def progress(phase,detail,**extra):
    value={'workflow':WORKFLOW,'updated_utc':now(),'phase':phase,'detail':detail,
           'benchmark_rollouts_started':False,'training_started':False,**extra}
    write('progress.json',value);print(json.dumps(value),flush=True)


def run(name,argv,*,cwd=None,env=None,timeout=1800,required=True):
    path=ARCHIVE/'logs'/f'{name}.log';path.parent.mkdir(parents=True,exist_ok=True)
    progress(name,'Running '+name.replace('_',' '))
    with path.open('w') as log:
        process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Stop only this process group; do not signal unrelated jobs.
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    data=path.read_bytes()
    value={'returncode':code,'log_sha256':hashlib.sha256(data).hexdigest(),'log_path':str(path),
           'log_tail':data[-2200:].decode(errors='replace')}
    if required and code:
        write('setup-error.json',{'stage':name,**value})
        raise RuntimeError(f'{name} failed with code {code}; '+value['log_tail'][-600:])
    return value


def fetch(spec,path):
    req=urllib.request.Request(spec['url'],headers={'User-Agent':'EFE-reproduction'})
    with urllib.request.urlopen(req,timeout=180) as response,path.open('wb') as out:
        while chunk:=response.read(1024*1024):out.write(chunk)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('Downloaded binary checksum mismatch: '+path.name)


def control_verdict(agent,result):
    if agent not in {'oracle','nop'}:raise ValueError('Only diagnostic agents are allowed')
    rewards=(result.get('verifier_result') or {}).get('rewards') or {}
    score=rewards.get('success',rewards.get('reward'))
    if result.get('exception_info') or score not in (0,1):
        return 'infrastructure_failure',score
    if any(rewards.get(k) for k in ('infra_error','grader_error','instance_mismatch','resource_monitor_missing')):
        return 'infrastructure_failure',score
    return ('passed' if score==int(agent=='oracle') else 'unexpected_verdict'),score


def find_trial_result(results,task):
    matches=[r for r in results if r.get('task_name') in {task,'embodied-first-exam/'+task}]
    if len(matches)>1:raise ValueError('Multiple trial results for one diagnostic control')
    return matches[0] if matches else None


def setup(spec):
    ROOT.mkdir(parents=True,exist_ok=True);ARCHIVE.mkdir(parents=True,exist_ok=True)
    driver=spec.get('podman_storage_driver','vfs')
    if driver not in {'vfs','overlay'}:raise ValueError('Unsupported container storage driver')
    packages='podman uidmap slirp4netns git curl ca-certificates'+(' fuse-overlayfs' if driver=='overlay' else '')
    run('system_packages',['bash','-c','apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq '+packages],timeout=600)
    progress('install_pinned_container_tools','Installing checksum-pinned Podman 5.8.0 and Compose 2.40.3')
    fetch(spec['podman'],ROOT/'podman.tar.gz')
    with tarfile.open(ROOT/'podman.tar.gz') as tar:tar.extractall(ROOT/'podman',filter='data')
    binaries=list((ROOT/'podman').glob('*/usr/local/bin/podman'))
    if len(binaries)!=1:raise ValueError('Unexpected static Podman archive layout')
    binary=binaries[0];prefix=binary.parents[3]
    bindir=ROOT/'bin';bindir.mkdir(exist_ok=True)
    fetch(spec['compose'],bindir/'docker-compose');(bindir/'docker-compose').chmod(0o755)
    conmon=next(prefix.rglob('conmon'))
    conf=ROOT/'containers.conf'
    conf.write_text('[engine]\ncompose_warning_logs = false\nhelper_binaries_dir = ['+json.dumps(str(prefix/'usr/local/lib/podman'))+']\nconmon_path = ['+json.dumps(str(conmon))+']\ncgroup_manager = "cgroupfs"\nevents_logger = "file"\n[engine.runtimes]\ncrun = ['+json.dumps(str(binary.parent/'crun'))+']\n')
    storage=ROOT/'storage.conf'
    storage_text=('[storage]\ndriver='+json.dumps(driver)+'\nrunroot='+json.dumps(str(ROOT/'run'))+
                  '\ngraphroot='+json.dumps(str(ROOT/'containers'))+'\n')
    if driver=='overlay':
        # Explicit opt-in on fresh isolated jobs. Overlay shares image layers;
        # VFS copied them until the Isaac jobs exceeded 800 GiB of scratch.
        if not Path('/dev/fuse').is_char_device():
            raise RuntimeError('Requested fuse-overlayfs storage needs /dev/fuse in the allocated job')
        storage_text+='[storage.options.overlay]\nmount_program="/usr/bin/fuse-overlayfs"\n'
    storage.write_text(storage_text)
    env={**os.environ,'PATH':str(binary.parent)+':'+str(bindir)+':'+os.environ['PATH'],
         'CONTAINERS_CONF':str(conf),'CONTAINERS_STORAGE_CONF':str(storage),
         'PODMAN_COMPOSE_PROVIDER':str(bindir/'docker-compose'),
         # Harbor parses stdout from compose exec for package-manager detection.
         # Podman's provider banner otherwise contaminates the returned value.
         'PODMAN_COMPOSE_WARNING_LOGS':'false',
         'DOCKER_HOST':'unix:///tmp/efe-native/podman.sock','PYTHONUNBUFFERED':'1'}
    log=(ARCHIVE/'logs/podman-service.log').open('w')
    service=subprocess.Popen(['podman','system','service','--time=0',env['DOCKER_HOST']],env=env,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(60):
        if (ROOT/'podman.sock').exists():break
        if service.poll() is not None:raise RuntimeError('Podman API exited before socket creation')
        time.sleep(1)
    run('compose_socket',['podman','compose','ls'],env=env,timeout=60)
    run('container_storage_info',['podman','info','--format','json'],env=env,timeout=60)
    reference=ROOT/'reference'
    run('clone_reference',['git','clone','--filter=blob:none','--no-checkout',spec['repository'],str(reference)],env=env,timeout=300)
    run('sparse_init',['git','sparse-checkout','init','--no-cone'],cwd=reference,env=env,timeout=60)
    patterns=['/README.md','/LICENSE','/pyproject.toml','/scripts/','/docs/','/ci_checks/','/embodied_first_exam/','/images/monitor/','/tasks/*/README.md']
    patterns += ['/'+r['path']+'/' for r in spec['tasks']]
    run('sparse_paths',['git','sparse-checkout','set','--no-cone',*patterns],cwd=reference,env=env,timeout=300)
    run('checkout_reference',['git','checkout','--detach',spec['revision']],cwd=reference,env=env,timeout=600)
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference,text=True).strip()
    if pin!=spec['revision']:raise ValueError('Wrong reference commit')
    for row in spec['tasks']:
        task=reference/row['path']
        if hashlib.sha256((task/'task.toml').read_bytes()).hexdigest()!=row['task_toml_sha256']:
            raise ValueError('Frozen task metadata changed: '+row['task'])
        if (task/'DIGEST').read_text().strip().removeprefix('sha256:')!=row['upstream_digest']:
            raise ValueError('Frozen task digest changed: '+row['task'])
    run('install_pinned_harbor',[sys.executable,'-m','pip','install','--disable-pip-version-check','harbor==0.23.0'],env=env,timeout=600)
    run('install_efe_hook',[sys.executable,'-m','pip','install','--no-deps','-e',str(reference)],env=env,timeout=300)
    env['EFE_ROOT']=str(reference)
    run('record_python_versions',[sys.executable,'-m','pip','freeze'],env=env,timeout=60)
    run('harbor_version',['harbor','--version'],env=env,timeout=60)
    # Import and task-selection failures are setup failures, before any trial.
    for index,row in enumerate(spec['tasks']):
        run(f'validate_task_{index:02d}',['harbor','run','-d','embodied-first-exam@all','-p',row['path'],'--list'],cwd=reference,env=env,timeout=60)
    return reference,env,service,log


def main():
    spec=json.loads(SPEC.read_text());write('spec.json',spec)
    results={'workflow':WORKFLOW,'purpose':'Development-only isolated oracle/nop controls; no teacher/student benchmark score',
             'trials':[],'all_controls_passed':False,'upstream_revision':spec['revision']}
    service=None;log=None
    try:
        reference,env,service,log=setup(spec)
        for index,row in enumerate(spec['tasks']):
            for agent in ('oracle','nop'):
                job=f'{index:02d}-{agent}'
                command=['harbor','run','-d','embodied-first-exam@all','-p',row['path'],'-a',agent,'-e','podman','-y',
                         '-k','1','-n','1','--max-retries','0','--job-name',job,'-o',str(ARCHIVE/'jobs')]
                receipt=run('control_'+job,command,cwd=reference,env=env,timeout=7200,required=False)
                candidates=list((ARCHIVE/'jobs'/job).glob('*/result.json'))
                parsed=[json.loads(p.read_text()) for p in candidates]
                trial=find_trial_result(parsed,row['task'])
                status,score=control_verdict(agent,trial) if trial else ('infrastructure_failure',None)
                if receipt['returncode']!=0:status='infrastructure_failure'
                record={'task':row['task'],'suite':row['suite'],'agent':agent,'status':status,'score':score,
                        'detail':f'Expected {int(agent=="oracle")}; actual {score}. Diagnostic only.',
                        'returncode':receipt['returncode'],'log_sha256':receipt['log_sha256'],
                        'error_tail':receipt['log_tail'] if status!='passed' else None,
                        'trial_result_sha256':hashlib.sha256(json.dumps(trial,sort_keys=True).encode()).hexdigest() if trial else None,
                        'video_count':len(list((ARCHIVE/'jobs'/job).rglob('*.mp4')))}
                results['trials'].append(record);write('results.json',results)
                progress('simulator_controls',f'{len(results["trials"])} of {2*len(spec["tasks"])} native controls finished; no teacher/student rollouts yet',current_task=row['task'])
        results['all_controls_passed']=len(results['trials'])==2*len(spec['tasks']) and all(r['status']=='passed' for r in results['trials'])
        run('resolved_images',['podman','images','--format','json'],env=env,timeout=60,required=False)
        results['completed_utc']=now();write('results.json',results)
        progress('controls_passed' if results['all_controls_passed'] else 'controls_need_review',
                 'Native simulator controls passed. Teacher image/tool round trip is next.' if results['all_controls_passed'] else 'Native simulator controls completed with failures; inspect infrastructure diagnostics before teacher collection.')
    except Exception as error:
        results['setup_error']=str(error);results['completed_utc']=now();write('results.json',results)
        progress('setup_failed','Native simulator setup failed: '+str(error)[-700:])
    finally:
        write('complete.json',{'completed_utc':now(),'all_controls_passed':results['all_controls_passed']})
        # Give the local collector time to verify and persist the summaries.
        for _ in range(60):
            if (ROOT/'collected').exists():break
            time.sleep(5)
        if service is not None:service.terminate()
        if log is not None:log.close()


if __name__=='__main__':main()
