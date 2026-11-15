"""Incrementally profile curated traces on an already allocated evaluation node."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write
from .monitor_efe_preflight import remote
from .launch_efe_leact_warm96 import collect
from ..embodied_first_exam.direct_control_corpus import inventory, refresh

DEST = ROOT / 'direct-control-v1/preparation-profile'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    summary, episodes, excluded, prompt = inventory()
    if not episodes or not summary['student_deployment_prompt_verified']:
        return None
    root = ROOT / 'direct-control-v1'
    sources = ['efe_direct_corpus_profile.py', 'efe_training_data.py', 'efe_budget_gateway.py']
    provenance = {'episodes': [(r['episode'], r['bundle']['sha256'], r['action_audit_sha256'],
        hashlib.sha256(json.dumps(r['trial'], sort_keys=True).encode()).hexdigest()) for r in episodes],
        'prompt': prompt, 'scene_audit': sha(root / 'scene-audit/result.json'),
        'sources': {n: sha(BASE / 'cluster' / n) for n in sources}}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    build = BUILD_ROOT / ('efe-direct-profile-' + signature)
    if (build / 'source-manifest.json').exists():
        return build
    build.mkdir(exist_ok=True)
    for row in episodes:
        row['action_audit_base64'] = base64.b64encode(Path(row.pop('action_audit_source')).read_bytes()).decode()
        row.pop('action_audit')
    spec = {'snapshot': signature, 'episodes': episodes, 'excluded_episodes': excluded,
        'scene_audit_sha256': provenance['scene_audit'],
        'instruction_suffix_sha256': prompt['instruction_suffix_sha256'],
        'base_model': '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-v1/base-model'}
    write(build / 'spec.json', spec)
    for name in sources:
        shutil.copyfile(BASE / 'cluster' / name, build / name)
    shutil.copyfile(BASE / 'cluster/efe_direct_control_prompt.txt', build / 'method.txt')
    (build / 'student-prompt').mkdir(exist_ok=True)
    for name in prompt['files']:
        shutil.copyfile(root / 'students/base-direct/student-prompt-requests' / name, build / 'student-prompt' / name)
    manifest = {str(p.relative_to(build)): sha(p) for p in build.rglob('*') if p.is_file()}
    write(build / 'source-manifest.json', manifest)
    return build


def launch(build, workflow):
    DEST.mkdir(exist_ok=True)
    signature = read(build / 'spec.json')['snapshot']
    dest = DEST / signature
    dest.mkdir(exist_ok=True)
    bundle = dest / ('efe-direct-profile-' + signature + '.tar.gz')
    manifest = read(build / 'source-manifest.json')
    with tarfile.open(bundle, 'w:gz') as archive:
        for name, digest in manifest.items():
            if sha(build / name) != digest:
                raise ValueError('Frozen CPU profile source changed')
            archive.add(build / name, arcname=name)
    with (dest / 'upload.log').open('a') as log:
        subprocess.run(['osmo', 'workflow', 'rsync', 'upload', workflow, 'preflight', str(bundle.resolve()) + ':/tmp',
            '--no-progress', '--timeout', '20'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    remote_root = '/tmp/efe-direct-profile-' + signature
    script = ('import pathlib,subprocess,os,json,tarfile,hashlib\nroot=pathlib.Path(' + repr(remote_root) + ')\n'
        + 'manifest=' + repr(manifest) + '\nbundle=pathlib.Path(' + repr('/tmp/' + bundle.name) + ')\n'
        + 'expected=' + repr(sha(bundle)) + '\n') + '''
if (root/'launch.json').exists():
 print('EFE_CAPTURE '+(root/'launch.json').read_text(),flush=True)
else:
 if hashlib.sha256(bundle.read_bytes()).hexdigest()!=expected:raise ValueError('Profile upload differs')
 root.mkdir(exist_ok=True)
 with tarfile.open(bundle) as archive:archive.extractall(root,filter='data')
 for name,digest in manifest.items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Profile source differs')
 environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
 with (root/'worker.log').open('a') as log:
  child=subprocess.Popen(['/usr/bin/python3',str(root/'efe_direct_corpus_profile.py'),'--spec',str(root/'spec.json')],
   cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result={'pid':child.pid,'root':str(root)}
 (root/'launch.json').write_text(json.dumps(result))
 print('EFE_CAPTURE '+json.dumps(result),flush=True)
'''
    receipt = remote(workflow, script)
    receipt.update(workflow=workflow, snapshot=signature, build=str(build),
        updated_utc=datetime.now(timezone.utc).isoformat())
    write(dest / 'launch.json', receipt)
    write(DEST / 'active.json', receipt)
    return receipt


def service(active):
    DEST.mkdir(exist_ok=True)
    build = prepare()
    available = [r for r in active if r['status']=='RUNNING' and (
        r['name'].startswith('leact-efe-followup-') or r['name'].startswith('leact-efe-direct-warm64-')
        or r['name'].startswith('leact-efe-direct-base-aligned-'))]
    available.sort(key=lambda r:not r['name'].startswith('leact-efe-followup-leact-rebuilt-'))
    current = read(DEST / 'active.json', {})
    if current:
        dest = DEST / current['snapshot']
        if not (dest / 'result.json').exists() and not (dest / 'error.json').exists():
            running = next((r for r in active if r['name'] == current['workflow']), None)
            if running and running['status'] == 'RUNNING':
                collect(current['workflow'], current['root'], dest,
                    ['progress.json', 'result.json', 'error.json', 'states-profile.json', 'trace-admission.json', 'student-interface.json'])
            else:
                write(dest / 'error.json', {'error': 'Hosting evaluation allocation ended before CPU profile collection; retained input bundles remain intact.'})
        if (dest / 'result.json').exists():
            write(DEST / 'latest.json', read(dest / 'result.json'))
            refresh()
        elif not (dest / 'error.json').exists():
            return False
        else:
            # Preserve the failed profile rather than silently changing inputs.
            write(DEST / 'needs-review.json', {'snapshot': current['snapshot'], 'error': read(dest / 'error.json')})
            if not build or read(build / 'spec.json')['snapshot'] == current['snapshot']:
                error=read(dest/'error.json',{}).get('error','')
                if build and available and error.startswith('Hosting evaluation allocation ended'):
                    (dest/'error.json').replace(dest/('host-ended-'+current['workflow']+'.json'))
                    launch(build,available[0]['name'])
                return False
    if not build or current.get('snapshot') == read(build / 'spec.json')['snapshot']:
        return bool(build and (DEST / current['snapshot'] / 'result.json').exists())
    if available:
        launch(build, available[0]['name'])
    return False


if __name__ == '__main__':
    from .scale_efe import command
    active = command(['osmo', 'workflow', 'list', '--name', 'leact', '--status', 'RUNNING', 'PENDING', 'WAITING', '--format-type', 'json'])['workflows']
    service(active)
    print(json.dumps(read(DEST / 'active.json', {})))
