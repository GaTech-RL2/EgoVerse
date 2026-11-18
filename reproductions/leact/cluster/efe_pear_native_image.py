"""Build/cache native simulator bases before starting timed Harbor trials.

The optional clean-venv repair changes installation order only: the release's
own requirements.lock and editable source checkouts remain authoritative.
"""
import base64
import fcntl
import hashlib
import json
import inspect
from pathlib import Path
import re
import textwrap

import yaml


def locked_venv_repair(dockerfile):
    try:from . import efe_pear_editable_restore as editable
    except ImportError:import efe_pear_editable_restore as editable
    lines=dockerfile.splitlines(keepends=True);start=None;stop=None
    for i,line in enumerate(lines):
        if line.startswith('RUN ')and'uv pip install --no-deps --index-strategy'in line:
            start=i;stop=i
            while lines[stop].rstrip().endswith('\\'):stop+=1
            if 'environment == requirements.lock'not in ''.join(lines[start:stop+1]):
                raise ValueError('Unreviewed native lock installation block')
            break
    if start is None:raise ValueError('Native locked environment installation block is missing')
    script=inspect.getsource(editable)+'\n'+textwrap.dedent('''\
        import importlib.metadata as md,json,os,pathlib,subprocess,tempfile
        saved_directory=pathlib.Path(tempfile.mkdtemp(prefix='native-editable-installations-'))
        editable_artifacts=capture_editables('/opt/venv',md.distributions(),saved_directory)
        interpreter=os.path.realpath(os.sys.executable)
        if interpreter.startswith('/opt/venv/'):raise ValueError('The venv does not have an external pinned interpreter')
        subprocess.run(['uv','venv','--clear','--python',interpreter,'/opt/venv'],check=True)
        subprocess.run(['uv','pip','install','--python','/opt/venv/bin/python','--no-deps',
            '--index-strategy','unsafe-best-match','--extra-index-url','https://download.pytorch.org/whl/cu128',
            '--extra-index-url','https://pypi.nvidia.com','-r','/opt/requirements.lock'],check=True)
        restore_editables('/opt/venv',saved_directory,editable_artifacts)
        actual=subprocess.check_output(['uv','pip','freeze','--python','/opt/venv/bin/python','--exclude-editable'],text=True)
        expected=pathlib.Path('/opt/requirements.lock').read_text()
        if actual.splitlines()!=expected.splitlines():raise ValueError('Repaired environment differs from the native release lock')
        print('environment == requirements.lock; fresh venv; editable source checkouts preserved',flush=True)
    ''')
    command='RUN python -c "import base64;exec(base64.b64decode(\''+base64.b64encode(script.encode()).decode()+'\'))"\n'
    repaired=''.join(lines[:start])+command+''.join(lines[stop+1:])
    # A serialized installer avoids uv's directory-creation race on the nested
    # overlay filesystem. This affects installation only; the release lock and
    # its exact freeze comparison above still decide every installed version.
    first_from=re.search(r'^FROM .+\n',repaired,re.M)
    if first_from is None:raise ValueError('Native base image declaration is missing')
    return repaired[:first_from.end()]+'ENV UV_CONCURRENT_INSTALLS=1\n'+repaired[first_from.end():]


def prepare_base(reference,row,spec,env,run):
    directory=Path(reference)/row['path']/'environment';recipe=directory/'sim/Dockerfile'
    original=recipe.read_text();compose=yaml.safe_load((directory/'docker-compose.yaml').read_text())
    caches=[tag for service in compose['services'].values()for tag in
            (service.get('build',{}).get('cache_from',[])if isinstance(service.get('build'),dict)else[])
            if 'efe-private:'in tag]
    if len(set(caches))!=1:raise ValueError('Native simulator base cache reference is ambiguous')
    tag=caches[0];match=re.search(r'^FROM .+? AS ([\w-]+)\s*$',original,re.M)
    if match is None:raise ValueError('Native simulator base target is missing')
    repaired=locked_venv_repair(original)if spec.get('clean_locked_venv')else original
    key=hashlib.sha256(repaired.encode()).hexdigest();cache=Path(spec['cache_root'])/row['suite']/key
    cache.mkdir(parents=True,exist_ok=True);archive=cache/'base.oci.tar';manifest=cache/'manifest.json'
    from efe_pear_checkpoint_archive import digest
    with (cache/'build.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if digest(archive)!=saved['archive_sha256']:raise ValueError('Native simulator image cache is corrupt')
            run('load_native_base',['podman','load','-i',str(archive)],env=env,timeout=3600)
            return saved
        local=cache/'Dockerfile';local.write_text(repaired)
        run('prebuild_native_base',['podman','build','--target',match[1],'-t',tag,'-f',str(local),str(recipe.parent)],
            env=env,timeout=6*3600)
        pending=archive.with_suffix('.partial')
        run('cache_native_base',['podman','save','--format','oci-archive','-o',str(pending),tag],env=env,timeout=3600)
        pending.replace(archive)
        saved={'tag':tag,'target':match[1],'original_recipe_sha256':hashlib.sha256(original.encode()).hexdigest(),
               'build_recipe_sha256':key,'clean_locked_venv':bool(spec.get('clean_locked_venv')),
               'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,
               'repair_scope':'Fresh locked venv and preserved editable source checkouts; no task, reward, action or agent-budget changes.'}
        manifest.write_text(json.dumps(saved,indent=2)+'\n');return saved
