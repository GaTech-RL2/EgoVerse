"""Submit a CPU-only check of the exact primary command and both node profiles."""
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .efe_pear_submission import submit_reconciled
from .efe_pear_worker_bundle import bootstrap
from .launch_efe_pear_scaled import DEST,REMOTE,TEMPLATE,prepare_training_bundle,require_unpaused
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,read,write


def launch():
    require_unpaused();old=read(DEST/'primary-launch.json')
    primary=read(Path(old['build'])/'spec.json')
    package=prepare_training_bundle();files={n:Path(package['bundle'])/n for n in package['files']}
    files['efe_pear_driver_check.py']=BASE/'cluster/efe_pear_driver_check.py'
    spec={'primary':primary,'output':REMOTE+'/primary-driver-checks',
        'script_sha256':hashlib.sha256((BASE/'cluster/efe_pear_scaled_train.sh').read_bytes()).hexdigest(),
        'files':{n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in files.items()}}
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-primary-driver-check-'+key)
    if not(build/'source-manifest.json').exists():
        build.mkdir()
        for n,p in files.items():shutil.copyfile(p,build/n)
        write(build/'spec.json',spec)
        (build/'bootstrap.py').write_text(bootstrap(build,[*files,'spec.json'],'efe_pear_driver_check.py'))
        image=yaml.safe_load(TEMPLATE.read_text())['workflow']['tasks'][0]['image']
        task={'name':'preflight','image':image,'command':['python3'],
            'args':['-u','/tmp/rp-jhh-driver-check/bootstrap.py'],
            'environment':{'EFE_WORKFLOW_ID':'{{workflow_id}}','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},
            'files':[{'localpath':'bootstrap.py','path':'/tmp/rp-jhh-driver-check/bootstrap.py'}]}
        value={'workflow':{'name':'leact-'+build.name,'resources':{'default':{
            'gpu':0,'cpu':4,'memory':'24Gi','storage':'10Gi','platform':'ovx-l40'}},
            'timeout':{'queue_timeout':'10m','exec_timeout':'30m'},'tasks':[task]}}
        (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False));freeze_sources(build)
    result=submit_reconciled(build,priority='HIGH',pool='groot-l40-02',nodes=1)
    value={**result,'build':str(build),'pool':'groot-l40-02','priority':'HIGH','nodes':1}
    write(DEST/'primary-driver-check-launch.json',value)
    return value


if __name__=='__main__':print(json.dumps(launch()))
