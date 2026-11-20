"""Stage an idle, native-service-only Dojo replacement; never launch a collector."""
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .launch_astra_wide_collection import CREDENTIAL, OWNER, freeze_worker
from .launch_efe_pear_primary import BUILD_ROOT, bootstrap
from .recover_efe_pear_graphics import frozen_files
from .scale_efe import read, write
from .stage_astra_rpc_heartbeat import PROVENANCE, patch_worker
from .switch_astra_wide_profile import PROFILE_ROOT


VERSION = 'astra-native-service-only-standby-v1'
RECEIPT = 'teacher-native-service-standby.json'


def native_only_task(task):
    task.setdefault('environment', {})['PEAR_KEEPER_AUTOSTART'] = '0'
    task.setdefault('credentials', {}).pop(CREDENTIAL, None)
    files = task.setdefault('files', [])
    bootstrap_path = next((item['path'] for item in files if item.get('localpath') == 'bootstrap.py'),
                          '/tmp/pear-native/bootstrap.py')
    item = {'localpath': 'source-manifest.json', 'path': str(Path(bootstrap_path).with_name('source-manifest.json'))}
    if item not in files:files.append(item)


def prepare(source, *, build_root=None):
    source = Path(source)
    files = frozen_files(source)
    spec = read(source/'spec.json')
    if (spec.get('owner_model_workflow') != OWNER or spec.get('teacher_root') != PROFILE_ROOT
        or spec.get('teacher_profile_id') != 'astra62-xhigh-v1'
        or spec.get('teacher_native_only') is not True or spec.get('suite') != 'robodojo'
        or spec.get('teacher_native_entry') != 'efe_pear_primary_rtx.py' or PROVENANCE not in files):
        raise ValueError('Standby requires the reviewed exact-profile teacher-only Dojo RPC bundle')
    original = files['efe_astra_original_rpc_worker.py'].read_text()
    if files['efe_pear_remote_worker.py'].read_text() != patch_worker(original):
        raise ValueError('Standby RPC source differs from the reviewed fix')
    new_spec = {**spec, 'teacher_native_service_only': True, 'teacher_activation_required': True}
    raw_spec = (json.dumps(new_spec, indent=2, sort_keys=True)+'\n').encode()
    key = hashlib.sha256((source/'source-manifest.json').read_bytes()+raw_spec+VERSION.encode()).hexdigest()[:12]
    build = Path(build_root or BUILD_ROOT)/('rp-jhh-astra-native-standby-'+key)
    if (build/'source-manifest.json').exists():
        frozen_files(build)
        return build
    build.mkdir(parents=True)
    for name, path in files.items():shutil.copyfile(path, build/name)
    (build/'spec.json').write_bytes(raw_spec)
    overlay = read(build/PROVENANCE)
    overlay.update(spec_byte_identical=False, spec_sha256=hashlib.sha256(raw_spec).hexdigest(),
                   standby_infrastructure_fields=['teacher_native_service_only', 'teacher_activation_required'],
                   pre_standby_spec_sha256=hashlib.sha256((source/'spec.json').read_bytes()).hexdigest())
    write(build/PROVENANCE, overlay)
    write(build/RECEIPT, {'version': VERSION, 'source_build': str(source),
        'source_manifest_sha256': hashlib.sha256((source/'source-manifest.json').read_bytes()).hexdigest(),
        'spec_changes': {'teacher_native_service_only': True, 'teacher_activation_required': True},
        'runtime_source_changes': [], 'model_collectors_started': 0, 'automatic_child_start': False,
        'activation_requires': 'All scheduler tasks RUNNING; frozen payload checksum verification on the new node; explicit native-only job activation.',
        'old_allocation_and_collector_preserved': True, 'submitted': False})
    names = [*files, RECEIPT]
    (build/'bootstrap.py').write_text(bootstrap(build, [*names, 'spec.json'], 'efe_pear_allocation_keeper.py'))
    workflow = yaml.safe_load((source/'workflow.yaml').read_text())
    wf = workflow['workflow']
    wf['name'] = 'rp-jhh-astra-wide-robodojo-native-only-'+key
    wf.setdefault('labels', {})['teacher_role'] = 'native_service_only'
    if len(wf['tasks']) != 1:raise ValueError('A native standby must retain its single qualified task')
    native_only_task(wf['tasks'][0])
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    freeze_worker(build, [*names, 'spec.json', 'bootstrap.py', 'workflow.yaml'])
    frozen_files(build)
    return build
