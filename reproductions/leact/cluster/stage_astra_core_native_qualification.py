"""Freeze an isolated four-GPU teacher-native qualification; never collect."""
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .launch_astra_wide_collection import OWNER, TEACHER_ROOT, freeze_worker
from .launch_efe_pear_primary import BUILD_ROOT, bootstrap
from .recover_efe_pear_graphics import frozen_files
from .scale_efe import read, write
from .stage_astra_native_gpu_shape import prepare as smaller_shape
from .stage_astra_native_standby import native_only_task


VERSION = 'astra-core4-native-qualification-v1'
SUITES = {'abc-sim', 'dexjoco', 'libero-max', 'metaworldplus', 'robocasa', 'vlabench'}
ARCHIVE_CREDENTIAL = 'rp-jhh-pear-swift-archive-20261118'


def cases_for(spec):
    cases = spec['teacher_plan']['cases']
    selected = []
    for suite in sorted(SUITES):
        rows = [c for c in cases if c['suite'] == suite and c['task'] in spec['teacher_task_names']
                and c.get('admitted') is True and c.get('partition') == 'train']
        if not rows:
            raise ValueError('Missing admitted training qualification scene: ' + suite)
        # Prefer two already observed native interfaces; every reset remains an
        # exact member of the frozen training plan, including its scene hash.
        preferred = {'libero-max': 'flat-stove-on-illumination-switch',
                     'metaworldplus': 'button-press-wall'}
        rows.sort(key=lambda c: (preferred.get(suite, '') not in c['task'], c['task'], c['config_id']))
        selected.append(rows[0])
    return selected


def prepare(source, *, build_root=None):
    source = Path(source)
    original = read(source / 'spec.json')
    if (set(original.get('suites', [])) != SUITES or original.get('control_interface')
            or original.get('native_membership_local')):
        raise ValueError('Qualification requires the original six-suite teacher native interface')
    shaped = smaller_shape(source, 4, build_root=build_root)
    files = frozen_files(shaped)
    spec = read(shaped / 'spec.json')
    selected = cases_for(spec)
    identity = hashlib.sha256((shaped / 'source-manifest.json').read_bytes()
        + json.dumps(selected, sort_keys=True).encode() + VERSION.encode()).hexdigest()[:12]
    build = Path(build_root or BUILD_ROOT) / ('rp-jhh-astra-core4-native-qual-' + identity)
    if (build / 'source-manifest.json').exists():
        frozen_files(build)
        return build
    root = TEACHER_ROOT + '/qualifications/core4-native-v1/' + identity
    changes = {'teacher_root': root, 'transport_root': root + '/transport',
        'head_heartbeat': root + '/head-heartbeat.json', 'finished': root + '/finished.json',
        'output': root + '/native', 'allocation_control_base': root + '/allocations',
        'episode_archive': original['episode_archive'].rstrip('/') + '/core4-qualification/' + identity,
        'teacher_qualification_only': True, 'teacher_native_service_only': True,
        'teacher_activation_required': True}
    spec = {**spec, **changes}
    build.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, build / name)
    write(build / 'spec.json', spec)
    write(build / 'qualification-cases.json', selected)
    workflow = yaml.safe_load((shaped / 'workflow.yaml').read_text())
    wf = workflow['workflow']
    wf['name'] = build.name
    wf['labels'].update(teacher_role='native_shape_qualification', qualification_version=VERSION)
    task = wf['tasks'][0]
    native_only_task(task)
    if (set(task.get('credentials', {})) != {ARCHIVE_CREDENTIAL}
            or 'NVIDIA_INFERENCE_API_KEY' in task.get('environment', {})):
        raise ValueError('A native qualification may receive only its existing archive credential')
    task['environment']['PEAR_ALLOCATION_BASE'] = spec['allocation_control_base']
    (build / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    receipt = {'version': VERSION, 'source_build': str(source), 'shaped_build': str(shaped),
        'source_manifest_sha256': hashlib.sha256((source / 'source-manifest.json').read_bytes()).hexdigest(),
        'owner': OWNER, 'changed_infrastructure_spec_fields': changes,
        'native_setup_byte_equivalent': spec['native_setup'] == original['native_setup'],
        'native_config_byte_equivalent': spec['native_config'] == original['native_config'],
        'teacher_plan_byte_equivalent': spec['teacher_plan'] == original['teacher_plan'],
        'runtime_sources_unchanged_except': ['efe_pear_primary_native.py', 'efe_pear_native_gpu_shape.py'],
        'qualification_case_ids': [c['case_id'] for c in selected],
        'gpu_count': 4, 'lanes_per_gpu': spec.get('lanes_per_gpu'),
        'future_priority': 'LOW', 'inference_credential': False, 'collector_started': False,
        'qualification_outputs_sft_eligible': False, 'submitted': False, 'watcher_changed': False}
    write(build / 'native-qualification-provenance.json', receipt)
    names = [*files, 'qualification-cases.json', 'native-qualification-provenance.json', 'spec.json']
    (build / 'bootstrap.py').write_text(bootstrap(build, names, 'efe_pear_allocation_keeper.py'))
    freeze_worker(build, [*names, 'bootstrap.py', 'workflow.yaml'])
    frozen_files(build)
    return build
