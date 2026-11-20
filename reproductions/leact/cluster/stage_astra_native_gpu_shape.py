"""Stage opt-in 2/4-GPU teacher native workers without submitting or adopting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .efe_pear_native_gpu_shape import VERSION, PLUGIN_DOC
from .launch_astra_wide_collection import OWNER, freeze_worker
from .launch_efe_pear_primary import BUILD_ROOT, bootstrap
from .recover_efe_pear_graphics import frozen_files
from .scale_efe import BASE, read, write


def prepare(source, gpu_count, *, build_root=None):
    source = Path(source)
    spec = read(source/'spec.json')
    if type(gpu_count) is not int or gpu_count not in {2, 4}:
        raise ValueError('Stage an explicit 2- or 4-GPU teacher native shape')
    if (spec.get('owner_model_workflow') != OWNER
            or spec.get('teacher_profile_id') != 'astra62-xhigh-v1'
            or spec.get('teacher_native_entry') != 'efe_pear_primary_native.py'
            or spec.get('teacher_native_only') is not True
            or spec.get('native_gpu_count', 8) != 8
            or spec.get('teacher_transport_diagnostics')):
        raise ValueError('Expected the original exact-profile native-only teacher EFE build')
    files = frozen_files(source)
    workflow = yaml.safe_load((source/'workflow.yaml').read_text())
    task = workflow['workflow']['tasks'][0]
    if (not task.get('privileged') or len(workflow['workflow']['tasks']) != 1
            or workflow['workflow']['resources']['default']['gpu'] != 8):
        raise ValueError('Expected the qualified privileged eight-GPU source worker')
    if any(name in task.get('environment', {}) for name in
           ['NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'NV_GPU']):
        raise ValueError('Workflow must not override scheduler-provided GPU assignment')
    # Replace only the frozen review bundle's entry. The live eight-GPU source
    # stays byte-identical to its native EEF qualification pins.
    files.update({'efe_pear_primary_native.py': BASE/'cluster/efe_astra_native_gpu_bootstrap.py',
                  'efe_pear_native_gpu_shape.py': BASE/'cluster/efe_pear_native_gpu_shape.py'})
    spec = {**spec, 'native_gpu_count': gpu_count, 'native_gpu_ownership': {
        'version': VERSION, 'required_assignment': 'Inherited full GPU UUIDs in NVIDIA_VISIBLE_DEVICES',
        'ambiguous_visibility': 'Fail before native setup; no selection by host GPU order'}}
    identity = hashlib.sha256(json.dumps([spec, {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}],
        sort_keys=True).encode()).hexdigest()[:12]
    build = Path(build_root or BUILD_ROOT)/f'rp-jhh-astra-native-gpu{gpu_count}-{identity}'
    if (build/'source-manifest.json').exists():
        frozen_files(build)
        return build
    build.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, build/name)
    write(build/'spec.json', spec)
    (build/'bootstrap.py').write_text(bootstrap(build, [*files, 'spec.json'], 'efe_pear_allocation_keeper.py'))
    workflow['workflow']['name'] = f'rp-jhh-astra-native-gpu{gpu_count}-{identity}'
    resources = workflow['workflow']['resources']['default']
    resources.update(gpu=gpu_count, cpu=8*gpu_count, memory=f'{64*gpu_count}Gi')
    workflow['workflow'].setdefault('labels', {})['native_gpu_shape'] = str(gpu_count)
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    write(build/'native-gpu-shape-provenance.json', {
        'version': VERSION, 'source_build': str(source), 'native_gpu_count': gpu_count,
        'source_manifest_sha256': hashlib.sha256((source/'source-manifest.json').read_bytes()).hexdigest(),
        'unchanged_sources': {name: digest for name, digest in read(source/'source-manifest.json').items()
                              if name in files and name != 'efe_pear_primary_native.py'},
        'changed_spec_keys': ['native_gpu_count', 'native_gpu_ownership'],
        'teacher_concurrency_unchanged': spec.get('teacher_concurrency'),
        'parallelism_review': 'Review collector concurrency against smaller native lane count before adoption.',
        'gpu_assignment_overrides_added': False, 'runtime_ownership_proof_required': True,
        'documentation': PLUGIN_DOC, 'future_requests_priority': 'LOW',
        'submitted': False, 'watcher_modified': False, 'hero_modified': False})
    freeze_worker(build, [*files, 'spec.json', 'bootstrap.py', 'workflow.yaml', 'native-gpu-shape-provenance.json'])
    frozen_files(build)
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_build', type=Path)
    parser.add_argument('--gpus', type=int, choices=[2, 4], required=True)
    args = parser.parse_args()
    print(json.dumps({'build': str(prepare(args.source_build, args.gpus)), 'submitted': False}))


if __name__ == '__main__':
    main()
