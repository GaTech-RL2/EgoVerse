"""Freeze a reviewable teacher-only diagnostics overlay; never submit or switch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .efe_astra_transport_diagnostics import VERSION
from .launch_astra_wide_collection import OWNER, freeze_worker
from .launch_efe_pear_primary import BUILD_ROOT, bootstrap
from .recover_efe_pear_graphics import frozen_files
from .scale_efe import BASE, read, write


def prepare(source, *, build_root=None):
    source = Path(source)
    spec = read(source/'spec.json')
    if (spec.get('owner_model_workflow') != OWNER
            or spec.get('teacher_profile_id') != 'astra62-xhigh-v1'
            or spec.get('teacher_native_entry') != 'efe_pear_primary_native.py'
            or spec.get('teacher_native_only') is not True):
        raise ValueError('Expected an independently owned exact-profile native-only Astra EFE build')
    files = frozen_files(source)
    if 'teacher_transport_diagnostics' in spec or 'efe_astra_original_remote_worker.py' in files:
        raise ValueError('This teacher build already contains a transport overlay')
    original = files['efe_pear_remote_worker.py']
    files = {**files, 'efe_astra_original_remote_worker.py': original,
             'efe_pear_remote_worker.py': BASE/'cluster/efe_astra_transport_entry.py',
             'efe_astra_transport_diagnostics.py': BASE/'cluster/efe_astra_transport_diagnostics.py'}
    spec = {**spec, 'teacher_transport_diagnostics': {
        'version': VERSION, 'source_build': str(source),
        'original_remote_worker_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'activation': 'Only at independently verified failed/finished teacher child boundary; retain allocation.'}}
    identity = hashlib.sha256(json.dumps([spec, {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}],
        sort_keys=True).encode()).hexdigest()[:12]
    build = Path(build_root or BUILD_ROOT)/('rp-jhh-astra-transport-diag-'+identity)
    if (build/'source-manifest.json').exists():
        frozen_files(build)  # Existing staged bundles must still verify.
        return build
    build.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, build/name)
    write(build/'spec.json', spec)
    entry = spec['allocation_job_entry']
    (build/'bootstrap.py').write_text(bootstrap(build, [*files, 'spec.json'], 'efe_pear_allocation_keeper.py'))
    workflow = yaml.safe_load((source/'workflow.yaml').read_text())
    workflow['workflow']['name'] = 'rp-jhh-astra-transport-diag-'+identity
    workflow['workflow'].setdefault('labels', {})['teacher_transport_diagnostics'] = VERSION
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    original_manifest = read(source/'source-manifest.json')
    provenance = {'version': VERSION, 'source_build': str(source), 'entry_preserved': entry,
                  'source_manifest_sha256': hashlib.sha256((source/'source-manifest.json').read_bytes()).hexdigest(),
                  'spec_added_keys': ['teacher_transport_diagnostics'],
                  'unchanged_shared_sources': {name: digest for name, digest in original_manifest.items()
                      if name in files and name != 'efe_pear_remote_worker.py'},
                  'changed_frozen_entry': 'efe_pear_remote_worker.py',
                  'original_entry_preserved_as': 'efe_astra_original_remote_worker.py',
                  'submitted': False, 'watcher_modified': False, 'active_workers_modified': False}
    write(build/'diagnostic-provenance.json', provenance)
    freeze_worker(build, [*files, 'spec.json', 'bootstrap.py', 'workflow.yaml', 'diagnostic-provenance.json'])
    frozen_files(build)
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_build', type=Path)
    args = parser.parse_args()
    print(json.dumps({'build': str(prepare(args.source_build)), 'submitted': False}))


if __name__ == '__main__':
    main()
