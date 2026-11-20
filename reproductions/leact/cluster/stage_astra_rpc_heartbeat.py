"""Freeze the Dojo teacher's exact RPC worker with one heartbeat-race fix.

No qualified shared source is edited, and this helper never submits, retargets,
restarts, or changes the watcher. An operator must coordinate safe adoption.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .efe_astra_heartbeat_io import VERSION
from .launch_astra_wide_collection import OWNER, freeze_worker
from .launch_efe_pear_primary import BUILD_ROOT, bootstrap
from .recover_efe_pear_graphics import frozen_files
from .scale_efe import BASE, read, write
from .switch_astra_wide_profile import PROFILE_ROOT


ORIGINAL_CHECK = 'if heartbeat.exists()and time.time()-heartbeat.stat().st_mtime>300:head_lost.set()'
FIXED_CHECK = 'if head_heartbeat_expired(heartbeat):head_lost.set()'
ORIGINAL_IMPORT = 'from efe_pear_file_rpc import atomic_json,claim_episode,serve_episode\n'
HELPER_IMPORT = 'from efe_astra_heartbeat_io import head_heartbeat_expired\n'
PROVENANCE = 'teacher-heartbeat-overlay.json'
OVERLAY_FILES = {PROVENANCE, 'efe_astra_heartbeat_io.py', 'efe_astra_original_rpc_worker.py'}


def patch_worker(original):
    if original.count(ORIGINAL_CHECK) != 1 or original.count(ORIGINAL_IMPORT) != 1:
        raise ValueError('Frozen RPC worker differs from the reviewed heartbeat-race source')
    if HELPER_IMPORT in original:
        raise ValueError('Heartbeat overlay is already present')
    patched = original.replace(ORIGINAL_IMPORT, HELPER_IMPORT+ORIGINAL_IMPORT).replace(ORIGINAL_CHECK, FIXED_CHECK)
    compile(patched, 'teacher-only-efe_pear_remote_worker.py', 'exec')
    return patched


def recovery_sources(files):
    """Remove reviewed overlay packaging before an unchanged teacher refresh.

    The watcher refreshes its pinned baseline RPC source, then applies this
    overlay again. Old metadata must not claim that a different build is fixed.
    """
    if PROVENANCE not in files:
        return dict(files)
    original = files['efe_astra_original_rpc_worker.py'].read_text()
    if files['efe_pear_remote_worker.py'].read_text() != patch_worker(original):
        raise ValueError('Previous teacher heartbeat overlay differs from its preserved source')
    if read(files[PROVENANCE]).get('version') != VERSION:
        raise ValueError('Unexpected teacher heartbeat overlay version')
    sources = {name: path for name, path in files.items() if name not in OVERLAY_FILES}
    sources['efe_pear_remote_worker.py'] = files['efe_astra_original_rpc_worker.py']
    return sources


def for_recovery(build):
    """Apply only to the independently owned exact-profile Dojo native lane."""
    spec = read(Path(build)/'spec.json')
    if (spec.get('suite') == 'robodojo' and spec.get('teacher_profile_id') == 'astra62-xhigh-v1'
            and spec.get('teacher_native_entry') == 'efe_pear_primary_rtx.py'):
        return prepare(build)
    return build


def prepare(source, *, build_root=None):
    source = Path(source)
    spec_bytes = (source/'spec.json').read_bytes()
    spec = json.loads(spec_bytes)
    if (spec.get('owner_model_workflow') != OWNER or spec.get('teacher_root') != PROFILE_ROOT
            or spec.get('teacher_profile_id') != 'astra62-xhigh-v1'
            or spec.get('teacher_native_entry') != 'efe_pear_primary_rtx.py'
            or spec.get('teacher_native_only') is not True or spec.get('suite') != 'robodojo'
            or spec.get('teacher_transport_diagnostics')):
        raise ValueError('Expected the independently owned, unchanged exact-profile Dojo native teacher build')
    files = frozen_files(source)
    if PROVENANCE in files:
        raise ValueError('Review the existing overlay instead of layering another fix')
    original_bytes = files['efe_pear_remote_worker.py'].read_bytes()
    patched_bytes = patch_worker(original_bytes.decode()).encode()
    helper = BASE/'cluster/efe_astra_heartbeat_io.py'
    identity = hashlib.sha256(spec_bytes+patched_bytes+helper.read_bytes()+
        (source/'source-manifest.json').read_bytes()).hexdigest()[:12]
    build = Path(build_root or BUILD_ROOT)/('rp-jhh-astra-rpc-heartbeat-'+identity)
    if (build/'source-manifest.json').exists():
        frozen_files(build)
        return build
    build.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, build/name)
    (build/'spec.json').write_bytes(spec_bytes)
    (build/'efe_astra_original_rpc_worker.py').write_bytes(original_bytes)
    (build/'efe_pear_remote_worker.py').write_bytes(patched_bytes)
    shutil.copyfile(helper, build/helper.name)
    names = [*files, helper.name, 'efe_astra_original_rpc_worker.py']
    (build/'bootstrap.py').write_text(bootstrap(build, [*names, 'spec.json'], 'efe_pear_allocation_keeper.py'))
    workflow = yaml.safe_load((source/'workflow.yaml').read_text())
    workflow['workflow']['name'] = 'rp-jhh-astra-rpc-heartbeat-'+identity
    workflow['workflow'].setdefault('labels', {})['teacher_rpc_heartbeat_fix'] = VERSION
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    write(build/PROVENANCE, {
        'version': VERSION, 'source_build': str(source),
        'source_manifest_sha256': hashlib.sha256((source/'source-manifest.json').read_bytes()).hexdigest(),
        'original_rpc_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'patched_rpc_sha256': hashlib.sha256(patched_bytes).hexdigest(),
        'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
        'spec_byte_identical': True, 'spec_sha256': hashlib.sha256(spec_bytes).hexdigest(),
        'unchanged_sources': {name: digest for name, digest in read(source/'source-manifest.json').items()
                              if name in files and name != 'efe_pear_remote_worker.py'},
        'runtime_change': 'Replace only head-heartbeat exists/stat check with a single-stat FileNotFoundError guard; same 300-second expiry and missing-file behavior.',
        'qualified_shared_source_modified': False, 'submitted': False, 'processes_changed': False,
        'exhausted_case_retry_pass': False, 'watcher_adoption': False,
        'future_recovery_requirement': 'Reapply/preserve this teacher-only overlay on later teacher recovery; the current watcher still copies its original shared RPC source.'})
    freeze_worker(build, [*names, 'spec.json', 'bootstrap.py', 'workflow.yaml', PROVENANCE])
    frozen_files(build)
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_build', type=Path)
    args = parser.parse_args()
    print(json.dumps({'build': str(prepare(args.source_build)), 'submitted': False}))


if __name__ == '__main__':
    main()
