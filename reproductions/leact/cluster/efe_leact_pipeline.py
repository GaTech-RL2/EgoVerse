"""Run LeAct and its native evaluation on one allocation using local storage."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from .efe_training_data import sha, write
except ImportError:
    from efe_training_data import sha, write


def export_portable(root, ready):
    source = Path(ready['checkpoint'])
    manifest = json.loads(Path(ready['manifest']).read_text())
    if not ready['final'] or manifest['revision'] != ready['revision']:
        raise ValueError('Only the fixed final checkpoint can be exported')
    dest = root / 'portable-model'
    dest.mkdir(exist_ok=True)
    for name, receipt in manifest['files'].items():
        if Path(name).name != name or sha(source / name) != receipt['sha256']:
            raise ValueError('Invalid portable checkpoint member: ' + name)
        if (dest/name).exists():
            if sha(dest/name)!=receipt['sha256']:raise ValueError('Existing portable checkpoint differs: '+name)
        else:os.link(source / name, dest / name)
    shutil.copyfile(ready['manifest'], dest / 'checkpoint-manifest.json')
    record = {'checkpoint': str(dest), 'manifest': str(dest / 'checkpoint-manifest.json'),
              'revision': ready['revision'], 'bytes': sum(r['bytes'] for r in manifest['files'].values()),
              'manifest_sha256': sha(dest / 'checkpoint-manifest.json'),
              'optimizer_persistence': 'Final Adam state remains node-local. Portable model weights are collected to the tracking host; no existing checkpoint is deleted.'}
    write(root / 'portable-ready.json', record)
    return record


def main():
    spec = json.loads(Path('/tmp/efe-training-spec.json').read_text())
    workflow = os.environ['EFE_WORKFLOW_ID']
    root = Path(spec['archive_root']) / workflow
    root.mkdir(parents=True, exist_ok=True)
    native = Path(spec['evaluation_archive_root']) / workflow
    native.mkdir(parents=True, exist_ok=True)
    def progress(phase, detail):
        value = {'workflow': workflow, 'phase': phase, 'detail': detail,
                 'updated_utc': datetime.now(timezone.utc).isoformat()}
        write(root / 'pipeline-progress.json', value)
        print(json.dumps(value), flush=True)
    receipt = {'success': False, 'workflow': workflow, 'training_updates': None,
               'capability_screen_enabled': False}
    try:
        progress('leact_started', 'Generating student reasoning from the fixed step-96 checkpoint; all new files use node-local disk.')
        env = {**os.environ, 'NCCL_P2P_DISABLE': '1'}
        with (root / 'leact-pipeline.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '/tmp/efe_leact_vl.py'], env=env,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        complete = json.loads((root / 'complete.json').read_text())
        if complete.get('no_admissions'):
            receipt.update(success=True, no_admissions=True, training_updates=0)
            progress('no_admissions', 'No candidate passed the fixed LeAct filter. No model update or duplicate warmstart evaluation is performed.')
        else:
            if not complete.get('success'):
                raise ValueError('LeAct did not complete successfully')
            ready = json.loads((root / 'leact-ready.json').read_text())
            portable = export_portable(root, ready)
            receipt.update(training_updates=ready['step'], portable=portable)
            progress('native_evaluation', 'LeAct training complete. Evaluating the same seven development tasks; the portable model is ready for durable collection.')
            evaluation = json.loads(Path('/tmp/efe-evaluation-template.json').read_text())
            evaluation.update(student_checkpoint=portable['checkpoint'], student_manifest=portable['manifest'],
                student_revision=ready['revision'], archive_root=spec['evaluation_archive_root'],
                agent_label='Qwen warmstart step 96 + LeAct R1', agent_slug='qwen-warm96-leact-r1',
                training_parent=workflow, local_episode_outputs=True, retry_of={},
                purpose='Fixed LeAct round from step-96 warmstart; matched seven-task development evaluation. No core or capability screen.')
            write(Path('/tmp/efe-native-spec.json'), evaluation)
            with (root / 'native-pipeline.log').open('w') as log:
                subprocess.run(['bash', '/tmp/efe-native-bootstrap.sh'], env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            results = json.loads((native / 'results.json').read_text())
            trials = results['trials']
            valid = len(trials) == 7 and all(t.get('status') == 'graded' for t in trials)
            receipt.update(success=valid, graded=sum(t.get('status') == 'graded' for t in trials),
                           successes=sum(t.get('score') or 0 for t in trials), native_archive=str(native))
            sources = {'episodes': [dict(t, role='student') for t in trials],
                       'student_protocol': results.get('protocol_label', '')}
            write(root / 'episode-sources.json', sources)
            subprocess.run([sys.executable, '/tmp/export_efe_viewer.py', '--sources', str(root / 'episode-sources.json'),
                            '--output', str(root / 'public-episodes')], check=True)
            progress('native_complete', f"Native development evaluation complete: {receipt['successes']}/7 successes. Verifying durable collection.")
    except Exception as error:
        receipt['error'] = str(error)
        progress('needs_review', str(error))
    finally:
        receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write(root / 'pipeline-complete.json', receipt)
        # Public receipts survive an interrupted local collector in OSMO logs.
        print('EFE_PIPELINE_COMPLETE ' + json.dumps(receipt), flush=True)
        for _ in range(spec.get('final_collection_wait_seconds', 3600) // 5):
            if (root / 'pipeline-collected').exists():
                break
            time.sleep(5)
    if not receipt['success']:
        raise RuntimeError(receipt.get('error', 'Native evaluation did not grade all seven tasks'))


if __name__ == '__main__':
    main()
