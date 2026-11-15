"""Check graphics and nested-container prerequisites on an allocated GPU node.

No benchmark episode or model training runs here. Results distinguish an OSMO
allocation from successful simulator/harness validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path('/tmp/efe-preflight')


def command(name, argv, *, timeout=180, env=None):
    print(json.dumps({'stage': name, 'started_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                env={**os.environ, **(env or {})})
        output = result.stdout + result.stderr
        record = {'returncode': result.returncode, 'passed': result.returncode == 0,
                  'output_tail': output[-2500:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        output = str(error)
        record = {'returncode': None, 'passed': False, 'error': str(error)}
    (ROOT / (name + '.log')).write_text(output)
    record['log_sha256'] = hashlib.sha256(output.encode()).hexdigest()
    return record


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    checks = {}
    checks['gpu'] = command('gpu', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    checks['user_namespace'] = command('user_namespace', ['unshare', '--user', '--map-root-user', 'true'])
    checks['packages'] = command('packages', ['bash', '-c',
        'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq '
        'podman uidmap slirp4netns libegl1 libgl1'], timeout=600)
    checks['python_packages'] = command('python_packages', [sys.executable, '-m', 'pip', 'install',
        '--disable-pip-version-check', 'mujoco==3.14.0', 'harbor==0.23.0'], timeout=600)
    if checks['python_packages']['passed']:
        render = '''import hashlib,json
import mujoco
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size=".2" rgba="0 1 0 1"/></worldbody></mujoco>')
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
with mujoco.Renderer(m,height=64,width=64) as renderer:
 renderer.update_scene(d);pixels=renderer.render()
 assert pixels.shape==(64,64,3) and int(pixels.max())>int(pixels.min())
 print(json.dumps({'shape':list(pixels.shape),'sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'mujoco':mujoco.__version__}))
'''
        if os.environ.get('EFE_CHECK_GRAPHICS', '1') == '1':
            checks['egl_render'] = command('egl_render', [sys.executable, '-c', render], env={'MUJOCO_GL': 'egl'})
        checks['harbor'] = command('harbor', ['harbor', '--version'])
    if shutil.which('podman'):
        checks['podman_version'] = command('podman_version', ['podman', '--version'])
        checks['podman_info'] = command('podman_info', ['podman', '--storage-driver=vfs',
            '--root', str(ROOT/'containers'), '--runroot', str(ROOT/'run'), 'info', '--format', 'json'])
        if checks['podman_info']['passed']:
            checks['container_smoke'] = command('container_smoke', ['podman', '--storage-driver=vfs',
                '--root', str(ROOT/'containers'), '--runroot', str(ROOT/'run'),
                'run', '--rm', '--network=none', 'docker.io/library/alpine:3.22',
                'sh', '-c', 'test -r /etc/os-release && echo EFE_CONTAINER_OK'], timeout=300)
    result = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'workflow': os.environ.get('EFE_WORKFLOW_ID'), 'pool': os.environ.get('EFE_POOL', 'groot-l40s-01'),
        'purpose': 'Allocated GPU-node graphics/container-runtime prerequisite check; no benchmark trial',
        'benchmark_rollouts_started': False, 'checks': checks,
        'graphics_passed': checks.get('egl_render', {}).get('passed', False),
        'nested_container_passed': checks.get('container_smoke', {}).get('passed', False),
        'official_task_oracle_nop_passed': False,
        'notes': ['A basic EGL render does not validate RTX/Isaac or Nyx.',
                  'A container smoke test does not validate Harbor multi-container isolation or a benchmark task.'],
    }
    path=ROOT/'result.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print('EFE_PREFLIGHT_RESULT '+json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
