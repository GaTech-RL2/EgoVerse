"""Start one native evaluation on a fresh allocation with matching CUDA libraries."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys

lane = Path('/tmp/efe-training-evaluation-v1/warm96')
version = re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)', Path('/proc/driver/nvidia/version').read_text()).group(1)
libraries = [Path('/usr/lib/x86_64-linux-gnu') / (name + '.so.' + version)
             for name in ['libcuda', 'libnvidia-ptxjitcompiler', 'libnvidia-nvvm']]
if not all(p.is_file() for p in libraries):
    raise RuntimeError('The matching host driver/compiler libraries are required')
env = {**os.environ, 'LD_PRELOAD': ':'.join(map(str, libraries)),
       'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', ''),
       'EFE_SERVING_PYTHON': sys.executable, 'NCCL_P2P_DISABLE': '1'}
environment = lane / 'emimic'
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
python = str(environment / 'bin/python')
subprocess.run([python, '-m', 'pip', 'install', '--disable-pip-version-check', 'aiohttp==3.13.3'], env=env, check=True)
env.update(VIRTUAL_ENV=str(environment), PATH=str(environment / 'bin') + ':' + env['PATH'])
(lane / 'fresh-bootstrap.json').write_text(json.dumps({'driver':version, 'libraries':list(map(str,libraries)),
    'storage_limit_gib':800, 'one_simulator_store_per_node':True}))
os.execve(python, [python, '-u', str(lane / 'worker.py'), '--run'], env)
