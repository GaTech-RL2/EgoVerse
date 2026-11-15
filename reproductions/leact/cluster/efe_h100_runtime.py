"""Use the host-matched CUDA driver and compiler libraries for EFE model work."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = None
ARCHIVE = None


def driver_environment():
    kernel = Path('/proc/driver/nvidia/version').read_text()
    match = re.search(r'\b(\d{3}\.\d+(?:\.\d+)?)\b', kernel)
    if not match:raise RuntimeError('Cannot identify the allocated host driver')
    version = match.group(1)
    directory = Path('/usr/lib/x86_64-linux-gnu')
    libraries = [directory / (name + '.so.' + version)
                 for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
    if not all(path.is_file() for path in libraries):
        raise RuntimeError('Matching host CUDA driver/compiler libraries missing')
    env = {**os.environ,'LD_PRELOAD':':'.join(map(str,libraries)),
           'LD_LIBRARY_PATH':str(directory)+':'+os.environ.get('LD_LIBRARY_PATH',''),
           'CUDA_VISIBLE_DEVICES':','.join(map(str,range(8))),'NCCL_P2P_DISABLE':'1'}
    probe = subprocess.run([sys.executable,'-c',
        'import torch,json;torch.cuda.init();print(json.dumps({"devices":torch.cuda.device_count(),"device":torch.cuda.get_device_name(0),"cuda":torch.version.cuda}))'],
        env=env,text=True,capture_output=True,timeout=120)
    record={'kernel_driver':version,'libraries':list(map(str,libraries)),
            'returncode':probe.returncode,'stdout':probe.stdout[-1500:],'stderr':probe.stderr[-1500:]}
    for root in {ROOT,ARCHIVE}:
        if root is not None:(Path(root)/'cuda-preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    if probe.returncode:raise RuntimeError('Host-matched CUDA initialization failed')
    if json.loads(probe.stdout.strip().splitlines()[-1])['devices']!=8:
        raise RuntimeError('The validated training layout requires eight allocated GPUs')
    return env
