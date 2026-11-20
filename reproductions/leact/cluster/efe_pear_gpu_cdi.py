"""Expose allocated GPUs to nested Podman through NVIDIA's supported CDI path.

Runs inside an isolated OSMO job, never on the cluster host. It does not change
drivers, simulator code, action spaces, benchmark limits, or native rewards.
"""
from pathlib import Path
import shutil
import subprocess
import urllib.request


TOOLKIT_VERSION='1.20.1-1'
DOC='https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html'


def ensure_gpu_cdi(run):
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('GPU allocation does not expose nvidia-smi')
    if shutil.which('nvidia-ctk') is None:
        run('cdi_prerequisites',['apt-get','install','-y','-qq','gnupg2','ca-certificates'],timeout=300)
        key=Path('/tmp/efe-nvidia-container-key.asc')
        with urllib.request.urlopen('https://nvidia.github.io/libnvidia-container/gpgkey',timeout=60) as response:
            key.write_bytes(response.read())
        keyring='/usr/share/keyrings/efe-nvidia-container-toolkit.gpg'
        run('cdi_repository_key',['gpg','--batch','--yes','--dearmor','-o',keyring,str(key)],timeout=60)
        with urllib.request.urlopen('https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list',timeout=60) as response:
            sources=response.read().decode()
        sources=sources.replace('deb https://',f'deb [signed-by={keyring}] https://')
        Path('/etc/apt/sources.list.d/efe-nvidia-container-toolkit.list').write_text(sources)
        run('cdi_repository_update',['apt-get','update','-qq'],timeout=300)
        run('cdi_toolkit_install',['apt-get','install','-y','-qq',
            'nvidia-container-toolkit-base='+TOOLKIT_VERSION],timeout=300)
    directory=Path('/var/run/cdi');directory.mkdir(parents=True,exist_ok=True)
    run('cdi_generate',['nvidia-ctk','cdi','generate','--output',str(directory/'nvidia.yaml')],timeout=60)
    listing=subprocess.run(['nvidia-ctk','cdi','list'],capture_output=True,text=True,check=True,timeout=30)
    if 'nvidia.com/gpu=all' not in listing.stdout:
        raise RuntimeError('Allocated GPUs are absent from generated CDI devices')
    return {'toolkit':subprocess.check_output(['nvidia-ctk','--version'],text=True,timeout=30).strip(),
            'devices':listing.stdout.splitlines(),'spec':str(directory/'nvidia.yaml'),'documentation':DOC}
