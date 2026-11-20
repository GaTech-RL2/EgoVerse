"""Private CPU renderer setup for the independent teacher reasoning stratum."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root=Path('/tmp/rp_jhh_astra_reasoning_dependencies')
    root.mkdir(exist_ok=True)
    marker=root/'ready'
    if not marker.exists():
        subprocess.run(['apt-get','update','-qq'],check=True,timeout=300)
        subprocess.run(['apt-get','install','-y','-qq','libosmesa6','libgl1','libglib2.0-0'],check=True,timeout=300)
        subprocess.run([shutil.which('uv')or'/root/.local/bin/uv','pip','install','--python',sys.executable,'--target',str(root),
                        'numpy==2.2.6','mujoco==3.3.7','pillow==12.0.0'],check=True,timeout=600)
        marker.touch()
    sys.path.insert(0,str(root));os.environ.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    from efe_astra_wide_worker import main as collect
    collect()


if __name__=='__main__':main()
