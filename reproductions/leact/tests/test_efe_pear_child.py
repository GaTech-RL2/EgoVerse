import os
from pathlib import Path
import subprocess
import sys

from reproductions.leact.cluster import efe_pear_child


def test_exec_bootstrap_keeps_inherited_transport_pipe(tmp_path):
    module=tmp_path/'transport_probe.py'
    module.write_text('import os,sys\nos.write(int(sys.argv[1]), b"ready")\n')
    read_fd,write_fd=os.pipe()
    try:
        result=subprocess.run([sys.executable,str(Path(efe_pear_child.__file__)),str(os.getpid()),
            'transport_probe',str(write_fd)],env={**os.environ,'PYTHONPATH':str(tmp_path)},
            pass_fds=(write_fd,),capture_output=True,text=True,timeout=10)
        assert result.returncode==0,result.stderr
        assert 'PEAR_SIMULATOR_EXEC_READY'in result.stdout
        assert os.read(read_fd,5)==b'ready'
    finally:os.close(read_fd);os.close(write_fd)


def test_exec_bootstrap_rejects_a_lost_or_different_owner_before_loading_env(tmp_path):
    flag=tmp_path/'loaded'
    (tmp_path/'owner_probe.py').write_text('from pathlib import Path\nPath('+repr(str(flag))+').touch()\n')
    result=subprocess.run([sys.executable,str(Path(efe_pear_child.__file__)),str(os.getpid()+999999),
        'owner_probe'],env={**os.environ,'PYTHONPATH':str(tmp_path)},capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and 'owner disappeared'in result.stderr
    assert not flag.exists()
