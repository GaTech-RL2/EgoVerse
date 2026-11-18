import json
from pathlib import Path
import subprocess
import sys

import pytest

from reproductions.leact.cluster.efe_pear_worker_bundle import bootstrap


def test_worker_receives_byte_identical_inputs_and_executes_its_entry(tmp_path):
    source=tmp_path/'source';source.mkdir()
    expected={'task':'scene-001','nested':{'json_braces':'{{literal}}'},'rewards':[0,1]}
    raw=json.dumps(expected).encode()+b'\n'
    (source/'spec.json').write_bytes(raw)
    entry="from pathlib import Path\np=Path(__file__).resolve().parent\n(p/'copied.json').write_bytes((p/'spec.json').read_bytes())\n"
    (source/'worker.py').write_text(entry)
    text=bootstrap(source,['spec.json','worker.py'],'worker.py')
    assert text==bootstrap(source,['worker.py','spec.json'],'worker.py')
    assert '{{literal}}'not in text
    target=tmp_path/'target';target.mkdir();(target/'bootstrap.py').write_text(text)
    subprocess.run([sys.executable,str(target/'bootstrap.py')],check=True,timeout=10)
    assert(target/'spec.json').read_bytes()==raw
    assert(target/'copied.json').read_bytes()==raw
    assert(target/'worker.py').read_bytes()==(source/'worker.py').read_bytes()


def test_worker_bundle_rejects_paths_outside_its_frozen_directory(tmp_path):
    with pytest.raises(ValueError,match='flat frozen filenames'):
        bootstrap(tmp_path,['../outside.py'],'../outside.py')
