import json
from pathlib import Path
import shutil

import pytest

from reproductions.leact.cluster.efe_pear_editable_restore import capture_editables,restore_editables


class Distribution:
    def __init__(self,root,files):self.root=root;self.files=files
    def read_text(self,name):return json.dumps({'dir_info':{'editable':True}})
    def locate_file(self,entry):return self.root/entry


def test_editable_import_loader_and_entry_point_survive_a_clean_venv(tmp_path):
    prefix=tmp_path/'venv';pure=prefix/'lib/python3.11/site-packages'
    paths={'__editable__.native.pth':'import native_finder\n','native_finder.py':'ROOT="/opt/native"\n',
           'native.dist-info/direct_url.json':'{"dir_info":{"editable":true}}\n',
           '../../../bin/native-cli':'#!/opt/venv/bin/python\n'}
    for relative,body in paths.items():
        target=pure/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(body)
    saved=capture_editables(prefix,[Distribution(pure,list(paths))],tmp_path/'saved')
    shutil.rmtree(prefix);prefix.mkdir()
    restore_editables(prefix,tmp_path/'saved',saved)
    assert all((pure/name).read_text()==body for name,body in paths.items())


def test_editable_restore_rejects_a_changed_artifact(tmp_path):
    prefix=tmp_path/'venv';prefix.mkdir();(prefix/'loader.pth').write_text('/opt/native\n')
    saved=capture_editables(prefix,[Distribution(prefix,['loader.pth'])],tmp_path/'saved')
    (tmp_path/'saved/loader.pth').write_text('/opt/other\n')
    with pytest.raises(ValueError,match='changed'):restore_editables(prefix,tmp_path/'saved',saved)


def test_editable_capture_rejects_an_escape(tmp_path):
    prefix=tmp_path/'venv';prefix.mkdir();(tmp_path/'outside').write_text('outside')
    with pytest.raises(ValueError,match='escapes'):
        capture_editables(prefix,[Distribution(prefix,['../outside'])],tmp_path/'saved')
