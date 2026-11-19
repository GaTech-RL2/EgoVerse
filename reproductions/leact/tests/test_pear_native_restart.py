import io
import json
import tarfile

from reproductions.leact.cluster import efe_native_pilot as native


def test_a_new_job_stages_tools_and_socket_without_overwriting_old_executables_or_image_cache(tmp_path,monkeypatch):
    old=tmp_path/'previous';old.mkdir();(old/'conmon').write_bytes(b'old running binary')
    current=tmp_path/'current';archive=tmp_path/'archive'
    (archive/'logs').mkdir(parents=True)
    monkeypatch.setattr(native,'ROOT',current);monkeypatch.setattr(native,'ARCHIVE',archive)
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz')as tar:
        for path in ['usr/local/bin/podman','usr/local/bin/crun','usr/local/lib/podman/conmon']:
            info=tarfile.TarInfo('podman-linux-amd64/'+path);info.size=3;info.mode=0o755
            tar.addfile(info,io.BytesIO(b'new'))
    def fetch(spec,path):path.write_bytes(data.getvalue()if path.suffix=='.gz'else b'compose')
    monkeypatch.setattr(native,'fetch',fetch)
    monkeypatch.setattr(native,'run',lambda *a,**k:None)
    monkeypatch.setattr(native.subprocess,'check_output',lambda *a,**k:'pinned')
    class Service:
        def __init__(self,*a,**k):
            assert k['env']['DOCKER_HOST']=='unix://'+str(current/'podman.sock')
            (current/'podman.sock').touch()
        def poll(self):return None
    monkeypatch.setattr(native.subprocess,'Popen',Service)
    reference,env,service,log=native.setup({'podman':{},'compose':{},'repository':'unused','revision':'pinned',
        'tasks':[],'container_storage_root':str(old)})
    try:
        assert(old/'conmon').read_bytes()==b'old running binary'
        assert reference==current/'reference'
        assert 'graphroot='+json.dumps(str(old/'containers'))in(current/'storage.conf').read_text()
        assert str(current/'podman')in env['PATH']
    finally:log.close()


def test_each_retained_node_job_gets_its_own_bounded_unix_socket_path(monkeypatch):
    from pathlib import Path
    import importlib.util
    folder=Path(__file__).parents[1]/'cluster';monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('native_restart_paths',folder/'efe_pear_primary_native.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.runtime_root('old-job')!=module.runtime_root('new-job')
    assert len(str(module.runtime_root('very-long-workflow-id'*20)/'podman.sock'))<108
