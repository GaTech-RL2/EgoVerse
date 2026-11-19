from concurrent.futures import ThreadPoolExecutor
import time

from PIL import Image
import pytest

from reproductions.leact.cluster.efe_pear_file_rpc import (
    FileEpisodeClient,RemoteEpisodeLost,claim_episode,serve_episode,atomic_json,retire_abandoned_pending)


def test_cross_worker_ownership_and_real_image_action_reward_roundtrip(tmp_path):
    class Environment:
        def __init__(self,task):assert task=='cpu_transport_probe'
        def open(self):return {'images':[Image.new('RGB',(32,24),(100,20,50))],'done':False,'rewards':[0.]}
        def step(self,action):
            assert action=={'action':[.25],'repeat':2}
            return {'done':True,'rewards':[1.],'executed':2}
        def close(self):pass
    client=FileEpisodeClient(tmp_path,'test',{'task':'cpu_transport_probe'},timeout=5)
    def worker():
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            claimed=claim_episode(tmp_path,'test-worker')
            if claimed:
                folder,request=claimed;serve_episode(folder,Environment,request['env_args']);return
            time.sleep(.01)
        raise TimeoutError('No test episode arrived')
    with ThreadPoolExecutor(max_workers=1)as pool:
        running=pool.submit(worker)
        initial=client.open();assert initial['images'][0].getpixel((0,0))==(100,20,50)
        assert claim_episode(tmp_path,'other-worker')is None
        assert client.step({'action':[.25],'repeat':2})=={'done':True,'rewards':[1.],'executed':2}
        client.close();running.result(timeout=5)


def test_native_infrastructure_error_is_never_converted_to_zero_reward(tmp_path):
    client=FileEpisodeClient(tmp_path,'test',{},timeout=5)
    def worker():
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            claimed=claim_episode(tmp_path,'test-worker')
            if claimed:
                def broken():raise RuntimeError('missing native verifier output')
                with pytest.raises(RuntimeError):serve_episode(claimed[0],broken,{})
                return
            time.sleep(.01)
    with ThreadPoolExecutor(max_workers=1)as pool:
        running=pool.submit(worker)
        with pytest.raises(RemoteEpisodeLost,match='missing native verifier'):client.open()
        running.result(timeout=5)


def test_cancelled_pending_requests_are_preserved_without_starting_a_simulator(tmp_path):
    client=FileEpisodeClient(tmp_path,'efe/metaworldplus',{})
    pending=tmp_path/'pending'/(client.uid+'.json')
    atomic_json(pending,{'kind':client.kind,'env_args':{}})
    client.close()
    assert claim_episode(tmp_path,'worker')is None
    assert not pending.exists()
    assert(client.folder/'retired-pending-request.json').exists()
    assert not(client.folder/'response-000000.pkl').exists()


def test_legacy_expiry_respects_the_longer_native_yam_open_timeout(tmp_path):
    import os
    clients=[FileEpisodeClient(tmp_path,kind,{})for kind in ['efe/metaworldplus','native_yam/task']]
    for client in clients:
        p=tmp_path/'pending'/(client.uid+'.json');atomic_json(p,{'kind':client.kind,'env_args':{}})
        os.utime(p,(1000,1000))
    retired=retire_abandoned_pending(tmp_path,now=3200)
    assert len(retired)==1 and retired[0]['kind']=='efe/metaworldplus'
    assert retired[0]['reason']=='open_deadline_expired'
    assert(tmp_path/'pending'/(clients[1].uid+'.json')).exists()


def test_explicit_deadline_and_cancellation_after_claim_do_not_create_rewards(tmp_path):
    client=FileEpisodeClient(tmp_path,'custom/test',{})
    deadline=time.time()+1000
    pending=tmp_path/'pending'/(client.uid+'.json')
    atomic_json(pending,{'kind':client.kind,'env_args':{},'open_deadline_unix':deadline})
    assert retire_abandoned_pending(tmp_path,now=deadline+200)==[]
    claimed=claim_episode(tmp_path,'worker');assert claimed is not None
    client.close()
    serve_episode(claimed[0],lambda:pytest.fail('Cancelled episode started physics'),{})
    assert not(client.folder/'response-000000.pkl').exists()
    other=FileEpisodeClient(tmp_path,'custom/test',{})
    atomic_json(tmp_path/'pending'/(other.uid+'.json'),{'kind':other.kind,'env_args':{},'open_deadline_unix':deadline})
    assert len(retire_abandoned_pending(tmp_path,now=deadline+301))==1
