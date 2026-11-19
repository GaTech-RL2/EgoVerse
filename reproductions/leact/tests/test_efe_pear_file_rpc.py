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


def test_two_workers_retiring_the_same_cancelled_request_keep_serving(tmp_path):
    from reproductions.leact.cluster.efe_pear_file_rpc import retire_abandoned_request
    client=FileEpisodeClient(tmp_path,'efe/robodojo',{})
    pending=tmp_path/'pending'/(client.uid+'.json')
    request={'kind':client.kind,'env_args':{}}
    atomic_json(pending,request);client.close()
    assert retire_abandoned_request(tmp_path,pending,request)['reason']=='cancelled'
    # A second worker already read the request before the first rename.
    assert retire_abandoned_request(tmp_path,pending,request)is None
    assert claim_episode(tmp_path,'surviving-worker')is None
    assert not(client.folder/'response-000000.pkl').exists()


def test_replayed_retired_request_does_not_launch_another_episode(tmp_path):
    client=FileEpisodeClient(tmp_path,'efe/robodojo',{})
    pending=tmp_path/'pending'/(client.uid+'.json')
    request={'kind':client.kind,'env_args':{}}
    atomic_json(pending,request);client.close()
    assert claim_episode(tmp_path,'first-worker')is None
    original=(client.folder/'retired-pending-request.json').read_bytes()
    receipt=(client.folder/'retired-pending-receipt.json').read_bytes()
    atomic_json(pending,request)
    assert claim_episode(tmp_path,'second-worker')is None
    assert not pending.exists()
    assert(client.folder/'retired-pending-request.json').read_bytes()==original
    assert(client.folder/'retired-pending-receipt.json').read_bytes()==receipt
    assert not(client.folder/'claimed.json').exists()


def test_changed_replay_is_preserved_for_investigation(tmp_path):
    client=FileEpisodeClient(tmp_path,'efe/robodojo',{})
    pending=tmp_path/'pending'/(client.uid+'.json')
    atomic_json(pending,{'kind':client.kind,'env_args':{}});client.close()
    assert claim_episode(tmp_path,'first-worker')is None
    atomic_json(pending,{'kind':client.kind,'env_args':{'task':'different'}})
    with pytest.raises(ValueError,match='different contents'):claim_episode(tmp_path,'second-worker')
    assert pending.exists()
    assert not(client.folder/'claimed.json').exists()


def test_native_queue_claims_the_oldest_live_request_instead_of_the_smallest_uuid(tmp_path):
    now=time.time()
    for uid,age in [('f'*32,900),('0'*32,10)]:
        (tmp_path/'episodes'/uid).mkdir(parents=True)
        (tmp_path/'pending').mkdir(exist_ok=True)
        atomic_json(tmp_path/'pending'/(uid+'.json'),{'kind':'efe/metaworldplus','env_args':{},
            'open_deadline_unix':now-age+1800})
    assert claim_episode(tmp_path,'first',{'efe/metaworldplus'})[0].name=='f'*32
    assert claim_episode(tmp_path,'second',{'efe/metaworldplus'})[0].name=='0'*32


def test_expired_request_in_retirement_grace_does_not_displace_a_live_request(tmp_path):
    now=time.time()
    for uid,deadline in [('0'*32,now-10),('f'*32,now+300)]:
        (tmp_path/'episodes'/uid).mkdir(parents=True)
        (tmp_path/'pending').mkdir(exist_ok=True)
        atomic_json(tmp_path/'pending'/(uid+'.json'),{'kind':'efe/metaworldplus','env_args':{},
            'open_deadline_unix':deadline})
    assert claim_episode(tmp_path,'worker',{'efe/metaworldplus'})[0].name=='f'*32
    assert(tmp_path/'pending'/('0'*32+'.json')).exists()
    assert not(tmp_path/'episodes'/('0'*32)/'retired-pending-request.json').exists()


def test_legacy_queue_age_falls_back_to_mtime_and_respects_worker_kinds(tmp_path):
    import os
    for uid,mtime,kind in [('f'*32,10,'custom/eligible'),('0'*32,20,'custom/eligible'),('1'*32,1,'custom/other')]:
        (tmp_path/'episodes'/uid).mkdir(parents=True)
        (tmp_path/'pending').mkdir(exist_ok=True)
        p=tmp_path/'pending'/(uid+'.json');atomic_json(p,{'kind':kind,'env_args':{}});os.utime(p,(mtime,mtime))
    assert claim_episode(tmp_path,'worker',{'custom/eligible'})[0].name=='f'*32
    assert(tmp_path/'pending'/('1'*32+'.json')).exists()


def test_native_termination_closes_the_episode_without_fabricating_a_verdict(tmp_path):
    from reproductions.leact.cluster.efe_pear_remote_worker import stop_signal
    closed=[]
    class Environment:
        def open(self):stop_signal()
        def close(self):closed.append(True)
    with pytest.raises(SystemExit)as error:serve_episode(tmp_path,Environment,{})
    assert error.value.code==143 and closed==[True]
    assert not(tmp_path/'response-000000.pkl').exists()


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
