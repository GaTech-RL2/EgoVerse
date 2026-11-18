from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import pickle
import time

import pytest

from reproductions.leact.cluster.efe_pear_object_transport import (
    Store,NativeBridge,ObjectEpisodeClient,checked_result,LostEpisode)


class MissingObject(Exception):
    response={'Error':{'Code':'NoSuchKey'}}


class FakeS3:
    def __init__(self):self.objects={}
    def put_object(self,Bucket,Key,Body):self.objects[(Bucket,Key)]=bytes(Body)
    def get_object(self,Bucket,Key):
        if(Bucket,Key)not in self.objects:raise MissingObject()
        return {'Body':io.BytesIO(self.objects[(Bucket,Key)])}
    def delete_object(self,Bucket,Key):self.objects.pop((Bucket,Key),None)
    def list_objects_v2(self,Bucket,Prefix,ContinuationToken=None):
        keys=sorted(k for b,k in self.objects if b==Bucket and k.startswith(Prefix))
        start=int(ContinuationToken or 0);page=keys[start:start+2]
        result={'Contents':[{'Key':k}for k in page],'IsTruncated':start+2<len(keys)}
        if result['IsTruncated']:result['NextContinuationToken']=str(start+2)
        return result


@pytest.fixture
def transport(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    store=Store('s3://private/study/attempt',FakeS3())
    return store,NativeBridge(tmp_path/'native',store)


def pump(bridge,future,timeout=5):
    deadline=time.monotonic()+timeout
    while not future.done()and time.monotonic()<deadline:
        bridge.tick();time.sleep(.005)
    return future.result(timeout=1)


def test_cross_filesystem_episode_preserves_observations_actions_and_native_reward(transport):
    from efe_pear_file_rpc import claim_episode,publish_result,wait_command
    store,bridge=transport
    client=ObjectEpisodeClient(store,'native_yam/task',{'task':'task','reset_id':'train-019'},poll_seconds=.005)
    observation={'done':False,'rewards':[0.],'state':{'joints':[.1,.2]},'rgb':bytes(range(255))}
    terminal={'done':True,'rewards':[1.],'state':None,'native_success':True}
    with ThreadPoolExecutor(max_workers=1)as pool:
        opening=pool.submit(client.open)
        deadline=time.monotonic()+5
        claimed=None
        while claimed is None and time.monotonic()<deadline:
            bridge.tick();claimed=claim_episode(bridge.root,'native-worker',{'native_yam/task'});time.sleep(.005)
        assert claimed is not None
        folder,request=claimed
        assert request['env_args']=={'task':'task','reset_id':'train-019'}
        publish_result(folder,0,observation)
        assert pump(bridge,opening)==observation
        action={'action_type':'joint_position','action':[.1,.2],'repeat':2}
        stepping=pool.submit(client.step,action)
        deadline=time.monotonic()+5
        while not(folder/'request-000001.json').exists()and time.monotonic()<deadline:
            bridge.tick();time.sleep(.005)
        assert wait_command(folder,1,timeout=1)==action
        publish_result(folder,1,terminal)
        assert pump(bridge,stepping)==terminal
    client.close();bridge.tick()
    assert(folder/'cancel.json').exists()
    assert client.uid not in bridge.active
    assert not list(store.keys('responses/'))


def test_bridge_restart_does_not_duplicate_an_already_claimed_episode(transport):
    from efe_pear_file_rpc import claim_episode
    store,bridge=transport
    uid='a'*32;key='pending/'+uid+'.json'
    request={'kind':'efe/robodojo','env_args':{'task':'task'}}
    store.put_json(key,request);bridge.tick()
    assert claim_episode(bridge.root,'worker',{'efe/robodojo'})is not None
    # Simulate retrying a pending upload after local claim has committed.
    store.put_json(key,request)
    restarted=NativeBridge(bridge.root,store);restarted.tick()
    assert claim_episode(bridge.root,'other-worker',{'efe/robodojo'})is None


def test_infrastructure_failure_propagates_without_a_training_reward(transport):
    from efe_pear_file_rpc import atomic_json
    store,bridge=transport;client=ObjectEpisodeClient(store,'native_yam/task',{},poll_seconds=.005)
    store.put_json('pending/'+client.uid+'.json',{'kind':'native_yam/task','env_args':{}});bridge.tick()
    atomic_json(bridge.root/'episodes'/client.uid/'infrastructure-error.json',{'error':'simulator unavailable'})
    bridge.tick()
    with pytest.raises(LostEpisode,match='simulator unavailable'):client._receive()


@pytest.mark.parametrize('value',[{'done':True,'rewards':[]},{'done':True,'rewards':[.5]},
    {'done':True,'rewards':[True]},{'done':'true','rewards':[1.]}])
def test_malformed_results_cannot_be_used_as_native_rewards(value):
    with pytest.raises(LostEpisode):checked_result(pickle.dumps(value))


def test_object_listing_paginates_without_losing_pending_episodes():
    store=Store('s3://private/study/attempt',FakeS3())
    for i in range(5):store.put_json('pending/'+str(i),{'id':i})
    assert list(store.keys('pending/'))==['pending/'+str(i)for i in range(5)]


def test_malformed_episode_key_cannot_escape_the_private_filesystem(transport):
    store,bridge=transport
    store.put_json('pending/../escape.json',{'kind':'task','env_args':{}})
    with pytest.raises(ValueError,match='Unexpected episode object key'):bridge.tick()


def test_file_model_relay_preserves_native_control_and_leaves_other_families_local(transport,tmp_path):
    from efe_pear_file_rpc import FileEpisodeClient,claim_episode,publish_result,wait_command,atomic_json
    from reproductions.leact.cluster.efe_pear_file_relay import FileModelRelay,RelayNativeBridge
    store,_=transport;native=RelayNativeBridge(tmp_path/'graphics',store)
    model=FileModelRelay(tmp_path/'model',store,{'native_yam/task'})
    client=FileEpisodeClient(model.root,'native_yam/task',{'task':'task'},timeout=3)
    other=FileEpisodeClient(model.root,'efe/libero',{},timeout=3)
    atomic_json(model.root/'pending'/(other.uid+'.json'),{'kind':other.kind,'env_args':{}})
    def cycle():model.tick();native.tick()
    def until(predicate):
        deadline=time.monotonic()+3
        while not predicate()and time.monotonic()<deadline:cycle();time.sleep(.005)
        assert predicate()
    initial={'done':False,'rewards':[0.],'rgb':bytes(range(255))}
    terminal={'done':True,'rewards':[1.],'native_termination_terms':{'success':True}}
    try:
        with ThreadPoolExecutor(max_workers=1)as pool:
            opening=pool.submit(client.open)
            until(lambda:(native.root/'pending'/(client.uid+'.json')).exists())
            folder,request=claim_episode(native.root,'actual-worker',{'native_yam/task'})
            assert request['env_args']=={'task':'task'}
            heartbeat_time=time.time()-20
            atomic_json(native.root/'workers/actual-worker.json',{'worker_id':'actual-worker','updated_unix':heartbeat_time})
            publish_result(folder,0,initial);until(opening.done)
            assert opening.result()==initial
            assert json.loads((model.root/'workers/actual-worker.json').read_text())['updated_unix']==heartbeat_time
            action={'action_type':'joint_position','action':[.1,.2],'repeat':1}
            stepping=pool.submit(client.step,action)
            until(lambda:(folder/'request-000001.json').exists())
            assert wait_command(folder,1,timeout=1)==action
            publish_result(folder,1,terminal);until(stepping.done)
            assert stepping.result()==terminal
        assert(model.root/'pending'/(other.uid+'.json')).exists()
        client.close();cycle();assert(folder/'cancel.json').exists()
        assert client.uid not in model.active
    finally:model.close()


def test_file_relay_recovers_an_atomic_claim_before_progress_is_saved(transport,tmp_path):
    from efe_pear_file_rpc import FileEpisodeClient,atomic_json,claim_episode
    from reproductions.leact.cluster.efe_pear_file_relay import FileModelRelay,RelayNativeBridge
    store,_=transport;root=tmp_path/'model';native=RelayNativeBridge(tmp_path/'graphics',store)
    relay=FileModelRelay(root,store,{'native_yam/task'})
    client=FileEpisodeClient(root,'native_yam/task',{})
    # A crash can happen just after atomic ownership moves off the local queue.
    atomic_json(relay.owned/(client.uid+'.json'),{'kind':client.kind,'env_args':{}})
    relay.close();restored=FileModelRelay(root,store,{'native_yam/task'})
    try:
        restored.tick();native.tick()
        assert claim_episode(native.root,'worker',{'native_yam/task'})is not None
        restored.tick();native.tick()
        assert claim_episode(native.root,'other',{'native_yam/task'})is None
    finally:restored.close()


def test_file_relay_propagates_infrastructure_errors_without_an_outcome(transport,tmp_path):
    from efe_pear_file_rpc import FileEpisodeClient,RemoteEpisodeLost,atomic_json
    from reproductions.leact.cluster.efe_pear_file_relay import FileModelRelay
    store,_=transport;relay=FileModelRelay(tmp_path/'model',store,{'native_yam/task'})
    client=FileEpisodeClient(relay.root,'native_yam/task',{})
    atomic_json(relay.root/'pending'/(client.uid+'.json'),{'kind':client.kind,'env_args':{}})
    store.put_json('errors/'+client.uid+'.json',{'error':'native renderer failed'})
    try:
        relay.tick()
        with pytest.raises(RemoteEpisodeLost,match='renderer failed'):client._receive()
        assert not list(client.folder.glob('response-*.pkl'))
    finally:relay.close()


def test_file_relay_does_not_refresh_an_expired_native_heartbeat(transport,tmp_path):
    from efe_pear_file_rpc import FileEpisodeClient,RemoteEpisodeLost,atomic_json
    from reproductions.leact.cluster.efe_pear_file_relay import FileModelRelay
    store,_=transport;relay=FileModelRelay(tmp_path/'model',store,{'native_yam/task'})
    client=FileEpisodeClient(relay.root,'native_yam/task',{})
    atomic_json(relay.owned/(client.uid+'.json'),{'kind':client.kind,'env_args':{}})
    relay.active[client.uid]={'index':0,'request':1,'sent':False}
    store.put_json('claims/'+client.uid+'.json',{'worker_id':'native-worker'})
    store.put_json('workers/native-worker.json',{'worker_id':'native-worker','updated_unix':1.})
    try:
        relay.tick()
        with pytest.raises(RemoteEpisodeLost,match='heartbeat expired'):client._receive()
    finally:relay.close()


def test_file_relay_refuses_to_resume_with_a_different_remote_owner(transport,tmp_path):
    from reproductions.leact.cluster.efe_pear_file_relay import FileModelRelay
    store,_=transport;root=tmp_path/'model'
    relay=FileModelRelay(root,store,{'native_yam/task'});relay.close()
    with pytest.raises(ValueError,match='ownership differs'):
        FileModelRelay(root,Store('s3://private/other/attempt',store.client),{'native_yam/task'})
    with pytest.raises(ValueError,match='ownership differs'):
        FileModelRelay(root,store,{'native_yam/held-out-task'})
