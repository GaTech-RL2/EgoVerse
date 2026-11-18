from concurrent.futures import ThreadPoolExecutor
import time

from PIL import Image
import pytest

from reproductions.leact.cluster.efe_pear_file_rpc import (
    FileEpisodeClient,RemoteEpisodeLost,claim_episode,serve_episode)


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
