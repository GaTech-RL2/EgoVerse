import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from reproductions.leact.cluster.efe_pear_native_finalization import finalize_episode,MissingNativeVerdict


def episode():
    e=SimpleNamespace(uid='episode',env_args={'task':'training-task'},done=False,success=False,
        actions=[{'action_type':'ee_delta'}],num_chunks=1,chunk_rewards=[0.])
    e.is_success=lambda value:bool(value['done']and sum(value['rewards'])>0)
    return e


@pytest.mark.parametrize('reward',[0.,1.])
def test_budget_or_parse_end_collects_native_grade_without_adding_a_sampled_action(reward):
    e=episode();calls=[]
    class Client:
        async def step(self,uid,action):
            calls.append((uid,action))
            return{'done':True,'rewards':[reward],'actions':[],'executed':0,'native_trial_path':'/native/result.json'}
    receipt=asyncio.run(finalize_episode(Client(),e))
    assert e.done and e.success==bool(reward)and receipt['success']==bool(reward)
    assert e.num_chunks==1 and len(e.actions)==1 and e.chunk_rewards==[0.]
    assert calls[0][1]['stop']is True
    assert asyncio.run(finalize_episode(Client(),e))is None and len(calls)==1


def test_a_missing_native_grade_is_an_error_not_a_zero_reward():
    e=episode()
    class Client:
        async def step(self,*args):return{'done':False,'rewards':[0.],'actions':[],'executed':0}
    with pytest.raises(MissingNativeVerdict):asyncio.run(finalize_episode(Client(),e))
    assert not e.done


def test_actual_tool_close_collects_the_reward_before_upstream_cleanup(tmp_path,monkeypatch):
    calls=[]
    class Client:
        async def step(self,uid,action):
            calls.append('verdict')
            return{'done':True,'rewards':[1.],'actions':[],'executed':0,'native_trial_path':'/native/result.json'}
    class BaseTool:
        def __init__(self):self.client=Client();self.log_path=tmp_path/'actions.jsonl'
        async def close(self,e):
            assert e.success and e.done
            calls.append('cleanup')
    protocol=ModuleType('pear.agent_loop.protocol');protocol.InvalidAction=ValueError
    base=ModuleType('pear.envs.tool');base.EnvironmentTool=BaseTool
    monkeypatch.setitem(sys.modules,'pear.agent_loop.protocol',protocol)
    monkeypatch.setitem(sys.modules,'pear.envs.tool',base)
    folder=Path(__file__).parents[1]/'cluster';monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('tested_native_finalizing_tool',folder/'efe_pear_native_tool.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tool=module.NativeEfeTool();e=episode();tool.contracts[e.uid]={}
    asyncio.run(tool.close(e))
    assert calls==['verdict','cleanup']and e.uid not in tool.contracts
    row=json.loads(next(tmp_path.glob('native_finalizations_*.jsonl')).read_text())
    assert row['success']and row['harness_finalization']and not row['sampled_action_added']
