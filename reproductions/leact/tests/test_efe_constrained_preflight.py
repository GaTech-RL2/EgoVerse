import importlib.util
from pathlib import Path
import sys

import pytest

from reproductions.leact.cluster import efe_action_loop
from reproductions.leact.cluster.launch_efe_coverage_eval import constrained_preflight_source


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setitem(sys.modules,'efe_action_loop',efe_action_loop)
    path=Path(__file__).parents[1]/'cluster/efe_constrained_preflight.py'
    spec=importlib.util.spec_from_file_location('constrained_preflight_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_direct_action_eval_does_not_select_policies_by_tool_call_ability(probe):
    bodies=[]
    def response(url,body,token):
        bodies.append(body)
        return {'status':'completed','output':[{'type':'function_call','name':'wrong','arguments':'{}'}]}
    result=probe.check({'request_model':'fixture','upstream':'fixture'},'token',response)
    assert result['transport_status']=='passed' and result['benchmark_execution_allowed']
    assert result['status']=='semantic_failure'
    assert len(result['trials'])==2 and not any(t['valid_action'] for t in result['trials'])
    assert all('tools' not in body and body['instructions']==efe_action_loop.SYSTEM for body in bodies)


def test_direct_action_probe_still_blocks_transport_and_incomplete_responses(probe):
    def failed(*args):raise RuntimeError('HTTP failure')
    with pytest.raises(RuntimeError,match='HTTP failure'):
        probe.check({'request_model':'fixture','upstream':'fixture'},'token',failed)
    with pytest.raises(RuntimeError,match='did not complete'):
        probe.check({'request_model':'fixture','upstream':'fixture'},'token',lambda *args:{'status':'incomplete'})


def test_launcher_replaces_only_preflight_dispatch():
    source="        return vision_tool_preflight({**spec,'upstream':'http://127.0.0.1:19862/v1'},token)\n\n# Native execution follows unchanged.\nrun_native(spec)\n"
    result=constrained_preflight_source(source)
    assert 'from efe_constrained_preflight import check' in result
    assert result.endswith('\n\n# Native execution follows unchanged.\nrun_native(spec)\n')
    with pytest.raises(ValueError):constrained_preflight_source(result)
