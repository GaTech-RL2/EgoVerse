import ast
import copy
from pathlib import Path
from types import SimpleNamespace

from reproductions.leact.cluster.efe_pear_yam_contract import terminal_result,observe_termination_computes


def test_native_success_survives_auto_reset_and_needs_no_new_camera(tmp_path):
    class Scalar:
        def __init__(self,value):self.value=value
        def item(self):return self.value
    manager=SimpleNamespace(active_terms=['success','time_out'],value=False)
    manager.get_term=lambda name:[Scalar(manager.value if name=='success'else False)]
    manager.compute=lambda:setattr(manager,'value',True)
    captured=[];observe_termination_computes(manager,captured.append)
    manager.compute();manager.value=False  # native auto-reset clears live buffers
    result=terminal_result({'action_types':{}},'task','independent-scene',captured[-1],
        {'action_type':'joint_position','action':[0.],'repeat':3,'summary':'complete'},2,tmp_path/'verdict.json')
    assert result['done']and result['rewards']==[1.]
    assert result['native_termination_terms']['success']is True
    assert result['images']is None and result['executed_control_steps']==2


def test_terminal_policy_branch_does_not_read_post_reset_sensors(tmp_path):
    path=Path(__file__).resolve().parents[1]/'cluster/efe_pear_yam_session_policy.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='YamSessionPolicy')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_public_result'))
    namespace={'terminal_result':terminal_result}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    policy=SimpleNamespace(contract={},instruction='task',scene_sha256='independent-scene',
        terms={'success':True,'time_out':False},folder=tmp_path)
    # No env/sensor object is available after a native auto-reset. A terminal
    # result must depend exclusively on the captured live termination terms.
    value=namespace['_public_result'](policy,None,None,done=True,action={'stop':True,'summary':'done'})
    assert value['rewards']==[1.]and value['public_blocks']is None
