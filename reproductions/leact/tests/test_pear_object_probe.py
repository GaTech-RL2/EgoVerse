from pathlib import Path

import pytest

from reproductions.leact.cluster.efe_pear_object_probe import hold_action,stop_action


def test_native_probe_actions_obey_the_real_control_parser(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    from efe_action_loop import parse_action
    import json
    contract={'action_types':{'joint_position':{'dim':2,'lower':[-1.,-1.],'upper':[1.,1.]}},'max_repeat':1}
    action=hold_action([.2,-.3],contract)
    assert action['action']==[.2,-.3]and action['repeat']==1 and action['summary']
    assert parse_action(json.dumps(action),contract=contract,allow_stop=True)==action
    assert stop_action(contract)['stop']is True
    with pytest.raises(ValueError,match='bounds'):hold_action([2.,0.],contract)
