import json
from pathlib import Path

import pytest

from reproductions.leact.cluster.efe_pear_float32_json import bits,public_number,public_contract,verify_contract


def test_short_bounds_accept_the_same_native_gripper_command_without_clipping(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    from efe_action_loop import parse_action
    original={'action_types':{'joint_position':{'dim':1,'layout':['gripper'],
        'lower':[-0.04749999940395355],'upper':[0.]}},'max_repeat':25}
    actual=public_contract(original);verify_contract(actual,original)
    assert actual['action_types']['joint_position']['lower']==[-.0475]
    action={'action_type':'joint_position','action':[-.0475],'repeat':1,'summary':'Open gripper.'}
    with pytest.raises(ValueError,match='bounds'):parse_action(json.dumps(action),contract=original,allow_stop=True)
    accepted=parse_action(json.dumps(action),contract=actual,allow_stop=True)
    assert accepted==action and bits(accepted['action'][0])==bits(original['action_types']['joint_position']['lower'][0])
    for value in [-.048,.007]:
        action['action']=[value]
        with pytest.raises(ValueError,match='bounds'):parse_action(json.dumps(action),contract=actual,allow_stop=True)


def test_observation_values_and_bounds_keep_identical_float32_bits():
    import struct
    values=[struct.unpack('!f',struct.pack('!f',v))[0]for v in [-.0475,0.,3.66519,-2.61799,.07845018,1e-15]]
    assert all(bits(public_number(v))==bits(v)for v in values)
    assert public_number(values[0])==-.0475
    with pytest.raises(ValueError,match='float64'):public_number(.1)
    with pytest.raises(ValueError,match='finite'):public_number(float('nan'))
