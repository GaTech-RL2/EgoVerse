from pathlib import Path

import pytest


def test_case_replay_uses_the_production_final_json_decoder_and_native_limits(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    from efe_pear_matched_smoke import decode_action
    contract={'action_types':{'ee_delta':{'dim':1,'lower':[-1.],'upper':[1.]}},'max_repeat':25}
    text='<think>Student diagnostic thought.</think>\nMove toward the object.\n```json\n{"action_type":"ee_delta","action":[0.1],"repeat":1,"summary":"Approach."}\n```'
    action=decode_action(text,contract)
    assert action['action']==[.1]and action['action_type']=='ee_delta'
    assert decode_action(text.replace('<think>',''),contract)==action
    with pytest.raises(ValueError,match='bounds'):decode_action(text.replace('[0.1]','[2.0]'),contract)
