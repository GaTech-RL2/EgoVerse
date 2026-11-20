from pathlib import Path
import io
import urllib.error

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


def test_replay_limits_completion_to_the_actual_context_and_keeps_http_error_details(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    from efe_pear_matched_smoke import api_json,completion_capacity
    # Accumulated camera turns can leave less room than the per-call 32k cap.
    assert completion_capacity({'count':170000,'max_model_len':176384},160000)==6384
    assert completion_capacity({'count':10000,'max_model_len':176384},700)==700
    assert completion_capacity({'count':176384,'max_model_len':176384},160000)==0
    class FailedAPI:
        def open(self,*args,**kwargs):
            raise urllib.error.HTTPError('http://local',400,'Bad Request',{},
                io.BytesIO(b'{"error":"prompt exceeds the serving context"}'))
    with pytest.raises(RuntimeError,match='prompt exceeds the serving context'):
        api_json(FailedAPI(),'http://local/v1/chat/completions',{})
