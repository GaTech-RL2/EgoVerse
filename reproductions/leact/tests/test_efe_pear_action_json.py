import json

import pytest

from reproductions.leact.cluster.efe_pear_action_json import final_action_json


@pytest.mark.parametrize('text',[
    '{"answer_type":"point","point":[0.2,0.3]}',
    '<think>consider {the image}</think>\n{"answer_type":"point","point":[0.2,0.3]}<|im_end|>',
    '<think>consider</think>\nExplanation.\n```json\n{"answer_type":"point","point":[0.2,0.3]}\n```<|im_end|>',
])
def test_warmstart_and_pear_framing_preserve_the_identical_action(text):
    assert json.loads(final_action_json(text))=={'answer_type':'point','point':[.2,.3]}


@pytest.mark.parametrize('text',[
    'Example {"point":[0.2,0.3]} extra prose',
    '{"point":[0.2,0.3]} {"point":[0.4,0.5]}',
    '```json\n{"point":[0.2,0.3]}\n``` and another action',
    '{"point":[NaN,0.3]}',
    '{"point":[0.2,0.3],"point":[0.4,0.5]}',
    '[0.2,0.3]',
])
def test_never_repair_ambiguous_or_invalid_actions(text):
    with pytest.raises(ValueError):final_action_json(text)


def test_never_unwrap_or_coerce_schema_fields():
    assert json.loads(final_action_json('{"answer":{"point":[200,300]}}'))=={'answer':{'point':[200,300]}}
