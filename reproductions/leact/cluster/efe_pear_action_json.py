"""Decode a single final action without changing its keys, values or verifier.

The existing warmstart emits bare JSON; upstream PEAR emits fenced JSON. Both
represent the same constrained action. Never repair malformed JSON, unwrap an
answer, coerce coordinates, or recover an object from arbitrary answer prose.
"""
import json
import re


def final_action_json(text):
    answer=text.rpartition('</think>')[2] if '</think>' in text else text
    answer=answer.strip()
    for marker in ('<|im_end|>','<|endoftext|>'):
        if answer.endswith(marker):answer=answer[:-len(marker)].rstrip()
    blocks=re.findall(r'```json\s*([\s\S]*?)```',answer)
    if blocks:
        # A final fenced object may follow explanatory prose, as in PEAR.
        if answer[answer.rfind('```')+3:].strip():
            raise ValueError('Text follows the final JSON block')
        payload=blocks[-1].strip()
    else:payload=answer
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON action key: '+key)
            result[key]=value
        return result
    def finite_constant(value):raise ValueError('Nonfinite JSON action value: '+value)
    value=json.loads(payload,object_pairs_hook=unique_keys,parse_constant=finite_constant)
    if not isinstance(value,dict):raise ValueError('The final action must be one JSON object')
    return payload
