import base64
import json
import time

from reproductions.leact.cluster.collect_results import REMOTE
from reproductions.leact.data import write_json


def test_coldstart_collector_keeps_actual_leact_selection_and_length_admission(tmp_path, monkeypatch, capsys):
    training = tmp_path / 'leact/round-1/sft'
    write_json(training/'manifest.json', {'sft_examples':2})
    selection = dict(state_id='state', baseline_valid=True, selected=[
        dict(candidate_id=1, admitted_to_sft=True), dict(candidate_id=3, admitted_to_sft=False)])
    (training/'selection.jsonl').write_text(json.dumps(selection)+'\n')
    write_json(tmp_path/'complete.json', {})
    monkeypatch.setattr(time, 'sleep', lambda _:None)
    exec(REMOTE, dict(ROOT=str(tmp_path), DURATION=0, INTERVAL=1, COLDSTART=True,
                     GATE_ONLY=False, INCLUDE_ROLLOUTS=False, PROMPT_DIAGNOSTIC=False))
    records = [json.loads(line[len('LEACT_FILE '):]) for line in capsys.readouterr().out.splitlines()
               if line.startswith('LEACT_FILE ')]
    record = next(r for r in records if r['relative']=='leact/round-1/sft/admission.json')
    payload = json.loads(base64.b64decode(record['base64']))
    assert payload == {'state':{'baseline_valid':True,'selected':selection['selected']}}
