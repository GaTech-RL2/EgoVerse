import copy

import pytest

from reproductions.leact.embodied_first_exam.capability_retention import DEST, read, jsonl, stratified
from reproductions.leact.embodied_first_exam.capability_score import mmlu_answer, paired_comparison, require_complete


def test_frozen_suite_has_all_questions_and_no_answers_in_generation_inputs():
    required = ['suite.json', 'inputs/prompts.jsonl', 'inputs/mmlu-pro-280.json']
    if not all((DEST / name).is_file() for name in required):
        pytest.skip('Requires the separately prepared 985-question capability suite')
    plan = read(DEST / 'suite.json')
    prompts = jsonl(DEST / 'inputs/prompts.jsonl')
    assert sum(b['count'] for b in plan['benchmarks']) == len(prompts) == 985
    assert len({r['id'] for r in prompts}) == 985
    assert all(not ({'answer', 'answer_index', 'canonical_solution', 'base_input', 'plus_input'} & r.keys()) for r in prompts)
    selected = read(DEST / 'inputs/mmlu-pro-280.json')
    assert len({r['question_id'] for r in selected}) == 280
    assert len({r['category'] for r in selected}) == 14


def test_selection_is_independent_of_input_order_and_labels():
    data = [{'question_id': i, 'category': str(i % 3), 'answer': 'A'} for i in range(90)]
    selected = [r['question_id'] for r in stratified(data, 10)]
    relabeled = [dict(r, answer='J') for r in reversed(data)]
    assert selected == [r['question_id'] for r in stratified(relabeled, 10)]


def test_missing_http_failure_and_duplicate_are_not_scored_as_model_failures():
    prompts = [{'id': 'a'}, {'id': 'b'}]
    with pytest.raises(ValueError):
        require_complete(prompts, [{'id': 'a', 'status': 'completed'}])
    with pytest.raises(ValueError):
        require_complete(prompts, [{'id': 'a', 'status': 'completed'}, {'id': 'b', 'status': 'http_error'}])
    with pytest.raises(ValueError):
        require_complete(prompts, [{'id': 'a', 'status': 'completed'}] * 2)
    # A complete but empty/truncated model answer remains a scored item.
    require_complete(prompts, [{'id': k, 'status': 'completed', 'final': '', 'finish_reason': 'length'} for k in ['a', 'b']])


def test_mmlu_unparseable_answers_are_wrong_without_random_fallback():
    assert mmlu_answer('After checking, the answer is (C).') == 'C'
    assert mmlu_answer('Answer: J') == 'J'
    assert mmlu_answer('There is insufficient information.') is None


def test_paired_comparison_counts_regressions_and_rejects_protocol_drift():
    base = {'complete': True, 'suite_sha256': 'frozen', 'benchmarks': {'coding': {}},
            'items': [{'id': str(i), 'benchmark': 'coding', 'correct': True} for i in range(10)]}
    candidate = copy.deepcopy(base)
    for row in candidate['items'][:4]: row['correct'] = False
    result = paired_comparison(base, candidate, draws=500)['coding']
    assert result['delta_percentage_points'] == -40
    assert result['correct_to_wrong'] == 4 and result['wrong_to_correct'] == 0
    assert result['paired_bootstrap_95_ci_pp'][1] <= 0
    candidate['suite_sha256'] = 'changed'
    with pytest.raises(ValueError): paired_comparison(base, candidate)


def test_pairing_uses_ids_not_generation_completion_order():
    base = {'complete': True, 'suite_sha256': 'frozen', 'benchmarks': {'coding': {}},
            'items': [{'id': str(i), 'benchmark': 'coding', 'correct': bool(i % 2)} for i in range(20)]}
    reordered = dict(base, items=list(reversed(base['items'])))
    result = paired_comparison(base, reordered, draws=100)['coding']
    assert result['delta_percentage_points'] == 0
    assert result['paired_bootstrap_95_ci_pp'] == [0, 0]


def test_existing_text_checkpoints_wait_for_their_own_base_not_efe_vl():
    from reproductions.leact.cluster.launch_efe_capability import reference_ids
    registry = {'families': {'text': {'base_checkpoint': 'text-base'}}}
    assert reference_ids({'id': 'text-base', 'family': 'text', 'parent': None}, registry) == set()
    assert reference_ids({'id': 'text-sft', 'family': 'text', 'parent': 'text-base'}, registry) == {'text-base'}
    assert reference_ids({'id': 'text-leact', 'family': 'text', 'parent': 'text-sft'}, registry) == {'text-base', 'text-sft'}
    assert reference_ids({'id': 'qwen-leact', 'parent': 'qwen-warmstart'}, registry) == {'qwen-base', 'qwen-warmstart'}


def test_text_snapshot_links_require_frozen_path_and_bytes(tmp_path):
    import hashlib
    import json
    from reproductions.leact.cluster.text_capability_eval import verify_checkpoint
    snapshot = tmp_path / 'snapshot'; snapshot.mkdir()
    blob = tmp_path / 'weights-blob'; blob.write_bytes(b'frozen checkpoint fixture')
    name = 'model.safetensors'
    (snapshot / name).symlink_to(blob)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {name: {'resolved_path': str(blob),
        'bytes': blob.stat().st_size, 'sha256': hashlib.sha256(blob.read_bytes()).hexdigest()}}}))
    assert verify_checkpoint(snapshot, manifest)
    blob.write_bytes(b'changed checkpoint bytes!')
    with pytest.raises(ValueError, match='Checkpoint mismatch'):
        verify_checkpoint(snapshot, manifest)
    second = tmp_path / 'other-blob'; second.write_bytes(b'frozen checkpoint fixture')
    (snapshot / name).unlink(); (snapshot / name).symlink_to(second)
    with pytest.raises(ValueError, match='resolved path'):
        verify_checkpoint(snapshot, manifest)
