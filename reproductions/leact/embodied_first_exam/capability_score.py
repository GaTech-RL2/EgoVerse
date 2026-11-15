"""Official benchmark scoring, executed only inside the isolated scorer container."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mmlu_answer(text):
    # Exact three extraction patterns in the pinned upstream local evaluator.
    # An unparseable answer is wrong; never guess or retry using the test label.
    for pattern, flags in [(r'answer is \(?([A-J])\)?', 0),
                           (r'.*[aA]nswer:\s*([A-J])', 0),
                           (r'\b([A-J])\b(?!.*\b[A-J]\b)', re.DOTALL)]:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1)
    return None


def require_complete(prompts, generations):
    expected = {r['id'] for r in prompts}
    actual = [r['id'] for r in generations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate generations; do not publish an aggregate score')
    if any(r.get('status') != 'completed' for r in generations):
        raise ValueError('Inference infrastructure failures remain; do not treat them as policy failures')


def paired_comparison(reference, candidate, *, draws=4000):
    """Paired question bootstrap, preserving subject strata for MMLU-Pro-280."""
    import numpy as np
    if reference['suite_sha256'] != candidate['suite_sha256']:
        raise ValueError('Cannot compare different frozen suites')
    if not reference.get('complete') or not candidate.get('complete'):
        raise ValueError('Cannot compare incomplete results')
    comparisons = {}
    for bench in reference['benchmarks']:
        left = {r['id']: r for r in reference['items'] if r['benchmark'] == bench}
        right = {r['id']: r for r in candidate['items'] if r['benchmark'] == bench}
        if left.keys() != right.keys() or not left:
            raise ValueError('Mismatched paired question IDs')
        ids = sorted(left)
        diffs = np.array([int(right[k]['correct']) - int(left[k]['correct']) for k in ids])
        rng = np.random.default_rng(20261026)
        groups = sorted({left[k].get('category', 'all') for k in ids})
        total = np.zeros(draws)
        for group in groups:
            values = diffs[[left[k].get('category', 'all') == group for k in ids]]
            total += values[rng.integers(0, len(values), (draws, len(values)))].sum(axis=1)
        interval = np.quantile(total / len(ids), [0.025, 0.975]) * 100
        comparisons[bench] = {
            'questions': len(ids), 'delta_percentage_points': float(diffs.mean() * 100),
            'paired_bootstrap_95_ci_pp': interval.tolist(),
            'correct_to_wrong': int((diffs == -1).sum()), 'wrong_to_correct': int((diffs == 1).sum()),
            'interpretation': 'Descriptive question uncertainty for one inference seed; not training-run uncertainty or a guarantee of preserved general ability.'}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--generations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite-sha256', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(args.inputs / 'prompts.jsonl')
    generations = read_jsonl(args.generations)
    require_complete(prompts, generations)
    responses = {r['id']: r for r in generations}
    items = []

    from evalplus.sanitize import sanitize
    human = read_jsonl(args.inputs / 'humaneval.jsonl')
    samples = args.output / 'humaneval-samples.jsonl'
    with samples.open('w') as out:
        for row in human:
            text = responses['humaneval/' + row['task_id']]['final']
            out.write(json.dumps({'task_id': row['task_id'],
                                 'solution': sanitize(text, entrypoint=row['entry_point'])}) + '\n')
    # This module must only run in the network-disabled, unprivileged child container.
    subprocess.run([sys.executable, '-m', 'evalplus.evaluate', '--dataset', 'humaneval',
                    '--samples', str(samples), '--parallel', '8',
                    '--output_file', str(args.output / 'humaneval-grades.json')], check=True, timeout=1800)
    grades = json.loads((args.output / 'humaneval-grades.json').read_text())
    for row in human:
        grade = grades['eval'][row['task_id']]
        if len(grade) != 1:
            raise ValueError('Expected exactly one candidate per HumanEval problem')
        items.append({'id': 'humaneval/' + row['task_id'], 'benchmark': 'humaneval_plus',
                      'correct': grade[0]['base_status'] == grade[0]['plus_status'] == 'pass',
                      'base_correct': grade[0]['base_status'] == 'pass'})

    from instruction_following_eval import evaluation_lib
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    random.seed(20261026)
    for row in read_jsonl(args.inputs / 'ifeval.jsonl'):
        inp = evaluation_lib.InputExample(**row)
        text = responses['ifeval/' + str(row['key'])]['final']
        strict = evaluation_lib.test_instruction_following_strict(inp, {inp.prompt: text})
        loose = evaluation_lib.test_instruction_following_loose(inp, {inp.prompt: text})
        items.append({'id': 'ifeval/' + str(row['key']), 'benchmark': 'ifeval',
                      'correct': bool(strict.follow_all_instructions),
                      'instruction_correct': [bool(x) for x in strict.follow_instruction_list],
                      'loose_correct': bool(loose.follow_all_instructions)})

    for row in json.loads((args.inputs / 'mmlu-pro-280.json').read_text()):
        identity = 'mmlu_pro/' + str(row['question_id'])
        prediction = mmlu_answer(responses[identity]['final'])
        items.append({'id': identity, 'benchmark': 'mmlu_pro_280', 'category': row['category'],
                      'correct': prediction == row['answer'], 'prediction': prediction,
                      'parsed': prediction is not None})
    benchmarks = {}
    for bench in ['humaneval_plus', 'ifeval', 'mmlu_pro_280']:
        subset = [r for r in items if r['benchmark'] == bench]
        generated = [responses[r['id']] for r in subset]
        benchmarks[bench] = {
            'count': len(subset), 'correct': sum(r['correct'] for r in subset),
            'score': statistics.mean(r['correct'] for r in subset),
            'truncated': sum(r['finish_reason'] == 'length' for r in generated),
            'empty_final': sum(not r['final'].strip() for r in generated),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in generated),
            'mean_request_seconds': statistics.mean(r['elapsed_seconds'] for r in generated)}
    instructions = [x for r in items if r['benchmark'] == 'ifeval' for x in r['instruction_correct']]
    benchmarks['ifeval']['strict_instruction_accuracy'] = statistics.mean(instructions)
    benchmarks['ifeval']['loose_prompt_accuracy'] = statistics.mean(r['loose_correct'] for r in items if r['benchmark'] == 'ifeval')
    benchmarks['humaneval_plus']['original_test_pass_at_1'] = statistics.mean(r['base_correct'] for r in items if r['benchmark'] == 'humaneval_plus')
    result = {'complete': True, 'suite_sha256': args.suite_sha256, 'benchmarks': benchmarks, 'items': items,
              'note': 'MMLU-Pro-280 is a stratified subset, not an official full MMLU-Pro score. No combined score.'}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
