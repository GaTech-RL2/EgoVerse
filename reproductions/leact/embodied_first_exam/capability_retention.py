"""Freeze a small, evaluation-only checkpoint capability regression study."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from .report import ROOT, read, write

DEST = ROOT / 'capability-retention'
PINS = {
    'evalplus': '26d6d00bb1fd0fa37f39c99d5290da67891d1c5e',
    'ifeval': '700de9630a39cf1c0ded1ea7d1092ec0f467c7a7',
    'mmlu_pro_code': 'f418b116db00b065c2aea046518d8fcf74d39872',
    'mmlu_pro_data': 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
}
SUITE = 'efe-capability-retention-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.download')
        with urllib.request.urlopen(url, timeout=180) as inp, temp.open('wb') as out:
            while chunk := inp.read(1024 * 1024):
                out.write(chunk)
        temp.replace(path)
    return path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratified(rows, per_category=20):
    categories = sorted({r['category'] for r in rows})
    selected = []
    for category in categories:
        pool = [r for r in rows if r['category'] == category]
        if len(pool) < per_category:
            raise ValueError('Insufficient examples: ' + category)
        pool.sort(key=lambda r: hashlib.sha256(
            (SUITE + '/' + str(r['question_id'])).encode()).hexdigest())
        selected.extend(pool[:per_category])
    return selected


def mmlu_question(row, answer=False):
    options = [x for x in row['options'] if x != 'N/A']
    result = 'Question:\n' + row['question'] + '\nOptions:\n'
    result += ''.join(f'{chr(65+i)}. {opt}\n' for i, opt in enumerate(options))
    if answer:
        return result + row['cot_content'].replace("A: Let's think step by step.",
                                                   "Answer: Let's think step by step.") + '\n\n'
    return result + "Answer: Let's think step by step."


def prepare():
    """Pin all data and evaluators before the first model answer is generated."""
    import pyarrow.parquet as pq

    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'suite.json').exists():
        plan = read(DEST / 'suite.json')
        for name, digest in plan['file_sha256'].items():
            if sha(DEST / name) != digest:
                raise ValueError('Frozen suite changed: ' + name)
        return plan
    inputs = DEST / 'inputs'
    vendor = DEST / 'vendor'
    upstream = DEST / 'upstream'
    gz = fetch('https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz',
               upstream / 'HumanEvalPlus-v0.1.10.jsonl.gz')
    inputs.mkdir(exist_ok=True)
    (inputs / 'humaneval.jsonl').write_bytes(gzip.decompress(gz.read_bytes()))
    evaltar = fetch('https://codeload.github.com/evalplus/evalplus/tar.gz/' + PINS['evalplus'],
                    upstream / 'evalplus.tar.gz')
    extract = vendor / 'evalplus-source'
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(evaltar) as tar:
        tar.extractall(extract, filter='data')
    evalsrc = next(extract.glob('evalplus-*'))
    # Preserve upstream license and exact source; PYTHONPATH uses this directory.
    evalsrc.rename(vendor / 'evalplus')
    extract.rmdir()
    google = 'https://raw.githubusercontent.com/google-research/google-research/' + PINS['ifeval']
    for name in ['evaluation_lib.py', 'instructions.py', 'instructions_registry.py',
                 'instructions_util.py', 'README.md']:
        fetch(google + '/instruction_following_eval/' + name, vendor / 'instruction_following_eval' / name)
    fetch(google + '/LICENSE', vendor / 'instruction_following_eval' / 'LICENSE')
    fetch(google + '/instruction_following_eval/data/input_data.jsonl', inputs / 'ifeval.jsonl')
    hf = 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/' + PINS['mmlu_pro_data'] + '/data/'
    test = pq.read_table(fetch(hf + 'test-00000-of-00001.parquet', upstream / 'mmlu-test.parquet')).to_pylist()
    validation = pq.read_table(fetch(hf + 'validation-00000-of-00001.parquet', upstream / 'mmlu-validation.parquet')).to_pylist()
    initial = fetch('https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/' + PINS['mmlu_pro_code'] +
                    '/cot_prompt_lib/initial_prompt.txt', upstream / 'mmlu-initial-prompt.txt').read_text()
    selected = stratified(test)
    write(inputs / 'mmlu-pro-280.json', selected)
    write(inputs / 'mmlu-validation.json', validation)
    human = jsonl(inputs / 'humaneval.jsonl')
    instructions = jsonl(inputs / 'ifeval.jsonl')
    assert len(human) == 164 and len(instructions) == 541 and len(selected) == 280
    prompts = []
    for row in human:
        prompt = ('Please provide a self-contained Python script that solves the following problem in a markdown code block:'
                  + '\n```python\n' + row['prompt'].strip() + '\n```')
        prompts.append({'id': 'humaneval/' + row['task_id'], 'benchmark': 'humaneval_plus',
                        'task_id': row['task_id'], 'prompt': prompt})
    for row in instructions:
        prompts.append({'id': 'ifeval/' + str(row['key']), 'benchmark': 'ifeval', 'prompt': row['prompt']})
    for row in selected:
        fewshot = [r for r in validation if r['category'] == row['category']][:5]
        assert len(fewshot) == 5
        prompt = initial.replace('{$}', row['category'])
        prompt += ''.join(mmlu_question(r, answer=True) for r in fewshot) + mmlu_question(row)
        prompts.append({'id': 'mmlu_pro/' + str(row['question_id']), 'benchmark': 'mmlu_pro_280',
                        'category': row['category'], 'prompt': prompt})
    (inputs / 'prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prompts))
    # No test answers or executable tests are sent to the model.
    plan = {
        'id': SUITE, 'created_utc': datetime.now(timezone.utc).isoformat(), 'pins': PINS,
        'model': 'Qwen/Qwen3-VL-8B-Thinking',
        'base_revision': '92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b',
        'benchmarks': [
            {'id': 'humaneval_plus', 'name': 'HumanEval+', 'count': 164, 'metric': 'pass@1',
             'source': 'https://github.com/evalplus/evalplus', 'scope': 'Full v0.1.10 expanded test suite'},
            {'id': 'ifeval', 'name': 'IFEval', 'count': 541, 'metric': 'strict prompt accuracy',
             'source': 'https://github.com/google-research/google-research/tree/master/instruction_following_eval',
             'scope': 'Full set; also report strict instruction accuracy and loose prompt accuracy'},
            {'id': 'mmlu_pro_280', 'name': 'MMLU-Pro-280', 'count': 280, 'metric': 'accuracy',
             'source': 'https://github.com/TIGER-AI-Lab/MMLU-Pro',
             'scope': 'Fixed 20 questions per subject, five validation examples; NOT the full official score'}],
        'generation': {'max_tokens': 8192, 'temperature': 1.0, 'top_p': 0.95,
                       'top_k': 20, 'repetition_penalty': 1.0, 'seed': 20261026, 'n': 1},
        'prompting': 'Same base tokenizer and native thinking chat template for every checkpoint; no EFE system prompt or tools. Grade final answer only.',
        'selection': {'seed': SUITE, 'mmlu_categories': dict(Counter(r['category'] for r in selected)),
                      'test_ids_sha256': hashlib.sha256(json.dumps([r['id'] for r in prompts]).encode()).hexdigest()},
        'comparison': 'Report per-benchmark changes vs untouched base and immediate parent, paired question bootstrap 95% intervals, correct-to-wrong and wrong-to-correct counts. One inference seed is a screen, not a training replicate.',
        'truncation': 'All checkpoints use 8192 output tokens. Report truncated/empty finals separately; any longer-budget diagnostic reruns the same items on every compared checkpoint and retains original results.',
        'exclusion': 'Evaluation-only: no training admission, no benchmark-driven checkpoint selection, no retrying wrong answers. HTTP failures invalidate completeness; valid wrong/truncated answers remain in denominator.',
        'stages': ['qwen-base', 'qwen-warmstart', 'qwen-leact'],
        'stage_note': 'Warmstart is reasoning SFT in this EFE recipe. Additional distinct SFT/BC checkpoints may be registered as separate branches; identical weight hashes are aliases, not independent runs.',
        'compute': 'One bounded spare H100 node, within six-node total cap. Main EFE training takes precedence; incomplete generations can resume without replacing valid answers.',
    }
    plan['file_sha256'] = {str(p.relative_to(DEST)): sha(p) for folder in [inputs, vendor]
                           for p in sorted(folder.rglob('*')) if p.is_file()}
    write(DEST / 'suite.json', plan)
    write(DEST / 'checkpoints.json', {'checkpoints': [
        {'id': 'qwen-base', 'name': 'Untouched Qwen', 'parent': None, 'status': 'ready',
         'path': '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-v1/base-model',
         'manifest': '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-v1/checkpoint-manifest.json'},
        {'id': 'qwen-warmstart', 'name': 'Astra warmstart / reasoning SFT', 'parent': 'qwen-base',
         'status': 'waiting_for_checkpoint'},
        {'id': 'qwen-leact', 'name': 'Warmstart + LeAct', 'parent': 'qwen-warmstart',
         'status': 'waiting_for_checkpoint'}]})
    return plan


def dashboard():
    plan = read(DEST / 'suite.json')
    if not plan:
        return None
    rows = []
    registry = read(DEST / 'checkpoints.json')
    for stage in registry['checkpoints']:
        output = DEST / 'runs' / stage['id']
        family=registry.get('families',{}).get(stage.get('family'),{})
        rows.append({**stage, 'base_id':family.get('base_checkpoint','qwen-base'),
                     'progress': read(output / 'progress.json', {}),
                     'scheduler': read(output / 'scheduler.json', {}),
                     'result': read(output / 'results.json'),
                     'comparison': read(output / 'comparisons.json', {})})
    policy=read(DEST/'execution-policy.json',{})
    interpretation = ([] if policy.get('enabled',True) else
        ['Paused by user: prioritize native EFE robot evaluation and warmstart continuation. Historical screen results remain available; no new capability jobs will launch.'])
    base_ids = {'qwen-base', *(f['base_checkpoint'] for f in registry.get('families', {}).values())}
    if not any((r.get('result') or {}).get('complete') for r in rows if r['id'] not in base_ids):
        interpretation.append('No trained checkpoint has completed this screen yet, so there is no measured training regression or improvement.')
    for base in (r for r in rows if r['id'] in base_ids and (r.get('result') or {}).get('complete')):
        interpretation.append(base['name'] + ':')
        for benchmark in plan['benchmarks']:
            value = base['result']['benchmarks'].get(benchmark['id'], {})
            if value.get('truncated', 0):
                interpretation.append(f"{benchmark['name']}: {value['truncated']}/{value['count']} base answers reached the 8192-token limit; {value['empty_final']} had no final answer. These remain in the denominator.")
    for stage in rows:
        result=stage.get('result') or {}
        if stage['id'] in base_ids or not result.get('complete'):continue
        measures=list(result.get('benchmarks',{}).values())
        count=sum(r.get('count',0) for r in measures);empty=sum(r.get('empty_final',0) for r in measures)
        if count and empty==count:
            interpretation.append(f"{stage['name']} returned no final answer on all {count} questions, producing zero scores. This establishes a generation failure under this protocol; it does not isolate the cause or show that all underlying knowledge was lost.")
        if stage['id']=='qwen-warmstart' and stage['comparison'].get('qwen-base'):
            changes=stage['comparison']['qwen-base']
            measured='; '.join(f"{benchmark['name']} {result['benchmarks'][benchmark['id']]['score']*100:.1f}% ({changes[benchmark['id']]['delta_percentage_points']:+.1f} percentage points versus base)" for benchmark in plan['benchmarks'])
            interpretation.append('EFE warmstart: '+measured+'.')
            if all(value['paired_bootstrap_95_ci_pp'][0]<=0<=value['paired_bootstrap_95_ci_pp'][1] for value in changes.values()):
                interpretation.append('This single-seed screen shows no clear general-capability drop after EFE warmstart; all paired question intervals include zero. EFE task success and LeAct retention are separate measurements.')
        if stage['id']=='bb-warmstart-e3-v2' and stage['comparison'].get('text-qwen-base-v2'):
            scores=result['benchmarks']
            interpretation.append(f"The saved BuilderBench text-Qwen warmstart shows a severe measured regression: coding {scores['humaneval_plus']['score']*100:.1f}%, instruction following {scores['ifeval']['score']*100:.1f}%, and MMLU-Pro subset {scores['mmlu_pro_280']['score']*100:.1f}%. This is a different model family and training recipe from EFE. The scores establish degraded performance under this protocol; the cause is not yet isolated.")
    interpretation.append('Scores measure performance within this thinking budget. If a checkpoint changes reasoning length, compare a matched longer-budget diagnostic before attributing the change to lost knowledge or coding skill.')
    if registry.get('families'):
        interpretation.append('BuilderBench and Leduc use text Qwen3-8B; EFE uses Qwen3-VL-8B-Thinking. Each family is compared only with its own untouched base. Reusing saved weights here does not resume stopped BuilderBench simulator experiments.')
    return {'id': plan['id'], 'benchmarks': plan['benchmarks'], 'stages': rows,
            'execution_policy':policy,
            'description': 'Does task training preserve general coding, instruction following and reasoning? Same 985 prompts and 8192-token thinking budget at every stage, with matched settings within each model family. Blank cells are unmeasured.',
            'interpretation': ' '.join(interpretation),
            'stage_note': plan['stage_note'], 'comparison_note': plan['comparison']}


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    value = prepare()
    print(json.dumps({'suite': value['id'], 'questions': sum(b['count'] for b in value['benchmarks']),
                      'path': str(DEST / 'suite.json')}))
