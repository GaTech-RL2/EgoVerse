"""Generate backward candidates and apply the BuilderBench action-reward gap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data import file_sha256, fingerprint, write_json, write_jsonl
from ..training import fits_context
from .student import Backend, action_reward, decision_seed, split_output
from .student_data import CUTOFF, backward_prompt, export_dataset, forward_prompt, load_corpus, pairs


def parse_sample(sample, state, *, backward=False):
    try:
        if sample['finish_reason'] != 'stop':
            raise ValueError('Generation hit its token limit')
        reasoning, action = split_output(sample['text'], backward=backward, cubes=state['cubes'])
        return dict(reasoning=reasoning, action=action, valid=True, error=None)
    except (ValueError, TypeError, KeyError) as error:
        return dict(reasoning='', action=None, valid=False, error=str(error))


def generate(args):
    states, corpus = load_corpus(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = dict(checkpoint=str(args.checkpoint), corpus_sha256=corpus['states_sha256'], method=args.method,
                n=8, k=2, seed=args.seed, shard=args.shard, shards=args.shards,
                baseline='One decoded action with an empty reasoning block, Appendix D.2.1',
                reward='Action-type/cube/position overlap, positions in meters; invalid samples score zero')
    path = args.output / f'shard-{args.shard:02d}.jsonl'
    meta_path = path.with_suffix('.meta.json')
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError('Generation specification differs')
    write_json(meta_path, meta)
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    subset = states[args.shard::args.shards]
    if [r['state_id'] for r in existing] != [s['state_id'] for s in subset[:len(existing)]]:
        raise ValueError('Generation shard differs from its immutable corpus')
    backend = Backend(args.checkpoint) if len(existing) < len(subset) else None
    for state in subset[len(existing):]:
        record = dict(state_id=state['state_id'], task=state['task'], candidates=[], method=args.method,
                      corpus_sha256=corpus['states_sha256'])
        try:
            if args.method == 'leact':
                baselines, _ = backend.generate(forward_prompt(state), reasoning='',
                    seed=decision_seed(args.seed, state['state_id'], 0, 'baseline'))
                baseline = parse_sample(baselines[0], state)
                # Empty forced reasoning is deliberate and is accepted for the baseline.
                try:
                    if baselines[0]['finish_reason'] != 'stop':
                        raise ValueError('Truncated baseline')
                    _, action = split_output(baselines[0]['text'], thinking=False, cubes=state['cubes'])
                    baseline = dict(reasoning='', action=action, valid=True, error=None)
                except (ValueError, TypeError, KeyError) as error:
                    baseline = dict(reasoning='', action=None, valid=False, error=str(error))
                record['baseline'] = {**baseline, 'sample':baselines[0],
                                      'reward':action_reward(baseline['action'], state['expert_action'])}
                generated, _ = backend.generate(backward_prompt(state), n=8,
                    seed=decision_seed(args.seed, state['state_id'], 0, 'backward'))
                for index, raw in enumerate(generated):
                    candidate = dict(candidate_id=index, backward=raw, **parse_sample(raw, state, backward=True))
                    if candidate['valid']:
                        try:
                            forward, _ = backend.generate(forward_prompt(state), reasoning=candidate['reasoning'],
                                seed=decision_seed(args.seed, state['state_id'], index, 'conditioned'))
                            parsed = parse_sample(forward[0], state)
                            candidate.update(forward=forward[0], action=parsed['action'], valid=parsed['valid'], error=parsed['error'])
                        except OverflowError as error:
                            candidate.update(valid=False, error=str(error))
                    candidate['reward'] = action_reward(candidate['action'], state['expert_action'])
                    candidate['delta'] = candidate['reward'] - record['baseline']['reward']
                    record['candidates'].append(candidate)
            else:
                generated, _ = backend.generate(forward_prompt(state), n=8,
                    seed=decision_seed(args.seed, state['state_id'], 0, 'exit'))
                for index, raw in enumerate(generated):
                    candidate = dict(candidate_id=index, forward=raw, **parse_sample(raw, state))
                    candidate['reward'] = action_reward(candidate['action'], state['expert_action'])
                    record['candidates'].append(candidate)
        except OverflowError as error:
            record['context_error'] = str(error)
        with path.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(dict(method=args.method, state_id=state['state_id'], candidates=len(record['candidates']))), flush=True)
    write_json(path.with_suffix('.complete.json'), dict(sha256=file_sha256(path), states=len(subset)))


def select_and_export(corpus_path, generation, output, *, method, tokenizer):
    states, corpus = load_corpus(corpus_path)
    raw = []
    for path in sorted(Path(generation).glob('shard-*.jsonl')):
        complete = json.loads(path.with_suffix('.complete.json').read_text())
        if complete['sha256'] != file_sha256(path):
            raise ValueError('Unfinished or modified generation shard')
        raw.extend(json.loads(line) for line in path.read_text().splitlines())
    if len(raw) != len(states) or {r['state_id'] for r in raw} != {s['state_id'] for s in states}:
        raise ValueError('Generation must cover the frozen iterative corpus exactly')
    by_id = {s['state_id']:s for s in states}
    data, audit = [], []
    for row in sorted(raw, key=lambda r:r['state_id']):
        if row['method'] != method or row['corpus_sha256'] != corpus['states_sha256']:
            raise ValueError('Generation belongs to another method or corpus')
        state = by_id[row['state_id']]
        candidates = [c for c in row['candidates'] if c['valid'] and c['reasoning']]
        if method == 'leact':
            # Invalid baselines do not justify admission merely by receiving zero reward.
            candidates = [c for c in candidates if c['delta'] > 0] if row.get('baseline', {}).get('valid') else []
        candidates.sort(key=lambda c:(-c['delta'] if method=='leact' else -c['reward'], c['candidate_id']))
        admitted = []
        for candidate in candidates[:2]:
            examples = pairs(state, candidate['reasoning'], method=method,
                             action=state['expert_action'] if method=='leact' else candidate['action'])
            if all(fits_context(p, tokenizer, CUTOFF) for p in examples):
                data.extend(examples); admitted.append(candidate['candidate_id'])
        audit.append({**row, 'selected_ids':admitted, 'selected_traces':len(admitted)})
    if not data:
        write_json(Path(output) / 'no_admissions.json', dict(method=method, states=len(states),
                   reason='No valid selected traces fit the training context; no empty M-step',
                   generation_sha256=fingerprint(raw)))
        write_jsonl(Path(output) / 'selection.jsonl', audit)
        return None
    return export_dataset(output, data, method=method, corpus=corpus, selection=audit, tokenizer=tokenizer)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'corpus', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--method', choices=['leact','exit'], required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8)
    generate(p.parse_args())


if __name__ == '__main__':
    main()
