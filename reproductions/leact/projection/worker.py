"""One GPU's resumable projection chains, with original and proposed traces."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from ..coldstart.collect_full import nested_states
from ..core import parse_policy
from ..data import fingerprint, load_oracle, read_jsonl, write_json
from ..prompts import forward_prompt
from ..training import fits_context, supervised_pairs
from .backend import Backend, fixed_completion_text
from .sampler import SamplingConfig, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['oracle', 'corpus', 'checkpoint', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.batch_size < 1:
        raise ValueError('Invalid sharding or batch size')
    states, metadata = load_oracle(args.oracle)
    ordered = nested_states(states)
    corpus = read_jsonl(args.corpus)
    by_id = {r['state_id']: r for r in corpus}
    assert len(by_id) == len(corpus) == len(states) == 4032
    assert set(by_id) == {s['state_id'] for s in states}
    config = SamplingConfig()
    all_states = ordered[args.shard::args.shards]
    requested = ordered[:args.limit] if args.limit is not None else ordered
    selected_ids = {s['state_id'] for s in requested}
    selected = [s for s in all_states if s['state_id'] in selected_ids]
    manifest = dict(method='projection', oracle_sha256=metadata['sha256'], n=2, m=0, seed=config.seed,
                    prompt_variant='concise_v1', checkpoint=str(args.checkpoint),
                    state_ids=[s['state_id'] for s in all_states], shard=args.shard, shards=args.shards,
                    sampling=asdict(config), source_corpus_sha256=fingerprint(corpus),
                    acceptance='strictly higher mean token log likelihood; Appendix C.2 approximation',
                    source_revision='e74ab5eea6448bf78a4cff57179bbab8687cd243')
    manifest['proposal_format'] = 'v3: rewrite explanation only in non-thinking mode; deterministically preserve the original CFR Action/Policy suffix'
    manifest['score_representation'] = 'Full thinking-enabled completion: candidate explanation plus fixed expert answer, including terminal EOS'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix('.meta.json')
    if meta_path.exists() and json.loads(meta_path.read_text()) != manifest:
        raise ValueError('Projection resume protocol changed')
    write_json(meta_path, manifest)
    done = read_jsonl(args.output) if args.output.exists() else []
    if len({r['state_id'] for r in done}) != len(done):
        raise ValueError('Duplicate completed projection record')
    if any(r['state_id'] not in manifest['state_ids'] or r['seed'] != config.seed for r in done):
        raise ValueError('Completed projection record belongs to a different shard')
    seen = {r['state_id'] for r in done}
    selected = [s for s in selected if s['state_id'] not in seen]
    if not selected:
        return
    backend = Backend(args.checkpoint)
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        items, originals = [], []
        for state in batch:
            original = by_id[state['state_id']]
            pairs = supervised_pairs(state, original, metadata['rules'], prompt_variant='concise_v1')
            reference = pairs[0]['conversations'][1]['value']
            tokens = backend.tokenizer.encode(original['reasoning'], add_special_tokens=False)
            tokens.append(backend.tokenizer.eos_token_id)
            items.append(dict(state_id=state['state_id'], question=forward_prompt(state, metadata['rules'],
                              variant='concise_v1'), reference=reference, original_tokens=tokens))
            originals.append(original)
        original_scores = backend.score(items, [i['original_tokens'] for i in items])
        began = time.monotonic()
        sampled = project(backend, items, config)
        elapsed = time.monotonic() - began
        with args.output.open('a') as stream:
            for state, original, item, original_score, result in zip(batch, originals, items, original_scores, sampled):
                text = fixed_completion_text(backend.tokenizer, result['tokens'], backend.eos_ids, item['reference'])
                parsed = parse_policy(text, state['legal_actions'])
                reasoning = text[len('<think>\n'):].split('</think>', 1)[0].strip()
                reasons = []
                if not result['finished']:
                    reasons.append('generation reached the token cap')
                if text.count('</think>') != 1 or not reasoning or '<think>' in reasoning:
                    reasons.append('invalid reasoning delimiters')
                if 'Action:' in reasoning or 'Policy:' in reasoning:
                    reasons.append('explanation contains action output')
                if len(reasoning.split()) > 200:
                    reasons.append('reasoning exceeds the matched 200-word instruction')
                if parsed.fallback is not None:
                    reasons.append('invalid generated policy')
                elif max(abs(parsed.policy[a] - state['expert_policy'][a]) for a in state['legal_actions']) > 1e-6:
                    reasons.append('generated policy differs from the CFR target')
                candidate = dict(reasoning=reasoning, finish_reason='stop', fallback=None)
                pairs = supervised_pairs(state, candidate, metadata['rules'], prompt_variant='concise_v1')
                if not all(fits_context(pair, backend.tokenizer, 2048) for pair in pairs):
                    reasons.append('paired SFT example exceeds 2048 tokens')
                use_rewrite = not reasons
                row = dict(state_id=state['state_id'], seed=config.seed, baselines=[],
                           prompt_variant='concise_v1', method='projection',
                           candidates=[
                               dict(reasoning=original['reasoning'], forward=dict(text=item['reference'],
                                    finish_reason='stop', generated_tokens=len(item['original_tokens']))),
                               dict(reasoning=reasoning, forward=dict(text=text,
                                    finish_reason='stop' if result['finished'] else 'length',
                                    generated_tokens=len(result['tokens'])))],
                           projection={**result, 'original_mean_logprob': original_score,
                                       'mean_logprob_change': result['mean_logprob'] - original_score,
                                       'used_rewrite': use_rewrite, 'rejection_reasons': reasons,
                                       'selected_candidate': int(use_rewrite),
                                       'selected_reasoning': reasoning if use_rewrite else original['reasoning'],
                                       'original_reference': item['reference'], 'question': item['question'],
                                       'score_representation': manifest['score_representation'],
                                       'scored_tokens': backend.scored_tokens(item, result['tokens']),
                                       'original_scored_tokens': backend.scored_tokens(item, item['original_tokens']),
                                       'fixed_action_target': True,
                                       'batch_elapsed_seconds': elapsed, 'batch_examples': len(batch)})
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps(dict(state_id=state['state_id'], used_rewrite=use_rewrite,
                                       accepted=result['accepted'], attempts=result['attempts'],
                                       token_count=len(result['tokens']), reasons=reasons)), flush=True)


if __name__ == '__main__':
    main()
