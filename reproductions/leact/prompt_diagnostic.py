"""Matched prompt experiment; generation and SFT admission, without weight updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .core import kl, mean_policy, parse_policy
from .data import file_sha256, fingerprint, load_oracle, write_json
from .experiment import MODEL_REVISION
from .pipeline import generate_shard, load_shards, score_record
from .prompts import PROMPT_VARIANTS
from .training import prepare_data, supervised_pairs


def analyze(states, metadata, raw, tokenizer, variant):
    by_id = {s['state_id']: s for s in states}
    audit, baseline_kls = [], []
    for record in raw:
        state = by_id[record['state_id']]
        scored = score_record(state, record, method='leact', k=2)
        selected = {c['candidate_id'] for c in scored['selected']}
        clean_baseline = (scored['baseline_truncations'] == 0
                          and all(f is None for f in scored['baseline_fallbacks']))
        baseline_kls.append(kl(state['expert_policy'], mean_policy([
            parse_policy(g['text'], state['legal_actions']).policy for g in record['baselines']])))
        for candidate, generation in zip(scored['candidates'], record['candidates']):
            lengths = []
            for pair in supervised_pairs(state, candidate, metadata['rules'], prompt_variant=variant):
                messages = [{'role': 'user' if c['from'] == 'human' else 'assistant', 'content': c['value']}
                            for c in pair['conversations']]
                lengths.append(len(tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=True)))
            audit.append({
                'state_id': state['state_id'], 'candidate_id': candidate['candidate_id'],
                'complete': candidate['complete'], 'nonempty_reasoning': bool(candidate['reasoning'].strip()),
                'backward_closed_thinking': '</think>' in generation['backward']['text'],
                'fallback': candidate['fallback'], 'score': candidate['score'],
                'conditioned_kl': kl(state['expert_policy'], candidate['predicted_policy']),
                'selected_before_length_filter': candidate['candidate_id'] in selected,
                'clean_baseline': clean_baseline,
                'backward_generated_tokens': generation['backward']['generated_tokens'],
                'reasoning_words': len(candidate['reasoning'].split()),
                'forward_sft_tokens': lengths[0], 'backward_sft_tokens': lengths[1],
                'both_pairs_fit_2048': max(lengths) <= 2048,
                'both_pairs_fit_4096': max(lengths) <= 4096,
            })
    selected = [c for c in audit if c['selected_before_length_filter']]
    admission = {}
    for cutoff in (2048, 4096):
        admitted = [c for c in selected if c[f'both_pairs_fit_{cutoff}']]
        admission[str(cutoff)] = {
            'selected_traces': len(admitted), 'sft_examples': 2*len(admitted),
            'states_with_admitted_traces': len({c['state_id'] for c in admitted}),
            'overlength_dropped': len(selected)-len(admitted),
            'selected_policy_fallbacks': sum(c['fallback'] is not None for c in admitted),
            'selected_traces_with_clean_baselines': sum(c['clean_baseline'] and c['fallback'] is None for c in admitted),
            'states_with_clean_baselines': len({c['state_id'] for c in admitted if c['clean_baseline'] and c['fallback'] is None}),
        }
    return {
        'prompt_variant': variant, 'states': len(raw), 'candidates': len(audit),
        'usable_candidates': sum(c['complete'] and c['nonempty_reasoning'] and c['fallback'] is None for c in audit),
        'positive_selected': len(selected),
        'positive_selected_with_clean_baselines': sum(c['clean_baseline'] and c['fallback'] is None for c in selected),
        'clean_baseline_states': len({c['state_id'] for c in audit if c['clean_baseline']}),
        'truncated_candidates': sum(not c['complete'] for c in audit),
        'empty_reasoning_candidates': sum(not c['nonempty_reasoning'] for c in audit),
        'forward_policy_fallbacks': sum(c['fallback'] is not None for c in audit),
        'baseline_truncations': sum(g['finish_reason'] != 'stop' for r in raw for g in r['baselines']),
        'baseline_policy_fallbacks': sum(parse_policy(g['text'], by_id[r['state_id']]['legal_actions']).fallback is not None
                                         for r in raw for g in r['baselines']),
        'mean_baseline_capped_kl': statistics.mean(baseline_kls),
        'mean_conditioned_capped_kl': statistics.mean(c['conditioned_kl'] for c in audit),
        'mean_backward_generated_tokens': statistics.mean(c['backward_generated_tokens'] for c in audit),
        'reasoning_within_200_words': sum(c['nonempty_reasoning'] and c['reasoning_words'] <= 200 for c in audit),
        'admission': admission, 'audit': audit,
    }


def worker(args, states, metadata):
    from .inference import VLLMBackend
    backend = VLLMBackend(str(args.checkpoint), max_tokens=4096)
    for variant in PROMPT_VARIANTS:
        output = args.root/args.model_label/variant/'generation'/f'shard-{args.shard:02d}.jsonl'
        generate_shard(states, metadata, backend, output, checkpoint=str(args.checkpoint),
                       method='leact', n=2, m=2, seed=0, shard=args.shard, shards=args.gpus,
                       batch_size=4, max_tokens=4096, prompt_variant=variant)


def parallel_workers(args, model_label, checkpoint):
    directory = args.root/model_label
    directory.mkdir(parents=True, exist_ok=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(args.gpus)))).split(',')
    if len(visible) < args.gpus:
        raise RuntimeError('Not enough visible GPUs')
    processes, streams = [], []
    try:
        for shard in range(args.gpus):
            command = [sys.executable, '-m', 'reproductions.leact.prompt_diagnostic', '--worker',
                       '--root', str(args.root), '--oracle', str(args.oracle), '--checkpoint', str(checkpoint),
                       '--model-label', model_label, '--shard', str(shard), '--gpus', str(args.gpus)]
            log = directory/f'gpu-{shard:02d}.log'
            stream = log.open('a')
            streams.append(stream)
            write_json(log.with_suffix('.command.json'), command)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[shard], OMP_NUM_THREADS='2',
                       VLLM_DISABLE_COMPILE_CACHE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True))
        while any(p.poll() is None for p in processes):
            failed = [i for i, p in enumerate(processes) if p.poll() not in (None, 0)]
            if failed:
                for i in failed:
                    print((directory/f'gpu-{i:02d}.log').read_text()[-6000:], flush=True)
                raise RuntimeError(f'Prompt diagnostic workers failed: {failed}')
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError('Prompt diagnostic generation failed')
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()


def run(args, states, metadata):
    from transformers import AutoTokenizer
    original = args.saved_experiment
    base = Path(json.loads((original/'base_model.json').read_text())['path'])
    models = {'base': base, 'masked-bc': args.bc_checkpoint}
    for label, model in models.items():
        if not (model/'config.json').is_file() or not list(model.glob('*.safetensors')):
            raise ValueError(f'Incomplete {label} checkpoint: {model}')
    if MODEL_REVISION not in str(base):
        raise ValueError('Base model path does not contain the pinned revision')
    package = Path(__file__).parent
    protocol = {
        'scope': '16-state matched prompt diagnostic; no weight updates or full-tree evaluation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': fingerprint({p.name: p.read_text() for p in package.glob('*.py')}),
        'oracle_sha256': metadata['sha256'], 'models': {k: str(v) for k, v in models.items()},
        'model_revision': MODEL_REVISION, 'state_ids': [s['state_id'] for s in states],
        'variants': list(PROMPT_VARIANTS), 'seed': 0, 'n': 2, 'm': 2, 'k': 2,
        'max_generated_tokens': 4096, 'sft_cutoff_primary': 2048, 'sft_cutoff_fallback': 4096,
        'temperature': 1.0, 'top_p': 0.95, 'top_k': -1, 'repetition_penalty': 1.0,
        'fallback_rule': 'Export 4096-token data only if the concise arm has zero admitted traces with clean baselines at 2048 and has some at 4096.',
        'valid_baseline_audit': 'Report original selection unchanged, plus a stricter diagnostic requiring both baseline rollouts to terminate and parse, and no selected policy fallback. Export only clean-baseline states.',
        'success_scope': 'Positive complete candidates admitted to paired SFT; this does not establish policy improvement after training.',
        'training_launched': False,
    }
    if (args.root/'protocol.json').exists():
        raise ValueError('Use a fresh diagnostic root')
    write_json(args.root/'protocol.json', protocol)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    results = {'protocol': protocol, 'arms': {}}
    for label, checkpoint in models.items():
        print(json.dumps({'stage': 'generation', 'model': label, 'checkpoint': str(checkpoint)}), flush=True)
        # Warm shared-storage reads once before eight independent model loads.
        started = time.monotonic()
        for path in sorted(checkpoint.glob('*.safetensors')):
            with path.open('rb') as stream:
                while stream.read(16*1024*1024):
                    pass
        print(json.dumps({'model': label, 'checkpoint_preread_seconds': time.monotonic()-started}), flush=True)
        parallel_workers(args, label, checkpoint)
        for variant in PROMPT_VARIANTS:
            directory = args.root/label/variant
            paths = [directory/'generation'/f'shard-{i:02d}.jsonl' for i in range(args.gpus)]
            raw = load_shards(paths, metadata, method='leact')
            result = analyze(states, metadata, raw, tokenizer, variant)
            result['input_sha256'] = {p.name: file_sha256(p) for p in paths}
            result['checkpoint'] = str(checkpoint)
            cutoff = 2048 if result['admission']['2048']['selected_traces_with_clean_baselines'] else None
            if variant == 'concise_v1' and cutoff is None and result['admission']['4096']['selected_traces_with_clean_baselines']:
                cutoff = 4096
            if cutoff:
                clean_ids = {c['state_id'] for c in result['audit'] if c['clean_baseline']}
                clean_raw = [r for r in raw if r['state_id'] in clean_ids]
                clean_states = [s for s in states if s['state_id'] in clean_ids]
                # Do not export fallback-selected examples as validated reasoning.
                if any(c['selected_before_length_filter'] and c['fallback'] is not None
                       and c['clean_baseline'] for c in result['audit']):
                    result['dataset_export_skipped'] = 'Selected policy fallback needs inspection; all diagnostics retained.'
                else:
                    prepared = prepare_data(clean_states, metadata, directory/f'sft-{cutoff}', method='leact', raw=clean_raw,
                                            tokenizer=tokenizer, cutoff=cutoff, prompt_variant=variant)
                    if prepared['selected_traces'] != result['admission'][str(cutoff)]['selected_traces_with_clean_baselines']:
                        raise AssertionError('SFT export disagrees with admission audit')
                    result['prepared_dataset'] = prepared
            result['training_launched'] = False
            write_json(directory/'result.json', result)
            summary = {k: v for k, v in result.items() if k not in {'audit', 'input_sha256'}}
            results['arms'][label+'/'+variant] = summary
            write_json(args.root/'result.json', results)
            print('LEACT_PROMPT_RESULT '+json.dumps({'arm': label+'/'+variant, **summary}), flush=True)
    write_json(args.root/'complete.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                         'result_sha256': file_sha256(args.root/'result.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--saved-experiment', type=Path)
    parser.add_argument('--bc-checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-label', choices=['base', 'masked-bc'])
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    states, metadata = load_oracle(args.oracle)
    states = states[:16]
    if args.worker:
        worker(args, states, metadata)
    else:
        if args.saved_experiment is None or args.bc_checkpoint is None:
            parser.error('Parent driver requires saved experiment and BC checkpoint')
        run(args, states, metadata)


if __name__ == '__main__':
    main()
