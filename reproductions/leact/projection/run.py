"""Matched frontier-corpus ablation: projection sampling, joint SFT, two LeAct rounds."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from ..coldstart.collect_full import nested_states
from ..coldstart.run import export_corpus, train_and_evaluate
from ..compare import write_comparison
from ..data import file_sha256, fingerprint, load_oracle, read_jsonl, write_json
from ..experiment import MODEL_REVISION, parallel_generate
from ..pipeline import load_shards
from ..reproduction import phase
from ..training import prepare_data
from .sampler import SamplingConfig


def generate(args, base, limit=None):
    directory = args.root / 'projection_generation'
    directory.mkdir(parents=True, exist_ok=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(8)))).split(',')
    assert len(visible) == 8
    workers, logs = [], []
    try:
        for shard in range(8):
            command = [sys.executable, '-m', 'reproductions.leact.projection.worker',
                       '--oracle', str(args.oracle), '--corpus', str(args.corpus), '--checkpoint', str(base),
                       '--output', str(directory / f'shard-{shard:02d}.jsonl'), '--shard', str(shard)]
            if limit is not None:
                command += ['--limit', str(limit)]
            log = (directory / f'gpu-{shard:02d}.log').open('a')
            logs.append(log)
            workers.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': visible[shard], 'OMP_NUM_THREADS': '2'},
                           start_new_session=True))
        while any(worker.poll() is None for worker in workers):
            failed = [i for i, worker in enumerate(workers) if worker.poll() not in (None, 0)]
            if failed:
                for i in failed:
                    print((directory / f'gpu-{i:02d}.log').read_text()[-10000:], flush=True)
                raise RuntimeError('Projection workers failed: ' + str(failed))
            time.sleep(2)
        if any(worker.returncode for worker in workers):
            raise RuntimeError('Projection worker exited unsuccessfully')
    finally:
        for worker in workers:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
        for log in logs:
            log.close()
    rows = [r for path in sorted(directory.glob('shard-*.jsonl')) for r in read_jsonl(path)]
    if len({r['state_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate projection states across shards')
    return rows


def projection_metrics(rows):
    return dict(states=len(rows), rewrites_used=sum(r['projection']['used_rewrite'] for r in rows),
                original_fallbacks=sum(not r['projection']['used_rewrite'] for r in rows),
                proposals=sum(r['projection']['attempts'] for r in rows),
                accepted_proposals=sum(r['projection']['accepted'] for r in rows),
                mean_original_logprob=statistics.mean(r['projection']['original_mean_logprob'] for r in rows),
                mean_projected_logprob=statistics.mean(r['projection']['mean_logprob'] for r in rows),
                fallback_reasons={reason: sum(reason in r['projection']['rejection_reasons'] for r in rows)
                                  for reason in sorted({why for r in rows for why in r['projection']['rejection_reasons']})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'oracle', 'corpus', 'saved-experiment', 'scratch']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    states, metadata = load_oracle(args.oracle)
    corpus = read_jsonl(args.corpus)
    assert len(states) == len(corpus) == 4032
    assert fingerprint(corpus) == '170aa95acc3093369eec50f716ce4f2a8d6c3c9c67109cc38d109704fa0f774f'
    base = Path(json.loads((args.saved_experiment / 'base_model.json').read_text())['path'])
    assert MODEL_REVISION in str(base) and list(base.glob('*.safetensors'))
    args.root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    protocol = dict(paper='https://arxiv.org/abs/2610.02140', base_model=str(base),
                    implementation_sha256=fingerprint({name: Path(__file__).with_name(name).read_text()
                        for name in ['sampler.py', 'backend.py', 'worker.py', 'run.py']}),
                    initialization='Same untouched Qwen3-8B as the original frontier warmstart',
                    source_corpus_sha256=fingerprint(corpus), oracle_sha256=metadata['sha256'],
                    sampling=asdict(SamplingConfig()), proposal_temperature=0.6, repetition_penalty=1.05,
                    target_score='Mean raw token log likelihood of candidate explanation plus fixed CFR answer under ordinary task prompt; no privileged prompt context',
                    acceptance='Greedy Appendix C.2 / released-code approximation; not exact Metropolis-Hastings',
                    proposal_format='v3 Leduc adaptation: non-thinking explanation-only proposals; wrapper preserves thinking delimiters and exact original CFR answer',
                    terminal_eos='Proposal EOS ends explanation; scored full completion has EOS after the fixed action target; both token sequences are recorded',
                    expert_preservation='Generated policy must match CFR within 1e-6; paired examples must fit 2048 tokens',
                    invalid_rewrite='Use the original Astra rationale to preserve exactly the same 4032 states and 8064 examples',
                    semantic_preservation='Prompted preservation of card facts and strategy; policy agreement is checked, prose equivalence is not proven',
                    coldstart_epochs=3, coldstart_lr=1e-4, effective_batch=32, cutoff_len=2048,
                    evaluation_seeds=[0, 1, 2], n=8, m=4, k=2, rounds=2, planned_methods=['leact'],
                    generation_tokens=4096, prompt_variant='concise_v1',
                    pilot=dict(states=32, minimum_usable_rewrites=16),
                    adaptation='Sampler follows the new paper; model, SFT budget, joint targets, and Leduc evaluation match our existing LeAct reconstruction')
    path = args.root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Projection study protocol changed')
    write_json(path, protocol)
    report_path = args.root / 'results.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        scope='Leduc 6r2s independent reproduction with projection-sampled frontier warmstart',
        protocol=protocol, oracle=metadata, coldstarts={}, methods={'leact': []})
    if 'projection' not in report:
        phase(args.root, 'projection_pilot', states=32)
        pilot = generate(args, base, limit=32)
        assert len(pilot) >= 32
        report['pilot'] = projection_metrics(pilot)
        write_json(report_path, report)
        if report['pilot']['rewrites_used'] < 16 or report['pilot']['accepted_proposals'] == 0:
            phase(args.root, 'projection_pilot_needs_inspection', metrics=report['pilot'])
            raise RuntimeError('Projection pilot did not meet the predeclared usability check')
        phase(args.root, 'projection_sampling', states=4032, pilot=report['pilot'])
        raw = generate(args, base)
        assert len(raw) == len(states) and {r['state_id'] for r in raw} == {s['state_id'] for s in states}
        report['projection'] = projection_metrics(raw)
        write_json(report_path, report)
    else:
        raw = [r for path in sorted((args.root / 'projection_generation').glob('shard-*.jsonl')) for r in read_jsonl(path)]
    candidates = {r['state_id']: [dict(reasoning=r['projection']['selected_reasoning'],
                                      finish_reason='stop', fallback=None)] for r in raw}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False, local_files_only=True)
    label = 'projection_4032_n1'
    training = args.root / label / 'sft'
    manifest = export_corpus(nested_states(states), metadata, candidates, training, tokenizer,
                             retained_per_state=1, source='frontier_projection')
    assert manifest['sft_examples'] == 8064 and manifest['selected_traces'] == 4032
    if label not in report['coldstarts']:
        result = train_and_evaluate(args, label, training, base, states, metadata, lr=1e-4, all_epochs=True)
        report['coldstarts'][label] = result
        report['coldstart_thinking'], report['coldstart_thinking_all_epochs'] = result['best'], result['all_epochs']
        write_json(report_path, report)
    current = Path(report['coldstarts'][label]['selected_checkpoint'])
    for index in [1, 2]:
        previous = next((r for r in report['methods']['leact'] if r['round'] == index), None)
        if previous:
            current = Path(previous['selected_checkpoint'])
            continue
        directory = args.root / 'leact' / f'round-{index}'
        phase(args.root, 'leact_generation', round=index, checkpoint=str(current))
        paths = parallel_generate(args.oracle, current, directory / 'generation', gpus=8, method='leact',
                                  seed=index, n=8, m=4, prompt_variant='concise_v1')
        generated = load_shards(paths, metadata, method='leact')
        training = directory / 'sft'
        if not training.exists():
            prepare_data(states, metadata, training, method='leact', raw=generated, tokenizer=tokenizer,
                         prompt_variant='concise_v1', require_valid_policies=True)
        result = train_and_evaluate(args, f'leact-round-{index}', training, current, states, metadata,
                                    lr=5e-6, all_epochs=True)
        current = Path(result['selected_checkpoint'])
        report['methods']['leact'].append({'round': index, **result})
        write_json(report_path, report)
        write_comparison(report, args.root)
    phase(args.root, 'complete', role='projection_frontier')
    write_json(args.root / 'complete.json', {'results_sha256': file_sha256(report_path)})


if __name__ == '__main__':
    main()
