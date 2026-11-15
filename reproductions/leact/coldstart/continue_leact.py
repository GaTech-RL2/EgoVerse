"""Continue a completed self-generated warmstart with two matched LeAct rounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..compare import write_comparison
from ..data import file_sha256, fingerprint, load_oracle, write_json
from ..experiment import parallel_generate
from ..pipeline import load_shards
from ..reproduction import phase
from ..training import prepare_data
from .run import train_and_evaluate


def verify_anchor(source_root, registration, oracle_sha256):
    """Require the registered completed warmstart and its retained weight bytes."""
    report_path = source_root / 'results.json'
    if file_sha256(report_path) != registration['source_results_sha256']:
        raise ValueError('Source warmstart report changed')
    completion = json.loads((source_root / 'complete.json').read_text())
    if completion['results_sha256'] != registration['source_results_sha256']:
        raise ValueError('Source warmstart is not completely published')
    report = json.loads(report_path.read_text())
    if report['protocol']['role'] != 'self' or report['oracle']['sha256'] != oracle_sha256:
        raise ValueError('Expected self-generated warmstart with the same CFR oracle')
    arm = registration['source_arm']
    if arm != 'self_4032_n4':
        raise ValueError('This registered continuation uses the largest self-generated warmstart')
    anchor = report['coldstarts'][arm]
    if anchor['checkpoint_rule'] != 'fixed epoch 3':
        raise ValueError('Source checkpoint must be the fixed final epoch')
    training = source_root / arm / 'sft'
    manifest_path = training / 'manifest.json'
    if file_sha256(manifest_path) != registration['source_manifest_sha256']:
        raise ValueError('Source warmstart manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['data_sha256'] != registration['source_data_sha256']
            or file_sha256(training / 'data.json') != manifest['data_sha256']):
        raise ValueError('Source warmstart data changed')
    checkpoint = Path(anchor['selected_checkpoint'])
    if checkpoint != training / 'checkpoints' or checkpoint.is_symlink():
        raise ValueError('Expected the persisted final self-generated checkpoint')
    retention = json.loads((checkpoint / 'weight_retention.json').read_text())
    if retention['retained_checkpoint'] != str(checkpoint) or not retention['weight_sha256']:
        raise ValueError('Source checkpoint retention record is incomplete')
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    expected = set(index['weight_map'].values())
    if set(retention['weight_sha256']) != expected:
        raise ValueError('Retained weight list differs from model shard index')
    for name, sha in retention['weight_sha256'].items():
        if Path(name).name != name or file_sha256(checkpoint / name) != sha:
            raise ValueError('Source checkpoint weight checksum mismatch')
    return report, anchor, retention


def continue_rounds(args, report, current, states, metadata, tokenizer):
    for index in [1, 2]:
        previous = next((r for r in report['methods']['leact'] if r['round'] == index), None)
        if previous:
            # The oracle-BoN-selected checkpoint may differ from the final epoch.
            current = Path(previous['selected_checkpoint'])
            continue
        directory = args.root / 'leact' / f'round-{index}'
        phase(args.root, 'leact_generation', round=index, checkpoint=str(current))
        paths = parallel_generate(args.oracle, current, directory / 'generation', gpus=8,
                                  method='leact', seed=index, n=8, m=4, prompt_variant='concise_v1')
        generated = load_shards(paths, metadata, method='leact')
        training = directory / 'sft'
        if not training.exists():
            prepare_data(states, metadata, training, method='leact', raw=generated,
                         tokenizer=tokenizer, prompt_variant='concise_v1', require_valid_policies=True)
        result = train_and_evaluate(args, f'leact-round-{index}', training, current,
                                    states, metadata, lr=5e-6, all_epochs=True)
        current = Path(result['selected_checkpoint'])
        report['methods']['leact'].append(dict(round=index, **result))
        write_json(args.root / 'results.json', report)
        write_comparison(report, args.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'source-root', 'oracle', 'scratch', 'registration']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve() == args.source_root.resolve():
        raise ValueError('Continuation must not overwrite the warmstart study')
    registration = json.loads(args.registration.read_text())
    states, metadata = load_oracle(args.oracle)
    if len(states) != 4032 or not metadata['converged']:
        raise ValueError('The full converged Leduc 6r2s oracle is required')
    source, anchor, retention = verify_anchor(args.source_root, registration, metadata['sha256'])
    checkpoint = Path(anchor['selected_checkpoint'])
    code_root = Path(__file__).resolve().parents[1]
    protocol = dict(role='self_warmstart_leact', initialization=str(checkpoint),
        source_registration=registration, oracle_sha256=metadata['sha256'],
        implementation_sha256=fingerprint({name:file_sha256(code_root / name) for name in [
            'coldstart/continue_leact.py', 'coldstart/run.py', 'reproduction.py', 'experiment.py',
            'training.py', 'pipeline.py', 'core.py']}),
        rounds=2, planned_methods=['leact'], epochs=3, learning_rate=5e-6,
        effective_batch=32, cutoff_len=2048, generation_tokens=4096,
        n=8, m=4, k=2, prompt_variant='concise_v1', generation_seeds=[1, 2],
        evaluation_seeds=[0, 1, 2], evaluate_all_epochs=True,
        checkpoint_rule='lowest oracle-BoN mean across epochs; report fixed final epochs separately',
        matched_round_settings='Same LeAct round settings and CFR oracle as frontier and projection controls',
        causal_limits=['Warmstart was chosen after observing the completed scaling results.',
                      'Self warmstart uses more examples and compute than frontier/projection warmstarts.',
                      'Selected trace counts and training steps can differ between arms after filtering.',
                      'Seen-state evaluation and one trained model per setting; no generalization claim.'])
    args.root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    protocol_path = args.root / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Continuation protocol changed; choose a new study root')
    write_json(protocol_path, protocol)
    write_json(args.root / 'anchor-ready.json', dict(checkpoint=str(checkpoint),
        source_results_sha256=registration['source_results_sha256'],
        verified_weights=retention['weight_sha256'], measurement=anchor['final_epoch']))
    report_path = args.root / 'results.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        scope='Leduc 6r2s independent reproduction', arm='self-generated warmstart plus LeAct',
        protocol=protocol, oracle=metadata, coldstart_thinking=anchor['final_epoch'],
        methods={'leact': []})
    if report['protocol'] != protocol:
        raise ValueError('Continuation report belongs to another protocol')
    write_json(report_path, report)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True)
    continue_rounds(args, report, checkpoint, states, metadata, tokenizer)
    phase(args.root, 'complete', role='self_warmstart_leact')
    write_json(args.root / 'complete.json', dict(results_sha256=file_sha256(report_path)))


if __name__ == '__main__':
    main()
