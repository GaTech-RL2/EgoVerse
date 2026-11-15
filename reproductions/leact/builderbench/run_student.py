"""Durable train/evaluate queues for the authorized two-node BuilderBench study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..data import file_sha256, fingerprint, write_json
from ..experiment import epoch_checkpoints, run_logged, verify_training_update
from ..training import write_recipe
from .evaluate_student import summarize
from .iterate import select_and_export
from .student_data import export_warmstart, load_corpus
from .speed import summarize as summarize_speed


def phase(root, role, name, **details):
    value = dict(phase=name, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    write_json(root / f'{role}-progress.json', value)
    print(json.dumps(value), flush=True)


def weights(path):
    files = sorted(Path(path).glob('*.safetensors'))
    if not files:
        raise ValueError('Missing checkpoint weights')
    return {p.name:file_sha256(p) for p in files}


def publish_stage(root, identifier, checkpoint, *, method, label, final, parent=None):
    path = root / 'ready' / f'{identifier}.json'
    spec = dict(id=identifier, checkpoint=str(checkpoint), method=method, label=label,
                final_epoch=final, parent_stage=parent, planned_seeds=list(range(64 if final else 3)),
                checkpoint_sha256=weights(checkpoint), selection='Fixed epoch; no selection on held-out results')
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Published checkpoint identity changed')
    write_json(path, spec)
    return spec


def train_stage(root, identifier, training, base, *, method, lr, parent=None):
    training.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((training / 'manifest.json').read_text())
    spec = dict(initial_checkpoint=str(base), method=method, epochs=3, learning_rate=lr,
                gpus=8, microbatch=1, accumulation=4, effective_batch=32,
                data_sha256=manifest['data_sha256'])
    complete = training / 'training_complete.json'
    models = training / 'checkpoints'
    if complete.exists():
        if json.loads(complete.read_text()) != spec:
            raise ValueError('Completed SFT parameters changed')
    else:
        if models.exists() and any(models.iterdir()):
            raise RuntimeError('Interrupted SFT requires an explicit recovery; no silent model-only optimizer resume')
        phase(root, 'train', 'training', stage=identifier, examples=manifest['sft_examples'], **spec)
        recipe = write_recipe(training, str(base), gpus=8, epochs=3, learning_rate=lr)
        import yaml
        config = yaml.safe_load(recipe.read_text())
        config.update(per_device_train_batch_size=1, gradient_accumulation_steps=4,
                      preprocessing_num_workers=2, dataloader_num_workers=1)
        recipe.write_text(yaml.safe_dump(config, sort_keys=False))
        environment = dict(os.environ, FORCE_TORCHRUN='1', NPROC_PER_NODE='8', NNODES='1', NODE_RANK='0',
            MASTER_ADDR='127.0.0.1', MASTER_PORT='29631', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4', WANDB_DISABLED='true')
        run_logged(['llamafactory-cli', 'train', str(recipe)], training / 'train.log', environment)
        expected = math.ceil(math.ceil(manifest['sft_examples'] / 8) / 4) * 3
        verification = verify_training_update(base, models, expected)
        write_json(training / 'training_verification.json', verification)
        write_json(complete, spec)
    checkpoints = epoch_checkpoints(models, expected_epochs=3, require_weights=True)
    checkpoints = sorted(checkpoints, key=lambda p:int(p.name.split('-')[-1]))
    for epoch, checkpoint in enumerate(checkpoints, 1):
        publish_stage(root, f'{identifier}-e{epoch}', checkpoint, method=method,
                      label=f'{identifier} · epoch {epoch}', final=epoch == 3, parent=parent)
    return checkpoints[-1]


def workers(module, arguments, directory):
    directory.mkdir(parents=True, exist_ok=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(8)))).split(',')
    if len(visible) != 8:
        raise RuntimeError('This study requires one eight-H100 node per role')
    children, streams = [], []
    try:
        for index in range(8):
            command = [sys.executable, '-u', '-m', module, *arguments, '--shard', str(index), '--shards', '8']
            stream = (directory / f'gpu-{index}.log').open('a'); streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[index], OMP_NUM_THREADS='2',
                       TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                       VLLM_PORT=str(20000 + index * 1000))
            children.append(subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError(f'BuilderBench worker failed; inspect {directory}')
            time.sleep(2)
        if any(c.returncode for c in children):
            raise RuntimeError(f'BuilderBench worker failed; inspect {directory}')
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for stream in streams:
            stream.close()


def train(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    export_warmstart(args.corpus, args.root / 'training', tokenizer)
    publish_stage(args.root, 'base', args.base, method='base', label='Qwen3-8B · base', final=True)
    anchor = train_stage(args.root, 'joint', args.root / 'training/joint/sft', args.base,
                         method='joint', lr=1e-4, parent='base')
    write_json(args.root / 'anchor-ready.json', dict(checkpoint=str(anchor), stage='joint-e3'))
    train_stage(args.root, 'bc', args.root / 'training/bc/sft', args.base,
                method='bc', lr=1e-4, parent='base')
    iteration = args.root / 'iteration-corpus'
    deadline = time.monotonic() + 6*3600
    while not (iteration / 'ready.json').exists():
        phase(args.root, 'train', 'waiting_for_iteration_corpus', dependency='New GPT-6 trajectories on fixed in-domain tasks')
        if time.monotonic() > deadline:
            raise TimeoutError('Iteration corpus did not arrive within six hours')
        time.sleep(20)
    states, corpus = load_corpus(iteration)
    _, initial = load_corpus(args.corpus)
    if corpus['partition'] != initial['partition'] or corpus['role'] != 'iteration':
        raise ValueError('Iterative data changed the task split')
    for method in ['leact', 'exit']:
        directory = args.root / 'training' / f'{method}-r1'
        phase(args.root, 'train', 'generation', stage=f'{method}-r1', checkpoint=str(anchor), states=len(states))
        workers('reproductions.leact.builderbench.iterate', ['--checkpoint',str(anchor), '--corpus',str(iteration),
            '--output',str(directory/'generation'),'--method',method,'--seed','1'], directory/'generation-logs')
        manifest = select_and_export(iteration, directory/'generation', directory/'sft', method=method, tokenizer=tokenizer)
        if manifest is not None:
            train_stage(args.root, f'{method}-r1', directory/'sft', anchor,
                        method=method, lr=1e-5, parent='joint-e3')
        else:
            phase(args.root, 'train', 'no_admissions', method=method, states=len(states))
    write_json(args.root / 'train-complete.json', dict(stages=sorted(p.stem for p in (args.root/'ready').glob('*.json')),
               next='Complete all queued evaluations; compare first-round outcomes before subsequent rounds'))
    phase(args.root, 'train', 'complete')


def evaluate(args):
    # Progressively evaluate each new checkpoint before extending older cohorts.
    # The immutable manifest always declares the complete seed set; partial
    # summaries explicitly name their smaller denominator.
    while True:
        available = [json.loads(p.read_text()) for p in sorted((args.root/'ready').glob('*.json'))]
        queue = []
        for stage in available:
            directory = args.root / 'evaluations' / stage['id']
            completed = set()
            for seed in stage['planned_seeds']:
                if (directory / f'seed-{seed}-complete.json').exists():
                    completed.add(seed)
            remaining = set(stage['planned_seeds']) - completed
            if remaining:
                queue.append((len(completed), 0 if stage['final_epoch'] else 1, stage['id'], stage, min(remaining)))
        if not queue:
            if (args.root/'train-complete.json').exists():
                break
            phase(args.root, 'evaluate', 'waiting_for_checkpoint')
            time.sleep(20); continue
        _, _, _, stage, seed = min(queue, key=lambda r:r[:3])
        output = args.root / 'evaluations' / stage['id']
        phase(args.root, 'evaluate', 'rollouts', stage=stage['id'], seed=seed, planned_seeds=len(stage['planned_seeds']))
        workers('reproductions.leact.builderbench.evaluate_student', ['--reference',str(args.reference),
            '--corpus',str(args.corpus),'--checkpoint',stage['checkpoint'],'--output',str(output),
            '--stage',str(args.root/'ready'/f'{stage["id"]}.json'),'--simulator-python',str(args.simulator_python),
            '--seeds',*map(str,stage['planned_seeds']),'--work-seeds',str(seed)], output/f'logs-seed-{seed}')
        write_json(output/f'seed-{seed}-complete.json', dict(seed=seed, checkpoint_sha256=stage['checkpoint_sha256']))
        done = [s for s in stage['planned_seeds'] if (output/f'seed-{s}-complete.json').exists()]
        summarize(output, args.corpus, done)
        write_json(output/'speed.json', summarize_speed(output))
        phase(args.root, 'evaluate', 'seed_complete', stage=stage['id'], completed_seeds=len(done),
              planned_seeds=len(stage['planned_seeds']))
    write_json(args.root / 'evaluate-complete.json', dict(stages=len(available)))
    phase(args.root, 'evaluate', 'complete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['train','evaluate'], required=True)
    for name in ['root','base','corpus','reference','simulator-python']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / f'{args.role}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = dict(base=str(args.base), corpus_sha256=load_corpus(args.corpus)[1]['states_sha256'],
        warmstart_lr=1e-4, iterative_lr=1e-5, epochs=3, effective_batch=32, cutoff=8192, generation_tokens=8192,
        n=8, k=2, initial_rounds=1, baseline='One action-only decode, Appendix D.2.1',
        evaluation='Fixed final epoch: 64 seeds; intermediate epochs: 3 seeds; progressive seed-complete summaries',
        checkpoint_selection='Fixed final epoch; held-out task scores do not choose training checkpoints',
        source_sha256={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')})
    path = args.root/f'{args.role}-protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Study source or protocol changed')
    write_json(path, protocol)
    (train if args.role=='train' else evaluate)(args)


if __name__ == '__main__':
    main()
