"""Gate a curated 64K warmstart on lossless storage, scoring and memory checks."""
import argparse
import base64
import fcntl
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback

from efe_training_data import encode_example, forward_target, backward_messages, sha, write

CONFIG = Path('/tmp/efe-direct-warmstart-config.json')


def preserve_training(root, production, ready, config):
    """Keep final weights, Adam state and the exact reusable corpus together."""
    if config.get('durable_from_start'):
        from efe_training_state import verify_files
        from efe_durable_training import dataset_receipt
        if not str(production.resolve()).startswith(('/mnt/amlfs-03/','/mnt/amlfs-01/home/rpunamiya/leact/')):
            raise ValueError('Production corpus is not on durable storage')
        saved=json.loads((production/'restart-ready.json').read_text())
        state=Path(saved['checkpoint'])
        receipt=json.loads((state/'training-state.json').read_text())
        if receipt['step']!=ready['step']:raise ValueError('Final model and optimizer steps differ')
        verify_files(state,receipt['files'])
        manifest=json.loads(Path(ready['manifest']).read_text())
        verify_files(ready['checkpoint'],manifest['files'])
        _,corpus=dataset_receipt(production)
        record={'all_files_verified':True,'checkpoint':ready['checkpoint'],'manifest':ready['manifest'],
            'revision':ready['revision'],'corpus':str(production),'optimizer':str(state/'optimizer.pt'),
            'restart_state':str(state),'files':manifest['files'],'corpus_receipt':corpus,
            'durable_from_start':True}
        write(root/'retained.json',record);write(root/'durable-training-ready.json',record)
        return
    destination = Path(config['durable_root']) / os.environ['EFE_WORKFLOW_ID']
    checkpoint = Path(ready['checkpoint'])
    checkpoint_manifest = json.loads(Path(ready['manifest']).read_text())
    sources = [(checkpoint / name, destination / 'model' / name)
               for name in checkpoint_manifest['files']]
    sources += [(Path(ready['manifest']), destination / 'model/checkpoint-manifest.json'),
                (checkpoint / 'optimizer.pt', destination / 'optimizer.pt')]
    for name in ['spec.json','processor.json','student-interface.json','trace-admission.json',
                 'pilot-membership.json','warmstart-states.jsonl','decisions.jsonl',
                 'dataset-manifest.json','corpus-manifest.json','train/configuration.json',
                 'train/schedule.json','train/metrics.jsonl','train/complete.json']:
        source = production / name
        if source.is_file():sources.append((source, destination / 'corpus' / name))
    for name in ['rationales','tensors','tensor-blobs']:
        sources += [(source, destination / 'corpus' / source.relative_to(production))
                    for source in (production / name).rglob('*') if source.is_file()]
    receipts = {}
    for index, (source, target) in enumerate(sources):
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.copy.tmp')
            shutil.copyfile(source, temporary)
            if sha(temporary) != expected:raise ValueError('Durable copy differs')
            temporary.replace(target)
        if sha(target) != expected:raise ValueError('Existing durable artifact differs')
        receipts[str(target.relative_to(destination))] = {'sha256':expected,'bytes':target.stat().st_size}
        if index % 50 == 0:
            write(root / 'durable-copy-progress.json', {'files':index+1,'total':len(sources),
                'bytes':sum(r['bytes'] for r in receipts.values())})
    record = {'all_files_verified':True,'checkpoint':str(destination / 'model'),
        'manifest':str(destination / 'model/checkpoint-manifest.json'),'revision':ready['revision'],
        'corpus':str(destination / 'corpus'),'optimizer':str(destination / 'optimizer.pt'),'files':receipts}
    write(destination / 'retained.json', record)
    write(root / 'durable-training-ready.json', record)


def storage_check(root, processor, states):
    import torch
    from efe_visual_tensor_store import save
    from efe_train_vl import load_tensor
    folder = root / 'storage-check'
    (folder / 'tensors').mkdir(parents=True, exist_ok=True)
    checked = []
    visual_states = [state for state in states if any(
        isinstance(message.get('content'), list) and any(part.get('type')=='image' for part in message['content'])
        for message in state['messages'])][:2]
    if len(visual_states)!=2:raise ValueError('Two actual visual examples are required for storage validation')
    for index, state in enumerate(visual_states):
        batch, _ = encode_example(processor, state['messages'], state['tools'],
            forward_target(state['action'], 'Use the observed state to choose the next action.'), 65536)
        if batch is None:
            raise ValueError('Storage-check example exceeds context')
        for copy in range(2):
            path = folder / 'tensors' / f'{index}-{copy}.pt'
            save(batch, path)
            actual = load_tensor(path)
            if batch.keys() != actual.keys() or any(not torch.equal(batch[k], actual[k]) for k in batch):
                raise ValueError('Shared visual storage changed model inputs or labels')
        legacy = folder / 'legacy.pt'
        torch.save(batch, legacy)
        if any(not torch.equal(batch[k], load_tensor(legacy)[k]) for k in batch):
            raise ValueError('Legacy tensor loading changed')
        checked.append({'state_id': state['id'], 'all_tensors_equal': True})
    # Check corruption rejection in this disposable test fixture only.
    blob = next((folder / 'tensor-blobs').glob('*.pt'))
    original = blob.read_bytes()
    try:
        blob.write_bytes(original + b'corruption-test')
        rejected = False
        for path in (folder / 'tensors').glob('*.pt'):
            try:
                load_tensor(path)
            except ValueError:
                rejected = True
        if not rejected:
            raise ValueError('Changed visual blob was not rejected')
    finally:
        blob.write_bytes(original)
    write(root / 'storage-check.json', {'examples': checked, 'corruption_rejected': True,
        'unique_blob_files': len(list((folder / 'tensor-blobs').glob('*.pt'))),
        'scope': 'Real complete multimodal examples round-trip tensor-for-tensor; legacy format and corruption rejection checked.'})


def score(root, spec):
    import torch
    from transformers import AutoProcessor
    from vllm import LLM
    from efe_leact_vl import action_score
    processor = AutoProcessor.from_pretrained(spec['base_model'], local_files_only=True)
    processor.image_processor.max_pixels = 1048576
    processor.image_processor.size['longest_edge'] = 1048576
    states = [json.loads(line) for line in (root / 'probe-states.jsonl').read_text().splitlines()]
    llm = LLM(model=spec['base_model'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=65536, max_num_seqs=1, max_num_batched_tokens=2048,
        gpu_memory_utilization=.8, enforce_eager=True, enable_prefix_caching=False,
        limit_mm_per_prompt={'image':64, 'video':0},
        mm_processor_kwargs={'size':dict(processor.image_processor.size)}, seed=spec['seed'])
    probes = []
    (root / 'preflight/probes').mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(states):
        result, batch, receipt = action_score(llm, processor, state, '', 65536, spec['seed'])
        if result is None:
            raise ValueError('Complete scoring probe exceeds context')
        path = root / 'preflight/probes' / f'{index}.pt'
        torch.save(batch, path)
        probes.append({'condition':'curated-64k-'+str(index), 'state_id':state['id'],
            'tensor_file':str(path.relative_to(root / 'preflight')), 'tensor_sha256':sha(path), **result})
    write(root / 'preflight/scoring-probes.json', probes)
    write(root / 'token-parity.json', {'exact_vllm_hf_tokens':True, 'probes':probes,
        'mm_processor_kwargs':{'size':dict(processor.image_processor.size)}})


def finish_pipeline(root, production, config, progress, env):
    import efe_training_pilot as pilot
    complete=json.loads((production/'complete.json').read_text())
    if not complete['success']:raise ValueError('Curated warmstart failed')
    ready=json.loads((production/'warmstart-ready.json').read_text())
    from efe_leact_pipeline import export_portable
    portable=export_portable(root,ready)
    if config.get('durable_from_start'):
        portable['optimizer_persistence']='Full FP32 master parameters, Adam moments and per-rank RNG are committed atomically on shared storage.'
        write(root/'portable-ready.json',portable)
    write(root/'training-complete.json',{'success':True,'ready':ready,'portable':portable,
        'corpus_snapshot':config['corpus_snapshot'],'optimizer_steps':ready['step']})
    progress('preserving_training','Verifying final model, Adam state and the exact curated corpus for subsequent LeAct.')
    preserve_training(root,production,ready,config)
    progress('native_evaluation','Curated warmstart saved. Starting six matched training-task rollouts with the verified image-processing configuration.')
    lane=Path('/tmp/efe-training-evaluation-v1/warm96')
    evaluation=json.loads((lane/'evaluation-template.json').read_text())
    evaluation.update(student_checkpoint=portable['checkpoint'],student_manifest=portable['manifest'],
        student_revision=ready['revision'],training_parent=os.environ['EFE_WORKFLOW_ID'])
    write(lane/'spec.json',evaluation)
    pilot.ROOT=root
    pilot.run_logged('native-evaluation',[sys.executable,str(lane/'bootstrap.py')],env,12*3600)
    native=lane/'results'/os.environ['EFE_WORKFLOW_ID']
    outcomes=json.loads((native/'worker-complete.json').read_text())
    if not outcomes.get('success'):raise ValueError('Native warmstart evaluation did not finish cleanly')
    write(root/'pipeline-complete.json',{'success':True,'training':ready,'native':outcomes,'workflow':os.environ['EFE_WORKFLOW_ID']})
    progress('complete','Curated warmstart and matched native rollouts finished; awaiting verified artifact collection.')


def validated_production(root, config):
    """Reuse completed preparation only for the identical saved training inputs."""
    source=root/'production-spec.json'
    if not source.exists():return None
    spec=json.loads(source.read_text())
    production=Path(spec['archive_root'])/spec.get('run_id',os.environ['EFE_WORKFLOW_ID'])
    if not (production/'dataset-manifest.json').exists():return None
    if production!=root/'production'/config['training']['run_id']:
        raise ValueError('Saved production namespace differs')
    for key,value in config['training'].items():
        if spec.get(key)!=value:raise ValueError('Recovery changed a frozen training condition: '+key)
    receipt=json.loads((root/'preflight-passed.json').read_text())
    if not receipt.get('storage') or not receipt.get('exact_tokens') or not receipt.get('memory_smoke',{}).get('success'):
        raise ValueError('Completed 64K preparation receipt is missing')
    for name,digest in config['corpus_source_manifest'].items():
        if sha(root/'corpus'/name)!=digest:raise ValueError('Frozen corpus input changed: '+name)
    if not (production/'training-lineage.json').exists():
        raise ValueError('Saved corpus lacks a frozen training lineage; review preparation recovery')
    return source,production


def main():
    config = json.loads(CONFIG.read_text())
    workflow = os.environ['EFE_WORKFLOW_ID']
    root = Path(config['archive_root']) / config.get('run_id',workflow)
    root.mkdir(parents=True, exist_ok=True)
    lock=(root/'pipeline-writer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def progress(phase, detail, **extra):
        write(root / 'pipeline-progress.json', {'phase':phase, 'detail':detail,
            'workflow':workflow, 'updated_utc':datetime.now(timezone.utc).isoformat(), **extra})
        print(phase + ': ' + detail, flush=True)
    try:
        # The completed smoke directories are immutable. A reclaimed training
        # allocation resumes its saved optimizer state instead of rerunning
        # the discarded smoke into those same directories.
        if config.get('durable_from_start'):
            saved=validated_production(root,config)
            if saved:
                source,production=saved
                attempt=root/'recovery-attempts'/workflow;attempt.mkdir(parents=True,exist_ok=True)
                failure=root/'pipeline-complete.json'
                if failure.exists() and not json.loads(failure.read_text()).get('success'):
                    shutil.copyfile(failure,attempt/'prior-pipeline-complete.json');failure.unlink()
                progress('resuming_curated_warmstart','Reusing the verified 64K gates and exact saved corpus. Verify and restore the latest committed FP32 model, Adam and RNG state.')
                write(attempt/'preparation-reuse.json',{'production_spec_sha256':sha(source),
                    'preflight_receipt_sha256':sha(root/'preflight-passed.json'),
                    'dataset_sha256':sha(production/'dataset-manifest.json'),
                    'teacher_explanations_regenerated':False,'discarded_smoke_repeated':False})
                import efe_cuda_helpers as helpers
                import efe_training_pilot as pilot
                helpers.ROOT=root;helpers.ARCHIVE=root;env=helpers.driver_environment()
                env.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',NCCL_P2P_DISABLE='1')
                pilot.SPEC=source;pilot.main()
                finish_pipeline(root,production,config,progress,env)
                return
        payload = base64.b64decode(Path('/tmp/efe-direct-corpus.b64').read_bytes())
        import hashlib
        if hashlib.sha256(payload).hexdigest() != config['corpus_bundle_sha256']:
            raise ValueError('Frozen corpus input package changed')
        corpus = root / 'corpus'
        corpus.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(corpus, filter='data')
        for name, expected in config['corpus_source_manifest'].items():
            if sha(corpus / name) != expected:
                raise ValueError('Frozen corpus input differs: ' + name)
        progress('curated_corpus_verification', 'Verifying successful executed targets, scene isolation and complete 64K context coverage.')
        from efe_direct_corpus_profile import profile
        profile(corpus / 'spec.json')
        result = json.loads((corpus / 'result.json').read_text())
        if result['decisions'] != config['expected_decisions'] or result['coverage']['65536']['complete_decisions'] != result['decisions']:
            raise ValueError('Frozen corpus is incomplete at the declared 64K context')
        states = [json.loads(line) for line in (corpus / 'decisions.jsonl').read_text().splitlines()]
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(config['training']['base_model'], local_files_only=True)
        processor.image_processor.max_pixels = 1048576
        processor.image_processor.size['longest_edge'] = 1048576
        storage_check(root, processor, states)
        profile_rows = json.loads((corpus / 'states-profile.json').read_text())
        ordered = sorted(profile_rows, key=lambda r:max(r['forward_tokens'],r['backward_tokens']))
        chosen = {ordered[int(.95*len(ordered))]['state_id'], ordered[-1]['state_id']}
        selected = [s for s in states if s['id'] in chosen]
        (root / 'probe-states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in selected))
        preflight = root / 'preflight'
        (preflight / 'tensors').mkdir(parents=True, exist_ok=True)
        from efe_visual_tensor_store import save
        rows = []
        for state in selected:
            # Discarded memory test only: reserve more explanation tokens than
            # the 120-word teacher annotation is expected to require.
            rationale = ' '.join(['observation'] * 500)
            pairs = [('forward', state['messages'], forward_target(state['action'], rationale)),
                     ('backward', backward_messages(state['messages'], state['action']),
                      {'role':'assistant','content':'<think>\n'+rationale+'\n</think>\n\nDONE'})]
            for direction, messages, target in pairs:
                batch, receipt = encode_example(processor, messages, state['tools'], target, 65536)
                if batch is None:
                    raise ValueError('Reserved-explanation memory probe exceeds 64K')
                ident = state['id']+'-'+direction
                path = preflight / 'tensors' / (ident+'.pt')
                save(batch, path)
                rows.append({'id':ident,'state_id':state['id'],'suite':state['suite'],'task':state['task'],
                    'episode':state['episode'],'direction':direction,'tensor_file':str(path.relative_to(preflight)),
                    'tensor_sha256':sha(path),**receipt})
        write(preflight / 'dataset-manifest.json', {'examples':rows})
        train_spec = dict(config['training'])
        smoke_spec = preflight / 'spec.json'
        write(smoke_spec, train_spec)
        import efe_cuda_helpers as helpers
        helpers.ROOT = root
        helpers.ARCHIVE = root
        env = helpers.driver_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),NCCL_P2P_DISABLE='1',
            OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        import efe_training_pilot as pilot
        pilot.ROOT = root
        progress('64k_scoring_preflight', 'Checking exact multimodal token IDs and complete expert-action likelihood at long context.')
        pilot.run_logged('scoring-preflight', [sys.executable, '/tmp/efe_direct_warmstart.py', '--score', str(root)],
            {**env,'CUDA_VISIBLE_DEVICES':'0','VLLM_WORKER_MULTIPROC_METHOD':'spawn'}, 3600)
        progress('64k_optimizer_preflight', 'Measuring full-parameter optimizer memory on the p95 and longest complete examples; these two updates are discarded.')
        pilot.run_logged('memory-preflight', [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            '/tmp/efe_train_vl.py','--spec',str(smoke_spec),'--root',str(preflight),'--mode','smoke'],env,3600)
        smoke = json.loads((preflight / 'smoke/complete.json').read_text())
        if not smoke['success']:
            raise ValueError('64K optimizer preflight failed')
        write(root / 'preflight-passed.json', {'storage':True,'exact_tokens':True,
            'likelihood_parity':json.loads((preflight / 'smoke/scoring-parity.json').read_text()),
            'memory_smoke':smoke,'production_training_started':False})
        extraction = json.loads((corpus / 'extraction-spec.json').read_text())
        train_spec.update(extraction, archive_root=str(root / 'production'),
            pilot_decisions=len(states), minimum_suites=2)
        production_spec = root / 'production-spec.json'
        write(production_spec, train_spec)
        progress('curated_warmstart', '64K checks passed. Generate explicit teacher backward explanations, then train from untouched Qwen with complete decision coverage.')
        pilot.SPEC = production_spec
        pilot.main()
        production = root / 'production' / train_spec.get('run_id',workflow)
        finish_pipeline(root,production,config,progress,env)
    except Exception as error:
        write(root / 'pipeline-complete.json', {'success':False,'error':str(error),'traceback':traceback.format_exc(),'workflow':workflow})
        progress('needs_review', str(error))
        raise
    finally:
        import time
        for _ in range(config.get('final_collection_wait_seconds',21600)//5):
            if (root / 'pipeline-collected').exists():
                break
            time.sleep(5)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', type=Path)
    args = parser.parse_args()
    if args.score:
        score(args.score, json.loads(CONFIG.read_text())['training'])
    else:
        main()
