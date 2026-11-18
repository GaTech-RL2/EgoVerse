"""Short inference-throughput qualification; never a benchmark accuracy result."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from retention_runner import cuda_environment, write


def run(spec):
    import datasets
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from efe_model_cache import stage

    def event(phase, **values):
        record = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'accuracy_evaluation': False, **values}
        write(Path(spec['shared_root'])/'progress.json', record)
        print('EFE_SERVING_PROBE '+json.dumps(record), flush=True)

    if spec['purpose'] != 'serving_throughput_probe' or spec['probe_output_tokens'] > 1024:
        raise ValueError('The short serving probe must be isolated from benchmark results')
    model = spec['models'][0]
    event('staging_model')
    local = stage(model['path'], model['manifest'], '/tmp/rp_jhh_retention/models', model['revision'])
    processor = AutoProcessor.from_pretrained(model['path'], local_files_only=True)
    processor.image_processor.size.update(spec['image_processor_size'])
    dataset = datasets.load_dataset(spec['dataset']['id'], revision=spec['dataset']['revision'], split='test')
    if len(dataset) != 765:
        raise ValueError('Dataset differs from the frozen evaluation')
    event('initializing_engine', batch_size=spec['batch_size'])
    llm = LLM(model=local['local_model'], tokenizer=model['path'], dtype='bfloat16',
              tensor_parallel_size=spec['gpu_count'], distributed_executor_backend='mp',
              disable_custom_all_reduce=True, enforce_eager=True, max_model_len=65536,
              max_num_seqs=spec['batch_size'],
              max_num_batched_tokens=spec['serving_profile']['max_num_batched_tokens'],
              gpu_memory_utilization=spec['serving_profile']['gpu_memory_utilization'],
              enable_prefix_caching=False, async_scheduling=False, disable_log_stats=False,
              limit_mm_per_prompt={'image': 1, 'video': 0},
              mm_processor_kwargs={'size': spec['image_processor_size']},
              generation_config='auto', seed=spec['seed'])
    batches = []
    for start in range(0, 2*spec['batch_size'], spec['batch_size']):
        inputs, params, expected = [], [], []
        for index in range(start, start+spec['batch_size']):
            row = dataset[index]
            image = row['image'].convert('RGB')
            text = spec['prompt'].format(question=row['question'])
            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text},
                                                      {'type': 'image', 'image': image}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
            expected.append(processor(text=[prompt], images=[image], return_tensors='pt')['input_ids'][0].tolist())
            inputs.append({'prompt': prompt, 'multi_modal_data': {'image': image}})
            params.append(SamplingParams(temperature=1., top_p=.95, top_k=20,
                                         max_tokens=spec['probe_output_tokens'], seed=spec['seed']+index))
        event('generating', batch=len(batches)+1, batch_size=spec['batch_size'])
        before = time.monotonic()
        outputs = llm.generate(inputs, params, use_tqdm=False)
        elapsed = time.monotonic()-before
        for out, tokens in zip(outputs, expected, strict=True):
            if list(out.prompt_token_ids) != tokens:
                raise ValueError('HF/vLLM multimodal input-token mismatch')
        n = sum(len(out.outputs[0].token_ids) for out in outputs)
        batches.append({'requests': len(outputs), 'output_tokens': n, 'seconds': elapsed,
                        'aggregate_output_tokens_per_second': n/elapsed})
        event('batch_completed', batches=batches)
    result = {'success': True, 'accuracy_evaluation': False, 'optimizer_updates': 0,
              'batch_size': spec['batch_size'], 'output_cap_for_probe_only': spec['probe_output_tokens'],
              'batching_profile': spec['serving_profile'], 'batches': batches,
              'completed_utc': datetime.now(timezone.utc).isoformat()}
    write(Path(spec['shared_root'])/'result.json', result)
    print('EFE_SERVING_PROBE '+json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--full-spec', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.child:
        return run(spec)
    if importlib.util.find_spec('datasets') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==4.4.1'], check=True, timeout=600)
    subprocess.run([sys.executable, '-u', __file__, '--spec', str(args.spec), '--child'],
                   env=cuda_environment(spec['gpu_count']), check=True)
    if args.full_spec:
        full = json.loads(args.full_spec.read_text())
        for field in ['models', 'dataset', 'prompt', 'sampling', 'seed', 'image_processor_size',
                      'gpu_count', 'batch_size', 'serving_profile', 'max_output_tokens']:
            if full[field] != spec[field]:
                raise ValueError('Serving qualification differs from evaluation: '+field)
        result = json.loads((Path(spec['shared_root'])/'result.json').read_text())
        if not result.get('success') or len(result['batches']) != 2 or min(
                batch['aggregate_output_tokens_per_second'] for batch in result['batches']) < 20:
            raise RuntimeError('Batching did not qualify against the measured 14 tokens/s serial baseline')
        # The probe child exited, so its GPU engine is gone. Reuse the same
        # node for the full panel, with the original 32768-token output limit.
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('retention_runner.py')),
                        '--spec', str(args.full_spec)], check=True)


if __name__ == '__main__':
    main()
