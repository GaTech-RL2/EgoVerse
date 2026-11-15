"""Audit HF/vLLM multimodal token alignment on a frozen training state."""
import json
import os
from pathlib import Path
import traceback

import efe_leact_vl as leact
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def main():
    spec = json.loads(Path('/tmp/efe-training-spec.json').read_text())
    root = Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-recovery-20261027-v1/leact-efe-recovery-1')
    destination = root / ('token-probe-size-memory.json' if os.environ.get('EFE_EXPLICIT_IMAGE_SIZE') else 'token-probe.json')
    report = {}
    try:
        processor = AutoProcessor.from_pretrained(spec['tokenizer_base'], local_files_only=True)
        processor.image_processor.max_pixels = 1048576
        processor.image_processor.size['longest_edge'] = 1048576
        report['hf_image_processor'] = processor.image_processor.to_dict()
        image_kwargs = ({'size':dict(processor.image_processor.size)} if os.environ.get('EFE_EXPLICIT_IMAGE_SIZE') else {'max_pixels':1048576})
        report['vllm_image_kwargs'] = image_kwargs
        llm = LLM(model=spec['base_model'], tokenizer=spec['tokenizer_base'], dtype='bfloat16',
                  tensor_parallel_size=1, max_model_len=65536, max_num_seqs=2,
                  max_num_batched_tokens=2048, gpu_memory_utilization=.65,
                  enable_prefix_caching=False, enforce_eager=True,
                  limit_mm_per_prompt={'image':40,'video':0},
                  mm_processor_kwargs=image_kwargs, seed=spec['seed'])
        states = [json.loads(line) for line in (Path(spec['archive_root']) / 'leact-efe-leact-pilot-1/states.jsonl').read_text().splitlines()]
        state = states[5]
        target = leact.forward_target(state['action'], '')
        encoded, receipt = leact.encode_example(processor, state['messages'], state['tools'], target, spec['context'], action_only=True)
        request = leact.prompt(processor, state['messages'], state['tools'], target=target)
        result = llm.generate([request], SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1),use_tqdm=False)[0]
        left = encoded['input_ids'][0].tolist(); right = list(result.prompt_token_ids)
        first = next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b), min(len(left),len(right)))
        report.update(state_id=state['id'], hf_tokens=len(left), vllm_tokens=len(right),
                      identical=left==right, first_difference=first, receipt=receipt,
                      hf_grid=encoded.get('image_grid_thw').tolist() if 'image_grid_thw' in encoded else None,
                      hf_context=processor.tokenizer.decode(left[max(0,first-8):first+12]),
                      vllm_context=processor.tokenizer.decode(right[max(0,first-8):first+12]),
                      hf_image_tokens=left.count(processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')),
                      vllm_image_tokens=right.count(processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')))
    except Exception:
        report['error'] = traceback.format_exc()
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='hf_image_processor'}), flush=True)


if __name__=='__main__':
    main()
