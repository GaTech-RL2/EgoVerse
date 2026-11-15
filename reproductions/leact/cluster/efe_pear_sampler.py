"""Native constrained-action sampler retaining exact PEAR training probabilities.

This is a local Responses transport for the existing robot loop, not a planner.
Only generated student tokens are training targets. Images and public prompts
are saved with their exact processor token IDs; no teacher answer is provided.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import asyncio
import base64
import gzip
import hashlib
import io
import json
import os
import time
import uuid


def digest_body(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def final_answer(text):
    # The pinned Qwen template opens <think> in the generation prompt. An
    # unfinished thought must never be interpreted as an executable command.
    return text.split('</think>', 1)[1].strip() if '</think>' in text else ''


def write(path, value, compressed=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if compressed:
        data = gzip.compress(data)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


class Sampler:
    def __init__(self, spec, root):
        from transformers import AutoProcessor
        from vllm import LLM
        from efe_model_cache import stage
        self.spec, self.root = spec, root
        self.serial = 0
        self.lock = asyncio.Lock()
        cache = stage(spec['student_checkpoint'], spec['student_manifest'],
                      '/tmp/efe-pear-native-model', spec['student_revision'])
        self.processor = AutoProcessor.from_pretrained(cache['local_model'], local_files_only=True)
        self.processor.image_processor.size.update(spec['image_processor_size'])
        # Preserve the compiled serving configuration used by the native base
        # evaluation. Ring/Simple remains required for FSDP TRAINING separately.
        self.llm = LLM(model=cache['local_model'], tokenizer=cache['local_model'], dtype='bfloat16',
            tensor_parallel_size=8, distributed_executor_backend='mp', disable_custom_all_reduce=True,
            max_model_len=131072, max_num_seqs=1, max_num_batched_tokens=8192,
            gpu_memory_utilization=.90, max_logprobs=spec['pear']['capture_top_k'],
            limit_mm_per_prompt={'image':64, 'video':0},
            mm_processor_kwargs={'size':dict(self.processor.image_processor.size)},
            generation_config='auto', seed=spec['pear']['seed'])
        write(root/'ready.json', {'revision':spec['student_revision'], 'context':131072,
            'vocab_size':len(self.processor.tokenizer),
            'topk_log_probs':spec['pear']['capture_top_k'], 'optimizer_updates':0,
            'sampling':{'temperature':1., 'top_p':1., 'top_k':-1},
            'source_manifest_sha256':cache['source_manifest_sha256'],
            'ready_utc':datetime.now(timezone.utc).isoformat()})

    def generate(self, body, ident):
        from PIL import Image
        from vllm import SamplingParams
        from efe_training_data import chat_inputs
        from efe_pear_loss import validate_sampling
        start = time.monotonic()
        if body.get('tools') or body.get('model') != self.spec['request_model']:
            raise ValueError('Sampler accepts only the declared constrained-action policy')
        limit = body.get('max_output_tokens')
        if type(limit) is not int or not 0 < limit <= self.spec['max_output_tokens']:
            raise ValueError('Invalid output budget')
        messages = chat_inputs(body)
        images = [Image.open(io.BytesIO(base64.b64decode(part['image'].split(',', 1)[1]))).convert('RGB')
            for message in messages if isinstance(message.get('content'), list)
            for part in message['content'] if part['type'] == 'image']
        prompt = self.processor.apply_chat_template(messages, tools=[], tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=images, return_tensors='pt')
        expected = batch['input_ids'][0].tolist()
        if not prompt.rstrip().endswith('<think>'):
            raise ValueError('Pinned thinking template changed; review final-answer extraction')
        if len(expected) + limit > 131072:
            raise ValueError('Complete input plus declared generation budget exceeds context')
        prepared = time.monotonic()
        self.serial += 1
        seed = self.spec['pear']['seed'] + self.serial
        out = self.llm.generate([{'prompt':prompt, 'multi_modal_data':{'image':images}}],
            SamplingParams(temperature=1., top_p=1., top_k=-1, max_tokens=limit,
                logprobs=self.spec['pear']['capture_top_k'], seed=seed), use_tqdm=False)[0]
        generated = time.monotonic()
        if list(out.prompt_token_ids) != expected:
            raise ValueError('Sampler/HF multimodal prompt token mismatch')
        completion = out.outputs[0]
        if completion.finish_reason not in {'stop', 'length'}:
            raise ValueError('Incomplete or aborted sampler generation')
        if completion.logprobs is None or len(completion.token_ids) != len(completion.logprobs):
            raise ValueError('Missing sampler token probabilities')
        selected, heads, values = [], [], []
        for token, distribution in zip(completion.token_ids, completion.logprobs):
            selected.append(float(distribution[token].logprob))
            top = sorted(distribution, key=lambda t:distribution[t].logprob, reverse=True)[:self.spec['pear']['capture_top_k']]
            heads.append(top)
            values.append([float(distribution[t].logprob) for t in top])
        record = {'id':ident, 'request':body, 'request_sha256':digest_body(body),
            'prompt_token_ids':expected, 'token_ids':list(completion.token_ids),
            'sampled_logprobs':selected, 'topk_token_ids':heads, 'topk_logprobs':values,
            'sampler_revision':self.spec['student_revision'], 'temperature':1., 'top_p':1., 'top_k':-1,
            'seed':seed, 'finish_reason':completion.finish_reason, 'text':completion.text,
            'preprocessing_seconds':prepared-start, 'generation_seconds':generated-prepared,
            'model_seconds':generated-start, 'output_tokens':len(completion.token_ids),
            'input_tokens':len(expected), 'partition':'train_candidates',
            'role':self.spec['pear']['role'], 'optimizer_updates':0,
            'completed_utc':datetime.now(timezone.utc).isoformat()}
        validate_sampling(record, len(self.processor.tokenizer), self.spec['pear']['capture_top_k'])
        record['sampler_capture_complete'] = True
        write(self.root/(ident+'.json.gz'), record, compressed=True)
        write(self.root/'progress.json', {k:record[k] for k in
            ['id','sampler_revision','input_tokens','output_tokens','preprocessing_seconds',
             'generation_seconds','model_seconds','finish_reason','completed_utc']})
        return {'id':'resp_'+ident, 'object':'response', 'status':'completed',
            'model':self.spec['request_model'],
            'output':[{'id':'msg_'+ident, 'type':'message', 'role':'assistant', 'status':'completed',
                'content':[{'type':'output_text', 'text':final_answer(completion.text), 'annotations':[]}]}],
            'usage':{'input_tokens':len(expected), 'output_tokens':len(completion.token_ids),
                     'total_tokens':len(expected)+len(completion.token_ids)},
            # Transport completed even if the model spent its output budget.
            # The robot parser sees any incomplete JSON as an invalid action;
            # the tokens and native failed outcome remain in RL sample cost.
            'incomplete_details':{'reason':'max_output_tokens'} if completion.finish_reason=='length' else None}

    async def handle(self, request):
        from aiohttp import web
        if request.headers.get('Authorization') != 'Bearer '+os.environ['NVIDIA_INFERENCE_API_KEY']:
            raise web.HTTPUnauthorized()
        body = await request.json()
        if body.get('stream') is not True:
            raise web.HTTPBadRequest(text='Native sampler requires streaming transport')
        ident = 'pear_'+uuid.uuid4().hex
        out = web.StreamResponse(headers={'Content-Type':'text/event-stream', 'Cache-Control':'no-cache'})
        await out.prepare(request)
        async def emit(event):
            await out.write(b'data: '+json.dumps(event, allow_nan=False).encode()+b'\n\n')
        await emit({'type':'response.created', 'response':{'id':'resp_'+ident, 'status':'in_progress'}})
        try:
            async with self.lock:
                result = await asyncio.to_thread(self.generate, body, ident)
            await emit({'type':'response.completed', 'response':result})
        except Exception as error:
            write(self.root/(ident+'.error.json'), {'error':repr(error), 'optimizer_updates':0})
            await emit({'type':'response.failed', 'response':{'status':'failed', 'error':{'message':str(error)}}})
        await out.write_eof()
        return out


def main():
    from aiohttp import web
    p = argparse.ArgumentParser();p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True);p.add_argument('--port', type=int, default=19880)
    args = p.parse_args()
    sampler = Sampler(json.loads(args.spec.read_text()), args.root)
    app = web.Application(client_max_size=128*1024**2)
    app.router.add_get('/health', lambda request:web.json_response({'ready':True}))
    app.router.add_post('/v1/responses', sampler.handle)
    web.run_app(app, host='127.0.0.1', port=args.port, print=None, access_log=None)


if __name__ == '__main__':
    main()
