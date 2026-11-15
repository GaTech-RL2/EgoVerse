"""Pinned Qwen3 inference, strict action parsing and BuilderBench delta rewards."""
from __future__ import annotations

import json
import math
import time

from ..data import fingerprint
from .expert import validate_action

SETTINGS = dict(max_completion_tokens=8192, context_tokens=8192, temperature=1.0,
                top_p=0.95, top_k=-1, frequency_penalty=0.0, repetition_penalty=1.0)


def split_output(text, *, thinking=True, backward=False, cubes=None):
    if thinking:
        if not text.startswith('<think>') or text.count('</think>') != 1:
            raise ValueError('Missing or repeated reasoning delimiters')
        reasoning, answer = text[len('<think>'):].split('</think>')
        reasoning, answer = reasoning.strip(), answer.strip()
        if not reasoning:
            raise ValueError('Empty reasoning')
    else:
        reasoning, answer = '', text.strip()
        if answer.startswith('<think>'):
            prefix, separator, answer = answer.partition('</think>')
            if not separator or prefix[len('<think>'):].strip():
                raise ValueError('Action-only output contains reasoning')
            answer = answer.strip()
    if any(t in reasoning for t in ['<think', '<|im_']):
        raise ValueError('Unexpected control token')
    if backward:
        if answer != 'DONE':
            raise ValueError('Backward output must terminate with DONE')
        return reasoning, None
    action = json.loads(answer)
    validate_action(action, cubes)
    return reasoning, action


def action_reward(action, expert):
    """Appendix D.2.1, positions in meters; invalid outputs receive zero."""
    if not isinstance(action, dict) or action.get('type') != expert['type']:
        return 0.0
    if action.get('cube_id') != expert.get('cube_id'):
        return 0.1
    if 'pos' in expert:
        if not isinstance(action.get('pos'), list) or len(action['pos']) != 3:
            return 0.0
        distance = math.dist(action['pos'], expert['pos'])
        if not math.isfinite(distance):
            return 0.0
        return 1.0 if distance <= 0.01 else max(0.0, 1.0 - distance)
    # The appendix does not define low-level-vector grading; exact match only.
    return 1.0 if action == expert else 0.1


class Backend:
    def __init__(self, checkpoint):
        from vllm import LLM, SamplingParams
        started = time.perf_counter()
        self.params_type = SamplingParams
        self.model = LLM(model=str(checkpoint), dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=16384, max_num_seqs=32, gpu_memory_utilization=0.88,
            enable_prefix_caching=False, trust_remote_code=False, seed=0)
        self.tokenizer = self.model.get_tokenizer()
        self.load_seconds = time.perf_counter() - started

    def prepare(self, prompt, *, thinking=True, reasoning=None):
        rendered = self.tokenizer.apply_chat_template([dict(role='user', content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        if thinking:
            if rendered.endswith('<think>\n'):
                rendered = rendered[:-len('<think>\n')]
            header = '<think>\n' + (reasoning + '\n</think>\n\n' if reasoning is not None else '')
            rendered += header
        else:
            header = '<think>\n\n</think>\n\n'
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > SETTINGS['context_tokens']:
            raise OverflowError('Student prompt exceeds 8192 tokens; no silent truncation')
        return rendered, header, len(ids)

    def generate(self, prompt, *, seed, n=1, thinking=True, reasoning=None):
        rendered, header, prompt_tokens = self.prepare(prompt, thinking=thinking, reasoning=reasoning)
        params = self.params_type(n=n, seed=seed, max_tokens=8192, temperature=1.0, top_p=0.95,
                                 top_k=-1, frequency_penalty=0.0, repetition_penalty=1.0)
        started = time.perf_counter()
        outputs = self.model.generate([rendered], params, use_tqdm=False)[0].outputs
        elapsed = time.perf_counter() - started
        if len(outputs) != n:
            raise RuntimeError('Inference returned an incomplete sample group')
        return [dict(text=header + value.text, finish_reason=value.finish_reason,
                     prompt_tokens=prompt_tokens, generated_tokens=len(value.token_ids)) for value in outputs], elapsed


def decision_seed(seed, task, index, stage='rollout'):
    return int(fingerprint([seed, task, index, stage])[:8], 16) % (2**31)
