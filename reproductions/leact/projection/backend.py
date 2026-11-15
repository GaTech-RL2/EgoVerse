"""Pinned vLLM adapter for expert-conditioned proposals and unassisted scoring."""
from .sampler import continuation_mean


def completion_text(tokenizer, tokens, eos_ids):
    """Strip transport EOS only; preserve the reasoning delimiter tokens."""
    body = tokens[:-1] if tokens and tokens[-1] in eos_ids else tokens
    return '<think>\n' + tokenizer.decode(body, skip_special_tokens=False)


def fixed_answer_suffix(reference):
    """Preserve the serialized CFR target outside the rewrite search space."""
    if not reference.startswith('<think>\n') or reference.count('</think>') != 1:
        raise ValueError('Malformed expert reference')
    suffix = reference.split('</think>', 1)[1]
    if not suffix.startswith('\n\nAction: ') or '\nPolicy: ' not in suffix:
        raise ValueError('Missing fixed expert action target')
    return '\n</think>' + suffix


def fixed_completion_text(tokenizer, tokens, eos_ids, reference):
    return completion_text(tokenizer, tokens, eos_ids) + fixed_answer_suffix(reference)


class Backend:
    def __init__(self, checkpoint):
        from vllm import LLM, SamplingParams
        self.params = SamplingParams
        self.model = LLM(model=str(checkpoint), dtype='bfloat16', tensor_parallel_size=1,
                         max_model_len=12288, gpu_memory_utilization=0.88,
                         trust_remote_code=False, seed=0, enable_prefix_caching=False)
        self.tokenizer = self.model.get_tokenizer()
        self.eos_ids = {self.tokenizer.eos_token_id}

    def prompt_ids(self, question):
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if not text.endswith('<think>\n'):
            text += '<think>\n'
        return self.tokenizer.encode(text, add_special_tokens=False)

    def propose(self, items, prefixes, lengths, seeds):
        prompts, params = [], []
        for item, prefix, length, seed in zip(items, prefixes, lengths, seeds):
            partial = self.tokenizer.decode(prefix, skip_special_tokens=False)
            question = (
                'Paraphrase the explanation in the reference decision. Output only the explanation '
                'as one concise paragraph of 3-5 sentences, at most 200 words. Preserve the visible '
                'card facts and strategic claims; do not add hidden information or correct the reference. '
                'The action and probability JSON are copied separately and must not appear in your output. '
                'Do not include thinking tags, headings, JSON, a preamble, or discussion of these instructions. '
                'Continue the partial explanation without repeating it.\n\n'
                'TASK:\n' + item['question'] + '\n\nREFERENCE DECISION:\n' + item['reference'] +
                '\n\nPARTIAL EXPLANATION:\n' + partial)
            # The proposal emits a public explanation as its final answer. The
            # target scorer below still uses the ordinary thinking-enabled task.
            text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = self.tokenizer.encode(text, add_special_tokens=False) + prefix
            if len(ids) + length > 12288:
                raise ValueError('Proposal context exceeds the pinned model context')
            prompts.append({'prompt_token_ids': ids})
            params.append(self.params(max_tokens=length, temperature=0.6, top_p=1.0, top_k=-1,
                          repetition_penalty=1.05, stop_token_ids=list(self.eos_ids), seed=seed,
                          bad_words=['<think>', '</think>', 'Action:', 'Policy:']))
        outputs = self.model.generate(prompts, params, use_tqdm=False)
        return [prefix + list(output.outputs[0].token_ids) for prefix, output in zip(prefixes, outputs)]

    def scored_tokens(self, item, sequence):
        body = sequence[:-1] if sequence and sequence[-1] in self.eos_ids else sequence
        if any(token in self.eos_ids for token in body):
            raise ValueError('EOS inside a projected explanation')
        suffix = self.tokenizer.encode(fixed_answer_suffix(item['reference']), add_special_tokens=False)
        return body + suffix + [self.tokenizer.eos_token_id]

    def score(self, items, sequences):
        # Privileged context is absent from the task prompt. The fixed CFR
        # answer is part of the scored completion, just as in the SFT target.
        # Every partial candidate receives the same closing/action suffix.
        contexts = [self.prompt_ids(item['question']) for item in items]
        scored = [self.scored_tokens(item, sequence) for item, sequence in zip(items, sequences)]
        prompts = [{'prompt_token_ids': context + sequence} for context, sequence in zip(contexts, scored)]
        outputs = self.model.generate(prompts, self.params(max_tokens=1, temperature=0,
                                                           prompt_logprobs=0), use_tqdm=False)
        return [continuation_mean(output.prompt_logprobs, context, sequence)
                for output, context, sequence in zip(outputs, contexts, scored)]
