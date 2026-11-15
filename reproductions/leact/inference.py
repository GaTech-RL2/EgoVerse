"""Local vLLM inference with explicit assistant-prefix conditioning."""

from __future__ import annotations

from dataclasses import dataclass

from .data import fingerprint


@dataclass(frozen=True)
class Request:
    prompt: str
    n: int
    seed: int
    reasoning: str | None = None
    thinking: bool = True


def sample_seed(seed: int, state_id: str, stage: str) -> int:
    return int(fingerprint([seed, state_id, stage])[:8], 16) % (2**31)


class VLLMBackend:
    def __init__(self, checkpoint: str, *, revision: str | None = None,
                 tensor_parallel: int = 1, max_tokens: int = 4096):
        from vllm import LLM, SamplingParams

        self.params_type = SamplingParams
        self.max_tokens = max_tokens
        self.model = LLM(
            model=checkpoint, revision=revision, tokenizer_revision=revision,
            dtype="bfloat16", tensor_parallel_size=tensor_parallel,
            max_model_len=12288, gpu_memory_utilization=0.88,
            trust_remote_code=False, seed=0, enable_prefix_caching=True,
        )
        self.tokenizer = self.model.get_tokenizer()

    def generate(self, requests: list[Request]) -> list[list[dict]]:
        if not requests:
            return []
        prompts, params, headers = [], [], []
        for request in requests:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=request.thinking,
            )
            if request.thinking:
                header = "<think>\n"
                if request.reasoning is not None:
                    header += request.reasoning + "\n</think>\n\n"
                # Qwen revisions differ in whether the open tag is in the template.
                if prompt.endswith("<think>\n"):
                    prompt = prompt[:-len("<think>\n")]
                prompt += header
            else:
                header = "<think>\n\n</think>\n\n"
            if len(self.tokenizer.encode(prompt)) + self.max_tokens > 12288:
                raise ValueError("Prompt plus generation budget exceeds model context")
            prompts.append(prompt)
            headers.append(header)
            params.append(self.params_type(
                n=request.n, temperature=1.0, top_p=0.95, top_k=-1,
                frequency_penalty=0.0, repetition_penalty=1.0,
                max_tokens=self.max_tokens, seed=request.seed,
            ))
        results = self.model.generate(prompts, params, use_tqdm=False)
        if len(results) != len(requests):
            raise RuntimeError("Incomplete inference batch")
        grouped = []
        for header, request, result in zip(headers, requests, results):
            if len(result.outputs) != request.n:
                raise RuntimeError("Incomplete candidate generation")
            grouped.append([
                {"text": header + item.text, "finish_reason": item.finish_reason,
                 "generated_tokens": len(item.token_ids)}
                for item in result.outputs
            ])
        return grouped
