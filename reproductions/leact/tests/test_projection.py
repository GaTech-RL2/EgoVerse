from types import SimpleNamespace
import random

import pytest

from reproductions.leact.projection.backend import Backend
from reproductions.leact.projection.sampler import SamplingConfig, continuation_mean, project


def test_target_likelihood_excludes_prompt_and_requires_each_observed_token():
    rows = [None, {10: SimpleNamespace(logprob=-1000)},
            {20: SimpleNamespace(logprob=-1)}, {21: SimpleNamespace(logprob=-3)}]
    assert continuation_mean(rows, [9, 10], [20, 21]) == -2
    rows[-1] = {22: SimpleNamespace(logprob=-3)}
    with pytest.raises(ValueError, match='Missing observed'):
        continuation_mean(rows, [9, 10], [20, 21])


class FakeBackend:
    eos_ids = {0}

    def propose(self, items, prefixes, lengths, seeds):
        result = []
        for prefix, length, seed in zip(prefixes, lengths, seeds):
            rng = random.Random(seed)
            suffix = [rng.randrange(1, 10) for _ in range(length)]
            if len(prefix) + length >= 8:
                suffix[-1] = 0
            result.append(prefix + suffix)
        return result

    def score(self, items, sequences):
        return [-sum(sequence) / len(sequence) for sequence in sequences]


def test_projection_uses_greedy_mean_likelihood_and_independent_rngs():
    items = [dict(state_id=str(i)) for i in range(4)]
    config = SamplingConfig(block_size=4, max_tokens=12, steps=10, seed=7)
    rows = project(FakeBackend(), items, config)
    separate = [project(FakeBackend(), [item], config)[0] for item in items]
    assert rows == separate
    assert all(row['finished'] and len(row['tokens']) <= 12 for row in rows)
    assert any(row['accepted'] for row in rows)
    for row in rows:
        for event in row['events']:
            if event['step'] is not None:
                assert event['accepted'] == (event['candidate_mean_logprob'] > event['previous_mean_logprob'])


def test_target_backend_never_conditions_on_the_reference():
    backend = Backend.__new__(Backend)
    seen = []
    backend.prompt_ids = lambda text: [ord(c) for c in text]
    backend.params = lambda **kwargs: kwargs
    backend.eos_ids = {0}
    backend.tokenizer = SimpleNamespace(eos_token_id=0,
        encode=lambda text, **kwargs: [ord(c) for c in text])
    reference='<think>\nSECRET EXPERT EXPLANATION\n</think>\n\nAction: call\nPolicy: {"call": 1.0}'
    item=dict(question='state',reference=reference)

    def generate(prompts, params, **kwargs):
        seen.extend(prompts)
        return [SimpleNamespace(prompt_logprobs=[{token: SimpleNamespace(logprob=-2)}
                        for token in p['prompt_token_ids']]) for p in prompts]

    backend.model = SimpleNamespace(generate=generate)
    assert backend.score([item], [[1, 2, 0]]) == [-2]
    suffix='\n</think>\n\nAction: call\nPolicy: {"call": 1.0}'
    assert seen == [{'prompt_token_ids': [ord(c) for c in 'state'] + [1, 2] + [ord(c) for c in suffix] + [0]}]


def test_prefix_mutation_by_backend_is_rejected():
    class Broken(FakeBackend):
        def propose(self, items, prefixes, lengths, seeds):
            return [[99] for _ in items]
    with pytest.raises(ValueError, match='fixed prefix'):
        project(Broken(), [dict(state_id='a')], SamplingConfig(block_size=2, max_tokens=4, steps=0))


@pytest.mark.parametrize('settings', [dict(max_tokens=0), dict(block_size=0), dict(max_tokens=33), dict(steps=-1)])
def test_invalid_sampling_budget(settings):
    with pytest.raises(ValueError):
        SamplingConfig(**settings)


def test_terminal_eos_is_transport_not_policy_text():
    from reproductions.leact.projection.backend import completion_text
    from reproductions.leact.core import parse_policy
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens is False
            return ''.join({1:'A valid decision explanation.',2:'\n</think>\n\nAction: call\nPolicy: {"call": 1.0}',3:'<|im_end|>'}[t] for t in tokens)
    text=completion_text(Tokenizer(), [1,2,3], {3})
    assert text.startswith('<think>\nA valid decision explanation.\n</think>')
    assert parse_policy(text,['call']).fallback is None
    assert completion_text(Tokenizer(),[1,2],{3}) == text


def test_fixed_answer_survives_an_explanation_that_ends_without_formatting():
    from reproductions.leact.projection.backend import fixed_completion_text
    from reproductions.leact.core import parse_policy
    tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs: ''.join(chr(t) for t in tokens))
    reference='<think>\nExpert wording.\n</think>\n\nAction: fold\nPolicy: {"fold": 0.75, "call": 0.25}'
    explanation='The visible hand is weak, so preserving chips favors folding.'
    text=fixed_completion_text(tokenizer,[ord(c) for c in explanation]+[0],{0},reference)
    assert text == '<think>\n'+explanation+'\n</think>\n\nAction: fold\nPolicy: {"fold": 0.75, "call": 0.25}'
    parsed=parse_policy(text,['fold','call'])
    assert parsed.fallback is None and parsed.policy == {'fold':0.75,'call':0.25}


def test_partial_and_finished_explanations_use_the_same_scored_target():
    backend=Backend.__new__(Backend)
    backend.eos_ids={0}
    backend.tokenizer=SimpleNamespace(eos_token_id=0,encode=lambda text, **kwargs:[ord(c) for c in text])
    item={'reference':'<think>\nOriginal.\n</think>\n\nAction: call\nPolicy: {"call": 1.0}'}
    assert backend.scored_tokens(item,[1,2]) == backend.scored_tokens(item,[1,2,0])
    with pytest.raises(ValueError,match='EOS inside'):
        backend.scored_tokens(item,[1,0,2])


@pytest.mark.parametrize('reference',['no structure','<think>\nx\n</think>\nno answer','<think>\nx</think></think>'])
def test_fixed_answer_rejects_malformed_references(reference):
    from reproductions.leact.projection.backend import fixed_answer_suffix
    with pytest.raises(ValueError):
        fixed_answer_suffix(reference)
