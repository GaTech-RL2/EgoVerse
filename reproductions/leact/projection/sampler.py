"""The released paper's greedy, length-normalized approximation, not exact MH.

Reference: aakaran/finetuning-with-sampling, commit
e74ab5eea6448bf78a4cff57179bbab8687cd243, boost_sci.py and Appendix C.2.
The proposal receives the expert response; the scoring prompt never does.
"""
from dataclasses import dataclass
import math
import random

from ..inference import sample_seed


@dataclass(frozen=True)
class SamplingConfig:
    block_size: int = 32
    max_tokens: int = 1856
    steps: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.block_size <= 0 or self.max_tokens <= 0 or self.max_tokens % self.block_size or self.steps < 0:
            raise ValueError('Invalid projection-sampling budget')


def continuation_mean(prompt_logprobs, prompt_ids, completion_ids):
    """Mean raw model log-probability of completion tokens, excluding prompt."""
    offset = len(prompt_ids)
    if not completion_ids or len(prompt_logprobs) != offset + len(completion_ids):
        raise ValueError('Incomplete target log-probability response')
    values = []
    for i, token in enumerate(completion_ids):
        row = prompt_logprobs[offset + i]
        if row is None or token not in row:
            raise ValueError('Missing observed-token log probability')
        value = float(row[token].logprob)
        if not math.isfinite(value):
            raise ValueError('Nonfinite target likelihood')
        values.append(value)
    return sum(values) / len(values)


def project(backend, items, config=SamplingConfig()):
    """Batch independent chains while retaining per-example deterministic RNGs."""
    rngs = [random.Random(sample_seed(config.seed, item['state_id'], 'projection')) for item in items]
    records = [dict(state_id=item['state_id'], tokens=[], events=[], attempts=0, accepted=0,
                    mean_logprob=None, finished=False) for item in items]

    def propose(indices, cuts, lengths, block, step):
        prefixes = [records[i]['tokens'][:cut] for i, cut in zip(indices, cuts)]
        seeds = [rngs[i].randrange(2**31) for i in indices]
        proposals = backend.propose([items[i] for i in indices], prefixes, lengths, seeds)
        if len(proposals) != len(indices):
            raise ValueError('Incomplete proposal batch')
        for prefix, proposal, length in zip(prefixes, proposals, lengths):
            if proposal[:len(prefix)] != prefix or not len(prefix) < len(proposal) <= len(prefix) + length:
                raise ValueError('Proposal changed the fixed prefix or exceeded its budget')
        scores = backend.score([items[i] for i in indices], proposals)
        if len(scores) != len(indices) or not all(math.isfinite(x) for x in scores):
            raise ValueError('Invalid score batch')
        for i, cut, prefix, candidate, value, seed in zip(indices, cuts, prefixes, proposals, scores, seeds):
            row = records[i]
            before = row['mean_logprob']
            accepted = step is None or value > before
            row['events'].append(dict(block=block, step=step, cut=cut, seed=seed,
                                      suffix=candidate[len(prefix):], previous_mean_logprob=before,
                                      candidate_mean_logprob=value, accepted=accepted))
            if step is not None:
                row['attempts'] += 1
                row['accepted'] += int(accepted)
            if accepted:
                row['tokens'], row['mean_logprob'] = candidate, value

    for block in range(config.max_tokens // config.block_size):
        active = [i for i, row in enumerate(records) if not row['finished']]
        if not active:
            break
        propose(active, [len(records[i]['tokens']) for i in active],
                [config.block_size] * len(active), block, None)
        for step in range(config.steps):
            cuts = [rngs[i].randrange(len(records[i]['tokens'])) for i in active]
            lengths = [len(records[i]['tokens']) - cut for i, cut in zip(active, cuts)]
            propose(active, cuts, lengths, block, step)
        for i in active:
            row = records[i]
            endings = [j for j, token in enumerate(row['tokens']) if token in backend.eos_ids]
            if endings:
                # vLLM must stop at the first EOS; scoring a different sequence
                # from the saved one would invalidate its likelihood.
                if endings[0] != len(row['tokens']) - 1:
                    raise ValueError('Proposal contains tokens after EOS')
                row['finished'] = True
    return records
