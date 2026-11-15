"""PEAR-style REINFORCE with optional top-k score centering.

Formula: https://github.com/martin-marek/score-centering (README, PyTorch).
This is an EFE adaptation, not the upstream PEAR training engine. Inputs are
generated tokens only; q is the actual sampler distribution, never rescored
with the training engine. The unseen tail is modeled proportional to p.
"""
from __future__ import annotations
import math


def group_advantages(rewards, normalize=True):
    """Complete same-task/reset groups only; a constant group is DAPO-filtered."""
    if len(rewards) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in rewards):
        raise ValueError('A complete group of finite native rewards is required')
    mean = sum(rewards) / len(rewards)
    centered = [x - mean for x in rewards]
    if max(rewards) == min(rewards):
        return None
    scale = math.sqrt(sum(x*x for x in centered)/(len(rewards)-1)) + 1e-6 if normalize else 1.0
    return [x / scale for x in centered]


def validate_sampling(record, vocab_size, top_k=128):
    """Reject lossy or renormalized captures before they can enter training."""
    ids = record['token_ids']
    logq = record['sampled_logprobs']
    heads = record['topk_token_ids']
    values = record['topk_logprobs']
    if not ids or len({len(ids),len(logq),len(heads),len(values)}) != 1:
        raise ValueError('Sampler probability lengths differ')
    if record.get('temperature') != 1.0 or record.get('top_p') != 1.0 or record.get('top_k') != -1:
        raise ValueError('Qualification requires temperature 1, full-distribution sampling')
    if not record.get('sampler_revision') or not record.get('prompt_token_ids'):
        raise ValueError('Sampler revision and exact input token IDs are required')
    for token, sampled, head, probs in zip(ids, logq, heads, values):
        if type(token) is not int or not 0 <= token < vocab_size or not math.isfinite(sampled) or sampled > 1e-5:
            raise ValueError('Invalid sampled token or probability')
        if len(head) != top_k or len(probs) != top_k or len(set(head)) != top_k:
            raise ValueError('Top-k head must contain distinct exact token IDs')
        if any(type(t) is not int or not 0 <= t < vocab_size for t in head):
            raise ValueError('Invalid top-k token ID')
        if any(not math.isfinite(p) or p > 1e-5 for p in probs) or sum(map(math.exp, probs)) > 1.00005:
            raise ValueError('Invalid or renormalized sampling mass')
        if token in head and abs(probs[head.index(token)] - sampled) > 1e-5:
            raise ValueError('Sampled-token probability disagrees with the head')
    return len(ids)


def token_losses(train_logp, token_ids, advantages, *, score_centering,
                 topk_ids=None, topk_logq=None, eps=1e-6):
    """Per-token loss. Call on FP32 log-softmax in bounded vocabulary chunks.

    No PPO clipping, importance weights, KL or entropy bonus. Both arm variants
    use the same reduction: sum over generated tokens / global token count.
    """
    import torch
    if train_logp.dtype != torch.float32:
        raise ValueError('Score centering requires float32 log probabilities')
    selected = train_logp.gather(-1, token_ids[..., None])[..., 0]
    if score_centering:
        if topk_ids is None or topk_logq is None:
            raise ValueError('Score centering requires original sampler top-k logprobs')
        head = train_logp.gather(-1, topk_ids)
        p_head, q_head = head.exp(), topk_logq.detach().float().exp()
        rho = (1-q_head.sum(-1)).clamp_min(eps)/(1-p_head.sum(-1)).clamp_min(eps)
        residual = q_head - rho[..., None]*p_head
        correction = (residual.detach()*head).sum(-1)
        selected = selected-correction
    return -advantages.detach()*selected


def qualify(device='cuda'):
    """Numerical gradient checks, including drift under a mismatched sampler."""
    import torch
    torch.manual_seed(14)
    checks = {}
    # Constant reward must produce zero expected gradient after centering,
    # even when the sampler and trainer distributions differ.
    logits = torch.tensor([-.8,.4,1.2,-.2],device=device,requires_grad=True)
    q = torch.tensor([.5,.2,.2,.1],device=device)
    logp = logits.log_softmax(-1)
    ids = torch.arange(4,device=device)
    full = logp.expand(4,-1)
    head = ids.expand(4,-1)
    q_head = q.log().expand(4,-1)
    centered = token_losses(full,ids,torch.ones(4,device=device),score_centering=True,topk_ids=head,topk_logq=q_head)
    grad = torch.autograd.grad((q*centered).sum(),logits,retain_graph=True)[0]
    raw = token_losses(full,ids,torch.ones(4,device=device),score_centering=False)
    raw_grad = torch.autograd.grad((q*raw).sum(),logits)[0]
    checks['mismatch_centered_max_gradient'] = float(grad.abs().max())
    checks['mismatch_uncentered_max_gradient'] = float(raw_grad.abs().max())
    assert checks['mismatch_centered_max_gradient'] < 2e-6
    assert checks['mismatch_uncentered_max_gradient'] > .05
    # Verify the top-k tail approximation against its full-vocabulary formula.
    logits = torch.randn(11,device=device,requires_grad=True)
    logp = logits.log_softmax(-1);p=logp.detach().exp()
    head_ids=torch.tensor([1,4,8],device=device);qh=torch.tensor([.1,.2,.3],device=device)
    q=(.4/(1-p[head_ids].sum()))*p;q[head_ids]=qh
    adv=torch.tensor(1.7,device=device);sample=torch.tensor(10,device=device)
    sparse=token_losses(logp,sample,adv,score_centering=True,topk_ids=head_ids,topk_logq=qh.log())
    dense=-adv*(logp[sample]-(q.detach()*logp).sum())
    a=torch.autograd.grad(sparse,logits,retain_graph=True)[0];b=torch.autograd.grad(dense,logits)[0]
    checks['tail_approximation_gradient_error']=float((a-b).abs().max())
    assert checks['tail_approximation_gradient_error'] < 2e-6
    assert group_advantages([0]*8) is None and group_advantages([1]*8) is None
    assert abs(sum(group_advantages([0,0,0,0,1,1,1,1]))) < 1e-6
    checks['constant_groups_filtered']=True
    checks['success']=True
    return checks
