"""Future MoE post-training policy: freeze expert-selection router parameters."""
import re

ROUTER=re.compile(r'(?:^|\.)mlp\.gate\.(?:weight|bias)$')


def freeze_expert_routers(model):
    frozen=[];parameters=0
    for name,value in model.named_parameters():
        if ROUTER.search(name):
            value.requires_grad_(False);frozen.append(name);parameters+=value.numel()
    if not frozen:raise ValueError('No expert-selection router matched; qualify this architecture explicitly')
    return {'policy':'frozen_expert_selection_router','names':frozen,'parameters':parameters,
        'scope':'Expert-selection gates only; other trainable weights retain their existing policy.',
        'routing_replay':False}


def assert_expert_routers_frozen(model):
    routers=[(name,p)for name,p in model.named_parameters()if ROUTER.search(name)]
    if not routers or any(p.requires_grad for _,p in routers):
        raise ValueError('Expert-selection routers are missing or trainable')
