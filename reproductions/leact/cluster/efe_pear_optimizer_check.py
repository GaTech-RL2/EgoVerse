"""Optional observation of PEAR's actual AdamW step during qualification."""
import functools
import json
import os
from pathlib import Path


def install():
    if os.environ.get('PEAR_PRIMARY_TRAINING')!='0':
        raise ValueError('Diagnostic optimizer instrumentation is qualification-only')
    import torch
    import torch.distributed as dist
    from pear.trainer.engine import PearFSDPEngine
    from efe_pear_router import ROUTER,assert_expert_routers_frozen
    original=PearFSDPEngine._optimizer_update
    if getattr(original,'_rp_jhh_checked',False):return

    def local(tensor):return tensor.to_local()if hasattr(tensor,'to_local')else tensor

    @functools.wraps(original)
    def checked(engine):
        assert_expert_routers_frozen(engine.module)
        routers={n:local(p.detach()).cpu().clone()for n,p in engine.module.named_parameters()if ROUTER.search(n)}
        candidates=[]
        for name,p in engine.module.named_parameters():
            if ROUTER.search(name):
                if p.grad is not None:raise ValueError('Frozen router acquired a gradient')
                continue
            if p.grad is None:continue
            grad=local(p.grad).detach().flatten()
            if not grad.numel()or grad.numel()>65536:continue
            value,index=grad.abs().max(dim=0)
            if float(value)>0:
                candidates.append((name,p,int(index),local(p.detach()).flatten()[index].cpu().clone()))
                if len(candidates)==4:break
        if not candidates:raise ValueError('No local trainable gradient probe is available')
        result=original(engine)
        unchanged=all(torch.equal(old,local(dict(engine.module.named_parameters())[name].detach()).cpu())for name,old in routers.items())
        changed=any(not torch.equal(old,local(p.detach()).flatten()[index].cpu())for _,p,index,old in candidates)
        value={'rank':dist.get_rank(),'routers_unchanged':unchanged,'trainable_probe_changed':changed,
               'router_count':len(routers),'primary_rl_updates':0,'discarded_optimizer_updates':1}
        path=Path(os.environ['EFE_PEAR_OPTIMIZER_CHECK_DIR']);path.mkdir(parents=True,exist_ok=True)
        (path/f'rank-{dist.get_rank()}.json').write_text(json.dumps(value)+'\n')
        if not unchanged or not changed:raise ValueError('Qualification optimizer invariant failed')
        return result
    checked._rp_jhh_checked=True
    PearFSDPEngine._optimizer_update=checked
