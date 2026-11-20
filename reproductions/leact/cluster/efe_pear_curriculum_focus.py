"""Soft bootstrap sampling from confirmed training tasks; no batch admission gate."""
from collections import defaultdict

try:
    from .efe_pear_curriculum import usable_group_probability
except ImportError:
    from efe_pear_curriculum import usable_group_probability


def strength(step,decay_updates):
    return max(0.,1.-max(0,step-1)/decay_updates)


def focus_tasks(probabilities,base,confirmed,rates,*,step,decay_updates=8):
    selected=set(probabilities)&set(confirmed)
    if not selected:return dict(probabilities)
    learnability=max(usable_group_probability(rates[k])for k in selected)/usable_group_probability(.5)
    amount=strength(step,decay_updates)*learnability
    total=sum(probabilities[k]for k in selected)
    # The original task sampler already contains a 20% coverage component.
    # Mix focus into its adaptive component, preserving that full 20% floor.
    return{k:.2*base[k]+.8*((1-amount)*max(0.,(probabilities[k]-.2*base[k])/.8)
        +amount*(probabilities[k]/total if k in selected else 0.))for k in probabilities}


def focus_families(probabilities,targets,configs,confirmed,groups,*,step,decay_updates=8):
    confirmed=set(confirmed);histories=defaultdict(list)
    for g in groups.values():
        c=configs[g['config_id']]
        if c['task']in confirmed and g['infrastructure_ok']:histories[c['task']].extend(g['outcomes'])
    scores=defaultdict(float)
    for c in configs.values():
        if c['task']not in confirmed:continue
        values=histories[c['task']][-100:];p=(sum(values)+1)/(len(values)+2)
        scores[c['family']]=max(scores[c['family']],usable_group_probability(p)/usable_group_probability(.5))
    if not scores:return dict(probabilities)
    target_total=sum(targets.values());base={f:n/target_total for f,n in targets.items()}
    adaptive={f:max(0.,(probabilities[f]-.2*base[f])/.8)*(
        1+7*strength(step,decay_updates)*scores[f])for f in probabilities}
    total=sum(adaptive.values())
    return{f:.2*base[f]+.8*adaptive[f]/total for f in probabilities}
