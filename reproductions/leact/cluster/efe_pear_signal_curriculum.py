"""Opt-in task curriculum and balancing of available GRPO signal over updates."""
from collections import Counter,defaultdict

POLICY='task-signal-window-v2'
WINDOW_UPDATES=8


def active_weights(targets,present):
    if not present or not set(present)<=set(targets):raise ValueError('Unknown or empty active family set')
    total=sum(targets[f]for f in present)
    if total<=0:raise ValueError('Active families have no target weight')
    return{f:targets[f]/total for f in present}


def window_counts(history):
    counts=Counter()
    for row in history[-(WINDOW_UPDATES-1):]:counts.update(row['family_counts'])
    return counts


def select_available(groups,targets,history):
    """Keep the full 128x8 batch; an unavailable family cannot block it."""
    by_family=defaultdict(list);seen=set()
    for group in groups:
        uid=group['uid'];family=group['family'];values=group['outcomes']
        if uid in seen:raise ValueError('Duplicate prompt group')
        seen.add(uid)
        if family not in targets:raise ValueError('Unknown curriculum family')
        if not group.get('infrastructure_ok'):continue
        if len(values)!=8 or any(v not in (0,1)for v in values):raise ValueError('Incomplete or nonbinary group')
        if min(values)==max(values):continue
        by_family[family].append(group)
    if sum(map(len,by_family.values()))<128:return None
    for rows in by_family.values():rows.sort(key=lambda g:(g['dispatch_step'],g['uid']))
    previous=window_counts(history);selected=[];current=Counter()
    length=min(len(history),WINDOW_UPDATES-1)+1
    for _ in range(128):
        available=[f for f,rows in by_family.items()if current[f]<len(rows)]
        # Debt is normalized by the target so large families cannot permanently
        # crowd out a small family when its first informative group appears.
        family=max(available,key=lambda f:(
            (targets[f]*length-previous[f]-current[f])/targets[f],
            -by_family[f][current[f]]['dispatch_step'],f))
        selected.append(by_family[family][current[family]]['uid']);current[family]+=1
    return selected


def dispatch_weights(targets,history,outcomes,inflight,*,uniform_fraction=.2):
    """A full-mixture exploration floor plus signal- and debt-aware effort."""
    if not 0<uniform_fraction<=1:raise ValueError('A positive exploration floor is required')
    previous=window_counts(history);length=min(len(history),WINDOW_UPDATES-1)+1
    scores={}
    for family,target in targets.items():
        recent=outcomes.get(family,())[-100:]
        acceptance=(sum(v['retained']for v in recent)+1)/(len(recent)+2)
        debt=max(.25,(target*length-previous[family])/target)
        scores[family]=target*acceptance*min(4.,debt)/(1+inflight.get(family,0))
    total=sum(scores.values());target_total=sum(targets.values())
    return{f:uniform_fraction*targets[f]/target_total+(1-uniform_fraction)*scores[f]/total for f in targets}
