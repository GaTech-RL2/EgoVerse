"""Restartable dispatch and outcome state for the authorized PEAR++ mixture.

Only an admitted manifest is accepted. This module does not generate resets,
admit scenes, supply hidden state to a policy, or change evaluation sampling.
"""
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import random

try:
    from .efe_pear_curriculum import INITIAL_QUOTAS, REASONING_FAMILIES, SuccessHistory, sgs_probabilities, validate_quotas
except ImportError:
    from efe_pear_curriculum import INITIAL_QUOTAS, REASONING_FAMILIES, SuccessHistory, sgs_probabilities, validate_quotas


def identity(env_class, env_args):
    return json.dumps([env_class, env_args], sort_keys=True, separators=(',', ':'))


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


class CurriculumState:
    def __init__(self, manifest, seed=1701):
        if manifest.get('partition') != 'train' or manifest.get('qualification') != 'passed':
            raise ValueError('A qualified training manifest is required')
        self.stage=manifest.get('curriculum_stage','expanded_full')
        if self.stage not in {'expanded_full','control_bootstrap'}:raise ValueError('Unknown curriculum stage')
        self.manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        self.configs={}
        seen_envs=set()
        for entry in manifest['configs']:
            c=dict(entry);key=c['config_id']
            if (key in self.configs or c['family'] not in INITIAL_QUOTAS or
                c.get('partition') != 'train' or not c.get('admitted') or
                c.get('already_successful') or not c.get('observation_complete') or
                not c.get('scene_sha256') or not c.get('admission_receipt_sha256')):
                raise ValueError('Unqualified, duplicate, terminal or unobservable reset: '+key)
            if c['stratum'] not in {'full','grasped','near_goal'}:
                raise ValueError('Unknown reset stratum')
            if self.stage=='control_bootstrap'and(c['family']!='general_manipulation'or c['suite']!='metaworldplus'):
                raise ValueError('Control bootstrap is restricted to existing MetaWorld training tasks')
            for required in ['suite','task','env_class','env_args']:
                if not c.get(required):raise ValueError('Missing reset field: '+required)
            env_id=identity(c['env_class'],c['env_args'])
            if env_id in seen_envs:raise ValueError('Duplicate reset identity under different config IDs')
            seen_envs.add(env_id);self.configs[key]=c
        required={'general_manipulation'}if self.stage=='control_bootstrap'else set(INITIAL_QUOTAS)
        if {c['family'] for c in self.configs.values()} != required:
            raise ValueError('Every authorized control and reasoning family must be admitted')
        tasks={(c['family'],c['suite'],c['task']) for c in self.configs.values()}
        full={(c['family'],c['suite'],c['task']) for c in self.configs.values() if c['stratum']=='full'}
        if tasks != full:raise ValueError('Every semantic task needs a native full start')
        self.rng=random.Random(seed)
        self.quotas={'general_manipulation':128}if self.stage=='control_bootstrap'else dict(INITIAL_QUOTAS)
        self.history=SuccessHistory()
        self.groups={}
        self.dispatches=Counter()
        self.family_outcomes=defaultdict(list)
        self.adaptations=[]
        self.last_draw=None

    @property
    def loss_weights(self):
        return {key:n/128 for key,n in self.quotas.items()}

    def register(self, uid, config_id, dispatch_step):
        if uid in self.groups:
            old=self.groups[uid]
            if old['config_id']!=config_id:raise ValueError('Reissued group changed reset')
            return
        c=self.configs[config_id]
        self.groups[uid]={'uid':uid,'config_id':config_id,'family':c['family'],
                          'dispatch_step':dispatch_step,'infrastructure_ok':False,'outcomes':[]}
        self.dispatches[config_id]+=1

    def observe(self, uid, outcomes):
        group=self.groups[uid];c=self.configs[group['config_id']]
        # Sparse procedural scene histories may share an audited generator /
        # template / difficulty-bin ID. Control reset identities stay distinct.
        history_id=c.get('history_id',c['config_id'])
        if not self.history.observe(uid,history_id,outcomes):return
        values=list(outcomes)
        group.update(outcomes=values,infrastructure_ok=True)
        series=self.family_outcomes[c['family']]
        series.append({'uid':uid,'step':group['dispatch_step'],
                       'successes':sum(values),'retained':min(values)!=max(values)})
        del series[:-1000]

    def draw(self, ready=None, inflight=None, *, step=0):
        """Choose family, simulator, semantic task, start stratum, then reset.

        The replay buffer enforces final quotas; bounded acceptance correction
        affects dispatch effort only. It cannot replace a missing family.
        """
        ready=ready or {};inflight=inflight or {}
        weights={}
        for family,quota in self.quotas.items():
            recent=self.family_outcomes.get(family,())[-100:]
            acceptance=(sum(x['retained'] for x in recent)+1)/(len(recent)+2)
            deficit=max(0,quota-ready.get(family,0))
            weights[family]=max(0,deficit/max(.1,acceptance)-inflight.get(family,0))
        if not any(weights.values()):
            weights={f:max(0,n-ready.get(f,0)) for f,n in self.quotas.items()}
        if not any(weights.values()):weights=dict(self.quotas)
        family=self.rng.choices(list(weights),weights=list(weights.values()),k=1)[0]
        pool=[c for c in self.configs.values() if c['family']==family]
        # Balance coverage hierarchically. Reset-bank size never changes a
        # simulator or semantic task's sampling weight.
        def least_used(field, rows):
            counts={x[field]:0 for x in rows}
            for c in rows:counts[c[field]]+=self.dispatches[c['config_id']]
            minimum=min(counts.values())
            return self.rng.choice(sorted(k for k,v in counts.items() if v==minimum))
        suite=least_used('suite',pool);pool=[c for c in pool if c['suite']==suite]
        task=least_used('task',pool);pool=[c for c in pool if c['task']==task]
        counts=Counter()
        for c in pool:counts[c['stratum']]+=self.dispatches[c['config_id']]
        full_fraction=.75 if step>=256 else .5
        assisted=sorted({c['stratum'] for c in pool}-{'full'})
        stratum=('full' if not assisted or counts['full']<=full_fraction*sum(counts.values())
                 else self.rng.choice(assisted))
        pool=[c for c in pool if c['stratum']==stratum]
        rates={}
        for c in pool:
            values=self.history.history.get(c.get('history_id',c['config_id']),())
            # Beta(1,1) shrinking avoids treating one sparse group as certainty.
            rates[c['config_id']]=(sum(values)+1)/(len(values)+2)
        probabilities=sgs_probabilities(rates)
        chosen=self.rng.choices(sorted(probabilities),
            weights=[probabilities[k] for k in sorted(probabilities)],k=1)[0]
        self.last_draw={'family':family,'suite':suite,'task':task,'stratum':stratum,
            'selected_config_id':chosen,'selected_probability':probabilities[chosen],
            'estimated_success':rates[chosen],'reset_pool_size':len(probabilities),
            'effective_sample_size':1/sum(p*p for p in probabilities.values()),
            'uniform_fraction':.2,'history_horizon':self.history.horizon,
            'ready_groups':dict(ready),'inflight_groups':dict(inflight),
            'quotas':dict(self.quotas),'full_start_target_fraction':full_fraction}
        return chosen

    def adapt(self, quotas, *, step, evidence):
        """Accept measured training-only updates, never invent missing signals."""
        validate_quotas(quotas,self.stage)
        if step<32 or step%16 or evidence.get('partition')!='train':
            raise ValueError('Adaptation requires the scheduled training-only measurement')
        if evidence.get('step')!=step or set(evidence.get('families',{}))!=set(self.quotas):
            raise ValueError('Missing aligned per-family progress evidence')
        for family,old in self.quotas.items():
            # Integer quotas permit at least one group of movement.
            if abs(quotas[family]-old)>max(1,math.floor(.2*old)):
                raise ValueError('Family quota change exceeds the relative limit')
            row=evidence['families'][family]
            if (row.get('probe_episodes',0)<1 or
                not math.isfinite(row.get('success_rate',float('nan'))) or
                not math.isfinite(row.get('surrogate_improvement',float('nan')))):
                raise ValueError('Training probes and pre/post surrogate measurements are required')
        self.adaptations.append({'step':step,'quotas':dict(quotas),'evidence':evidence})
        self.quotas=dict(quotas)

    def state_dict(self):
        return {'manifest_sha256':self.manifest_sha256,'rng':self.rng.getstate(),
                'quotas':self.quotas,'history':self.history.state_dict(),'groups':self.groups,
                'dispatches':dict(self.dispatches),'family_outcomes':dict(self.family_outcomes),
                'adaptations':self.adaptations}

    def restore(self, state):
        if state['manifest_sha256']!=self.manifest_sha256:
            raise ValueError('Resume changed the admitted curriculum')
        validate_quotas(state['quotas'],self.stage)
        self.rng.setstate(_tuples(state['rng']));self.quotas=dict(state['quotas'])
        self.history=SuccessHistory.from_state_dict(state['history'])
        self.groups=dict(state['groups']);self.dispatches=Counter(state['dispatches'])
        self.family_outcomes=defaultdict(list,state['family_outcomes'])
        self.adaptations=list(state['adaptations'])
