"""Restartable dispatch and outcome state for the authorized PEAR++ mixture.

Only an admitted manifest is accepted. This module does not generate resets,
admit scenes, supply hidden state to a policy, or change evaluation sampling.
"""
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import random

try:
    from .efe_pear_curriculum import INITIAL_QUOTAS, REASONING_FAMILIES, SuccessHistory, sgs_probabilities, validate_quotas
    from . import efe_pear_signal_curriculum as signal_curriculum
    from . import efe_pear_curriculum_focus as curriculum_focus
except ImportError:
    from efe_pear_curriculum import INITIAL_QUOTAS, REASONING_FAMILIES, SuccessHistory, sgs_probabilities, validate_quotas
    import efe_pear_signal_curriculum as signal_curriculum
    import efe_pear_curriculum_focus as curriculum_focus


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
        self.sampling_policy=manifest.get('sampling_policy','strict-family-v1')
        if self.sampling_policy not in {'strict-family-v1',signal_curriculum.POLICY}:
            raise ValueError('Unknown curriculum sampling policy')
        if self.sampling_policy==signal_curriculum.POLICY and self.stage!='expanded_full':
            raise ValueError('Signal balancing requires the complete admitted mixture')
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
        self.selection_history=[]
        self.bootstrap_draw_cursor=0
        self.set_sampling_control(manifest.get('warmup_sampling',{}))

    def set_sampling_control(self,value):
        confirmed=value.get('confirmed_training_tasks',[])
        if(confirmed and(value.get('partition')!='train'or
            not set(confirmed)<={c['task']for c in self.configs.values()})):
            raise ValueError('Bootstrap sampling can only prioritize admitted training tasks')
        if value and value.get('decay_updates',8)<=0:
            raise ValueError('Bootstrap sampling needs a positive decay horizon')
        if not set(value.get('probe_config_ids',[]))<=set(self.configs):
            raise ValueError('Bootstrap probes must be admitted training configurations')
        self.warmup_sampling=dict(value)

    @property
    def signal_balancing(self):return self.sampling_policy==signal_curriculum.POLICY

    def select_groups(self,groups):
        if self.signal_balancing:
            return signal_curriculum.select_available(groups,self.quotas,self.selection_history)
        try:from .efe_pear_curriculum import retained_selection
        except ImportError:from efe_pear_curriculum import retained_selection
        return retained_selection(groups,self.quotas,self.stage)

    def record_selection(self,step,uids):
        if not self.signal_balancing:return
        if len(uids)!=128 or len(set(uids))!=128:raise ValueError('A selected batch needs 128 unique groups')
        row={'step':int(step),'uids':sorted(uids),
             'family_counts':dict(Counter(self.groups[uid]['family']for uid in uids))}
        existing=next((v for v in self.selection_history if v['step']==step),None)
        if existing:
            if existing!=row:raise ValueError('A retried step changed its selected groups')
            return
        self.selection_history.append(row)
        del self.selection_history[:-signal_curriculum.WINDOW_UPDATES]

    def batch_loss_weights(self,families):
        return(signal_curriculum.active_weights(self.loss_weights,set(families))
               if self.signal_balancing else self.loss_weights)

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
        control=os.environ.get('EFE_PEAR_SAMPLING_CONTROL')
        if control and Path(control).exists():
            self.set_sampling_control(json.loads(Path(control).read_text()))
        confirmed=set(self.warmup_sampling.get('confirmed_training_tasks',[]))
        if self.warmup_sampling:
            confirmed.update(self.configs[g['config_id']]['task']for g in self.groups.values()
                if g['infrastructure_ok']and any(g['outcomes'])and
                self.configs[g['config_id']]['suite']!='procedural_reasoning')
        probes=self.warmup_sampling.get('probe_config_ids',[])
        forced=self.configs[probes[self.bootstrap_draw_cursor]]if self.signal_balancing and self.bootstrap_draw_cursor<len(probes)else None
        if forced:self.bootstrap_draw_cursor+=1
        if self.signal_balancing:
            weights=signal_curriculum.dispatch_weights(self.quotas,self.selection_history,self.family_outcomes,inflight)
            if self.warmup_sampling:
                weights=curriculum_focus.focus_families(weights,self.quotas,self.configs,
                    confirmed,self.groups,
                    step=step,decay_updates=self.warmup_sampling.get('decay_updates',8))
        else:weights={}
        for family,quota in (()if self.signal_balancing else self.quotas.items()):
            recent=self.family_outcomes.get(family,())[-100:]
            acceptance=(sum(x['retained'] for x in recent)+1)/(len(recent)+2)
            deficit=max(0,quota-ready.get(family,0))
            weights[family]=max(0,deficit/max(.1,acceptance)-inflight.get(family,0))
        if not any(weights.values()):
            weights={f:max(0,n-ready.get(f,0)) for f,n in self.quotas.items()}
        if not any(weights.values()):weights=dict(self.quotas)
        family=forced['family']if forced else self.rng.choices(list(weights),weights=list(weights.values()),k=1)[0]
        pool=[c for c in self.configs.values() if c['family']==family]
        # Balance coverage hierarchically. Reset-bank size never changes a
        # simulator or semantic task's sampling weight.
        def least_used(field, rows):
            counts={x[field]:0 for x in rows}
            for c in rows:counts[c[field]]+=self.dispatches[c['config_id']]
            minimum=min(counts.values())
            return self.rng.choice(sorted(k for k,v in counts.items() if v==minimum))
        task_probability=None
        if self.signal_balancing:
            tasks={c['config_id']:(c['suite'],c['task'])for c in pool};recent=defaultdict(lambda:deque(maxlen=100))
            for g in self.groups.values():
                if g['config_id']in tasks and g['infrastructure_ok']:
                    recent[tasks[g['config_id']]].extend(g['outcomes'])
            rates={name:(sum(recent[name])+1)/(len(recent[name])+2)for name in sorted(set(tasks.values()))}
            learned=sgs_probabilities(rates,uniform_fraction=0.)
            suite_counts=Counter(key[0]for key in rates)
            base={key:1/(len(suite_counts)*suite_counts[key[0]])for key in rates}
            total=sum(base[key]*learned[key]for key in rates)
            task_probabilities={key:.2*base[key]+.8*base[key]*learned[key]/total for key in rates}
            if self.warmup_sampling:
                focus_names={name for name in rates if name[1]in confirmed}
                task_probabilities=curriculum_focus.focus_tasks(task_probabilities,base,focus_names,rates,
                    step=step,decay_updates=self.warmup_sampling.get('decay_updates',8))
            suite,task=self.rng.choices(sorted(task_probabilities),weights=[task_probabilities[k]for k in sorted(task_probabilities)],k=1)[0]
            task_probability=task_probabilities[suite,task]
        else:
            suite=least_used('suite',pool);pool=[c for c in pool if c['suite']==suite]
            task=least_used('task',pool)
        if forced:suite,task=forced['suite'],forced['task'];task_probability=1.
        pool=[c for c in pool if c['suite']==suite and c['task']==task]
        counts=Counter()
        for c in pool:counts[c['stratum']]+=self.dispatches[c['config_id']]
        full_fraction=.75 if step>=256 else .5
        assisted=sorted({c['stratum'] for c in pool}-{'full'})
        stratum=('full' if not assisted or counts['full']<=full_fraction*sum(counts.values())
                 else self.rng.choice(assisted))
        if forced:stratum=forced['stratum']
        pool=[c for c in pool if c['stratum']==stratum]
        rates={}
        for c in pool:
            values=self.history.history.get(c.get('history_id',c['config_id']),())
            # Beta(1,1) shrinking avoids treating one sparse group as certainty.
            rates[c['config_id']]=(sum(values)+1)/(len(values)+2)
        probabilities=sgs_probabilities(rates)
        chosen=self.rng.choices(sorted(probabilities),
            weights=[probabilities[k] for k in sorted(probabilities)],k=1)[0]
        if forced:chosen=forced['config_id']
        self.last_draw={'family':family,'suite':suite,'task':task,'stratum':stratum,
            'selected_config_id':chosen,'selected_probability':probabilities[chosen],
            'estimated_success':rates[chosen],'reset_pool_size':len(probabilities),
            'effective_sample_size':1/sum(p*p for p in probabilities.values()),
            'uniform_fraction':.2,'history_horizon':self.history.horizon,
            'ready_groups':dict(ready),'inflight_groups':dict(inflight),
            'quotas':dict(self.quotas),'full_start_target_fraction':full_fraction}
        self.last_draw.update(sampling_policy=self.sampling_policy,task_probability=task_probability,
                              sgs_scope='task_and_reset'if self.signal_balancing else'reset_only',
                              warmup_sampling=self.warmup_sampling,planned_training_probe=bool(forced))
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
                'adaptations':self.adaptations,'selection_history':self.selection_history,
                'warmup_sampling':self.warmup_sampling,'bootstrap_draw_cursor':self.bootstrap_draw_cursor}

    def restore(self, state):
        if state['manifest_sha256']!=self.manifest_sha256:
            raise ValueError('Resume changed the admitted curriculum')
        validate_quotas(state['quotas'],self.stage)
        self.rng.setstate(_tuples(state['rng']));self.quotas=dict(state['quotas'])
        self.history=SuccessHistory.from_state_dict(state['history'])
        self.groups=dict(state['groups']);self.dispatches=Counter(state['dispatches'])
        self.family_outcomes=defaultdict(list,state['family_outcomes'])
        self.adaptations=list(state['adaptations'])
        self.selection_history=list(state.get('selection_history',[]))
        self.set_sampling_control(state.get('warmup_sampling',self.warmup_sampling))
        self.bootstrap_draw_cursor=state.get('bootstrap_draw_cursor',0)
