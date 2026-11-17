"""Deterministic curriculum arithmetic for PEAR++, independent of GPU libraries.

This module does not admit environments, create resets, or replace the native
success verifier. Callers must supply admitted, nonterminal reset identities and
all valid on-policy outcomes, including constant-reward groups discarded by GRPO.
"""
from collections import deque
from dataclasses import dataclass, field
import math


REASONING_FAMILIES = (
    'object_pointing', 'set_pointing_counting', 'spatial_relations_frames',
    'valid_placement', 'cross_view_correspondence', 'affordance_part_selection',
    'physical_prediction', 'active_perception_memory',
)
INITIAL_QUOTAS = {'native_gr00t_yam':32, 'broader_bimanual':24, 'general_manipulation':24,
                  'language_conditioned_control':16, **{key:4 for key in REASONING_FAMILIES}}
FLOORS = {'native_gr00t_yam':24, 'broader_bimanual':16, 'general_manipulation':16,
          'language_conditioned_control':8, **{key:2 for key in REASONING_FAMILIES}}


def validate_quotas(quotas):
    if set(quotas) != set(INITIAL_QUOTAS):
        raise ValueError('Every authorized control and reasoning family is required')
    if any(type(n) is not int or n < FLOORS[key] for key,n in quotas.items()):
        raise ValueError('Family coverage floor violated')
    if sum(quotas.values()) != 128 or sum(quotas[k] for k in REASONING_FAMILIES) != 32:
        raise ValueError('The retained batch must contain 96 control and 32 reasoning groups')


def usable_group_probability(p, n=8):
    if not 0 <= p <= 1 or n < 2:
        raise ValueError('Invalid binary success probability or group size')
    return 1 - p**n - (1-p)**n


def sgs_probabilities(rates, *, uniform_fraction=.2, target=.5, kappa=1.,
                      temperature=2., epsilon=1e-4):
    """SGS weights with a uniform floor; an unseen reset has neutral p=0.5.

    This is applied *within* a task and a preselected start-state stratum. It must
    not silently favor tasks, easier start strata, or evaluation instances.
    """
    if not rates or not 0 <= uniform_fraction <= 1 or not 0 <= target <= 1:
        raise ValueError('Empty reset pool or invalid mixing/target probability')
    if temperature <= 0 or kappa <= 0 or epsilon <= 0:
        raise ValueError('Invalid SGS shape parameters')
    weights={}
    for key, rate in rates.items():
        p=.5 if rate is None else float(rate)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('SGS requires binary-success rates')
        score=(p+epsilon)**(kappa*target) * (1-p+epsilon)**(kappa*(1-target))
        weights[key]=(max(score,epsilon)+epsilon)**(1/temperature)
    total=sum(weights.values());uniform=uniform_fraction/len(weights)
    return {key:uniform+(1-uniform_fraction)*w/total for key,w in weights.items()}


@dataclass
class SuccessHistory:
    """Restartable histories keyed by an immutable task/reset/stratum identity."""
    horizon: int = 100
    history: dict = field(default_factory=dict)
    judged_groups: set = field(default_factory=set)

    def observe(self, uid, config_id, outcomes, *, infrastructure_ok=True):
        if uid in self.judged_groups:
            return False
        if not infrastructure_ok:
            # An interrupted group must not be recorded as eight policy failures.
            return False
        values=list(outcomes)
        if len(values) != 8 or any(x not in (0,1) for x in values):
            raise ValueError('Only complete, officially graded groups of eight are valid')
        q=deque(self.history.get(config_id,()), maxlen=self.horizon)
        q.extend(float(x) for x in values)
        self.history[config_id]=list(q)
        self.judged_groups.add(uid)
        return True

    def rate(self, config_id):
        values=self.history.get(config_id,())
        return sum(values)/len(values) if values else None

    def state_dict(self):
        return {'horizon':self.horizon,'history':self.history,
                'judged_groups':sorted(self.judged_groups)}

    @classmethod
    def from_state_dict(cls, value):
        if value['horizon'] < 8:
            raise ValueError('History must fit a complete group')
        return cls(value['horizon'],{k:list(v) for k,v in value['history'].items()},
                   set(value['judged_groups']))


def choose_reset(configs, history, rng, *, task, stratum, partition='train'):
    """Select an admitted reset; callers choose the task and start stratum first."""
    if partition != 'train':
        raise ValueError('SGS must never change frozen evaluation sampling')
    eligible=[c for c in configs if c['task']==task and c['stratum']==stratum]
    if not eligible or any(c.get('partition')!='train' or not c.get('admitted')
                           or c.get('already_successful') or not c.get('observation_complete')
                           for c in eligible):
        raise ValueError('Missing, unqualified, terminal, or unobservable reset')
    ids=sorted(c['config_id'] for c in eligible)
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate reset identities')
    probabilities=sgs_probabilities({key:history.rate(key) for key in ids})
    selected=rng.choices(ids,weights=[probabilities[key] for key in ids],k=1)[0]
    return selected, {'probabilities':probabilities,'adaptive_choice':len(ids)>1,
                      'task':task,'stratum':stratum}


def retained_selection(groups, quotas):
    """Oldest usable groups satisfying exact quotas, or None if a family is short.

    Group rewards are checked here even if the upstream DAPO filter already ran.
    No compensating quota transfer to an easier family is allowed.
    """
    validate_quotas(quotas)
    by_family={key:[] for key in quotas}
    ids=set()
    for group in groups:
        uid=group['uid']
        if uid in ids:
            raise ValueError('Duplicate prompt group')
        ids.add(uid)
        family=group['family'];values=group['outcomes']
        if family not in quotas:
            raise ValueError('Unknown curriculum family')
        if not group.get('infrastructure_ok'):
            continue
        if len(values)!=8 or any(x not in (0,1) for x in values):
            raise ValueError('Incomplete or nonbinary group')
        if min(values)==max(values):
            continue
        by_family[family].append(group)
    if any(len(by_family[key])<n for key,n in quotas.items()):
        return None
    selected=[]
    for family,n in quotas.items():
        selected.extend(sorted(by_family[family],key=lambda g:(g['dispatch_step'],g['uid']))[:n])
    return [g['uid'] for g in selected]


def token_mean_factors(global_generated_tokens, family_weights):
    """Scale advantages once to turn a global token mean into family means.

    L = sum_f w_f sum_{tokens in f}(loss_t)/T_f. The upstream actor divides by
    T=sum_f T_f, so multiply each family's *post-GRPO-normalization* advantages
    by w_f*T/T_f. Counts must cover the entire optimizer batch across DP ranks
    and include only generated tokens. Do not also weight the response mask.
    """
    if set(global_generated_tokens)!=set(family_weights):
        raise ValueError('Every weighted family must have a global token denominator')
    if not global_generated_tokens or any(type(n) is not int or n<=0 for n in global_generated_tokens.values()):
        raise ValueError('A family has no generated tokens')
    if any(not math.isfinite(w) or w<=0 for w in family_weights.values()):
        raise ValueError('Family loss weights must be finite and positive')
    if not math.isclose(sum(family_weights.values()),1.,rel_tol=0,abs_tol=1e-8):
        raise ValueError('Family loss weights must sum to one')
    total=sum(global_generated_tokens.values())
    return {key:family_weights[key]*total/n for key,n in global_generated_tokens.items()}


def allocate_bounded_quotas(scores, *, floors=FLOORS):
    """Largest-remainder allocation inside the fixed 96/32 control/reason split.

    The caller supplies measured learning-progress scores. This arithmetic does
    not invent a progress metric or read any held-out evaluation results.
    """
    if set(scores)!=set(INITIAL_QUOTAS) or set(floors)!=set(scores):
        raise ValueError('All families are required')
    if any(not math.isfinite(s) or s<0 for s in scores.values()):
        raise ValueError('Learning-progress scores must be nonnegative and finite')
    result=dict(floors)
    for keys,budget in [(sorted(set(scores)-set(REASONING_FAMILIES)),96),
                        (sorted(REASONING_FAMILIES),32)]:
        left=budget-sum(result[k] for k in keys)
        if left<0:raise ValueError('Coverage floors exceed the category budget')
        total=sum(scores[k] for k in keys)
        shares={k:left*(scores[k]/total if total else 1/len(keys)) for k in keys}
        for key in keys:result[key]+=math.floor(shares[key])
        spare=budget-sum(result[k] for k in keys)
        order=sorted(keys,key=lambda k:(-(shares[k]-math.floor(shares[k])),k))
        for key in order[:spare]:result[key]+=1
    validate_quotas(result)
    return result
