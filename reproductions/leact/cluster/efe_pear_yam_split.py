"""Freeze native YAM task groups while protecting every authored saved layout.

This selects task templates, not executable training resets. Native release
layouts stay evaluation-only, including for tasks whose templates may train.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import yaml

BASE=Path(__file__).resolve().parents[1]


def task_group(name,assignments):
    if name in assignments:return assignments[name]
    base=re.sub(r'_(?:contact_a[0-9]+|hold_a[0-9]+)$','',name)
    if base not in assignments:raise ValueError('Unreviewed native task or alias: '+name)
    return assignments[base]


def build(source,revision,policy):
    if revision!=policy['source_revision']:raise ValueError('Re-review the split for a changed native YAM revision')
    test_groups=set(policy['test_groups']);development_groups=set(policy['development_groups'])
    if test_groups&development_groups:raise ValueError('Test and development semantic groups overlap')
    if (test_groups|development_groups)-set(policy['group_assignments'].values()):
        raise ValueError('A reserved semantic group is missing from the native catalog')
    source=Path(source);root=source/'gr00t_sim/tasks/env_graphs/yam';tasks=defaultdict(list)
    for path in sorted(root.rglob('*.yaml')):
        raw=path.read_bytes();data=yaml.safe_load(raw);name=data.get('env_name',path.stem)
        tasks[name].append({'path':str(path.relative_to(source)),'sha256':hashlib.sha256(raw).hexdigest(),
                            'description':data.get('task',{}).get('description',name)})
    if set(tasks)!=set(policy['group_assignments']):raise ValueError('The native task-name catalog changed; re-review membership')
    for name in tasks:
        base=re.sub(r'_(?:contact_a[0-9]+|hold_a[0-9]+)$','',name)
        if base!=name and base in policy['group_assignments'] and policy['group_assignments'][name]!=policy['group_assignments'][base]:
            raise ValueError('A native alias crosses semantic groups')
    entries=[]
    for name,graphs in sorted(tasks.items()):
        group=task_group(name,policy['group_assignments'])
        split='test'if group in policy['test_groups']else'development'if group in policy['development_groups']else'train'
        entries.append({'env_name':name,'semantic_group':group,'split':split,
            'description':graphs[0]['description'],'graphs':graphs,'rl_admitted':False})
    groups=defaultdict(set)
    for row in entries:groups[row['semantic_group']].add(row['split'])
    if any(len(v)!=1 for v in groups.values()):raise ValueError('A semantic group crosses task splits')
    return {'id':'rp_jhh_native_yam_split_v1','source_revision':revision,
        'status':'task_membership_frozen_reset_generation_and_qualification_pending',
        'entries':entries,'counts':dict(Counter(r['split']for r in entries)),
        'semantic_groups':dict(Counter(next(iter(v))for v in groups.values())),
        'scene_policy':{
            'all_existing_saved_layouts':'evaluation_only_including_all_25_layouts_and_canonical_refs_1_to_10',
            'train':'Generate independent schema-3 scenes for TRAIN templates only. A different RNG seed with saved layouts does not change poses.',
            'validation':'Development task groups only; fixed native full starts, no SGS.',
            'test':'Unseen-task groups on seen YAM simulator. Freeze canonical reference order, family assignment and native scoring.',
            'seen_task_new_variant':'Separately report canonical saved layouts of train templates. This is not the unseen-task score.',
            'unseen_simulator':'Existing EFE PandaDyn, RoboPaint, RoboTwin-2 and RoboWits reserves; native YAM cannot be its own unseen simulator.',
            'admission_requires':['scene_state_fingerprint_disjointness','initialization_and_asset_ancestry_audit',
                'native_action_and_reward_qualification','observation_complete_nonterminal_resets'],
            'historical_warmstart_exposure_verified':False,
            'protected_genesis_sapien_holdouts_unchanged':True},
        'note':'Environment names include aliases and contact variants. Report semantic-group counts as well. Every graph hash is pinned; none is an admitted training reset.'}


def main():
    metadata=json.loads((BASE/'artifacts/embodied-first-exam/pear-scaled-v1/sources/gr00t-sim-source.json').read_text())
    directory=BASE/'artifacts/embodied-first-exam/pear-scaled-v1'
    policy=json.loads((directory/'native-yam-split-policy-v1.json').read_text())
    result=build(metadata['source'],metadata['revision'],policy)
    target=directory/'native-yam-split-v1.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'path':str(target),'counts':result['counts'],'semantic_groups':result['semantic_groups']}))


if __name__=='__main__':main()
