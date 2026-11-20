"""Prepare independent native reset inputs; simulation admission is separate.

Read native geometry/constraints and the task graph's authored home. Never
sample from saved evaluation scene poses. The native sampler and validator
remain authoritative for bounds, contacts, mechanisms and generation replay.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


def pose_fingerprint(poses):
    def canonical(value):
        if isinstance(value,dict):return {k:canonical(v)for k,v in sorted(value.items())}
        if isinstance(value,list):return [canonical(v)for v in value]
        if type(value)in (int,float):return round(float(value),7)
        return value
    return hashlib.sha256(json.dumps(canonical(poses),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def independent_spec(reference,authored,seed):
    if reference.get('schema_version')!=3:raise ValueError('This generator requires native schema-3 geometry')
    if set(authored)!=set(reference['base_object_init_poses']):raise ValueError('Graph/geometry actor inventories differ')
    distributions=[s['distribution']for s in reference['scenes']if s.get('distribution',{}).get('groups')]
    if not distributions:raise ValueError('No native bounded reset distribution is declared')
    distribution=copy.deepcopy(distributions[0])
    if distribution.get('yaw_degrees')!=0:raise ValueError('Only native translation groups are supported')
    # Saved scene poses, appearance choices, entry IDs and canonical-reference
    # ancestry are deliberately absent from the new scene's construction.
    result={key:copy.deepcopy(reference[key])for key in
        ['schema_version','task','translation_only_actors','placement_geometry','initial_constraints','robot_reset']
        if key in reference}
    result.update(base_object_init_poses=copy.deepcopy(authored),generation_seed=seed,
        scenes=[{'id':'rp_jhh_generated_training_scaffold','weight':1,
            'object_init_poses':copy.deepcopy(authored),'distribution':distribution}],
        state_kind='independently_sampled_reset_inputs_pending_native_simulation',velocities='zero')
    return result


def generate(source,inputs,split,output,count=64,seed=2026111801):
    source,inputs,output=map(Path,[source,inputs,output]);output.mkdir(parents=True,exist_ok=True)
    sampler_path=source/'gr00t_sim/tasks/tgen/reset_layouts/sampling.py'
    spec=importlib.util.spec_from_file_location('native_yam_reset_sampler',sampler_path)
    sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler)
    snapshot_path=source/'tools/tgen_eval_launcher/tgen_eval/snapshot.py'
    # The source's small pure authored_poses helper has package-relative imports
    # elsewhere in its module; compile that function alone without importing any
    # simulator, launcher, saved-layout selector or evaluation runtime.
    import ast
    tree=ast.parse(snapshot_path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='authored_poses')
    namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(snapshot_path),'exec'),namespace)
    authored_poses=namespace['authored_poses']
    catalog=json.loads((source/'configs/tgen/tgen_catalog.json').read_text())
    declarations={v['graph']:v for t in catalog['tasks'].values()for v in [t,*t.get('families',{}).values()]}
    protected=set();input_hashes={}
    for path in sorted(inputs.rglob('*.json')):
        document=json.loads(path.read_text())
        if document.get('schema_version')not in (2,3)or not document.get('entries'):continue
        input_hashes[str(path.relative_to(inputs))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for entry in document['entries']:
            protected.add(pose_fingerprint(sampler.entry_image(document,entry)['object_init_poses']))
    report={'qualification':'generation_only_pending_native_reset_reward_and_transport_checks',
        'primary_rl_updates':0,'partition':'train','tasks':{},'not_generated':{},
        'native_sampler_sha256':hashlib.sha256(sampler_path.read_bytes()).hexdigest(),
        'native_authored_pose_helper_sha256':hashlib.sha256(ast.unparse(fn).encode()).hexdigest(),
        'protected_pose_fingerprints':len(protected),'input_geometry_manifest_sha256':input_hashes,
        'saved_scene_poses_used_to_generate_training':False,'training_admitted':False}
    seen=set(protected)
    for row in json.loads(Path(split).read_text())['entries']:
        if row['split']!='train':continue
        graph=row['graphs'][0];declaration=declarations.get(graph['path'])
        if not declaration:
            report['not_generated'][row['env_name']]='No schema-3 native catalog reset definition';continue
        graph_path=source/graph['path']
        if hashlib.sha256(graph_path.read_bytes()).hexdigest()!=graph['sha256']:raise ValueError('Frozen training graph changed')
        document=yaml.safe_load(graph_path.read_text())
        reference=json.loads((inputs/declaration['initializations']).read_text())
        try:
            bank=independent_spec(reference,authored_poses(document),seed)
            bank['entries']=sampler.sample_entries(bank,seed,count)
            sampler.validate_manifest(bank)
            fingerprints=[pose_fingerprint(sampler.entry_image(bank,e)['object_init_poses'])for e in bank['entries']]
            if len(set(fingerprints))!=count or any(f in seen for f in fingerprints):
                raise ValueError('Generated physical reset repeats or overlaps a protected pose')
            seen.update(fingerprints)
            target=output/(row['env_name']+'.json');target.write_text(json.dumps(bank,indent=2,allow_nan=False)+'\n')
            report['tasks'][row['env_name']]={'graph':graph,'manifest':str(target),
                'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'generated_resets':count,'pose_fingerprints':fingerprints,'native_generator_replay_passed':True,
                'protected_pose_overlap':0,'training_admitted':False}
        except ValueError as exc:report['not_generated'][row['env_name']]=str(exc)
    (output/'generation-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['source','inputs','split','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--count',type=int,default=64);args=parser.parse_args()
    result=generate(args.source,args.inputs,args.split,args.output,args.count)
    print(json.dumps({'tasks':len(result['tasks']),'generated_resets':sum(v['generated_resets']for v in result['tasks'].values()),
        'not_generated':result['not_generated'],'training_admitted':False},indent=2))
