"""Generate and check every reasoning scene; diagnostics are never policy data."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import inspect

try:
    from .efe_pear_reasoning_scenes import SceneEpisode,REASONING_FAMILIES,VERSION
except ImportError:
    from efe_pear_reasoning_scenes import SceneEpisode,REASONING_FAMILIES,VERSION


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def qualify(output,*,train_scenes=64,development_scenes=8,cluster=False):
    import mujoco
    if mujoco.__version__!='3.3.7':raise ValueError('The generator is pinned to MuJoCo 3.3.7')
    if cluster and(platform.system()!='Linux'or not os.environ.get('EFE_WORKFLOW_ID')):
        raise ValueError('Cluster qualification must run in its recorded OSMO workflow')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'previews').mkdir(exist_ok=True)
    manifest={'generator_version':VERSION,'scenes':{},'qualification':'in_progress',
        'generator_sha256':hashlib.sha256(Path(inspect.getfile(SceneEpisode)).read_bytes()).hexdigest(),
        'runtime':{'mujoco':mujoco.__version__,'platform':platform.system(),
                   'render_backend':os.environ.get('MUJOCO_GL','platform_default')},
        'workflow':os.environ.get('EFE_WORKFLOW_ID'),'primary_rl_updates':0,
        'asset_ancestry':'Only independently authored primitive geometry; no imported assets, benchmark images, scans, annotations or scenes.',
        'scripted_diagnostics_are_policy_data':False,'rejected':[]}
    seen={};counts=Counter()
    for family in REASONING_FAMILIES:
        for partition,target in [('train',train_scenes),('development',development_scenes)]:
            accepted=0;seed=0
            while accepted<target and seed<target*6+100:
                e=None
                try:
                    e=SceneEpisode(family,seed,partition);obs=e.observation();scene=e.scene_sha256
                    if scene in seen:raise ValueError('A physical scene repeats or crosses partitions')
                    wrong={'answer_type':e.answer_type,'point':[0.,0.],'bbox':[0.,0.,.01,.01],
                           'points':[],'count':0,'summary':'Discarded negative verifier control.'}
                    if e.grade(wrong)!=0:raise ValueError('Negative verifier control was rewarded')
                    actions=[];answer=e.oracle_answer()
                    if answer.get('action_type')=='view':
                        if e.seen_target:raise ValueError('Unnecessary oracle view transition')
                        actions.append(answer);obs,reward,done=e.step(answer)
                        if done or reward!=0 or not e.seen_target:raise ValueError('Active perception did not reveal its target')
                        answer=e.oracle_answer()
                    if family=='set_pointing_counting'and len(answer['points'])>1:
                        duplicate={**answer,'points':[answer['points'][0]]*len(answer['points'])}
                        if e.grade(duplicate)!=0:raise ValueError('Duplicate set points earned reward')
                    if e.grade(answer)!=1:raise ValueError('Correct verifier control was rejected')
                    _,reward,done=e.step(answer)
                    if not done or reward!=1:raise ValueError('Native stateful finalization lost the verified answer')
                    key=f'{family}/{partition}/{seed}';seen[scene]=key
                    manifest['scenes'][key]={'family':family,'partition':partition,'seed':seed,
                        'scene_sha256':scene,'qualification':'passed'if cluster else'local_checks_passed',
                        'positive_control':1,'negative_control':0,'stateful_finalization_passed':True,
                        'view_transitions':len(actions)}
                    if accepted==0:
                        for i,img in enumerate(obs['images']):img.save(output/'previews'/f'{family}-{partition}-{i}.png')
                    accepted+=1;counts[family+'/'+partition]+=1
                except ValueError as exc:
                    manifest['rejected'].append({'family':family,'partition':partition,'seed':seed,'reason':str(exc)})
                finally:
                    if e:e.close()
                seed+=1
                if accepted%8==0:
                    write(output/'status.json',{'phase':'reasoning_scene_validation','counts':dict(counts),
                        'required_families':list(REASONING_FAMILIES),'primary_rl_updates':0})
            if accepted!=target:raise RuntimeError('Reasoning family lacks enough qualified scenes: '+family)
        print('PEAR_REASONING_CHECK '+json.dumps({'family':family,'counts':dict(counts),
            'phase':'family_checked','primary_rl_updates':0}),flush=True)
    manifest.update(qualification='passed'if cluster else'local_checks_passed_pending_cluster',
        completed_utc=datetime.now(timezone.utc).isoformat(),counts=dict(counts),
        train_development_scene_overlap=0,full_curriculum_qualified=False)
    write(output/'membership.json',manifest)
    report={'phase':'reasoning_validation_completed','counts':dict(counts),
        'qualification':manifest['qualification'],'primary_rl_updates':0,'workflow':manifest['workflow'],
        'membership_file':str(output/'membership.json'),
        'membership_sha256':hashlib.sha256((output/'membership.json').read_bytes()).hexdigest()}
    write(output/'status.json',report);print('PEAR_REASONING_CHECK '+json.dumps(report),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-scenes',type=int,default=64);parser.add_argument('--development-scenes',type=int,default=8)
    parser.add_argument('--cluster',action='store_true');args=parser.parse_args()
    qualify(args.output,train_scenes=args.train_scenes,development_scenes=args.development_scenes,cluster=args.cluster)
