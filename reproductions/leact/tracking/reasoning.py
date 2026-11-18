"""Build the reasoning workspace inventory and explicit PEAR planning estimates.

This module does not submit jobs or change train/evaluation membership.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import re
import statistics
import subprocess

import yaml

from .reasoning_media import simulator_families

BASE=Path(__file__).resolve().parents[1]
PAPER='https://mit-han-lab.github.io/pear-blog/#appendix'


def public_workspace(value):
    """Exclude unrelated experiment provenance from this independent site."""
    def private(text):
        return re.search(r'leact|yam-coverage-rpunamiya',text,re.I)is not None
    if isinstance(value,dict):
        return {k:public_workspace(v)for k,v in value.items()
                if k!='compute_usage'and not private(k)
                and not (isinstance(v,str)and private(v))}
    if isinstance(value,list):return [public_workspace(v)for v in value if not(isinstance(v,str)and private(v))]
    return value


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def allocation(query,spec):
    """Count only running tasks, applying per-task resource overrides."""
    workflow=spec.get('workflow',spec);resources=workflow.get('resources',{})
    definitions=list(workflow.get('tasks',[]))+[t for g in workflow.get('groups',[])for t in g.get('tasks',[])]
    lookup={t['name']:t for t in definitions};running=[]
    tasks=list(query.get('tasks',[]))+[t for g in query.get('groups',[])for t in g.get('tasks',[])]
    for task in tasks:
        if task.get('status')!='RUNNING':continue
        reference=lookup[task['name']].get('resource','default')
        resource=resources[reference]if isinstance(reference,str)else reference
        running.append({'name':task['name'],'status':'RUNNING','allocated_gpus':int(resource.get('gpu',0)),
                        'platform':resource.get('platform'),'node_name':task.get('node_name')})
    return {'name':query['name'],'pool':query.get('pool'),'status':query['status'],'tasks':running,
            'allocated_gpus':sum(t['allocated_gpus']for t in running)}


def collect_compute():
    """Read OSMO state; never submit, change priority or cancel workflows."""
    def command(args):
        return subprocess.run(['osmo','workflow',*args],capture_output=True,text=True,check=True,timeout=40).stdout
    names=[];offset=0
    while True:
        rows=json.loads(command(['list','--status','RUNNING','--user','rpunamiya',
            '--count','250','--offset',str(offset),'--order','desc','--format-type','json']))['workflows']
        names.extend(row['name']for row in rows)
        if len(rows)<250:break
        offset+=250
    def inspect(name):
        query=json.loads(command(['query',name,'--format-type','json']))
        spec=yaml.safe_load(command(['spec',name]))
        return allocation(query,spec)
    with ThreadPoolExecutor(max_workers=4)as executor:workflows=list(executor.map(inspect,names))
    return {'observed_utc':datetime.now(timezone.utc).isoformat(),'workflows':workflows,
        'allocated_gpus':sum(w['allocated_gpus']for w in workflows),
        'basis':'Running OSMO tasks only; queued tasks and completed tasks excluded. Allocation, not utilization.'}


def estimate(*,updates,groups=128,rollouts=8,control_fraction=.75,control_tokens=13000,
             reasoning_tokens=4000,retention=.65,rollout_gpus=4,trainer_gpus=8,
             simulator_gpus=4,tokens_per_node_second=2500,parallel_environments=16,
             episode_seconds=180,learner_seconds_per_update=900,availability=.85,setup_hours=2):
    values=locals()
    if any(not isinstance(x,(int,float))or isinstance(x,bool)or not math.isfinite(x)for x in values.values()):
        raise ValueError('Planning inputs must be finite numbers')
    if updates<=0 or groups<=0 or rollouts<=0 or rollout_gpus<4 or trainer_gpus<8 or simulator_gpus<0:
        raise ValueError('Positive work and an FSDP8 trainer / TP4 sampler are required')
    if rollout_gpus%4 or trainer_gpus%8 or not 0<retention<=1 or not 0<availability<=1 or not 0<=control_fraction<=1:
        raise ValueError('Invalid parallelism, filtering or availability assumption')
    if min(control_tokens,reasoning_tokens,tokens_per_node_second,parallel_environments,episode_seconds,learner_seconds_per_update)<=0 or setup_hours<0:
        raise ValueError('Timing/token assumptions must be positive')
    retained=updates*groups*rollouts
    control=retained*control_fraction/retention
    reasoning=retained*(1-control_fraction)/retention
    generated=control*control_tokens+reasoning*reasoning_tokens
    decode_hours=generated/(tokens_per_node_second*rollout_gpus/8)/3600
    # Occupancy includes inference waiting. Take a maximum, not an additional
    # charge for inference which is already present in the decode estimate.
    sessions=min(parallel_environments,rollout_gpus/4*16)
    simulator_hours=control*episode_seconds/sessions/3600
    learner_hours=updates*learner_seconds_per_update*(8/trainer_gpus)/3600
    bottlenecks={'generation':decode_hours,'environment occupancy':simulator_hours,'learner':learner_hours}
    wall=setup_hours+max(bottlenecks.values())/availability
    gpus=rollout_gpus+trainer_gpus+simulator_gpus
    return {'retained_episodes':int(retained),'attempted_episodes':control+reasoning,
        'generated_tokens':generated,'wall_hours':wall,'allocated_gpu_hours':gpus*wall,
        'gpus_per_run':gpus,'bottleneck':max(bottlenecks,key=bottlenecks.get),
        'phase_hours':bottlenecks,'effective_parallel_environments':sessions,
        'kind':'scenario estimate, not measured PEAR throughput'}


def timing_proxies(root):
    evidence=read(root/'fixed-action-v1/coverage-scale-v1/evidence.json',{})
    groups=defaultdict(list)
    for name,value in evidence.get('evaluations',{}).items():
        if 'astra'in name:continue
        model='warmstart496'if 'warmstart-step496'in name else 'warmstart737'if 'eval-warmstart-b'in name else 'base'
        for trial in value.get('results',{}).get('trials',[]):
            metrics=trial.get('action_metrics',{})
            if trial.get('status')=='graded'and metrics.get('model_requests',0)>0:groups[model].append(trial)
    result={}
    for name,rows in groups.items():
        metrics=[r['action_metrics']for r in rows];calls=sum(m['model_requests']for m in metrics)
        result[name]={'episodes':len(rows),'successes':sum(r['score']for r in rows),
            'median_episode_seconds':statistics.median(m['elapsed_seconds']for m in metrics),
            'median_episode_generated_tokens':statistics.median(m['output_tokens']for m in metrics),
            'tokens_per_action':sum(m['output_tokens']for m in metrics)/calls,
            'seconds_per_action':sum(m['model_seconds']for m in metrics)/calls,
            'basis':'Existing MoE EFE native evaluations; different sampling and episode caps from PEAR. Timing proxy only.'}
    return result


def inventory(root):
    general=root/'generalization-v2';scaled=root/'pear-scaled-v1'
    rows=read(general/'inventory.json',[]);splits=read(general/'splits.json',{})
    selected={r['task']for r in read(scaled/'proposed-candidate-membership-v2.json',{}).get('rows',[])}
    sets={key:set(splits.get(key,[]))for key in ['train_candidates','official_core_evaluation',
        'protected_core_all_modes','development','simulator_holdout_reserve']}
    reserved=set().union(*(values for key,values in sets.items()if key!='train_candidates'))
    if selected&reserved or selected-sets['train_candidates']:
        raise ValueError('PEAR candidates overlap reserved splits or fall outside the training catalog')
    contracts=read(scaled/'catalog-contracts.json',{})
    review=read(scaled/'holdout-review-v2.json',{})
    cohorts={r.get('task'):r.get('cohort',r.get('evaluation_cohort'))for k in ['core_entries','development_entries']for r in review.get(k,[])}
    result=[]
    for row in rows:
        task=row['task'];suite=row['suite']
        role=('excluded'if suite=='behavior-1k' else 'unseen simulator evaluation'if task in sets['simulator_holdout_reserve']
              else 'core evaluation'if task in sets['official_core_evaluation']
              else 'protected evaluation variant'if task in sets['protected_core_all_modes']
              else 'development'if task in sets['development']
              else 'PEAR training candidate'if task in selected
              else 'deferred training candidate'if task in sets['train_candidates']else 'unassigned / deferred')
        check=contracts.get(suite,{})
        checked=check.get('success')and check.get('task')==task
        result.append({'id':task,'source':'EFE','suite':suite,'role':role,
            'semantic_task':row.get('semantic_task'),'runtime':row.get('simulator_runtime'),
            'description':row.get('base_task',task),'source_path':row.get('path'),
            'native_wall_cap_seconds':row.get('cutoff_seconds'),'cohort':cohorts.get(task),
            'status':'one native interface check passed; RL admission pending'if checked else
                     'suite has one checked interface; this task is not qualified'if check.get('success')else
                     'interface needs repair'if check else 'not qualified',
            'native_action_types':list(check.get('action_contract',{}).get('action_types',{}))if checked else [],
            'rl_admitted':False})
    source=read(scaled/'sources/gr00t-sim-source.json',{})
    native=Path(source.get('source','/nonexistent'))
    yam_split=read(scaled/'native-yam-split-v1.json',{})
    yam_by_name={r['env_name']:r for r in yam_split.get('entries',[])}
    names=set();native_files=[]
    for path in sorted((native/'gr00t_sim/tasks/env_graphs/yam').rglob('*.yaml')):
        data=yaml.safe_load(path.read_text());name=data.get('env_name',path.stem);names.add(name)
        membership=yam_by_name.get(name,{})
        role={'train':'PEAR training candidate','test':'seen simulator unseen task evaluation',
              'development':'development'}.get(membership.get('split'),'native candidate; split pending')
        task=data.get('task',{});embodiment=data.get('embodiment',{})
        item={'id':'gr00t-yam/'+str(path.relative_to(native/'gr00t_sim/tasks/env_graphs/yam')),
            'source':'Native GR00T','suite':'native-gr00t-yam','role':role,
            'task_partition':membership.get('split'),'semantic_group':membership.get('semantic_group'),
            'semantic_task':name,'description':task.get('description',name),
            'runtime':'native GR00T simulator; adapter qualification pending',
            'source_path':str(path.relative_to(native)),'source_revision':source.get('revision'),
            'embodiment':embodiment.get('registry_name'),
            'status':'task split frozen; independent training resets and bridge qualification pending'if membership else'split pending',
            'rl_admitted':False}
        result.append(item);native_files.append(item)
    proposal=read(BASE/'proposals/efe_pearplusplus_multitask_v2.json',{})
    for family in proposal.get('reasoning_training',{}).get('families',[]):
        result.append({'id':family['id'],'source':'Generated reasoning','suite':family['id'],
            'role':'reasoning training planned','description':family['name'],
            'runtime':'independent training scenes','status':family.get('status','planned'),
            'rl_admitted':False,'answer_type':family.get('output')})
    for benchmark in proposal.get('holdouts',{}).get('benchmarks',[]):
        result.append({'id':benchmark,'source':'Reasoning evaluation','suite':benchmark,
            'role':'reasoning evaluation only','description':benchmark,'runtime':'benchmark-specific',
            'status':'reserved; all splits excluded from new training','rl_admitted':False})
    counts=Counter(r['suite']for r in rows)
    suites=[]
    for suite,n in sorted(counts.items()):
        subset=[r for r in result if r['source']=='EFE'and r['suite']==suite]
        suites.append({'suite':suite,'source':'EFE','catalog_entries':n,
            'pear_training_candidates':sum(r['role']=='PEAR training candidate'for r in subset),
            'evaluation_entries':sum('evaluation'in r['role']for r in subset),
            'development_entries':sum(r['role']=='development'for r in subset),
            'interface_status':'one task passed; suite-wide admission pending'if contracts.get(suite,{}).get('success')
                else 'repair needed'if contracts.get(suite)else 'not qualified'})
    return {'entries':result,'suites':suites,'counts':{'efe_catalog_entries':len(rows),
        'efe_pear_training_candidates':len(selected),'efe_general_train_candidates':len(sets['train_candidates']),
        'efe_core_eval_entries':len(sets['official_core_evaluation']),
        'efe_protected_eval_variants':len(sets['protected_core_all_modes']),
        'efe_development_entries':len(sets['development']),'unseen_simulator_reserve':len(sets['simulator_holdout_reserve']),
        'native_yam_graph_files':len(native_files),'native_yam_unique_env_names':len(names),
        'qualified_example_interfaces':sum(bool(v.get('success'))for v in contracts.values()),
        'rl_admitted_tasks':0},'limitations':[
        'Catalog entries, semantic tasks, scene variants and admitted environments are different counts.',
        'The native YAM catalog contains repeated env_name declarations; source YAML count is not a count of independently verified environments.',
        'Native YAM task membership is frozen. All existing saved layouts stay evaluation-only; train tasks require independent scenes before admission.',
        'Seen-task/new-variant and seen-simulator/new-task cohorts remain provisional until actual scene/exposure audits.',
        'Six example interfaces passed; this does not qualify every task in their simulator families.'],
        'holdout_review':{k:review.get(k)for k in ['core_cohort_counts','development_cohort_counts','unseen_simulator_counts']}}


def build(base=BASE):
    root=base/'artifacts/embodied-first-exam';scaled=root/'pear-scaled-v1'
    plan=read(scaled/'plan.json',{})
    control=read(scaled/'control.json',{})
    scaling=read(scaled/'warmstart-post-smoke-scaling-policy.json',{})
    enabled=control.get('enabled_training_arms',[])
    warm_authorized=(not control.get('paused',True)and not control.get('primary_training_paused',True)
                     and 'moe-warmstart496-pearplusplus'in enabled)
    catalog=inventory(root)
    proposal=read(base/'proposals/efe_pearplusplus_multitask_v2.json',{})
    reasoning=proposal.get('reasoning_training',{})
    reasoning_check=read(scaled/'warmstart-reasoning-check.json',{})
    reasoning_bank=read(scaled/'reasoning-remote-qualified/membership.json',{})
    checked_families=reasoning_check.get('worker',{}).get('families',{})
    reasoning_families=[]
    for family in reasoning.get('families',[]):
        scenes=[s for s in reasoning_bank.get('scenes',{}).values()if s['family']==family['id']]
        train=sum(s['partition']=='train'and s.get('qualification')=='passed'for s in scenes)
        dev=sum(s['partition']=='development'and s.get('qualification')=='passed'for s in scenes)
        passed=checked_families.get(family['id'],{}).get('passed')is True
        reasoning_families.append({**family,'admitted_tasks':int(bool(train and passed)),
            'qualified_train_scenes':train,'qualified_development_scenes':dev,
            'transport_passed':passed,'status':'procedural scenes and reward/transport checks passed'if passed else family['status'],
            'training_environment':'Independently generated MuJoCo primitive scenes; initial task templates. Published dataset sources below remain candidates.'if passed else family['training_environment']})
    control_validation=read(scaled/'expanded-validation-status.json',{})
    task_receipts={}
    for path in(scaled/'validation-receipts').glob('*/checks.json'):
        task_receipts.update(read(path,{}))
    result={'title':'rpunamiya reasoning','updated_utc':datetime.now(timezone.utc).isoformat(),
        'inventory':catalog,'timing_proxies':timing_proxies(root),
        'compute_usage':read(base/'artifacts/reasoning-workspace/compute-usage.json',{}),
        'simulators':simulator_families(catalog,read(scaled/'holdout-review-v2.json',{}),
            read(base/'artifacts/reasoning-workspace/family-images.json',{})),
        'embodied_reasoning':{**reasoning,'families':reasoning_families,
            'evaluation_benchmarks':proposal.get('holdouts',{}).get('benchmarks',[]),
            'existing_training_sources':read(base/'proposals/efe_reasoning_training_sources_v1.json',{}).get('sources',[]),
            'box_output_planned':any('bounding box'in f.get('output','').lower()or'bbox'in f.get('output','').lower()
                for f in reasoning.get('families',[]))},
        'warmstart_retention':read(base/'artifacts/reasoning-workspace/retention-496/live.json',{}),
        'native_yam_split':read(scaled/'native-yam-split-v1.json',{}),
        'live_pear':read(base/'artifacts/reasoning-workspace/live/pear.json',{}),
        'live_collectors':{key:read(base/f'artifacts/reasoning-workspace/live/{key}-collector.json',{})for key in ['pear','retention']},
        'pear':{'arms':plan.get('arms',[]),'primary_updates':plan.get('primary_rl_updates',0),
            'moe_router_policy':proposal.get('moe_router_policy',{}),
            'sgs':proposal.get('sgs',{}),'optimizer':proposal.get('optimizer',{}),
            'balance':proposal.get('multitask_balancing',proposal.get('multitask',{})),
            'recipe':plan.get('recipe',{}),'interface_context':plan.get('interface_context',{}),
            'planned_extra_h100_gpus_total':32,'planned_h100_gpus_per_arm':16,
            'launch_authorized':warm_authorized,'enabled_training_arms':enabled,
            'paused_training_arms':control.get('paused_training_arms',[]),
            'warmstart_scaling':scaling,
            'preflight':{'router':read(scaled/'warmstart-router-check.json',{}),
                'native_bridge':read(scaled/'bridge-status.json',{}),
                'reasoning':reasoning_check,'controls':control_validation,
                'native_yam':read(scaled/'warmstart-yam-check.json',{}),
                'on_policy':read(scaled/'onpolicy-check-status.json',read(scaled/'onpolicy-check-launch.json',{})),
                'camera_observations':read(scaled/'control-observation-audit.json',{}),
                'protected_scene_collisions':sum(not r['source_scene_check_passed']for r in read(scaled/'source-scene-audit.json',{}).get('tasks',{}).values()),
                'task_receipts':{k:{field:v[field]for field in ['success','native_interface_passed','qualification','error']if field in v}
                    for k,v in task_receipts.items()}},
            'allocation_note':('Warmstart launch authorized after RL smoke checks. Target '
                +str(scaling.get('initial_target_h100s',64))+' H100s, fallback '
                +str(scaling.get('fallback_target_h100s',48))
                +'; RTX simulator capacity is additional for Isaac RGB. Expand when useful capacity is available. Base-model arm remains paused. No primary allocation is confirmed.'
                if warm_authorized else 'Estimate only. Existing experiments remain independent; do not launch PEAR.'),
            'paper_reference':{'url':PAPER,'h100_node_generated_tokens_second':[2000,3000],
                'mean_episode_generated_tokens':92000,'total_training_h100_equivalent_hours':50000,
                'note':'Published dense-model baseline; not our measured PEAR rate. The paper uses more concurrent sessions than our initial recipe.'},
            'default_inputs':{'updates':32,'groups':128,'rollouts':8,'control_fraction':.75,
                'reasoning_tokens':4000,'retention':.65,'rollout_gpus':4,'trainer_gpus':8,
                'simulator_gpus':4,'tokens_per_node_second':2500,'parallel_environments':16,
                'availability':.85,'setup_hours':2},
            'arm_assumptions':{'moe':{'control_tokens':13000,'episode_seconds':180,'learner_seconds_per_update':900},
                               'dense':{'control_tokens':92000,'episode_seconds':1200,'learner_seconds_per_update':3600}},
            'estimate_limitations':[
                'All PEAR timing and retention assumptions are editable. These are scenarios, not measured training rates or learning guarantees.',
                'Warmstart token/episode proxies come from the current small MoE evaluation panel with different caps and sampling. Reasoning can lengthen during RL.',
                'The dense-base default uses the paper token length; our old base EFE rollouts were longer. This is not a guaranteed upper bound.',
                'Filtering discards constant-reward groups; 128 groups of eight are counted after filtering, so attempted episodes exceed retained episodes.',
                'GPU-hours charge the entire reserved run allocation, including idle/queue-within-run time. External scheduler queue, engineering and admission time are excluded.',
                'Mixed environments reserve 8 GPUs for training, 4 for TP4 rollout and 4 for GPU simulation per run. CPU-rendered environments may use an 8+8 split instead.',
                'No valid end-to-end full-model PEAR timing exists yet. Engine, environment, curriculum and long-context checks remain prerequisites.']}}
    for key,assumptions in result['pear']['arm_assumptions'].items():
        result['pear'].setdefault('default_estimates',{})[key]=estimate(**result['pear']['default_inputs'],**assumptions)
    result=public_workspace(result)
    write(base/'artifacts/reasoning-workspace/workspace.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-compute',action='store_true',help='Read current OSMO allocations without launching jobs')
    args=parser.parse_args()
    if args.refresh_compute:
        path=BASE/'artifacts/reasoning-workspace/compute-usage.json'
        try:write(path,collect_compute())
        except (OSError,ValueError,KeyError,subprocess.SubprocessError)as exc:
            old=read(path,{})
            write(path,{**old,'refresh_error':type(exc).__name__,
                'refresh_attempted_utc':datetime.now(timezone.utc).isoformat()})
    value=build();print(json.dumps({'updated_utc':value['updated_utc'],'counts':value['inventory']['counts'],
        'estimates':value['pear']['default_estimates']}))


if __name__=='__main__':main()
