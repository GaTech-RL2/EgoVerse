"""Qualify Astra and Qwen with the same fixed robot interface, then run tasks."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .scale_efe import BASE,ROOT,BUILD_ROOT,read,write,submit,command
from .compute_budget import reserved_nodes,node_limit
from .launch_efe_training import freeze_sources
from .launch_efe_native_followups import LANE,collect_arm,verified_native_completion
from .launch_efe_student_qualification import replace_once,retry_build


DEST = ROOT/'fixed-action-v1'
SOURCE = BUILD_ROOT/'efe-qwen36-native-qualification-20261103-v3'
TEACHER_SOURCE = BUILD_ROOT/'efe-direct-control-20261101-v2-a3-03-resume'
VERSION = 4
RUN_DIRECTORY = 'fixed-action-v1/runs/v4'


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_arm(model,mode):
    name = model+'-'+mode
    # The recorded Astra reference had no rejected commands. The parser replay
    # audit verifies its commands are unchanged by the presentation-wrapper fix.
    if model == 'astra':
        return BUILD_ROOT/('efe-fixed-action-20261103-v3-'+name)
    build = BUILD_ROOT/('efe-fixed-action-20261104-v4-'+name)
    if (build/'source-manifest.json').exists():
        return build
    build.mkdir(exist_ok=True)
    for filename in read(SOURCE/'source-manifest.json'):
        shutil.copyfile(SOURCE/filename,build/filename)
    spec = read(build/'spec.json')
    for key in ['retry_of','serving_comparison','instruction_suffix_file']:
        spec.pop(key,None)
    if mode == 'task':
        original = read(BUILD_ROOT/'efe-native-followups-20261030-v1-base-direct/spec.json')
        faucet = next(t for t in original['tasks'] if t['task']=='robocasa-turn-on-sink-faucet-i00-standard')
        spec['tasks'].append(faucet)
    label = ('Qwen3.6-35B-A3B' if model=='qwen36' else 'Astra')+' · constrained output · '+mode
    spec.update(model_id=name,agent_slug=name,agent_label=label,
        export_role='constrained_action_v4_'+name,local_results_directory=RUN_DIRECTORY+'/'+name,
        protocol_version='efe-constrained-action-v4-'+mode,
        protocol_label='Constrained JSON commands validated against each environment native action types, dimensions, bounds and control timestep. Shared current RGB images, public specification and proprioception. Astra and Qwen share the same bridge within each environment.',
        scope=('Nine directed movement checks across three preselected training scenes. Native manipulation grades are retained for provenance but are NOT movement results or benchmark scores.' if mode=='movement' else
            'Four fixed native training-task evaluations: three MetaWorld scenes and one RoboCasa faucet scene. Same constrained-output interface for Astra and Qwen, with each robot native action space. A new protocol comparison; no held-out score.'),
        purpose='Test whether a simpler shared interface establishes robot competence before further SFT.',
        fixed_action_mode=mode,exclude_from_task_leaderboard=mode=='movement',
        align_tool_instructions=False,native_mirror_enabled=False,
        action_durable_root='/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-constrained-action-20261104-v4',
        parser_change='Accept exactly one Markdown JSON fence around a complete command. All action fields, bounds, prompt, images, generation and budgets are unchanged. v3 outcomes remain recorded.',
        comparison_to_development='Agent loop, observation packaging, prompt, action interface and call budget differ from the native Codex baseline. Teacher and student share this new interface; do not attribute its effect to model weights alone.')
    if model=='astra':
        teacher = read(TEACHER_SOURCE/'spec.json')
        for key in ['student_checkpoint','student_manifest','reasoning_parser','role_adapter']:
            spec.pop(key,None)
        spec.update(request_model=teacher['request_model'],harbor_model=teacher['harbor_model'],
            student_revision='provider-model-id-recorded-per-response',serving_version='external-provider',
            upstream=teacher.get('upstream','https://inference-api.nvidia.com/v1'))
    config = {'mode':mode,'model':spec['request_model'],'max_output_tokens':32768,
        'max_calls_per_goal':4,'max_task_calls':96,'wall_seconds':600 if mode=='movement' else 1500,
        'movement_target_m':0.02,'movement_along_tolerance_m':[0.015,0.035],
        'movement_orthogonal_tolerance_m':0.015,'required_movement_passes':8,'planned_movement_checks':9,
        'action_space_source':'native robot.spec()', 'max_repeat':25,
        'reasoning_policy':'Up to 32768 output tokens per call. Record public action summaries and token usage; do not request or export provider-private reasoning.',
        'task_budget_note':'Fixed 96-call/1500-second budget, within the unchanged native 1800-second task deadline.'}
    write(build/'action-config.json',config)
    spec['fixed_action_config']=config
    write(build/'spec.json',spec)
    for filename in ['efe_action_agent.py','efe_action_loop.py','efe_action_artifacts.py']:
        shutil.copyfile(BASE/'cluster'/filename,build/filename)
    path=build/'native-pilot.py';text=path.read_text()
    begin=text.index("        source = Path(importlib.util.find_spec('harbor.agents.installed.codex').origin)")
    end=text.index('        address = socket.gethostbyname',begin)
    text=text[:begin]+"        runtime.write('fixed-action-config.json',spec['fixed_action_config'])\n"+text[end:]
    begin=text.index("        runtime.write('native-config.json',")
    end=text.index('        env.update(OPENAI_API_KEY=',begin)
    text=text[:begin]+"        runtime.write('native-config.json',{'agent':'efe_action_agent:FixedActionAgent','config':spec['fixed_action_config'],'resource_enforcement_requested':1})\n"+text[end:]
    text=replace_once(text,"'-a', 'codex', '-m', spec['harbor_model']", "'-a', 'efe_action_agent:FixedActionAgent', '-m', spec['harbor_model']")
    text=replace_once(text,
        "                       '-o', str(jobs_root), '--allow-agent-host', address,\n"
        "                       '--ak', 'version=' + spec['codex_version'], '--ak', 'reasoning_effort=high',\n"
        "                       '--ak', 'reasoning_summary=none', '--ak', 'web_search=disabled',\n"
        "                       '--ak', 'config=' + str(config)]",
        "                       '-o', str(jobs_root), '--allow-agent-host', address]")
    text=replace_once(text,"                   RCB_LIMITS_ENFORCE='1')",
        "                   RCB_LIMITS_ENFORCE='1',PYTHONPATH=str(Path(__file__).resolve().parent)+':'+os.environ.get('PYTHONPATH',''))")
    text=replace_once(text,"            results['trials'].append(public_value(record, (provider_key, token)))",
        "            metric_files=list((runtime.ARCHIVE/'jobs'/job).glob('*/agent/action-metrics.json'))\n"
        "            if len(metric_files)==1:\n"
        "                record['action_metrics']=json.loads(metric_files[0].read_text())\n"
        "            record['exclude_from_task_leaderboard']=spec.get('exclude_from_task_leaderboard',False)\n"
        "            results['trials'].append(public_value(record, (provider_key, token)))")
    text=replace_once(text,'for _ in range(4320):','for _ in range(120):')
    path.write_text(text)
    path=build/'native-worker.py';text=path.read_text()
    text=replace_once(text, 'import student\n', 'import student\n    from efe_action_artifacts import retain_trials, mirror_evidence\n')
    text=replace_once(text, "        if name == 'results.json':\n            publish_results(value)",
        "        if name == 'results.json':\n            retain_trials(archive,spec,value,runtime.WORKFLOW)\n            publish_results(value)")
    text=replace_once(text,"    def cached_run(name, argv, **kwargs):",
        "        mirror_evidence(archive,spec,runtime.WORKFLOW,name,value)\n\n    def cached_run(name, argv, **kwargs):")
    text=replace_once(text,"                'elapsed_s':metrics.get('elapsed_s'), 'public_episode':row['id']})",
        "                'elapsed_s':metrics.get('elapsed_s'), 'public_episode':row['id'],\n"
        "                'action_metrics':trial.get('action_metrics',{})})")
    text=replace_once(text,"            'scope':spec['scope']})",
        "            'scope':spec['scope'],'fixed_action_mode':spec['fixed_action_mode'],\n"
        "            'exclude_from_task_leaderboard':spec['exclude_from_task_leaderboard'],\n"
        "            'movement_passes':sum(r['action_metrics'].get('movement_passes',0) for r in rows),\n"
        "            'planned_movement_checks':9 if spec['fixed_action_mode']=='movement' else 0})")
    path.write_text(text)
    workflow=yaml.safe_load((build/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-fixed-'+name+'-v4'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'8h'}
    task=workflow['workflow']['tasks'][0]
    for filename in ['efe_action_agent.py','efe_action_loop.py','efe_action_artifacts.py','action-config.json']:
        task['files'].append({'localpath':filename,'path':LANE+'/'+filename})
    if model=='astra':
        teacher_workflow=yaml.safe_load((TEACHER_SOURCE/'workflow.yaml').read_text())
        task['credentials']=teacher_workflow['workflow']['tasks'][0]['credentials']
        path=build/'native-bootstrap.py'
        path.write_text(replace_once(path.read_text(),
            "subprocess.run([python, '-u', str(lane / 'efe_prepare_qualification_student.py')], env=env, check=True)\n",''))
        (build/'native-student.py').write_text('''"""Use the existing authenticated Astra endpoint with the fixed robot agent."""
import importlib.util
def main():
    source=importlib.util.spec_from_file_location('fixed_pilot','/tmp/efe-training-evaluation-v1/warm96/pilot.py')
    module=importlib.util.module_from_spec(source);source.loader.exec_module(module);module.main()
''')
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in build.glob('*.py'):
        compile(path.read_text(),str(path),'exec')
    freeze_sources(build)
    return build


def prepare():
    DEST.mkdir(exist_ok=True)
    arms=[]
    for model in ['qwen36','astra']:
        for mode in ['movement','task']:
            build=prepare_arm(model,mode)
            directory=(RUN_DIRECTORY if model=='qwen36' else 'fixed-action-v1/runs/v3')+'/'+model+'-'+mode
            arms.append({'id':model+'-'+mode,'model':model,'mode':mode,'build':str(build),
                'directory':directory,'require_retained':True,'retained_storage':'durable',
                'condition_label':('Qwen3.6 · JSON wrapper adapter' if model=='qwen36' else 'Astra · recorded v3 reference'),
                'depends_on':model+'-movement' if mode=='task' else None})
            (ROOT/arms[-1]['directory']).mkdir(parents=True,exist_ok=True)
    plan={'created_utc':now(),'enabled':True,'arms':arms,'maximum_concurrent_nodes':2,
        'priority':'HIGH','capacity_fallback':'LOW','training_enabled':False,
        'summary':'Astra passed 9/9 movement checks and solved 3/4 native tasks through constrained output. Qwen v3 was strongly affected by Markdown-wrapped JSON rejected by the parser. A v4 Qwen retry accepts one complete JSON wrapper while keeping action validation, prompts and budgets unchanged. The recorded Astra reference remains unchanged under this adapter.',
        'movement_gate':'Qualification target: eight of nine +2 cm world-axis movements, with 1.5–3.5 cm along-axis displacement and at most 1.5 cm orthogonal error. Three axes sequentially in each of three diagnostic scenes; up to four model decisions per goal. The native task comparison follows an infrastructure-clean diagnostic even if this target is missed.',
        'benchmark_separation':'Movement episodes are excluded from task scores and SFT. Task trials start in fresh native episodes and use the original grader.',
        'shared_interface':'Up to four declared RGB cameras, public spec/proprioception/task metadata, last four actions and feedback. Model emits action_type, action, repeat and a brief summary. The native specification determines component count, semantics, bounds and control_dt. The bridge validates then executes the exact command.',
        'limitations':'Astra is the recorded v3 reference, with every accepted command verified unchanged by the wrapper adapter. Qwen v4 is a new rollout condition; no retrospective success is assigned to previously rejected commands. These are training-scene diagnostics, not held-out scores.',
        'next':'Retain all outcomes. Curate successful teacher task trajectories with complete native action/observation evidence and scene isolation. Expand diverse robot/task coverage using the same constrained-output contract, then adapt and validate Qwen3.6 warmstart plus LeAct. Qwen3.6 training has not started.'}
    previous=read(DEST/'plan.json',{})
    plan['created_utc']=previous.get('created_utc',plan['created_utc'])
    plan['run_version']=VERSION
    plan['parser_replay_audit']='parser-replay-audit.json'
    plan['prior_results_directory']='fixed-action-v1/runs/v3'
    plan['native_timing_correction']='Read control_dt from each native spec: recorded MetaWorld 0.0125s; RoboCasa 0.05s. Earlier v2 dt=0.02 failures are retained as infrastructure attempts, not model scores.'
    plan['transport_correction']='Omit the unsupported reasoning.summary=none request field; retain high reasoning effort. Earlier HTTP400 and pre-action interruption are infrastructure attempts, not robot scores.'
    write(DEST/'plan.json',{**previous,**plan})
    return arms


def monitor():
    arms=prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from ..embodied_first_exam.report import publish
    while True:
        states=[]
        try:
            active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
            previous={r['id']:r for r in read(DEST/'status.json',{}).get('arms',[]) if r.get('directory','').startswith(RUN_DIRECTORY+'/')}
            finished=set()
            for arm in arms:
                local=ROOT/arm['directory'];control=local/'control';control.mkdir(exist_ok=True)
                current=read(control/'active-build.json',{'build':arm['build'],'retry_number':0,'priority':'HIGH'})
                build=Path(current['build']);receipt=read(build/'submission.json',{})
                if not receipt:
                    parent=arm['depends_on']
                    if parent and parent not in finished and not previous.get(parent,{}).get('complete',{}).get('success'):
                        states.append({**arm,'status':'WAITING','detail':'Automatically follows the completed movement diagnostic.'});continue
                    running=sum(bool(s.get('status') in {'RUNNING','PENDING','WAITING'} and s.get('workflow')) for s in states)
                    owned=sum(w.get('name','').startswith('leact-efe-fixed-') for w in active)
                    if reserved_nodes(active)>=node_limit() or max(running,owned)>=2:
                        states.append({**arm,'status':'WAITING','detail':'Waiting for an authorized node slot.'});continue
                    receipt=submit(build,priority=current['priority'])
                    write(control/'active-build.json',current)
                    active.append({'name':receipt['name'],'status':'PENDING'})
                workflow=receipt['name'];scheduler=command(['osmo','workflow','query',workflow,'--format-type','json'])
                write(control/'scheduler.json',scheduler)
                complete=read(local/'worker-complete.json',{}) or read(local/'verified-completion.json',{})
                if scheduler['status']=='RUNNING' and not complete:
                    complete=collect_arm(arm,workflow)
                if scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED'):
                    complete=complete or verified_native_completion(arm)
                behavior=read(local/'behavior.json',{})
                state={**arm,'workflow':workflow,'status':scheduler['status'],'complete':complete,
                    'progress':read(local/'progress.json',{}),'graded':behavior.get('graded',0),
                    'successes':behavior.get('successes',0),'control_steps':sum(r['control_steps'] for r in behavior.get('trials',[])),
                    'movement_passes':behavior.get('movement_passes',0),'planned_movement_checks':9 if arm['mode']=='movement' else 0,
                    'exclude_from_task_leaderboard':arm['mode']=='movement','updated_utc':now()}
                if complete.get('success'):
                    finished.add(arm['id'])
                elif scheduler['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} and not (local/'worker-start.json').exists():
                    number=current['retry_number']+1;retry=retry_build(build,number)
                    write(control/'active-build.json',{'build':str(retry),'retry_number':number,'priority':'LOW'})
                    state['detail']='Allocation failed before evaluation; LOW-priority retry prepared.'
                elif scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED'):
                    state['needs_review']=True
                    state['detail']='Runtime failure; retain recorded outcomes and inspect before expansion.'
                states.append(state)
                # Publish incrementally so completed stages are visible during
                # potentially slow collection of a different model's videos.
                processed={s['id']:s for s in states}
                current_rows=[processed.get(a['id'],previous.get(a['id'],a)) for a in arms]
                write(DEST/'status.json',{'updated_utc':now(),'arms':current_rows,'complete':False})
            done=len(states)==len(arms) and all(s.get('complete',{}).get('success') for s in states)
            write(DEST/'status.json',{'updated_utc':now(),'arms':states,'complete':done})
            # Reporting errors must not stop collection or launch the next stage twice.
            # Publish only meaningful state changes, not every poll timestamp.
            signature=hashlib.sha256(json.dumps([{k:s.get(k) for k in
                ['id','workflow','status','graded','successes','control_steps','movement_passes','detail','complete','progress']}
                for s in states],sort_keys=True).encode()).hexdigest()
            if read(DEST/'published-state.json',{}).get('signature')!=signature:
                try:
                    if any(s.get('graded') for s in states):
                        from ..embodied_first_exam.direct_control_viewer import build as build_viewer
                        build_viewer()
                    publish()
                    write(DEST/'published-state.json',{'signature':signature,'updated_utc':now()})
                except Exception as error:
                    safe_log('Dashboard update failed: '+str(error)[-600:])
            if done:return
        except Exception as error:
            safe_log('Monitor iteration failed: '+str(error)[-1800:])
            try:write(DEST/'monitor-error.json',{'updated_utc':now(),'error':str(error)[-1800:]})
            except OSError:pass
        time.sleep(30)


def safe_log(message):
    try:print(message,flush=True)
    except OSError:pass


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'monitor'],cwd=BASE.parents[1],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'updated_utc':now(),'pid':child.pid})
    return child.pid


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    elif args.action=='start':print(json.dumps({'pid':start()}))
    else:print(json.dumps(prepare()))
