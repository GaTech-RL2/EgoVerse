"""Stage model-free EEF checks without changing an existing keeper or teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from .scale_efe import BASE,BUILD_ROOT,read,write,command
from .launch_efe_pear_scaled import DEST,REMOTE
from .efe_pear_allocation_jobs import running_tasks
from .monitor_efe_preflight import remote


ROOT=DEST/'eef-interface-v1'
WORKFLOW='leact-rp-jhh-pear-worker-fd7cc8361488-allocation-r5-1'


def prepare(*,approach=False):
    plan=read(ROOT/'native-pilot-capacity-handoff.json')
    host=plan['retained_yam_host'];old=Path(host['native_template_build']);native=read(old/'spec.json')
    names=read(old/'source-manifest.json')
    files={name:old/name for name in names if name.endswith('.py')and name!='bootstrap.py'}
    current=['efe_pear_yam_server.py','efe_pear_yam_session_policy.py','efe_pear_yam_eef.py',
        'efe_pear_eef.py','efe_pear_eef_service.py','efe_action_loop.py','efe_pear_context.py',
        'efe_pear_eef_probe.py']
    files.update({name:BASE/'cluster'/name for name in current})
    spec={k:native[k]for k in ['source_archive_path','source_archive_sha256','generated_reset_archive',
        'generated_reset_archive_sha256','assets_release','tasks','generated_reset_tasks','runtime_image']}
    for key in ['asset_aliases','numeric_representation']:
        if key in native:spec[key]=native[key]
    cases=host['training_cases']
    # One independently generated training reset in each of two task graphs;
    # each direction is checked after a fresh reset of that exact scene.
    spec['probe_cases']=[case for case in cases if case['env_args']['reset_id']=='init-000']
    keep={case['task']for case in spec['probe_cases']}
    spec['tasks']=[row for row in spec['tasks']if row['env_name']in keep]
    if len(spec['tasks'])!=2:raise ValueError('The retained probe host has exactly two free GPUs')
    spec.update(control_interface='pear-eef-native-v1',teacher_calls_permitted=False,
        optimizer_updates_permitted=False,qualification_only=True)
    if approach:spec['probe_cartesian_approach']=True
    identity=hashlib.sha256(json.dumps([spec,{n:hashlib.sha256(p.read_bytes()).hexdigest()
        for n,p in files.items()}],sort_keys=True).encode()).hexdigest()[:12]
    experiment=REMOTE+'/rl-eef-probes/rp_jhh_pear_eef_'+identity
    spec.update(owner_model_workflow='rp_jhh_pear_eef_probe_'+identity,probe_root=experiment,
        probe_worker_id='rp_jhh_pear_eef_probe_'+identity,output=experiment+'/native',
        transport_root=experiment+'/transport',finished=experiment+'/finished.json',
        head_heartbeat=experiment+'/head-heartbeat.json')
    build=BUILD_ROOT/('rp-jhh-pear-eef-probe-'+identity);build.mkdir(exist_ok=True)
    for name,path in files.items():shutil.copyfile(path,build/name)
    write(build/'spec.json',spec)
    write(build/'source-manifest.json',{name:hashlib.sha256((build/name).read_bytes()).hexdigest()
        for name in [*files,'spec.json']})
    return build


def launch(build):
    query=command(['osmo','workflow','query',WORKFLOW,'--format-type','json']);running_tasks(query)
    build=Path(build);spec=read(build/'spec.json');bundle=spec['probe_root']+'/bundle'
    script='''from pathlib import Path
import json,subprocess
root=Path(__ROOT__);root.mkdir(mode=0o700,parents=True,exist_ok=True)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if gpu:raise RuntimeError('Probe GPUs are no longer idle; preserve existing work')
record=root/'process.json'
if record.exists():
 old=json.loads(record.read_text())
 try:
  command=(Path('/proc')/str(old['pid'])/'cmdline').read_bytes()
  if str(root/'bundle'/'efe_pear_eef_probe.py').encode()in command:
   print('EFE_CAPTURE '+json.dumps({**old,'already_running':True}),flush=True);raise SystemExit(0)
 except FileNotFoundError:pass
Path(__BUNDLE__).mkdir(mode=0o700,parents=True,exist_ok=True)
print('EFE_CAPTURE '+json.dumps({'ready':True}),flush=True)
'''.replace('__ROOT__',repr(spec['probe_root'])).replace('__BUNDLE__',repr(bundle))
    ready=remote(WORKFLOW,script,task='native')
    if ready.get('already_running'):return ready
    result=subprocess.run(['osmo','workflow','rsync','upload',WORKFLOW,'native',
        str(build)+'/:'+bundle+'/','--no-progress'],capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError('Native probe staging failed: '+result.stderr[-500:])
    script='''from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(__ROOT__);bundle=Path(__BUNDLE__)
for name,expected in json.loads((bundle/'source-manifest.json').read_text()).items():
 if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=expected:raise ValueError('Staged PEAR probe source changed')
if(root/'finished.json').exists():raise RuntimeError('This exact probe already finished; inspect its receipt before retrying')
env=dict(os.environ);env['PEAR_PRIMARY_TRAINING']='0';env['PYTHONPATH']=str(bundle)+':'+env.get('PYTHONPATH','')
for key in list(env):
 if key.startswith(('OPENAI_','ASTRA_')):env.pop(key,None)
with(root/'probe.log').open('a')as log:
 child=subprocess.Popen(['/opt/gr00t-sim/venv/bin/python','-u',str(bundle/'efe_pear_eef_probe.py')],
  env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':child.pid,'root':str(root),'bundle':str(bundle),'workflow':__WORKFLOW__,
 'started_unix':time.time(),'teacher_model_calls':0,'optimizer_updates':0,'allocation_preserved':True}
(root/'process.json').write_text(json.dumps(value));print('EFE_CAPTURE '+json.dumps(value),flush=True)
'''.replace('__ROOT__',repr(spec['probe_root'])).replace('__BUNDLE__',repr(bundle)).replace('__WORKFLOW__',repr(WORKFLOW))
    receipt=remote(WORKFLOW,script,task='native')
    write(ROOT/'yam-native-probe-launch.json',{**receipt,'build':str(build)})
    return receipt


def launch_rtx(suite='robodojo'):
    import yaml
    from .launch_efe_pear_primary import worker_build
    from .efe_pear_submission import submit_reconciled
    plan=read(ROOT/'native-pilot-capacity-handoff.json')
    if suite=='robodojo':old=Path(plan['dojo']['qualified_build'])
    else:
        states=read(DEST/'astra-wide-warmstart-v1/launches.json')
        options=[Path(v['build'])for v in states.values()if not v.get('superseded_by')
            and v.get('label')==suite]
        if not options:raise ValueError('No qualified native image/template for '+suite)
        old=options[-1]
    native=read(old/'spec.json');files={name:old/name for name in read(old/'source-manifest.json')
        if name.endswith('.py')and name!='bootstrap.py'and not name.startswith('efe_astra')}
    current=['efe_pear_rtx_env.py','efe_pear_native_entry.py','efe_pear_primary_rtx.py','efe_pear_eef.py',
        'efe_pear_eef_service.py','efe_action_loop.py','efe_pear_context.py',
        'efe_pear_eef_probe.py','efe_pear_eef_rtx_probe.py']
    files.update({name:BASE/'cluster'/name for name in current})
    for name,path in files.items():compile(path.read_text(),str(path),'exec')
    membership=read(old/'native-membership.json')
    candidates={k:v for k,v in membership['training_tasks'].items()if v['row']['suite']==suite}
    selected={name:candidates[name]for name in (
        ['robodojo-general-pickup-i00-standard','robodojo-align-blocks-i00-standard']
        if suite=='robodojo'else sorted(candidates)[:1])}
    identity=hashlib.sha256(json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest()
        for name,path in files.items()},sort_keys=True).encode()).hexdigest()[:12]
    experiment=REMOTE+'/rl-eef-probes/rp_jhh_pear_eef_'+suite+'_'+identity
    spec={k:native[k]for k in ['payload','payload_sha256','files','suite','native_tool_relative']}
    spec.update(owner_model_workflow='rp_jhh_pear_eef_dojo_'+identity,probe_root=experiment,
        output=experiment+'/native',transport_root=experiment+'/transport',
        finished=experiment+'/finished.json',head_heartbeat=experiment+'/head-heartbeat.json',
        control_interface='pear-eef-native-v1',qualification_only=True,teacher_calls_permitted=False,
        episode_archive='s3://rp-jhh-pear-checkpoints/rpunamiya/rp_jhh_pear_eef_probe_'+identity+'/native-episodes',
        probe_cases=[{'task':name,'env_args':{'task':name,'seed':item['row']['seeds']['default']},
            'scene_sha256':item['scene_sha256']}for name,item in selected.items()])
    frozen=ROOT/(suite+'-membership-'+identity+'.json');write(frozen,{'training_tasks':selected})
    files['native-membership.json']=frozen
    image=yaml.safe_load((old/'workflow.yaml').read_text())['workflow']['tasks'][0]['image']
    python=yaml.safe_load((old/'workflow.yaml').read_text())['workflow']['tasks'][0]['command'][0]
    build=worker_build(spec,files,'efe_pear_eef_rtx_probe.py',image,python,
        gpu=1,cpu=8,memory='64Gi',storage='150Gi',platform='ovx-l40')
    path=build/'workflow.yaml';workflow=yaml.safe_load(path.read_text())
    workflow['workflow']['name']='rp-jhh-pear-eef-'+suite+'-'+identity
    workflow['workflow']['labels']={'experiment':'rp_jhh_pear_eef_native_v1','phase':'model_free_native_qualification'}
    task=workflow['workflow']['tasks'][0]
    task['credentials']={'rp-jhh-pear-swift-archive-20261118':{'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}}
    path.write_text(yaml.safe_dump(workflow,sort_keys=False))
    write(build/'source-manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in build.iterdir()if p.is_file()and p.name!='source-manifest.json'})
    receipt=submit_reconciled(build,priority='LOW',pool='groot-l40-03',nodes=1)
    result={**receipt,'build':str(build),'root':experiment,'priority':'LOW','pool':'groot-l40-03',
        'teacher_model_calls':0,'optimizer_updates':0}
    write(ROOT/('dojo-native-probe-launch.json'if suite=='robodojo'else suite+'-native-probe-launch.json'),result)
    return result


def launch_core(*,reuse=False):
    import yaml
    from .launch_efe_pear_primary import worker_build
    from .efe_pear_submission import submit_reconciled
    from .launch_efe_pear_scaled import TEMPLATE,prepare_training_bundle
    package=prepare_training_bundle();common=Path(package['bundle'])
    files={name:common/name for name in package['files']}
    names=['efe_pear_primary_native.py','efe_pear_remote_worker.py','efe_pear_gpu_cdi.py',
        'efe_pear_eef_probe.py','efe_pear_eef_core_probe.py']
    files.update({name:BASE/'cluster'/name for name in names})
    for path in files.values():
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
    original=read(BUILD_ROOT/'rp-jhh-pear-primary-2490d773c60f/native-membership.json')
    chosen={}
    for suite in ['abc-sim','vlabench','dexjoco']:
        matching=sorted((k,v)for k,v in original['training_tasks'].items()if v['row']['suite']==suite)
        if not matching:raise ValueError('Missing admitted native suite '+suite)
        chosen.update(matching[:1])
    # Native nested containers need a privileged-capable platform. L40-03
    # rejects privileged jobs and the H100 platform requires whole nodes.
    layout={'gpu':8,'pool':'groot-h100-01','priority':'HIGH'}
    identity=hashlib.sha256(json.dumps([layout,chosen,{n:hashlib.sha256(p.read_bytes()).hexdigest()
        for n,p in files.items()}],sort_keys=True).encode()).hexdigest()[:12]
    experiment=REMOTE+'/rl-eef-probes/rp_jhh_pear_eef_core_'+identity
    template=read(BUILD_ROOT/'efe-pear-native-qualification-20261114-v1/spec.json')
    setup={k:template[k]for k in ['repository','revision','podman','compose']}
    setup.update(tasks=[v['row']for v in chosen.values()],podman_storage_driver='overlay')
    membership=ROOT/('core-membership-'+identity+'.json');write(membership,{'training_tasks':chosen})
    files['native-membership.json']=membership
    spec={'kind':'efe','owner_model_workflow':'rp_jhh_pear_eef_core_'+identity,'probe_root':experiment,
        'output':experiment+'/native','native_setup':setup,'lanes_per_gpu':1,'startup_stagger_seconds':10,
        'retry_native_infrastructure':False,'control_interface':'pear-eef-native-v1',
        'qualification_only':True,'teacher_calls_permitted':False,'native_membership_local':True,
        'native_config':{'qualification_only':False,'membership_file':experiment+'/bundle/native-membership.json',
            'action_config':{**template['fixed_action_config'],'model':'external-pear-policy'}},
        'suites':['abc-sim','vlabench','dexjoco'],'transport_root':experiment+'/transport',
        'finished':experiment+'/finished.json','head_heartbeat':experiment+'/head-heartbeat.json',
        'episode_archive':'s3://rp-jhh-pear-checkpoints/rpunamiya/rp_jhh_pear_eef_core_'+identity+'/native-episodes',
        'probe_cases':[{'suite':v['row']['suite'],'task':name,
            'env_args':{'task':name,'seed':v['row']['seeds']['default']},'scene_sha256':v['scene_sha256']}
            for name,v in chosen.items()]}
    # Primary-native uses lane-local adapter modules but the membership path
    # above must remain available from each independently spawned lane.
    image=yaml.safe_load(TEMPLATE.read_text())['workflow']['tasks'][0]['image']
    build=worker_build(spec,files,'efe_pear_eef_core_probe.py',image,'python3',
        gpu=8,cpu=64,memory='512Gi',storage='400Gi',platform='dgx-h100',privileged=True)
    value=yaml.safe_load((build/'workflow.yaml').read_text());workflow=value['workflow']
    workflow['name']='rp-jhh-pear-eef-core-'+identity
    workflow['labels']={'experiment':'rp_jhh_pear_eef_native_v1','phase':'model_free_native_qualification'}
    task=workflow['tasks'][0]
    task['credentials']={'rp-jhh-pear-swift-archive-20261118':{'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}}
    from .efe_pear_worker_bundle import bootstrap
    source_names=[p.name for p in build.iterdir()if p.is_file()and p.name not in ['bootstrap.py','workflow.yaml','source-manifest.json']]
    (build/'bootstrap.py').write_text(bootstrap(build,source_names,'efe_pear_allocation_keeper.py'))
    (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False))
    write(build/'source-manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in build.iterdir()if p.is_file()and p.name!='source-manifest.json'})
    previous=read(ROOT/'core-native-probe-launch.json',{})
    if reuse:
        from .efe_pear_allocation_jobs import submit as submit_job
        if not previous:raise ValueError('No retained core probe allocation to reuse')
        previous['workflow']=previous.get('workflow',previous['name'])
        receipt=submit_job(previous,build,replace_running=True)
        result={**previous,**receipt,'allocation_build':previous.get('allocation_build',previous['build']),
            'build':str(build),'root':experiment,'retained_allocation_reused':True}
    else:
        receipt=submit_reconciled(build,priority=layout['priority'],pool=layout['pool'],nodes=1)
        result={**receipt,'build':str(build),'root':experiment,'priority':layout['priority'],'pool':layout['pool']}
    result.update(teacher_model_calls=0,optimizer_updates=0)
    write(ROOT/'core-native-probe-launch.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    parser.add_argument('--approach',action='store_true')
    parser.add_argument('--reuse-core',action='store_true')
    parser.add_argument('--suite',choices=['yam','robodojo','robolab','yamlab','core'],default='yam');args=parser.parse_args()
    if args.suite=='core':
        if not args.launch:raise ValueError('Use --launch for the explicit model-free native probe')
        print(json.dumps(launch_core(reuse=args.reuse_core)),flush=True)
    elif args.suite!='yam':
        if not args.launch:raise ValueError('Use --launch for an explicitly requested RTX native probe')
        print(json.dumps(launch_rtx(args.suite)),flush=True)
    else:
        build=prepare(approach=args.approach)
        print(json.dumps(launch(build)if args.launch else{'build':str(build)}),flush=True)
