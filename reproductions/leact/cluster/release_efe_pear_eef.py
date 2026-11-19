"""Release the staged RL driver only after complete native EEF proof."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import runpy
import sys

from .scale_efe import BASE,read,write
from .launch_efe_pear_scaled import DEST
from .efe_pear_eef import VERSION,DELTA_SUITES
from .efe_pear_eef_admission import SUITES,qualify
from .efe_pear_full_admission import PendingAdmission,compile_admission


ROOT=DEST/'eef-interface-v1'
COMMON=['efe_pear_eef.py','efe_action_loop.py','efe_pear_context.py']
RTX=COMMON+['efe_pear_rtx_env.py','efe_pear_eef_service.py','efe_pear_native_entry.py']
YAM=COMMON+['efe_pear_yam_eef.py','efe_pear_yam_session_policy.py','efe_pear_yam_server.py']
CORE=COMMON+['efe_action_agent.py','efe_pear_bridge.py','efe_pear_native_episode.py','efe_pear_primary_native.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def executable_ast(path):
    tree=ast.parse(Path(path).read_text())
    if tree.body and isinstance(tree.body[0],ast.Expr)and isinstance(tree.body[0].value,ast.Constant)and isinstance(tree.body[0].value.value,str):
        tree.body.pop(0)
    return ast.dump(tree,include_attributes=False)


def contract_metadata_equivalent(before,after,suite,contracts):
    """Check recorded RTX schemas when only shared declaration text changed.

    The controller, parser, gripper conversion and all other helper functions
    must remain identical. Test both declarations against the actual measured
    vectors, retaining every execution field and excluding two prose fields.
    """
    if suite not in {'robodojo','robolab','yamlab'}or not contracts:return False
    def outside_contract(path):
        tree=ast.parse(Path(path).read_text())
        tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=='contract')]
        return ast.dump(tree,include_attributes=False)
    if outside_contract(before)!=outside_contract(after):return False
    def executable_fields(value):
        value=copy.deepcopy(value)
        for entry in value['action_types'].values():
            for field in ['units','native_semantics']:entry.pop(field,None)
        return value
    old_path=sys.path[:]
    try:
        sys.path.insert(0,str(BASE/'cluster'))
        implementations=[runpy.run_path(str(p))['contract']for p in [before,after]]
        for measured_contract in contracts:
            value=measured_contract['action_contract']
            spec={'control_dt':value['control_dt'],'max_chunk_s':value['max_repeat']*value['control_dt'],
                'cameras':{n:{'rgb':True}for n in value['cameras']},'action_types':value['action_types']}
            if any(executable_fields(fn(spec,suite,value['max_repeat']))!=executable_fields(value)for fn in implementations):return False
    finally:sys.path[:]=old_path
    return True


def source_pins(build,names,*,suite=None,contracts=None):
    result={};equivalent=[]
    for name in names:
        frozen=Path(build)/name;current=BASE/'cluster'/name
        if sha(frozen)!=sha(current):
            if executable_ast(frozen)!=executable_ast(current):
                if name!='efe_pear_eef.py'or not contract_metadata_equivalent(frozen,current,suite,contracts):
                    raise PendingAdmission('Native motion proof predates an executable adapter change: '+name)
                basis='Recorded native RTX execution schemas unchanged; only units/native_semantics prose differs for this suite. All code outside contract() is identical; Dex-specific declaration is separately measured in the current native core proof.'
            else:basis='Identical executable AST; module docstring, comments or formatting only.'
            equivalent.append({'file':name,'measured_sha256':sha(frozen),'current_sha256':sha(current),'basis':basis})
        result[name]=sha(current)
    return result,equivalent


def measured(status,*,suite=None,count):
    checks=[c for c in status.get('checks',[])if suite is None or c.get('suite')==suite]
    if len(checks)!=count or not all(c.get('passed')is True for c in checks):return False
    if suite is None:return status.get('phase')=='finished'and status.get('passed')is True
    tasks={c['task']for c in checks}
    return (status.get('phase')in {'finished','failed'}and
        tasks<=set(status.get('finalized_tasks',[]))and
        not(tasks&set(status.get('case_errors',{}))))


def assemble():
    admission=compile_admission(DEST);simulators={};pins={}
    for suite in sorted(DELTA_SUITES):
        receipt=DEST/'validation-receipts'/suite/'checks.json'
        values=read(receipt)
        tasks={k:v for k,v in admission['native_membership']['training_tasks'].items()if v['row']['suite']==suite}
        if not tasks or any(not values[k].get('success')or values[k].get('oracle_success')!=1 or
            values[k].get('nop_success')!=0 or set(values[k]['action_contract']['action_types'])!={'ee_delta'}for k in tasks):
            raise PendingAdmission('Existing Cartesian controls are not qualified: '+suite)
        simulators[suite]={'passed':True,'receipt':str(receipt),'receipt_sha256':sha(receipt),
            'basis':'Execution remains the previously qualified native Cartesian delta controller.',
            'admitted_training_tasks':len(tasks)}
    for label,suite,count,names in [('yam','native_gr00t_yam',14,YAM),('dojo','robodojo',12,RTX),
        ('robolab','robolab',6,RTX),('yamlab','yamlab',6,RTX),
        ('core','abc-sim',5,CORE),('core','vlabench',5,CORE+['efe_pear_vlabench_eef.py']),
        ('dexjoco-qualified','dexjoco',5,CORE)]:
        state=read(ROOT/(label+'-native-probe-launch.json'),{})
        latest=read(ROOT/(label+'-native-probe-latest.json'),{})
        status=latest.get('status',{})
        if latest.get('probe_root')!=state.get('root')or not measured(status,
            suite=suite if suite in {'abc-sim','vlabench','dexjoco'}else None,count=count):
            raise PendingAdmission('Native EEF motion/finalization still pending: '+suite)
        build=state.get('native_server_build',state['build'])
        contract_path=ROOT/(label+'-measured-contracts.json')
        contracts=read(contract_path,[])
        current,equivalent=source_pins(build,names,suite=suite,contracts=contracts);pins.update(current)
        receipt=ROOT/('qualified-'+suite+'.json')
        proof={'suite':suite,'passed':True,'control_interface':VERSION,'workflow':state.get('workflow',state.get('name')),
            'probe_root':state['root'],'source_build':build,'status':status,
            'source_pins':current,'source_equivalence':equivalent}
        if contracts:proof.update(measured_contracts=str(contract_path),measured_contracts_sha256=sha(contract_path))
        write(receipt,proof)
        simulators[suite]={'passed':True,'receipt':str(receipt),'receipt_sha256':sha(receipt),
            'basis':'Discarded model-free control and native finalization checks.'}
    if set(simulators)!=SUITES:raise PendingAdmission('Incomplete EEF simulator mixture')
    proof={'control_interface':VERSION,'scope':'pear_rl_only','teacher_collection_changed':False,
        'passed':True,'simulators':simulators,'adapter_sha256':pins}
    previous=read(ROOT/'qualification.json',{})
    if {k:v for k,v in previous.items()if k!='updated_unix'}==proof:return previous
    proof['updated_unix']=time.time()
    write(ROOT/'qualification.json',proof);qualify(DEST,BASE/'cluster')
    return proof


def publish():
    proof=assemble();state=read(ROOT/'model-runtime-launch.json');spec=read(Path(state['build'])/'spec.json')
    from .efe_pear_allocation_jobs import running_tasks
    from .scale_efe import command
    from .monitor_efe_preflight import remote
    running_tasks(command(['osmo','workflow','query',state['workflow'],'--format-type','json']))
    release={**proof,'run':spec['run'],'qualification_sha256':sha(ROOT/'qualification.json')}
    if any(spec['adapter_digests'].get(n)!=h for n,h in proof['adapter_sha256'].items()):
        raise ValueError('The staged model bundle changed after native qualification')
    script='''from pathlib import Path
import json,os
value=json.loads(__VALUE__);path=Path(__PATH__)
if path.exists():
 old=json.loads(path.read_text())
 if old['run']!=value['run']or old['adapter_sha256']!=value['adapter_sha256']:raise ValueError('Different EEF release already exists')
else:
 tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
print('EFE_CAPTURE '+json.dumps({'released':True,'path':str(path),'run':value['run']}),flush=True)
'''.replace('__VALUE__',repr(json.dumps(release))).replace('__PATH__',repr(spec['eef_release']))
    result=remote(state['workflow'],script,task='trainer')
    write(ROOT/'release.json',result)
    transition=read(ROOT/'transition.json');transition.update(phase='native_qualified_driver_released',
        training_activated=True,qualification_sha256=release['qualification_sha256'],updated_unix=time.time())
    write(ROOT/'transition.json',transition)
    return result


def tick(state):
    if state.get('control_interface')!=VERSION:return
    released=read(ROOT/'release.json',{})
    if released.get('released')and released.get('run')==state['run']:return released
    from .monitor_efe_pear_eef_probe import tick as observe
    observe()
    try:result=publish()
    except PendingAdmission as exc:
        result={'released':False,'phase':'waiting_for_complete_native_eef_qualification',
            'reason':str(exc),'run':state['run'],'updated_unix':time.time()}
    write(ROOT/'release-status.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();print(json.dumps(publish()if args.publish else assemble()),flush=True)
