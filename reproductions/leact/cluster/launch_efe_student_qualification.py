"""Qualify a coding-capable open-weight student on three fixed EFE train scenes."""
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

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, command, submit
from .compute_budget import reserved_nodes, node_limit
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect
from .launch_efe_native_followups import LANE, collect_arm, verified_native_completion
from .launch_efe_native_mirror import ensure as ensure_mirror

DEST = ROOT / 'structured-control-v2'
BUILD = BUILD_ROOT / 'efe-qwen36-native-qualification-20261103-v1'
SOURCE = BUILD_ROOT / 'efe-direct-base-aligned-20261030-v1'
MODEL_ROOT = '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-qwen36-20261103-v1'
ARCHIVE = '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-qwen36-qualification-20261103-v1'
MODEL = 'Qwen/Qwen3.6-35B-A3B'
REVISION = '995ad96eacd98c81ed38be0c5b274b04031597b0'
IMAGE = 'docker.io/vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90'
TASKS = ['metaworldplus-handle-press-side-i00-standard',
         'metaworldplus-push-wall-i00-standard',
         'metaworldplus-pick-out-of-hole-i00-standard']
ARM = {'id': 'qwen36-base', 'label': 'Qwen3.6-35B-A3B · native coding baseline',
       'directory': 'structured-control-v2/qwen36-base', 'build': str(BUILD)}


def now():
    return datetime.now(timezone.utc).isoformat()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen source hook changed: ' + old[:100])
    return text.replace(old, new)


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    (ROOT / ARM['directory']).mkdir(parents=True, exist_ok=True)
    if (BUILD / 'source-manifest.json').exists():
        return BUILD
    BUILD.mkdir(exist_ok=True)
    workflow = yaml.safe_load((SOURCE / 'workflow.yaml').read_text())
    for item in workflow['workflow']['tasks'][0]['files']:
        shutil.copyfile(SOURCE / item['localpath'], BUILD / item['localpath'])
    spec = read(BUILD / 'spec.json')
    by_task = {r['task']: r for r in spec['tasks']}
    splits = read(ROOT / 'generalization-v2/splits.json')
    if not set(TASKS).issubset(splits['train_candidates']):
        raise ValueError('Qualification must remain in the frozen training partition')
    spec.update(model_id=ARM['id'], agent_label=ARM['label'], agent_slug='qwen36-base',
        request_model=MODEL, harbor_model='openai/' + MODEL,
        student_checkpoint=MODEL_ROOT + '/base-model',
        student_manifest=MODEL_ROOT + '/checkpoint-manifest.json', student_revision=REVISION,
        serving_version='vllm==0.19.0', serving_image=IMAGE,
        tasks=[by_task[t] for t in TASKS], local_results_directory=ARM['directory'],
        export_role='training_qwen36_qualification', preflight_semantics_policy='stop',
        preflight_debug_seconds=60, protocol_version='native-coding-qualification-v1',
        reasoning_parser='qwen3',
        protocol_label='Three fixed MetaWorld training scenes, native Codex client, same direct-control suffix and grader; Qwen3.6 model and vLLM 0.19 serving.',
        scope='Robot-specific base-model qualification. Code execution, issued control, measured movement and task success are separate. No held-out score or training.',
        purpose='Establish whether an open-weight coding student already controls the robot before new SFT.',
        comparison_to_development='Different model architecture, tokenizer, vision encoder, parser, serving version and TP=8. Shared visible observations, instructions, per-turn output cap, native task deadlines and grader. Do not attribute differences to weights alone.',
        reasoning_token_measurement='Record provider usage; zero or missing reasoning count is not evidence of absent reasoning.',
        stream_transport='Reuse the recorded Responses event bridge; validate image/tool continuation before native tasks.',
        preflight_change='Require correct synthetic answers and valid native tool transport for this new model/runtime.',
        serving_transport={'mode': 'nccl_no_p2p_v1', 'runtime_workaround_passed': False,
            'reason': 'Retain the cluster NCCL transport workaround; the new model and runtime require validation.',
            'speed_comparison_eligible': False})
    spec['execution'] = {**spec['execution'], 'gpu_ids': '0,1,2,3,4,5,6,7'}
    write(BUILD / 'spec.json', spec)
    path = BUILD / 'native-student.py'
    source = path.read_text()
    replacements = [
        ("'--tensor-parallel-size','4'", "'--tensor-parallel-size','8'"),
        ("'--max-num-seqs','4'", "'--max-num-seqs','1'"),
        ("'--tool-call-parser','hermes','--reasoning-parser','deepseek_r1'",
         "'--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3'"),
        ("'CUDA_VISIBLE_DEVICES':'0,1,2,3'", "'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'"),
        ('Starting Qwen3-VL-8B-Thinking on four H100 GPUs; native development evaluation follows the image/tool preflight.',
         'Starting Qwen3.6-35B-A3B on eight H100 GPUs; three native training-scene trials follow image/tool qualification.'),
        ('Explicit Qwen3-VL image size matches the curated HF processor. Compare this condition to base-aligned; legacy controls used different image preprocessing.',
         'Model-native vision encoder with explicit 65536 to 1048576 pixel bounds, matching the previous input pixel bounds. This model has different patch/token encoding.'),
        ('deepseek_r1 uses the same think delimiters and accepts the opening tag already supplied by Qwen3-VL Thinking chat template; qwen3 in vLLM 0.16 incorrectly requires the opening tag in generated text.',
         'Qwen model-card serving configuration: qwen3 reasoning parser and qwen3_coder tool parser on vLLM 0.19.'),
    ]
    for before, after in replacements:
        source = replace_once(source, before, after)
    path.write_text(source)
    download = 'efe_prepare_qualification_student.py'
    shutil.copyfile(BASE / 'cluster' / download, BUILD / download)
    path = BUILD / 'native-bootstrap.py'
    source = replace_once(path.read_text(), "os.execve(python, [python, '-u', str(lane / 'worker.py'), '--run'], env)",
        "subprocess.run([python, '-u', str(lane / 'efe_prepare_qualification_student.py')], env=env, check=True)\n"
        "os.execve(python, [python, '-u', str(lane / 'worker.py'), '--run'], env)")
    path.write_text(source)
    workflow['workflow']['name'] = 'leact-efe-qwen36-qualification-v1'
    workflow['workflow']['timeout'] = {'queue_timeout': '20m', 'exec_timeout': '10h'}
    task = workflow['workflow']['tasks'][0]
    task['image'] = IMAGE
    task['files'].append({'localpath': download, 'path': LANE + '/' + download})
    task.pop('credentials', None)
    (BUILD / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    for path in BUILD.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    freeze_sources(BUILD)
    plan = {
        'created_utc': now(), 'stage': 'base_student_qualification', 'training_enabled': False,
        'summary': 'First establish valid code execution and robot movement with a coding-capable open-weight student. Then compare base, Astra warmstart and LeAct in a fixed shared interface.',
        'model': MODEL, 'revision': REVISION, 'model_url': 'https://huggingface.co/' + MODEL,
        'model_caveat': 'Candidate selected for coding and vision support; robot competence is unmeasured. The current Qwen3-VL trainer is not compatible without adaptation and a training smoke test.',
        'status': 'Prepared; submission tracked separately.', 'arm': ARM, 'tasks': TASKS,
        'baseline': 'Base Qwen3-VL-8B-Thinking on these same training scenes',
        'design': [
            'Use the existing native coding harness first, keeping the task instructions, direct-control suffix, visible observations and native grader.',
            'Report valid tool calls, executable client code, accepted motion commands, measured end-effector movement, recovery from errors and task success separately. Accepted control steps alone do not prove movement.',
            'Use these three existing training scenes to qualify the student. Do not choose held-out tasks by student success or train on published core solutions.',
            'If basic execution still fails, test a fixed observation/action interface with both students and Astra. Keep that interface comparison separate from the model comparison.',
            'After qualification and training-runtime validation, freeze the student/interface and compare its base model, Astra backward-reasoning warmstart, and warmstart plus LeAct.',
        ],
        'gates': [
            {'name': 'Image and tool transport', 'criterion': 'Pass image identification, function call, returned text and new tool-image checks before native trials.'},
            {'name': 'Native robot baseline', 'criterion': 'Run all three preselected scenes once; publish failures as well as successes. Inspect actual motion and code errors before declaring the candidate qualified.'},
            {'name': 'Directed movement check', 'criterion': 'Before new SFT, verify direction and displacement on nine short commands across the three scenes; target at least eight passes without human code repair. This diagnostic is separate from native task scores.', 'status': 'Planned; not implemented or running.'},
            {'name': 'Training compatibility', 'criterion': 'Adapt and validate the new MoE architecture, image preprocessing and likelihood scoring; verify a real parameter update and reload before full SFT.', 'status': 'Pending candidate behavior.'},
        ],
        'compute': {'nodes': 1, 'h100_per_node': 8, 'priority': 'HIGH', 'capacity_fallback': 'LOW', 'global_node_limit': node_limit()},
        'paused_work': ['BuilderBench', 'general capability regression'],
    }
    write(DEST / 'plan.json', plan)
    return BUILD


def retry_build(original, number):
    retry = Path(str(original) + '-low' + str(number))
    retry.mkdir(exist_ok=True)
    if not (retry / 'source-manifest.json').exists():
        for name in read(original / 'source-manifest.json'):
            shutil.copyfile(original / name, retry / name)
        workflow = yaml.safe_load((retry / 'workflow.yaml').read_text())
        workflow['workflow']['name'] += '-low' + str(number)
        workflow['workflow']['timeout']['queue_timeout'] = '12h'
        (retry / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
        freeze_sources(retry)
    return retry


def prepare_role_retry():
    """Version the confirmed first-request role error; preserve the failed arm."""
    build = BUILD_ROOT / 'efe-qwen36-native-qualification-20261103-v2'
    build.mkdir(exist_ok=True)
    if not (build / 'source-manifest.json').exists():
        for name in read(BUILD / 'source-manifest.json'):
            shutil.copyfile(BUILD / name, build / name)
        spec = read(build / 'spec.json')
        spec.update(developer_role_adapter=True,
            local_results_directory='structured-control-v2/qwen36-base-r2',
            protocol_version='native-coding-qualification-v2-role-adapter',
            protocol_label=spec['protocol_label'] + ' Explicit developer-to-system instruction adapter; original requests retained for audit.',
            preflight_change='Include a developer message in every image/tool check to exercise the full-agent role schema.',
            retry_of={TASKS[0]: {'workflow': 'leact-efe-qwen36-qualification-v1-1',
                'job': '00-qwen36-base', 'reason': 'HTTP 400 Unexpected message role before any successful model request or robot action.'}},
            role_adapter={'condition': 'developer-to-system-preamble-v1',
                'scope': 'Preserve developer text and source order in a labeled system-preamble section subordinate to original system instructions.',
                'original_requests_retained': True})
        write(build / 'spec.json', spec)
        shutil.copyfile(BASE / 'cluster/efe_budget_gateway.py', build / 'native-efe_budget_gateway.py')
        path = build / 'native-student.py'
        source = replace_once(path.read_text(), "'--flatten-namespaces','--terminal-output-events']",
                              "'--flatten-namespaces','--terminal-output-events','--developer-role-adapter']")
        source = replace_once(source, "    tools=[{'type':'namespace','name':'probe'",
            "    inputs=[{'role':'developer','content':[{'type':'input_text','text':'Use the supplied tools and answer from the observations. Do not invent an observation.'}]}]+inputs\n"
            "    tools=[{'type':'namespace','name':'probe'")
        path.write_text(source)
        path = build / 'native-pilot.py'
        source = replace_once(path.read_text(),
            "                + (['--align-tool-instructions'] if spec.get('align_tool_instructions') else [])",
            "                + (['--align-tool-instructions'] if spec.get('align_tool_instructions') else [])\n"
            "                + (['--developer-role-adapter'] if spec.get('developer_role_adapter') else [])")
        path.write_text(source)
        workflow = yaml.safe_load((build / 'workflow.yaml').read_text())
        workflow['workflow']['name'] = 'leact-efe-qwen36-qualification-v2'
        (build / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
        for path in build.glob('*.py'):
            compile(path.read_text(), str(path), 'exec')
        freeze_sources(build)
    history = DEST / 'attempts'
    history.mkdir(exist_ok=True)
    if not (history / 'attempt-1.json').exists():
        write(history / 'attempt-1.json', {'status': read(DEST / 'status.json', {}),
            'results': read(ROOT / ARM['directory'] / 'results.json', {}),
            'build': str(BUILD), 'excluded_from_task_success': True})
    write(DEST / 'active-build.json', {'build': str(build), 'attempt': 2, 'priority': 'HIGH'})
    plan = read(DEST / 'plan.json')
    plan.update(active_build=str(build), role_adapter=read(build / 'spec.json')['role_adapter'],
        status='Retry prepared after an unsupported developer role caused the first full-agent request to fail.',
        updated_utc=now())
    write(DEST / 'plan.json', plan)
    return build


def prepare_typed_comparison():
    """Keep the three graded v2 failures and test a corrected serving schema."""
    source = BUILD_ROOT / 'efe-qwen36-native-qualification-20261103-v2'
    build = BUILD_ROOT / 'efe-qwen36-native-qualification-20261103-v3'
    build.mkdir(exist_ok=True)
    if (build / 'source-manifest.json').exists():
        return build
    for name in read(source / 'source-manifest.json'):
        shutil.copyfile(source / name, build / name)
    spec = read(build / 'spec.json')
    spec.pop('retry_of', None)
    spec.update(agent_slug='qwen36-typed', export_role='training_qwen36_typed',
        local_results_directory='structured-control-v2/qwen36-base-r3',
        student_checkpoint='/tmp/efe-qwen36-model/base-model',
        student_manifest='/tmp/efe-qwen36-model/checkpoint-manifest.json',
        native_mirror_enabled=False,
        protocol_version='native-coding-qualification-v3-responses-tool-schema',
        protocol_label=spec['protocol_label'] + ' Responses-format function schemas retain integer, boolean and array argument types.',
        preflight_change='Require typed integer, boolean and array arguments as well as the existing image, developer-role and tool continuation checks.',
        serving_comparison={'parent_workflow': 'leact-efe-qwen36-qualification-v2-1',
            'parent_graded': 3, 'parent_successes': 0, 'parent_control_steps': 0,
            'change': 'Extend the Qwen3 coder parser schema lookup to Responses-format function tools. Preserve the upstream value conversion rules.',
            'parent_results_retained': True, 'task_outcomes_excluded': False},
        storage_policy='Pinned public model downloaded to node-local storage. Native traces, videos and result checksums collected locally before releasing the allocation.')
    write(build / 'spec.json', spec)
    plugin = 'efe_qwen36_tool_parser.py'
    shutil.copyfile(BASE / 'cluster' / plugin, build / plugin)
    path = build / 'native-student.py'
    text = replace_once(path.read_text(), "'--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3'",
        "'--tool-call-parser','qwen3_coder_responses','--tool-parser-plugin','/tmp/efe_qwen36_tool_parser.py','--reasoning-parser','qwen3'")
    text = replace_once(text,
        'Call report_color with the dominant color of this image. Use exactly one tool call.',
        'Call report_color with the dominant color of this image, count set to the integer 3, enabled set to boolean true, and values set to the array [1, 2]. Use exactly one tool call.')
    text = replace_once(text,
        "'properties':{'color':{'type':'string','enum':['red','green','blue']}},\n                          'required':['color']",
        "'properties':{'color':{'type':'string','enum':['red','green','blue']},\n"
        "                'count':{'type':'integer'},'enabled':{'type':'boolean'},\n"
        "                'values':{'type':'array','items':{'type':'number'}}},\n"
        "                          'required':['color','count','enabled','values']")
    text = replace_once(text, "set(arguments)!={'color'}", "set(arguments)!={'color','count','enabled','values'}")
    text = replace_once(text,
        "    check_answer('initial_image_color',arguments['color']=='green','Student initial image-color preflight failed')",
        "    check_answer('typed_tool_arguments',type(arguments['count']) is int and arguments['count']==3\n"
        "                 and arguments['enabled'] is True and isinstance(arguments['values'],list)\n"
        "                 and arguments['values']==[1,2] and all(type(v) in (int,float) for v in arguments['values']),\n"
        "                 'Student Responses tool arguments did not retain the declared JSON types')\n"
        "    check_answer('initial_image_color',arguments['color']=='green','Student initial image-color preflight failed')")
    text = replace_once(text, 'Qwen model-card serving configuration: qwen3 reasoning parser and qwen3_coder tool parser on vLLM 0.19.',
        'Qwen3 reasoning parser and Qwen3 coder tool parser with Responses-format schema lookup on vLLM 0.19.')
    path.write_text(text)
    workflow = yaml.safe_load((build / 'workflow.yaml').read_text())
    workflow['workflow']['name'] = 'leact-efe-qwen36-qualification-v3'
    workflow['workflow']['tasks'][0]['files'].append({'localpath': plugin, 'path': '/tmp/' + plugin})
    (build / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    for path in build.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    freeze_sources(build)
    return build


def monitor():
    prepare()
    lock = (DEST / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from ..embodied_first_exam.report import publish
    while True:
        try:
            active_build = read(DEST / 'active-build.json', {'build': str(BUILD), 'attempt': 1})
            build = Path(active_build['build'])
            spec = read(build / 'spec.json')
            arm = {**ARM, 'directory': spec['local_results_directory'], 'build': str(build)}
            local = ROOT / arm['directory']
            local.mkdir(parents=True, exist_ok=True)
            receipt = read(build / 'submission.json')
            if not receipt:
                active = command(['osmo', 'workflow', 'list', '--name', 'leact', '--status',
                                  'RUNNING', 'PENDING', 'WAITING', '--format-type', 'json'])['workflows']
                if reserved_nodes(active) >= node_limit():
                    write(DEST / 'status.json', {'updated_utc': now(), 'status': 'WAITING', 'detail': 'Waiting for an authorized node slot.', 'graded': 0, 'planned': 3})
                    publish()
                    time.sleep(30)
                    continue
                receipt = submit(build, priority=active_build.get('priority', 'HIGH' if active_build['attempt'] == 1 else 'LOW'))
                write(DEST / 'active-build.json', active_build)
            workflow = receipt['name']
            storage_hold = read(DEST / 'storage-recovery-hold.json', {})
            if storage_hold.get('workflow') == workflow and not storage_hold.get('complete'):
                # Keep a finished evaluator available until the independent
                # checksum/copy worker has finished; grading is unaffected.
                arm['remote_archive'] = LANE + '/results/' + workflow
            state = command(['osmo', 'workflow', 'query', workflow, '--format-type', 'json'])
            write(DEST / 'scheduler.json', state)
            complete = read(local / 'worker-complete.json', {})
            if state['status'] == 'RUNNING':
                collect(workflow, str(Path(spec['student_manifest']).parent), DEST / 'model-preparation', ['preparation.json'])
                if spec.get('native_mirror_enabled', True):
                    try:
                        ensure_mirror(workflow, ARCHIVE, DEST)
                    except Exception as error:
                        write(DEST / 'mirror-start-error.json', {'updated_utc': now(), 'workflow': workflow,
                              'error': str(error)[-1800:], 'local_collection_continues': True})
                if not complete:
                    complete = collect_arm(arm, workflow)
            behavior = read(local / 'behavior.json', {})
            progress = read(local / 'progress.json', {})
            preparation = read(DEST / 'model-preparation/preparation.json', {})
            status = {'updated_utc': now(), 'workflow': workflow, 'status': state['status'],
                      'detail': progress.get('detail', preparation.get('phase', state['status'])),
                      'graded': behavior.get('graded', 0), 'successes': behavior.get('successes', 0),
                      'planned': 3, 'control_steps': sum(t['control_steps'] for t in behavior.get('trials', [])),
                      'movement_verified': None, 'training_started': False, 'model': MODEL,
                      'directory': arm['directory'],
                      'attempt': active_build['attempt'], 'complete': complete, 'progress': progress}
            if state['status'].startswith('FAILED') or state['status'] == 'COMPLETED':
                complete = complete or verified_native_completion(arm)
                if not complete and state['status'] in {'FAILED_QUEUE_TIMEOUT', 'FAILED_PREEMPTED', 'FAILED_START_ERROR', 'FAILED_START_TIMEOUT'} and not (local / 'worker-start.json').exists():
                    number = active_build['attempt']
                    retry = retry_build(build, number)
                    write(DEST / 'active-build.json', {'build': str(retry), 'attempt': number + 1, 'priority': 'LOW'})
                    status['detail'] = 'Allocation failed before native trials; prepared LOW-priority retry.'
                elif not complete:
                    status['detail'] = 'Workflow ended without verified native completion; inspect runtime evidence before resuming.'
                    status['needs_runtime_review'] = True
            status['complete'] = complete
            write(DEST / 'status.json', status)
            publish()
            if complete or status.get('needs_runtime_review'):
                return
        except Exception as error:
            write(DEST / 'monitor-error.json', {'updated_utc': now(), 'error': str(error)[-1800:]})
        time.sleep(30)


def start():
    prepare()
    with (DEST / 'monitor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'monitor'],
            cwd=BASE.parents[1], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(DEST / 'monitor-launch.json', {'pid': process.pid, 'updated_utc': now()})
    return process.pid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'prepare-role-retry', 'prepare-typed-comparison', 'start', 'monitor'])
    args = parser.parse_args()
    if args.action == 'monitor':
        monitor()
    elif args.action == 'start':
        print(json.dumps({'pid': start(), 'build': read(DEST/'active-build.json',{}).get('build',str(BUILD))}))
    elif args.action == 'prepare-role-retry':
        print(prepare_role_retry())
    elif args.action == 'prepare-typed-comparison':
        print(prepare_typed_comparison())
    else:
        print(prepare())
