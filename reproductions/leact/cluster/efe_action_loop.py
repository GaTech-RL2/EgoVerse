"""A fixed public observation/action interface, run inside the native agent container.

The model chooses every action. This bridge does no planning, code repair,
object localization, inverse kinematics, or access to private simulator state.
"""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


PROTOCOL = 'efe-constrained-action-v3'
PARSER_VERSION = 'single-json-fence-v4'


def native_contract(spec, max_repeat=25):
    """Expose supported native vectors without inventing a robot-independent space."""
    dt = spec.get('control_dt')
    horizon = spec.get('max_chunk_s')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in [dt, horizon]):
        raise ValueError('Native control_dt and max_chunk_s must be finite and positive.')
    if type(max_repeat) is not int or max_repeat < 1:
        raise ValueError('Invalid repeat limit.')
    limit = min(max_repeat, math.floor(horizon / dt + 1e-9))
    if limit < 1:
        raise ValueError('Native chunk budget cannot hold one control step.')
    supported = {}
    for name, entry in spec['action_types'].items():
        if name == 'not_yet':
            continue
        if not isinstance(entry, dict):
            raise ValueError('Unsupported native action description: '+name)
        # ESE also publishes a non-vector termination operation. The common
        # bridge already exposes termination as its explicit stop object; do
        # not mistake the auxiliary operation for a malformed motion vector.
        if name=='done' and entry.get('dim') in (None,0) and not entry.get('layout'):
            continue
        dim = entry.get('dim')
        if type(dim) is not int or dim <= 0 or len(entry.get('layout', [])) != dim:
            raise ValueError('Missing native action dimension/layout: '+name)
        lower, upper = entry.get('lower'), entry.get('upper')
        if any(not isinstance(x, list) or len(x) != dim for x in [lower, upper]):
            raise ValueError('Missing native component bounds: '+name)
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in lower+upper):
            raise ValueError('Unsupported nonfinite native action bounds: '+name)
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError('Inverted native action bounds: '+name)
        supported[name] = entry
    if not supported:
        raise ValueError('No supported native vector actions.')
    cameras = spec['cameras']
    preferred = ['front', 'left', 'right', 'topview']
    names = [c for c in preferred if c in cameras] + sorted(c for c in cameras if c not in preferred)
    names = [c for c in names if cameras[c].get('rgb')][:4]
    if not names:
        raise ValueError('No declared RGB cameras.')
    return {'protocol': PROTOCOL, 'control_dt': dt, 'max_repeat': limit,
            'action_types': supported, 'cameras': names,
            'output_keys': ['action_type', 'action', 'repeat', 'summary']}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: '+key)
        result[key] = value
    return result


def parse_action(text, *, contract, allow_stop):
    # Accept a presentation wrapper around exactly one JSON document. Never
    # extract commands from prose, multiple blocks, or repair invalid fields.
    fenced = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*', text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError('Return one JSON object.')
    if value.get('stop') is True:
        if not allow_stop or set(value) != {'stop','summary'}:
            raise ValueError('This movement diagnostic requires an action, not stop.')
        if not isinstance(value['summary'], str) or len(value['summary']) > 2000:
            raise ValueError('summary must be a short string.')
        return value
    if set(value) != {'action_type','action','repeat','summary'}:
        raise ValueError('Required keys are action_type, action, repeat, summary.')
    kind = value['action_type']
    if not isinstance(kind, str) or kind not in contract['action_types']:
        raise ValueError('action_type must name a supported action in the supplied contract.')
    native = contract['action_types'][kind]
    action = value['action']
    if not isinstance(action, list) or len(action) != native['dim'] or any(
            type(x) not in (int,float) or not math.isfinite(x) for x in action):
        raise ValueError('action must contain exactly '+str(native['dim'])+' finite numbers.')
    if any(not lo <= x <= hi for x, lo, hi in zip(action, native['lower'], native['upper'])):
        raise ValueError('action exceeds the declared native component bounds.')
    if native.get('dtype', '').startswith(('int', 'uint')) and any(type(x) is not int for x in action):
        raise ValueError('This native action type requires integer components.')
    if type(value['repeat']) is not int or not 1 <= value['repeat'] <= contract['max_repeat']:
        raise ValueError('repeat must be an integer from 1 to '+str(contract['max_repeat'])+'.')
    if not isinstance(value['summary'], str) or len(value['summary']) > 2000:
        raise ValueError('summary must be a short string.')
    return value


def tcp(meta):
    proprio = meta['proprio']
    layout, values = proprio['layout'], proprio['now']
    xyz = [float(values[layout.index('tcp_' + axis)]) for axis in 'xyz']
    if not all(math.isfinite(x) for x in xyz):
        raise ValueError('Nonfinite public end-effector position.')
    return xyz


def optional_tcp(meta):
    try:
        return tcp(meta)
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def movement_verdict(start, end, axis):
    delta = [b-a for a,b in zip(start,end)]
    along = delta[axis]
    orthogonal = math.sqrt(sum(x*x for i,x in enumerate(delta) if i != axis))
    return {'delta_m':delta, 'along_axis_m':along, 'orthogonal_m':orthogonal,
            'passed':0.015 <= along <= 0.035 and orthogonal <= 0.015}


def response(body):
    request = urllib.request.Request(os.environ['OPENAI_BASE_URL'].rstrip('/') + '/responses',
        data=json.dumps(body, allow_nan=False).encode(),
        headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'], 'Content-Type':'application/json'})
    final = None
    with urllib.request.urlopen(request, timeout=900) as stream:
        for raw in stream:
            if not raw.startswith(b'data: '):
                continue
            data = raw[6:].strip()
            if data == b'[DONE]':
                continue
            event = json.loads(data)
            if event.get('type') == 'response.completed':
                final = event['response']
            if event.get('type') in {'error','response.failed','response.incomplete'}:
                raise RuntimeError('Model transport did not complete: ' + event['type'])
    if final is None:
        raise RuntimeError('Missing response.completed event.')
    # Only public final output is consumed or saved; provider reasoning is not.
    text = '\n'.join(c['text'] for item in final.get('output',[]) if item.get('type') == 'message'
        and item.get('channel') not in {'analysis','reasoning'} for c in item.get('content',[])
        if c.get('type') == 'output_text')
    return text, final.get('usage') or {}, final.get('model')


SYSTEM = '''Control a simulated robot through a constrained observation/action interface.
Every turn includes the current public robot state and named camera images. Choose
one short native action chunk from action_contract.action_types. You cannot execute
code, inspect files or invoke other tools. The bridge repeats your exact action at
action_contract.control_dt and returns a fresh observation. Read each robot's
action specification for its component order, dimensions, bounds, frames, units
and gripper semantics. Robots may have different native action spaces.
Return only one JSON object with action_type (a supported name), action (a numeric
array matching that type's layout and bounds), repeat (an integer from 1 to
action_contract.max_repeat), and summary (one short sentence explaining the action).
The bridge validates and executes your command without planning or repairing it.
Do not announce future work without providing the action JSON. Correct your
action using the next observed state. A movement diagnostic cannot be stopped
early. In task mode only, you may finish with {"stop":true,"summary":"reason"}.
Task success is determined solely by the native simulator grader.'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--instruction', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    mode = config['mode']
    if mode not in {'movement','task'}:
        raise ValueError('Unknown fixed-interface mode.')
    args.logs.mkdir(parents=True, exist_ok=True)
    import numpy as np
    from PIL import Image
    from rcb_client import Robot, EpisodeOver
    robot = Robot()
    spec = robot.spec()
    write(args.logs/'native-spec.json',spec)
    contract = native_contract(spec, config.get('max_repeat', 25))
    cameras = contract['cameras']
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    write(args.logs/'action-contract.json', {'contract':contract, 'sha256':contract_sha,
        'native_spec':spec, 'system_prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'bridge_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    instruction = args.instruction.read_text()
    trajectory = {'schema_version':PROTOCOL,'action_contract_sha256':contract_sha,'steps':[
        {'step_id':0,'source':'user','message':instruction,'timestamp':now()}]}
    stats = {'protocol':PROTOCOL,'mode':mode,'started_utc':now(),
        'parser_version':PARSER_VERSION,'accepted_fenced_actions':0,
        'action_contract_sha256':contract_sha, 'action_dimensions':{k:v['dim'] for k,v in contract['action_types'].items()},
        'control_dt':contract['control_dt'], 'model_seconds':0, 'submit_seconds':0,
        'model':config['model'],'model_requests':0,'valid_actions':0,'invalid_actions':0,
        'control_rows_requested':0,'input_tokens':0,'output_tokens':0,'goals':[],
        'original_task_success_is_not_movement_score':True}
    history = []
    started = time.monotonic()
    obs = robot.get_obs(cameras=cameras, include_depth=False)
    goals = list(range(3)) if mode == 'movement' else [None]
    try:
        for axis in goals:
            start = tcp(obs.meta) if axis is not None else None
            goal = {'axis':axis,'start_tcp_m':start,'attempts':0,'passed':False} if axis is not None else None
            if goal is not None:
                stats['goals'].append(goal)
            limit = config['max_calls_per_goal'] if goal is not None else config['max_task_calls']
            for _ in range(limit):
                if time.monotonic()-started >= config['wall_seconds']:
                    stats['end_reason'] = 'fixed_wall_budget'
                    return
                target = ('MOVEMENT DIAGNOSTIC, not the original task. Move the end effector +2 cm '
                    f'along world {"xyz"[axis]} from starting xyz={start}; keep other coordinates within '
                    '1.5 cm of their start. You may use up to four action chunks. '
                    'Do not solve the original manipulation task.') if goal is not None else instruction
                visible = {'mode':mode,'objective':target,'spec':spec,'action_contract':contract,'observation':obs.meta,
                           'recent_actions_and_feedback':history[-4:]}
                content = [{'type':'input_text','text':json.dumps(visible,allow_nan=False)}]
                for camera in cameras:
                    buf = io.BytesIO()
                    Image.fromarray(np.ascontiguousarray(obs.last[camera])).save(buf,format='PNG')
                    content += [{'type':'input_text','text':'Current camera: '+camera},
                                {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),'detail':'auto'}]
                body = {'model':config['model'],'instructions':SYSTEM,
                    'input':[{'role':'user','content':content}], 'stream':True,'store':False,
                    'max_output_tokens':config['max_output_tokens'], 'reasoning':{'effort':'high'}}
                before = optional_tcp(obs.meta)
                request_sha = hashlib.sha256(json.dumps(body, allow_nan=False).encode()).hexdigest()
                model_start = time.monotonic()
                text, usage, returned_model = response(body)
                model_seconds = time.monotonic()-model_start
                stats['model_seconds'] += model_seconds
                stats['model_requests'] += 1
                stats['input_tokens'] += usage.get('input_tokens',0)
                stats['output_tokens'] += usage.get('output_tokens',0)
                stats['returned_model'] = returned_model
                step = {'step_id':len(trajectory['steps']),'source':'agent','timestamp':now(),
                        'request_sha256':request_sha,'public_output':text,'usage':usage,
                        'model_seconds':model_seconds,'action_contract_sha256':contract_sha,
                        'tool_calls':[],'observation':{'results':[]}}
                call_id = 'action-'+str(stats['model_requests'])
                try:
                    action = parse_action(text,contract=contract,allow_stop=mode=='task')
                except (ValueError,TypeError) as error:
                    stats['invalid_actions'] += 1
                    feedback = {'error':str(error),'model_output':text}
                    history.append(feedback)
                    step['tool_calls'] = [{'tool_call_id':call_id,'function_name':'invalid_action','arguments':{'public_output':text}}]
                    step['observation']['results'] = [{'source_call_id':call_id,'content':json.dumps(feedback)}]
                    trajectory['steps'].append(step)
                    write(args.logs/'trajectory.json',trajectory)
                    write(args.logs/'action-metrics.json',stats)
                    continue
                if action.get('stop'):
                    step['tool_calls'] = [{'tool_call_id':call_id,'function_name':'stop','arguments':action}]
                    trajectory['steps'].append(step)
                    stats['end_reason'] = 'model_stop'
                    stats['stop_summary'] = action['summary']
                    return
                if text.strip().startswith('```'):
                    stats['accepted_fenced_actions'] += 1
                step['tool_calls'] = [{'tool_call_id':call_id,'function_name':'robot.'+action['action_type'],'arguments':action}]
                trajectory['steps'].append(step)
                write(args.logs/'trajectory.json',trajectory)
                controls_before = obs.meta.get('control_steps')
                submit_start = time.monotonic()
                obs = robot.submit(action['action_type'], [action['action']]*action['repeat'], dt=contract['control_dt'],
                    cameras=cameras,video='frames',include_depth=False,note=PROTOCOL)
                step['submit_seconds'] = time.monotonic()-submit_start
                stats['submit_seconds'] += step['submit_seconds']
                stats['valid_actions'] += 1
                stats['control_rows_requested'] += action['repeat']
                after = optional_tcp(obs.meta)
                feedback = {'action':action,'tcp_before_m':before,'tcp_after_m':after,
                            'observed_delta_m':[b-a for a,b in zip(before,after)] if before is not None and after is not None else None,
                            'control_steps_before':controls_before,'control_steps_after':obs.meta.get('control_steps'),
                            'execution_completed':True,'done':bool(obs.done)}
                history.append(feedback)
                step['observation']['results'] = [{'source_call_id':call_id,'content':json.dumps(feedback)}]
                if goal is not None:
                    goal.update(movement_verdict(start,after,axis),end_tcp_m=after)
                    goal['attempts'] += 1
                write(args.logs/'trajectory.json',trajectory)
                write(args.logs/'action-metrics.json',stats)
                if obs.done:
                    stats['end_reason'] = 'simulator_done'
                    return
                if goal is not None and goal['passed']:
                    break
        stats['end_reason'] = 'diagnostic_complete' if mode=='movement' else 'fixed_model_call_budget'
    except EpisodeOver as error:
        stats['end_reason'] = 'simulator_episode_over'
        stats['episode_over'] = str(error)
    except Exception as error:
        stats['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        stats['completed_utc'] = now()
        stats['elapsed_seconds'] = time.monotonic()-started
        stats['movement_passes'] = sum(g['passed'] for g in stats['goals'])
        stats['planned_movement_goals'] = 3 if mode=='movement' else 0
        write(args.logs/'trajectory.json',trajectory)
        write(args.logs/'action-metrics.json',stats)


if __name__ == '__main__':
    main()
