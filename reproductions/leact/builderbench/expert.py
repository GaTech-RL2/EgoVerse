"""GPT-6 BuilderBench decisions over the existing NVIDIA transport."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..astra_expert.client import ENDPOINT, MODEL, call_request, decode_json
from ..data import fingerprint

PROMPT_VERSION = 'builderbench-gpt6-public-summary-v2'
SETTINGS = dict(max_completion_tokens=8192, reasoning_effort='medium')
HISTORY_OBSERVATIONS = 16
INSTRUCTION = (
    'Choose the next action to accomplish all visible cube targets and the end-effector target, '
    'if specified, within the remaining simulator time. Use the observations to check the '
    'effects of earlier actions. Return exactly one JSON object with two fields: '
    '"decision_summary": a brief public explanation of the next action, in 1-3 sentences '
    'and at most 150 words; "action": the action object following the environment schema. '
    'Summarize the relevant visible evidence and tradeoff; do not provide private internal '
    'deliberation. The action must be inside the "action" field of this JSON wrapper. '
    'Do not output markdown, tags, extra fields, or a sequence of future actions.'
)


def validate_action(action, cubes):
    if not isinstance(action, dict):
        raise ValueError('Action must be an object')
    kind = action.get('type')
    schemas = {
        'pick_and_place': {'type', 'cube_id', 'grasp_yaw', 'pos', 'yaw'},
        'pick_and_hold': {'type', 'cube_id', 'grasp_yaw', 'pos', 'yaw'},
        'eef_target': {'type', 'pos', 'yaw', 'gripper'},
        'low_level': {'type', 'action'},
    }
    if kind not in schemas or set(action) != schemas[kind]:
        raise ValueError('Action schema differs from the requested contract')
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    def vector(value, length):
        return isinstance(value, list) and len(value) == length and all(number(v) for v in value)
    if kind == 'low_level':
        if not vector(action['action'], 5):
            raise ValueError('Invalid low-level action vector')
    else:
        if not vector(action['pos'], 3) or not number(action['yaw']):
            raise ValueError('Invalid action position or yaw')
        if kind == 'eef_target':
            if not number(action['gripper']) or not 0 <= action['gripper'] <= 1:
                raise ValueError('Invalid gripper value')
        elif (type(action['cube_id']) is not int or not 0 <= action['cube_id'] < cubes
              or type(action['grasp_yaw']) is not int or action['grasp_yaw'] not in (0, 1)):
            raise ValueError('Invalid cube or grasp identifier')
    return action


def decode(envelope, cubes):
    value = decode_json(envelope)
    if not isinstance(value, dict) or set(value) != {'decision_summary', 'action'}:
        raise ValueError('Expected summary and action fields')
    summary = value['decision_summary']
    if (not isinstance(summary, str) or not 1 <= len(summary.split()) <= 200
            or any(token in summary.lower() for token in ['<think', '</think', '<|im_'])):
        raise ValueError('Invalid public explanation')
    validate_action(value['action'], cubes)
    return value


def payload(system, observation, history):
    # Only public observations, prior executed actions and public summaries enter
    # the prompt. Simulator internals and scripted diagnostic actions are absent.
    original_format = 'Always output actions as a single, valid JSON object with the specified key/value structure.'
    if original_format not in system:
        raise ValueError('Upstream output format changed; review the wrapper adaptation')
    system = system.replace(original_format,
        'Always output a single JSON object with exactly two fields: "decision_summary" '
        '(a brief public explanation) and "action" (the action object following the schema '
        'and examples below). Do not output a bare action object. Do not provide private '
        'internal deliberation. The examples below describe the nested "action" field.')
    messages = [dict(role='system', content=system)]
    recent = history[-(HISTORY_OBSERVATIONS - 1):]
    for index, step in enumerate(recent):
        messages.append(dict(role='user', content='Observation:\n' + step['observation']))
        previous = {'action': step['action']}
        if index == len(recent) - 1:
            previous['decision_summary'] = step['decision_summary']
        messages.append(dict(role='assistant', content=json.dumps(previous, sort_keys=True)))
    messages.append(dict(role='user', content='Current Observation:\n' + observation + '\n\n' + INSTRUCTION))
    return dict(model=MODEL, messages=messages, stream=False, response_format={'type': 'json_object'},
                cache={'no-cache': True}, **SETTINGS)


def decide(system, observation, history, output: Path, *, cubes, identity, timeout=180):
    request = payload(system, observation, history)
    identity = {**identity, 'prompt_version': PROMPT_VERSION, 'endpoint': ENDPOINT,
                'request_sha256': fingerprint(request)}
    return call_request(request, output, identity=identity,
                        validate=lambda envelope: decode(envelope, cubes), timeout=timeout)
