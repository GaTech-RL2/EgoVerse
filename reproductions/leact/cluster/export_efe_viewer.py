"""Export existing EFE public action traces and recorded videos, without reruns.

Run inside an owned artifact-reader allocation. All writes go to node-local /tmp.
Provider reasoning, agent reasoning messages, raw sessions and credentials are
deliberately outside this export's allowlist.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scrub(value):
    if isinstance(value, dict):
        if value.get('type') in {'reasoning', 'input_image', 'image_url'}:
            return {'omitted': 'Private reasoning or image payload; see recorded video.'}
        return {k: scrub(v) for k, v in value.items() if k.lower() not in {
            'encrypted_content', 'reasoning_content', 'reasoning_details',
            'analysis', 'authorization', 'api_key', 'password', 'access_token'}}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'<think>.*?(?:</think>|$)', '[Reasoning omitted]', value, flags=re.S)
        value = re.sub(r'data:(?:image|video)/[^\s"\]]+', '[Embedded media omitted]', value)
        value = re.sub(r'\b(?:sk-|nvapi-)[A-Za-z0-9_-]{16,}', '[REDACTED]', value)
        value = re.sub(r'(?i)(Bearer\s+)[A-Za-z0-9._~-]{16,}', r'\1[REDACTED]', value)
    return value


def export_episode(trial, dest):
    job = Path(trial['archive'])
    paths = list(job.rglob('agent/trajectory.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one trajectory in {job}, got {len(paths)}')
    trajectory_path = paths[0]
    episode = trajectory_path.parent.parent
    trajectory = json.loads(trajectory_path.read_text())
    key = trial['task'] + '--' + trial['role']
    out = dest / key
    out.mkdir(exist_ok=True)
    steps = []
    prompts = []
    for step in trajectory['steps']:
        if step.get('source') == 'user':
            prompts.append(scrub(step.get('message')))
        if step.get('source') != 'agent' or not step.get('tool_calls'):
            continue
        # Do not copy step.message or step.extra: they can contain private reasoning.
        steps.append(scrub({k: step[k] for k in
            ['step_id', 'timestamp', 'tool_calls', 'observation'] if k in step}))
    requests = []
    for path in (job.parent.parent / 'requests' / trial['job']).glob('*.json'):
        record = json.loads(path.read_text())
        messages = []
        for item in record.get('public_output', []):
            if item.get('type') != 'message' or item.get('channel') in {'analysis', 'reasoning'}:
                continue
            for content in item.get('content', []):
                if content.get('type') in {'output_text', 'text'} and content.get('text'):
                    messages.append(content['text'])
        requests.append(scrub({k: record.get(k) for k in [
            'id', 'started_utc', 'status', 'elapsed_seconds', 'http_status',
            'usage', 'returned_model', 'error', 'incomplete_details']} |
            {'public_messages': messages, 'source_sha256': sha(path)}))
    requests.sort(key=lambda r: (r.get('started_utc') or '', r.get('id') or ''))
    metrics = trial.get('resource_metrics', {})
    counter = Counter()
    for step in steps:
        for call in step['tool_calls']:
            counter[(call.get('function_name', ''), json.dumps(call.get('arguments'), sort_keys=True))] += 1
    repeats = [{'name': name, 'arguments': json.loads(args), 'count': count}
               for (name, args), count in counter.most_common(12)]
    receipts = {}
    receipt_files = ['verifier/reward.json', 'verifier/verdict.json',
                     'verifier/record.json', 'artifacts/data/final.json',
                     'artifacts/data/video.json']
    for name in receipt_files:
        path = episode / name
        if path.exists():
            receipts[name] = {'source_sha256': sha(path), 'value': scrub(json.loads(path.read_text()))}
    episode_log = episode / 'verifier/episode.jsonl'
    sim_events = []
    if episode_log.exists():
        sim_events = [scrub(json.loads(line)) for line in episode_log.read_text().splitlines() if line.strip()]
    video = episode / 'verifier/replay.mp4'
    if not video.is_file():
        raise FileNotFoundError(video)
    shutil.copyfile(video, out / 'replay.mp4')
    duplicate = episode / 'artifacts/data/replay.mp4'
    payload = {
        'task': trial['task'], 'role': trial['role'], 'partition': trial['partition'],
        'score': trial['score'], 'resource_metrics': metrics,
        'model_requests': trial.get('model_requests'), 'model_errors': trial.get('model_errors'),
        'completed_utc': trial.get('completed_utc'), 'returned_models': trial.get('returned_models'),
        'recorded_caps': trial.get('recorded_caps'), 'prompts': prompts, 'steps': steps,
        'requests': requests, 'repeated_calls': repeats, 'simulator_events': sim_events,
        'grader_receipts': receipts,
        'provenance': {'archive': str(job), 'trajectory': str(trajectory_path),
                       'trajectory_sha256': sha(trajectory_path),
                       'trial_result_sha256': trial.get('trial_result_sha256'),
                       'episode_log_sha256': sha(episode_log) if episode_log.exists() else None},
        'video': {'source': str(video), 'sha256': sha(video), 'bytes': video.stat().st_size,
                  'duplicate_copy_matches': duplicate.exists() and sha(duplicate) == sha(video),
                  'kind': 'Original benchmark verifier replay; not newly simulated'},
        'trace_policy': 'All saved public tool calls and their recorded feedback. Agent reasoning messages and provider-private reasoning are omitted. Public assistant messages come from the gateway public_output field.'}
    (out / 'trace.json').write_text(json.dumps(payload, ensure_ascii=False) + '\n')
    return {'id': key, 'task': trial['task'], 'role': trial['role'], 'score': trial['score'],
            'tool_turns': len(steps), 'model_requests_collected': len(requests),
            'video_sha256': sha(out / 'replay.mp4'), 'trace_sha256': sha(out / 'trace.json'),
            'video_bytes': video.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/efe-viewer-export'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    rows = []
    for trial in sources['episodes']:
        row = export_episode(trial, args.output)
        rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'episodes': rows,
                'sources_sha256': sha(args.sources), 'exporter_sha256': sha(Path(__file__)),
                'student_protocol': sources['student_protocol'], 'new_rollouts': 0}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
