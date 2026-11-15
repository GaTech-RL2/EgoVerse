"""Export original teacher training episodes and their admitted decision markers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from .export_efe_viewer import export_episode, scrub, sha
except ImportError:
    from export_efe_viewer import export_episode, scrub, sha


def preview(value):
    if isinstance(value, dict):
        if value.get('type') in {'image', 'image_url', 'input_image'}:
            return {'type': 'image', 'display_note': 'Image omitted from text preview; original training retained it.'}
        return {k: preview(v) for k, v in value.items()}
    if isinstance(value, list):
        return [preview(v) for v in value]
    if isinstance(value, str):
        value = scrub(value)
        return value if len(value) <= 4000 else value[:4000] + '\n[Display preview truncated; original training retained full content.]'
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sources', 'output', 'warmstart', 'leact']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    frozen = json.loads((args.warmstart / 'dataset-manifest.json').read_text())
    state_path = args.warmstart / 'warmstart-states.jsonl'
    if sha(state_path) != frozen['states_sha256']:
        raise ValueError('Frozen warmstart states changed')
    sampled = set(json.loads((args.leact / 'membership.json').read_text())['states'])
    leact_dataset = json.loads((args.leact / 'dataset-manifest.json').read_text())
    examples = {row['id']: row for row in leact_dataset['examples']}
    schedule = json.loads((args.leact / 'train/schedule.json').read_text())
    counts = Counter(schedule['exposures'])
    by_state = defaultdict(list)
    for identity, count in counts.items():
        row = examples[identity]
        by_state[row['decision_id']].append({'example': identity, 'source': row['source'],
            'direction': row['direction'], 'draws': count})
    states = defaultdict(list)
    with state_path.open() as stream:
        for line in stream:
            state = json.loads(line)
            selected = json.loads((args.leact / 'candidates' / (state['id'] + '.json')).read_text())['selected'] if state['id'] in sampled else []
            item = {k: scrub(state.get(k)) for k in ['id', 'task', 'suite', 'episode', 'action',
                'public_commentary', 'rationale', 'scene_sha256', 'source_request_sha256', 'source_output_sha256']}
            item.update(partition='training', leact_sampled=state['id'] in sampled,
                leact_selected_candidates=selected, leact_training_exposures=by_state[state['id']],
                context_preview=[preview(m) for m in state['messages'] if m.get('role') != 'system'][-6:])
            states[state['episode']].append(item)
    if set(states) != {t['source_episode'] for t in sources['episodes']}:
        raise ValueError('Training episode membership does not match frozen states')
    rows = []
    for trial in sources['episodes']:
        row = export_episode(trial, args.output)
        path = args.output / row['id'] / 'trace.json'
        episode = json.loads(path.read_text())
        episode_states = states[trial['source_episode']]
        calls = {c['tool_call_id']: step['step_id'] for step in episode['steps'] for c in step['tool_calls']}
        requests = {r['source_sha256']: r for r in episode['requests']}
        for state in episode_states:
            target_calls = [c['id'] for c in state['action']]
            if any(identity not in calls for identity in target_calls):
                raise ValueError('Retained action cannot be located in original episode: ' + state['id'])
            request = requests.get(state['source_output_sha256'])
            if not request:
                raise ValueError('Retained decision response checksum missing: ' + state['id'])
            state.update(trace_step_ids=sorted({calls[identity] for identity in target_calls}),
                         response_id=request['id'], started_utc=request['started_utc'])
        episode_states.sort(key=lambda s: (s['started_utc'], s['id']))
        episode.update(training_decisions=episode_states, source_episode=trial['source_episode'],
            source_partition=trial['source_partition'], source_results_sha256=trial['source_results_sha256'],
            source_results=trial['source_results'])
        path.write_text(json.dumps(episode, ensure_ascii=False) + '\n')
        row.update(trace_sha256=sha(path), source_episode=trial['source_episode'],
            retained_decisions=len(episode_states), leact_sampled_states=sum(s['leact_sampled'] for s in episode_states),
            leact_updated_states=sum(any(e['source'] == 'leact' for e in s['leact_training_exposures']) for s in episode_states))
        rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'episodes': rows,
        'sources_sha256': sha(args.sources), 'exporter_sha256': sha(Path(__file__)),
        'warmstart_states_sha256': frozen['states_sha256'], 'leact_schedule_sha256': sha(args.leact / 'train/schedule.json'),
        'retained_decisions': sum(r['retained_decisions'] for r in rows),
        'leact_sampled_states': sum(r['leact_sampled_states'] for r in rows),
        'leact_updated_states': sum(r['leact_updated_states'] for r in rows),
        'new_rollouts': 0, 'scope': sources['student_protocol']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
