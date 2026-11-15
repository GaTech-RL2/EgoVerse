"""Export immutable expert states and observed generations with rescored decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import forward_delta, kl, mean_policy, parse_policy
from ..data import load_oracle
from ..pipeline import score_record
from ..prompts import backward_prompt, forward_prompt
from .export import BASE, read


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def records(base: Path = BASE, *, observed_chunks: list | None = None):
    states, metadata = load_oracle(base/'artifacts/leduc6r2s/oracle.jsonl')
    by_id = {s['state_id']: s for s in states}
    for state in states:
        yield {'kind': 'expert', 'oracle_sha256': metadata['sha256'], 'state_id': state['state_id'],
               'state': state, 'solver': {k: metadata.get(k) for k in
                                         ['exploitability', 'iterations', 'converged', 'rules', 'game']}}
    build = base/'cluster/.build'
    overnight = build/'overnight-20261022'
    frontier_admission = read(build/'overnight-20261022-frontier/collected/latest/frontier_4032_n1/sft/admission.json', {})
    if (overnight/'astra/actions').exists():
        from ..astra_expert.data import tracking_records
        for record in tracking_records(states, metadata, overnight/'astra', label='Astra full-tree solver test'):
            if record['kind'] == 'trace':
                record['scope'] = 'Full-tree teacher collection; expert experiments require exploitability <=0.10'
                record['candidates'][0]['sft_admission'] = {
                    'admitted_to_sft': False, 'reason': 'Teacher policy, not a selected student trace'}
            yield record
    for path in sorted((overnight/'frontier/frontier').glob('*.json')):
        row = json.loads(path.read_text())
        state = by_id[row['state_id']]
        from ..coldstart.data import request_for, validate_candidate
        value = validate_candidate(row['response'], state)
        if (not row.get('accepted') or row.get('oracle_sha256') != metadata['sha256']
                or row['request'] != request_for(state, metadata['rules']) or row['result'] != value):
            raise ValueError('Frontier coldstart provenance changed')
        yield {'kind': 'trace', 'run': 'frontier-coldstart/overnight-20261022', 'seed': 0,
               'label': 'Frontier coldstart · Astra rationale, CFR action target',
               'scope': 'Full-coverage warmstart collection; GPU context validation and SFT tracked separately',
               'method': 'frontier_coldstart', 'state_id': state['state_id'], 'info_state': state['info_state'],
               'oracle_sha256': metadata['sha256'], 'baselines': [], 'mean_baseline_policy': None, 'baseline_kl': None,
               'prompts': {'forward': row['request']['messages'][-1]['content'], 'backward': None,
                           'note': 'Frontier generation sees no solver policy; SFT replaces the proposed action target with CFR.'},
               'candidates': [{'candidate_id': 0, 'selected': True, 'complete': True,
                               'reasoning': value['reasoning'], 'predicted_policy': value['policy'],
                               'sft_target_policy': state['expert_policy'], 'fallback': None, 'score': None,
                               'kl': kl(state['expert_policy'], value['policy']),
                               'forward': {'text': row['response']['choices'][0]['message']['content'], 'finish_reason': 'stop'},
                               'sft_admission': {'admitted_to_sft': (0 in frontier_admission[state['state_id']]['selected']
                                                     if state['state_id'] in frontier_admission else None),
                                                 'reason': 'GPU context audit' if frontier_admission else 'Awaiting GPU context audit'}}],
               'provenance': {'api_record': row, 'source': str(path.relative_to(base))}}
    teachers = {}
    for teacher_path in build.glob('astra-*/teacher.jsonl'):
        if teacher_path.with_suffix('.meta.json').exists():
            teacher_rows, teacher_meta = load_oracle(teacher_path)
            if teacher_meta.get('cfr_evaluation_oracle_sha256') != metadata['sha256']:
                raise ValueError('Queued Astra teacher uses a different CFR evaluator')
            teachers[teacher_meta['sha256']] = {s['state_id']: s for s in teacher_rows}
            collection = teacher_path.parent / 'teacher-collection'
            if (collection / 'summaries-complete.json').exists():
                from ..astra_expert.data import tracking_records
                for record in tracking_records(states, metadata, collection, label='Astra teacher · frozen actions and explanations'):
                    if record['kind'] == 'trace':
                        record['scope'] = 'Full teacher collection; independent CFR reference; teacher labels are not student SFT traces'
                        record['candidates'][0]['sft_admission'] = {
                            'admitted_to_sft': False, 'reason': 'Teacher label; SFT admission applies to student-generated traces'}
                    yield record
    groups = [
        ('base-diagnostic', 'Untouched Qwen3-8B · diagnostic',
         build/'diagnostic-1/readback/base_check/generation', '16-state reasoning diagnostic'),
        ('masked-bc-diagnostic', 'Corrected BC · diagnostic',
         build/'full-5/gate-collected/latest/coldstart_check/generation', '16-state reasoning diagnostic'),
    ]
    for root in build.glob('*/collected/latest'):
        collection = read(root.parent/'collection-policy.json', {})
        coverage = ('Bounded live preview; complete raw records persist at '+collection['complete_raw_artifacts']
                    if collection.get('max_preview_bytes_per_shard')
                    else 'Collected generation records; coverage shown in trace browser')
        parents = {p.parent for p in root.glob('**/shard-*.jsonl')}
        parents |= {p.parent.parent for p in root.glob('**/shard-*.jsonl.parts/*.jsonl')}
        for directory in sorted(parents):
            relative = str(directory.relative_to(root))
            groups.append((root.parents[1].name+'/'+relative, root.parents[1].name+' · '+relative,
                           directory, coverage))
    admission = {(r['state_id'], r['candidate_id']): r for r in
                 read(base/'artifacts/base-gate-sft-admission.json', {}).get('audit', [])}
    for row in read(base/'artifacts/base-baseline-validity-audit.json', {}).get('audit', []):
        key = (row['state_id'], row['candidate_id'])
        admission[key] = {**admission.get(key, {}), **row}
    scoring_hash = hashlib.sha256((base/'core.py').read_bytes()+(base/'pipeline.py').read_bytes()).hexdigest()
    for group, label, directory, scope in groups:
        coldstart_admission = {p.parent.parent.name: read(p, {})
                               for p in directory.parent.glob('self_*/sft/admission.json')} if directory.name == 'self_generation' else {}
        training_admission = read(directory.parent/'sft/admission.json', {})
        projection_admission = directory.parent/'projection_4032_n1/sft/manifest.json'
        local_admission = {(r['state_id'], r['candidate_id']): r for r in
                           read(directory.parent/'result.json', {}).get('audit', [])}
        sources = [(p, p) for p in directory.glob('shard-*.jsonl')]
        sources += [(p, p.parent.with_suffix('')) for p in directory.glob('shard-*.jsonl.parts/*.jsonl')]
        for path, logical in sorted(sources):
            manifest = read(logical.with_suffix('.meta.json'))
            if not manifest:
                continue
            teacher_table = teachers.get(manifest.get('oracle_sha256'))
            if manifest.get('oracle_sha256') != metadata['sha256'] and teacher_table is None:
                raise ValueError(f'Trace oracle mismatch: {path}')
            method = manifest['method']
            source_bytes = path.read_bytes()
            for line in source_bytes.decode().splitlines():
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    if path.parent.name.endswith('.jsonl.parts'):
                        raise ValueError(f'An immutable transfer chunk contains invalid JSON: {path}')
                    continue  # A live append may still be incomplete.
                state = by_id[raw['state_id']]
                scoring_state = teacher_table[raw['state_id']] if teacher_table is not None else state
                if raw['state_id'] not in manifest['state_ids'] or raw['seed'] != manifest['seed']:
                    raise ValueError(f'Trace state/seed differs from manifest: {path}')
                if len(raw['candidates']) != manifest['n'] or (method == 'leact' and len(raw['baselines']) != manifest['m']):
                    raise ValueError(f'Trace sample count differs from manifest: {path}')
                baselines = []
                for sample in raw['baselines']:
                    parsed = parse_policy(sample['text'], state['legal_actions'])
                    baselines.append({**sample, 'policy': parsed.policy, 'fallback': parsed.fallback,
                                      'kl': kl(state['expert_policy'], parsed.policy)})
                warmstart_pool = directory.name == 'self_generation'
                scored = score_record(scoring_state, raw, method=method) if method in {'leact', 'exit'} and not warmstart_pool else None
                selected = {c['candidate_id'] for c in scored['selected']} if scored else set()
                actual_admission = training_admission.get(raw['state_id'])
                if actual_admission is not None:
                    selected = {c['candidate_id'] for c in actual_admission['selected']}
                candidates = []
                for index, candidate in enumerate(raw['candidates']):
                    parsed = parse_policy(candidate['forward']['text'], state['legal_actions'])
                    detail = scored['candidates'][index] if scored else {
                        'candidate_id': index, 'predicted_policy': parsed.policy, 'fallback': parsed.fallback,
                        'complete': candidate['forward']['finish_reason'] == 'stop',
                        'score': None,
                    }
                    candidates.append({**candidate, **detail, 'selected': index in selected,
                                       'kl': kl(state['expert_policy'], parsed.policy),
                                       'sft_admission': (admission if group == 'base-diagnostic' else local_admission).get((state['state_id'], index))})
                    if warmstart_pool and coldstart_admission:
                        membership = {arm: index in table.get(state['state_id'], {}).get('selected', [])
                                      for arm, table in coldstart_admission.items()}
                        candidates[-1]['selected'] = any(membership.values())
                        candidates[-1]['sft_admission'] = {'admitted_to_sft': any(membership.values()),
                            'actual_training_selection': True, 'by_coldstart': membership,
                            'rule': 'Fixed first eligible traces; no CFR-based ranking'}
                    if method == 'projection':
                        projection = raw['projection']
                        chosen = index == projection['selected_candidate']
                        candidates[-1]['selected'] = chosen
                        candidates[-1]['projection_mean_logprob'] = (projection['mean_logprob'] if index
                            else projection['original_mean_logprob'])
                        candidates[-1]['sft_target_policy'] = state['expert_policy']
                        candidates[-1]['sft_admission'] = {
                            'admitted_to_sft': chosen if projection_admission.exists() else None,
                            'actual_training_selection': projection_admission.exists(),
                            'reason': ('Projected rationale passed policy and context checks' if index
                                else 'Original rationale retained as fallback') if chosen
                                else 'Not selected for the matched warmstart corpus',
                            'rewrite_rejections': projection['rejection_reasons'],
                        }
                    if teacher_table is not None:
                        candidates[-1]['delta_0'] = detail['score']
                        candidates[-1]['scoring_reference'] = 'Frozen Astra action policy'
                        assisted = raw.get('assisted_baselines', [])
                        if assisted and candidate.get('assisted_forward'):
                            parsed_assisted = [parse_policy(b['text'], state['legal_actions']) for b in assisted]
                            pred_r = parse_policy(candidate['assisted_forward']['text'], state['legal_actions'])
                            valid_r = (pred_r.fallback is None and candidate['assisted_forward']['finish_reason'] == 'stop'
                                       and all(b.fallback is None for b in parsed_assisted)
                                       and all(b['finish_reason'] == 'stop' for b in assisted))
                            candidates[-1]['delta_r'] = forward_delta(scoring_state['expert_policy'], pred_r.policy,
                                [b.policy for b in parsed_assisted]) if valid_r else None
                    if actual_admission is not None:
                        actual = next((c for c in actual_admission['selected'] if c['candidate_id'] == index), None)
                        candidates[-1]['sft_admission'] = {
                            'actual_training_selection': True, 'selected_before_length_filter': actual is not None,
                            'admitted_to_sft': bool(actual and actual['admitted_to_sft']),
                            'clean_baseline': actual_admission['baseline_valid'], 'cutoff_len': 2048,
                        }
                average = mean_policy([b['policy'] for b in baselines]) if baselines else None
                run = group+'/seed-'+str(raw['seed'])
                yield {'kind': 'trace', 'run': run, 'label': label+' · seed '+str(raw['seed']),
                       'scope': scope, 'method': method, 'state_id': raw['state_id'], 'seed': raw['seed'],
                       'oracle_sha256': metadata['sha256'], 'info_state': state['info_state'],
                       'baselines': baselines, 'candidates': candidates,
                       'astra_policy': scoring_state['expert_policy'] if teacher_table is not None else None,
                       'assisted_baselines': raw.get('assisted_baselines'),
                       'mean_baseline_policy': average,
                       'baseline_kl': kl(state['expert_policy'], average) if average else None,
                       'prompts': ({**raw['prompts'], 'note': 'Exact prompt text recorded during this generation.'}
                                   if 'prompts' in raw else
                                   {'forward': forward_prompt(state, metadata['rules'], thinking=manifest.get('thinking', True), variant=manifest.get('prompt_variant', 'original')),
                                    'backward': backward_prompt(state, metadata['rules'], variant=manifest.get('prompt_variant', 'original')) if method == 'leact' else None,
                                    'note': 'Reconstructed from stored state/rules and the local template. Raw generation text is preserved verbatim.'}),
                       'provenance': {'source': str(logical.relative_to(base)),
                                      'projection': raw.get('projection'),
                                      'manifest': manifest, 'manifest_sha256': sha(manifest),
                                      'raw_row_sha256': sha(raw), 'scoring_source_sha256': scoring_hash,
                                      'scoring_note': ('Admission deltas use frozen Astra targets; displayed KL and expert states use independent CFR.'
                                                       if teacher_table is not None else
                                                       'Scores recomputed from the recorded expert policy and raw generations.')}}
            if observed_chunks is not None and path.parent.name.endswith('.jsonl.parts'):
                observed_chunks.append((path, hashlib.sha256(source_bytes).hexdigest()))
