"""Build the adaptive-selector report from a complete, verified result archive."""

import argparse
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from astra_reversal.codex_accounting import summarize_codex_calls
from .worker import safe_relative, sha256
from .xiaomi_teacher import METHODS, SYSTEM_PROMPT


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def reconcile_interrupted_jobs(records, jobs):
    """Charge completed local jobs even if their response did not reach the robot."""
    indexed = {job['request_fingerprint']: job for job in jobs}
    if len(indexed) != len(jobs) or len(records) != len(jobs):
        raise ValueError('Interrupted job receipts are incomplete or duplicated')
    seen, tokens, rows = set(), dict.fromkeys(('input_tokens','output_tokens','reasoning_tokens','total_tokens'),0), []
    for record in records:
        key = record['request_fingerprint']
        if key in seen or key not in indexed:
            raise ValueError('Interrupted request does not match a unique local completion')
        seen.add(key)
        job = indexed[key]
        if job['delivered_and_accepted_by_worker'] is not record['accepted']:
            raise ValueError('Local receipt cannot change worker acceptance')
        usage = job['token_usage']
        if record['accepted'] and usage != record['token_usage']:
            raise ValueError('Local and remote token receipts disagree')
        if any(type(usage[k]) is not int or usage[k] < 0 for k in tokens):
            raise ValueError('Cannot silently fill unknown interrupted token usage')
        if usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens']:
            raise ValueError('Reasoning must not be added twice to total tokens')
        for key in tokens:
            tokens[key] += usage[key]
        rows.append(dict(request_fingerprint=record['request_fingerprint'],
                         delivered=record['accepted'], token_usage=usage))
    return dict(completed_local_jobs=len(jobs), delivered_proposals=sum(r['accepted'] for r in records),
                undelivered_completed_jobs=sum(not r['accepted'] for r in records),token_usage=tokens,
                worker_wait_seconds=sum(r['latency_seconds'] for r in records),receipts=rows)


def interruption_lineage(results, protocol):
    """Bind every incomplete attempt to its registered, checksum-pinned parent."""
    current, evidence, seen, rows = protocol, results/'continuation', set(), []
    while 'continuation' in current:
        amendment = current['continuation']
        workflow = amendment['parent_workflow']
        if workflow in seen:
            raise ValueError('Cyclic continuation lineage')
        seen.add(workflow)
        if sha256(evidence/'parent_archive_receipt.json') != amendment['parent_receipt_sha256']:
            raise ValueError('Continuation receipt does not match its registered parent')
        parent_protocol = evidence/'parent/protocol.json'
        if amendment.get('parent_protocol_sha256') and sha256(parent_protocol) != amendment['parent_protocol_sha256']:
            raise ValueError('Continuation parent protocol changed')
        rows.append(amendment)
        current = read(parent_protocol)
        evidence = results/'lineage'/workflow/'continuation'
    return rows


def interruption_totals(rows):
    token_keys = ('input_tokens','output_tokens','reasoning_tokens','total_tokens')
    fields = ('completed_local_jobs','delivered_proposals','undelivered_completed_jobs',
              'worker_wait_seconds','physical_episodes','zero_action_incomplete_starts',
              'rejected_reset_starts','controls','chunks')
    return dict({k:sum(r[k] for r in rows) for k in fields},
                token_usage={k:sum(r['token_usage'][k] for r in rows) for k in token_keys})


def aggregate(rows, cases):
    """Distinguish per-episode SR from paired first-attempt and within-two SR."""
    groups = defaultdict(list)
    case_ids = {c['id'] for c in cases}
    for row in rows:
        if row['case'] not in case_ids or row['arm'] not in ('native', 'astra') or type(row['success']) is not bool:
            raise ValueError('Unexpected episode identity or success field')
        groups[row['case'], row['arm']].append(row)
    result = []
    for arm in ('native', 'astra'):
        arm_rows = [r for r in rows if r['arm'] == arm]
        for case in cases:
            attempts = sorted(groups[case['id'], arm], key=lambda r: r['attempt'])
            if not attempts or attempts[0]['attempt'] != 1:
                raise ValueError('Missing first attempt for ' + case['id'] + '/' + arm)
            expected = [1] if attempts[0]['success'] else [1, 2]
            if [r['attempt'] for r in attempts] != expected:
                raise ValueError('Unfinished or extra attempts for ' + case['id'] + '/' + arm)
        first = [r for r in arm_rows if r['attempt'] == 1]
        rescued = sum(any(r['success'] for r in groups[c['id'], arm]) for c in cases)
        result.append(dict(arm=arm, cases=len(cases), first_successes=sum(r['success'] for r in first),
                           within_two_successes=rescued, successes=sum(r['success'] for r in arm_rows),
                           episodes=len(arm_rows), controls=sum(r['steps'] for r in arm_rows),
                           chunks=sum(r['policy_queries'] for r in arm_rows),
                           wall_seconds=sum(r['wall_seconds'] for r in arm_rows),
                           policy_seconds=sum(r['policy_seconds'] for r in arm_rows),
                           teacher_seconds=sum(r['teacher_seconds'] for r in arm_rows),
                           calls=sum(r['teacher_calls'] for r in arm_rows),
                           tokens=sum(r['tokens'] for r in arm_rows)))
    return result


def build(results, output):
    receipt = read(results / 'archive_receipt.json')
    for relative, item in receipt['files'].items():
        path = results / safe_relative(relative)
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Result archive differs: ' + relative)
    if not read(results / 'evaluation/completed.json')['complete']:
        raise ValueError('Cannot present an unfinished experiment as complete')
    protocol, summary = read(results / 'protocol.json'), read(results / 'evaluation/summary.json')
    preflight = read(results / 'evaluation/intervention_preflight.json')
    if preflight['passed'] is not True:
        raise ValueError('Actual-checkpoint preflight did not pass')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('media', 'evidence', 'provenance'):
        (output / name).mkdir()

    def copy(path, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return relative

    rows, provider, all_decisions, queries = [], [], [], []
    for row in summary['episodes']:
        if row['completed'] is not True:
            raise ValueError('An incomplete attempt cannot be scored as a completed episode')
        identity = f"{row['case']}-{row['arm']}{row['attempt']}"
        folder = results / 'evaluation' / row['case'] / f"{row['arm']}{row['attempt']}"
        seed_folder = folder / f"seed{row['seed']}"
        pairing = read(seed_folder / 'paired_reset.json')
        anchor = read(folder.parent / 'anchor.json')
        if pairing.pop('verified') is not True or pairing != anchor:
            raise ValueError('Episode does not match its saved anchor: ' + identity)
        reset = read(seed_folder / 'reset.json')
        records = jsonl(folder / 'provider.jsonl')
        accounting = summarize_codex_calls(records)
        if len(records) != row['teacher_calls'] or accounting['accepted_proposals'] != len(records):
            raise ValueError('Completed episode contains unmatched/rejected teacher calls')
        token_count = accounting['tokens']['total_tokens']
        if not token_count['complete']:
            raise ValueError('Missing token usage must be reported explicitly')
        provider += records
        decision_rows = []
        for path in sorted((folder / 'guidance').glob('proposal_*.json')):
            proposal = read(path)
            decision = dict(proposal, case=row['case'], arm=row['arm'], attempt=row['attempt'], episode_id=identity)
            index = int(path.stem.split('_')[-1])
            usage_record = records[index]
            if usage_record['request_fingerprint'] != proposal['request_fingerprint']:
                raise ValueError('Per-review proposal and usage are not bound to the same observation')
            decision.update(tokens=usage_record['token_usage']['total_tokens'],
                            teacher_seconds=usage_record['latency_seconds'])
            for name in ('observed', 'source', 'applied'):
                p = path.parent / f'vision_{index:02d}_{name}.png'
                if p.exists():
                    decision[name + '_image'] = copy(p, f'media/{identity}-review{index:02d}-{name}.png')
            decision_rows.append(decision)
            copy(path, f'evidence/{identity}/{path.name}')
        if len(decision_rows) != len(records):
            raise ValueError('Proposal and provider receipts disagree')
        all_decisions += decision_rows
        episode_queries = jsonl(seed_folder / 'policy_queries.jsonl')
        if len(episode_queries) != row['policy_queries']:
            raise ValueError('Action-chunk audit is incomplete')
        for q in episode_queries:
            q.update(episode_id=identity, arm=row['arm'], case=row['case'], attempt=row['attempt'],
                     controls_executed=min(16, row['steps'] - q['control_step']))
        queries += episode_queries
        native_video = folder / row['task'] / f"episode_{row['episode']:03d}_seed_{row['seed']}_{'success' if row['success'] else 'failure'}.mp4"
        value = dict(row, id=identity, instruction=reset['instruction'], pairing_verified=True,
                     video=copy(native_video, f'media/{identity}.mp4'),
                     image=copy(seed_folder / 'starting_image.png', f'media/{identity}.png'),
                     tokens=token_count['sum'], teacher_seconds=accounting['latency_seconds'],
                     input_tokens=accounting['tokens']['input_tokens']['sum'],
                     output_tokens=accounting['tokens']['output_tokens']['sum'],
                     decisions=decision_rows, queries=episode_queries)
        rows.append(value)
        for name in ('reset.json', 'paired_reset.json', 'result.json', 'policy_queries.jsonl', 'initial_model.xml.gz'):
            copy(seed_folder / name, f'evidence/{identity}/{name}')
        copy(folder / 'episode.json', f'evidence/{identity}/episode.json')
        for path in folder.glob('reset_verification*'):
            copy(path,f'evidence/{identity}/{path.name}')
        # Provider receipt data is public evidence; local CLI events/reasoning are not copied.
        if (folder / 'provider.jsonl').exists():
            copy(folder / 'provider.jsonl', f'evidence/{identity}/provider.jsonl')
        (output / 'evidence' / identity / 'usage.json').write_text(json.dumps(accounting, indent=2) + '\n')
    arms = aggregate(rows, protocol['cases'])
    prefix_checks = []
    for guided in (r for r in rows if r['arm'] == 'astra'):
        native = next((r for r in rows if r['arm'] == 'native' and r['case'] == guided['case']
                       and r['attempt'] == guided['attempt']), None)
        if native is None:
            continue
        matches = []
        for first, second in zip(native['queries'], guided['queries']):
            if second['method'] != 'native':
                break
            if (first['noise_seed'],first['control_step']) != (second['noise_seed'],second['control_step']):
                raise ValueError('Claimed common noise schedule differs')
            matches.append(first['actions_sha256'] == second['actions_sha256'])
        if matches:
            prefix_checks.append(dict(case=guided['case'],attempt=guided['attempt'],chunks_compared=len(matches),
                identical_chunks=sum(matches),first_difference_query=next((i+1 for i,v in enumerate(matches) if not v),None)))
    method_stats = []
    for method in METHODS:
        samples = [q for q in queries if q['arm'] == 'astra' and q['method'] == method]
        times = [q['seconds'] for q in samples]
        method_stats.append(dict(method=method, reviews=sum(d['method'] == method for d in all_decisions),
            chunks=len(samples), controls=sum(q['controls_executed'] for q in samples),
            query_mean_seconds=float(np.mean(times)) if times else None,
            query_p50_seconds=float(np.median(times)) if times else None,
            query_p95_seconds=float(np.percentile(times, 95)) if times else None))
    pairings = []
    for case in protocol['cases']:
        group = [r for r in rows if r['case'] == case['id']]
        pairings.append(dict(case=case['id'], task=case['task'], seed=case['seed'], instruction=group[0]['instruction'],
            native_first=next(r['success'] for r in group if r['arm'] == 'native' and r['attempt'] == 1),
            astra_first=next(r['success'] for r in group if r['arm'] == 'astra' and r['attempt'] == 1),
            native_within_two=any(r['success'] for r in group if r['arm'] == 'native'),
            astra_within_two=any(r['success'] for r in group if r['arm'] == 'astra'),
            astra_tokens=sum(r['tokens'] for r in group if r['arm'] == 'astra'),
            native_attempts=sum(r['arm'] == 'native' for r in group), astra_attempts=sum(r['arm'] == 'astra' for r in group)))
    report = dict(schema='astra-xiaomi-selector-report-1', complete=True, protocol=protocol, arms=arms,
        cases=pairings, episodes=rows, method_stats=method_stats, preflight=preflight, resets=summary['resets'],
        accounting=summarize_codex_calls(provider), native_prefix_checks=prefix_checks,
        source=read(results / 'worker_started.json'),
        worker=read(results / 'worker_finished.json'), archive_files_verified=len(receipt['files']),
        archive_receipt_sha256=sha256(results / 'archive_receipt.json'), prompt=SYSTEM_PROMPT)
    generator_files = ('xiaomi_selector_report.py','xiaomi_selector_dashboard.html')
    repo = Path(__file__).resolve().parents[2]
    generator_status = subprocess.check_output(['git','status','--porcelain','--',
        *('astra_reversal/complex_manipulation/'+name for name in generator_files)],cwd=repo,text=True).strip()
    if generator_status:
        raise ValueError('Commit the report generator before publishing reproducible results')
    report['report_generation'] = dict(source_revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),source_files={})
    for name in generator_files:
        path = Path(__file__).parent / name
        report['report_generation']['source_files'][name] = sha256(path)
        copy(path,'provenance/report_generator/'+name)
    for path in (results / 'archive_receipt.json', results / 'protocol.json', results / 'worker_started.json',
                 results / 'worker_finished.json', results / 'stage_receipt.json', results / 'evaluation/intervention_preflight.json'):
        copy(path, 'evidence/' + path.name)
    for case in protocol['cases']:
        for name in ('anchor.json', 'anchor_model.xml.gz', 'anchor_state.npz', 'anchor_rng.json'):
            copy(results / 'evaluation' / case['id'] / name, f"evidence/{case['id']}/{name}")
    for name in ('xiaomi_teacher.py', 'xiaomi_interventions.py', 'xiaomi_selector_eval.py', 'xiaomi_selector_worker.py',
                 'xiaomi_selector_protocol.json', 'xiaomi_selector_provider_preflight.json'):
        content = subprocess.check_output(['git', 'show', report['source']['source_revision'] +
            ':astra_reversal/complex_manipulation/' + name], cwd=Path(__file__).resolve().parents[2])
        (output / 'provenance' / name).write_bytes(content)
    report['provider_probe'] = read(output / 'provenance/xiaomi_selector_provider_preflight.json')
    completed_tokens = report['accounting']['tokens']['total_tokens']['sum']
    report['spending'] = dict(completed_episode_tokens=completed_tokens,
                             completed_episode_jobs=len(provider),interrupted_tokens=0,interrupted_jobs=0,
                             rollout_tokens_including_interruption=completed_tokens,
                             rollout_jobs_including_interruption=len(provider),
                             calibration_tokens=18630,calibration_jobs=1,
                             all_teacher_tokens_including_calibration=completed_tokens+18630)
    if 'continuation' in protocol:
        interruption_path = Path(__file__).parent / 'xiaomi_selector_interruption.json'
        interruption = read(interruption_path)
        interruptions = []
        for amendment in interruption_lineage(results,protocol):
            case = amendment['interrupted_trial']
            identity = f"{case['case']}-{case['arm']}{case['attempt']}"
            workflow = amendment['parent_workflow']
            folder = results / 'interrupted' / workflow / case['case'] / f"{case['arm']}{case['attempt']}"
            partial = read(folder / 'incomplete.json')
            if partial['counts_as_completed_failure'] is not False:
                raise ValueError('Interrupted trial cannot enter the success-rate denominator')
            local = []
            if partial['teacher_calls']:
                if (workflow != interruption['workflow'] or
                        interruption['archive_receipt_sha256'] != amendment['parent_receipt_sha256']):
                    raise ValueError('Missing registered local receipts for interrupted teacher calls')
                local = [r for r in interruption['job_records'] if r['episode_id'] == identity]
            item = reconcile_interrupted_jobs(jsonl(folder / 'provider.jsonl'),local)
            item.update(case=case['case'],attempt=case['attempt'],arm=case['arm'],
                        physical_episodes=int(partial['steps']>0),
                        zero_action_incomplete_starts=int(partial['steps']==0),
                        rejected_reset_starts=int((folder/'reset_mismatch.json').exists()),
                        controls=partial['steps'],chunks=partial['queries'],
                        incomplete=partial,workflow=workflow,reason=amendment['reason'])
            if item['completed_local_jobs'] != partial['teacher_calls']:
                raise ValueError('Interrupted call count differs from local receipt reconciliation')
            interruptions.append(item)
        overhead = interruption_totals(interruptions)
        report['interruptions'] = interruptions
        report['interruption'] = overhead
        report['spending'].update(interrupted_tokens=overhead['token_usage']['total_tokens'],
            interrupted_jobs=overhead['completed_local_jobs'],
            rollout_tokens_including_interruption=completed_tokens+overhead['token_usage']['total_tokens'],
            rollout_jobs_including_interruption=len(provider)+overhead['completed_local_jobs'],
            all_teacher_tokens_including_calibration=completed_tokens+overhead['token_usage']['total_tokens']+18630)
        for item in report['cases']:
            item['interrupted_tokens'] = sum(r['token_usage']['total_tokens'] for r in interruptions if r['case']==item['case'])
            item['astra_tokens_including_interruption'] = item['astra_tokens']+item['interrupted_tokens']
        for name in ('interrupted','continuation','lineage'):
            for path in (results/name).rglob('*'):
                if path.is_file():
                    copy(path,'evidence/'+str(path.relative_to(results)))
        copy(interruption_path,'evidence/'+interruption_path.name)
        for name in (protocol['continuation'].get('protocol_source_file','xiaomi_selector_continuation_protocol.json'),
                     'xiaomi_selector_resume.py'):
            (output/'provenance'/name).write_bytes(subprocess.check_output(['git','show',
                report['source']['source_revision']+':astra_reversal/complex_manipulation/'+name],
                cwd=Path(__file__).resolve().parents[2]))
    report['physical_accounting'] = dict(completed_policy_episodes=len(rows),
        incomplete_policy_episodes=report.get('interruption',{}).get('physical_episodes',0),
        zero_action_incomplete_starts=report.get('interruption',{}).get('zero_action_incomplete_starts',0),
        rejected_reset_starts=report.get('interruption',{}).get('rejected_reset_starts',0),
        total_policy_controls=sum(r['steps'] for r in rows)+report.get('interruption',{}).get('controls',0),
        total_policy_chunks=sum(r['policy_queries'] for r in rows)+report.get('interruption',{}).get('chunks',0),
        policy_restore_resets=sum(r['policy_restore_resets'] for r in summary['resets']),
        additional_verification_restore_resets=sum(r.get('additional_verification_restore_resets',0) for r in summary['resets']))
    physical = report['physical_accounting']
    physical['total_restores_for_policy_starts'] = physical['policy_restore_resets']+physical['additional_verification_restore_resets']
    if physical['policy_restore_resets'] != (physical['completed_policy_episodes']+
            physical['incomplete_policy_episodes']+physical['zero_action_incomplete_starts']):
        raise ValueError('A physical policy reset is missing from episode accounting')
    infrastructure = Path(__file__).parent / 'xiaomi_selector_infrastructure.json'
    if infrastructure.exists():
        report['infrastructure'] = read(infrastructure)
        copy(infrastructure, 'evidence/' + infrastructure.name)
    replay_audit = Path(__file__).parent / 'xiaomi_replay_audit_summary.json'
    if replay_audit.exists():
        report['replay_audit'] = read(replay_audit)
        copy(replay_audit, 'evidence/' + replay_audit.name)
    observation_audit = Path(__file__).parent / 'xiaomi_observation_audit_summary.json'
    if observation_audit.exists():
        report['observation_audit'] = read(observation_audit)
        copy(observation_audit,'evidence/'+observation_audit.name)
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    columns = ['case', 'arm', 'attempt', 'success', 'steps', 'policy_queries', 'teacher_calls', 'tokens',
               'input_tokens', 'output_tokens', 'wall_seconds', 'policy_seconds', 'teacher_seconds', 'instruction']
    with (output / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairings[0]))
        writer.writeheader()
        writer.writerows(pairings)
    template = (Path(__file__).parent / 'xiaomi_selector_dashboard.html').read_text()
    (output / 'index.html').write_text(template.replace('__DATA__', json.dumps(report).replace('<', '\\u003c')))
    manifest = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = build(args.results, args.output)
    print(json.dumps({'output': str(args.output), 'arms': report['arms']}))
