"""Collect frozen EFE train/development episodes with the native Codex agent."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time

import efe_native_pilot as runtime
from efe_budget_gateway import public_value


def resume_trials(spec):
    """Retain graded successes AND failures after an infrastructure interruption."""
    resume = spec.get('resume_completed_trials')
    if not resume:
        return [], None
    sources=[resume,*resume.get('fallback_sources',[])]
    resume=next((candidate for candidate in sources
                 if (Path(candidate['archive'])/'results.json').is_file()),None)
    if resume is None:
        raise ValueError('No persisted results found for infrastructure recovery')
    source = Path(resume['archive'])
    previous_spec = json.loads((source / 'spec.json').read_text())
    digest = hashlib.sha256(json.dumps(previous_spec, sort_keys=True).encode()).hexdigest()
    if digest != resume['source_spec_sha256']:
        raise ValueError('Resume source specification checksum mismatch')
    for key in ('tasks', 'student_checkpoint', 'student_manifest', 'protocol_version',
                'max_output_tokens', 'max_images', 'sampling', 'preflight_semantics_policy'):
        if previous_spec.get(key) != spec.get(key):
            raise ValueError('Resume changed evaluation condition: ' + key)
    raw = (source / 'results.json').read_bytes()
    previous = json.loads(raw)
    if previous.get('workflow') != resume['workflow']:
        raise ValueError('Resume workflow mismatch')
    planned = {row['task']: row for row in spec['tasks']}
    retained, seen = [], set()
    for trial in previous.get('trials', []):
        task = trial['task']
        if task not in planned or task in seen:
            raise ValueError('Unknown or duplicate resume task: ' + task)
        seen.add(task)
        if trial['status'] == 'infrastructure_failure':
            continue
        if trial['status'] != 'graded' or trial['score'] not in (0, 1):
            raise ValueError('Invalid resume trial status/score')
        if any(trial.get(k) != planned[task][k] for k in ('suite', 'partition')):
            raise ValueError('Resume task identity mismatch')
        retained.append({**trial, 'source_workflow': trial.get('source_workflow', previous['workflow'])})
    return retained, {'source_workflow': previous['workflow'], 'archive': str(source),
                      'results_sha256': hashlib.sha256(raw).hexdigest(),
                      'retained_graded_trials': len(retained),
                      'policy': 'Retain every graded outcome, including zero scores; retry only unfinished infrastructure work.'}


def main():
    spec = json.loads(runtime.SPEC.read_text())
    runtime.ARCHIVE = Path(spec['archive_root']) / runtime.WORKFLOW
    runtime.ROOT.mkdir(parents=True, exist_ok=True)
    runtime.ARCHIVE.mkdir(parents=True, exist_ok=True)
    results = {'workflow': runtime.WORKFLOW, 'purpose': spec['purpose'], 'trials': [],
               'protocol_label': spec['protocol_label'], 'training_started': False,
               'upstream_revision': spec['revision'], 'planned_trials': len(spec['tasks'])}
    benchmark_started = False
    label = spec.get('agent_label', 'Astra')
    provider_key = os.environ['NVIDIA_INFERENCE_API_KEY']
    token = secrets.token_urlsafe(36)
    service = service_log = None

    def progress(phase, detail, **extra):
        value = {'workflow': runtime.WORKFLOW, 'phase': phase, 'detail': detail,
                 'updated_utc': runtime.now(), 'benchmark_rollouts_started': benchmark_started,
                 'training_started': False, 'completed_trials': len(results['trials']),
                 'planned_trials': len(spec['tasks']), **extra}
        runtime.write('progress.json', public_value(value, (provider_key, token)))
        print(json.dumps(public_value(value, (provider_key, token))), flush=True)

    runtime.progress = progress
    runtime.write('spec.json', spec)
    gateway = gateway_log = None
    try:
        results['trials'], resume_receipt = resume_trials(spec)
        if resume_receipt:
            results['resume'] = resume_receipt
            runtime.write('results.json', results)
            runtime.write('resume-evaluation.json', resume_receipt)
        completed_tasks = {row['task'] for row in results['trials']}
        reference, env, service, service_log = runtime.setup(spec)
        source = Path(importlib.util.find_spec('harbor.agents.installed.codex').origin)
        raw = source.read_bytes()
        patch = spec['model_routing']
        if hashlib.sha256(raw).hexdigest() != patch['upstream_file_sha256']:
            raise ValueError('Harbor Codex source differs from the reviewed routing patch')
        replaced = raw.replace(patch['old_line'].encode(), patch['new_line'].encode())
        if hashlib.sha256(replaced).hexdigest() != patch['patched_file_sha256']:
            raise ValueError('Harbor routing patch checksum mismatch')
        source.write_bytes(replaced)
        runtime.write('model-routing.json', {**patch, 'status': 'applied', 'applied_utc': runtime.now()})
        address = socket.gethostbyname(socket.gethostname())
        port = 18861
        config_input = runtime.ROOT / 'codex-base.toml'
        config_text = 'model_context_window = 131072\nmodel_auto_compact_token_limit = '+str(spec.get('auto_compact_tokens',100000))+'\n'
        if spec.get('local_compaction'):
            config_text += ('model_provider = "efe_vllm"\n[model_providers.efe_vllm]\n'
                            'name = "EFE local vLLM"\n'
                            f'base_url = "http://{address}:{port}/v1"\n'
                            'env_key = "OPENAI_API_KEY"\nwire_api = "responses"\n'
                            '[model_providers.efe_vllm.capabilities]\nremote_compaction = "unsupported"\n')
        config_input.write_text(config_text)
        config = runtime.ROOT / 'codex.toml'
        safe_config = subprocess.check_output([sys.executable, str(reference / 'scripts/codex_tools_off.py'),
                                               str(config_input)], text=True)
        config.write_text(safe_config)
        runtime.write('native-config.json', {'codex_version': spec['codex_version'],
                                           'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                                           'config': safe_config, 'reasoning_effort': 'high',
                                           'reasoning_summary': 'none', 'web_search': 'disabled',
                                           'per_request_output_cap': spec['max_output_tokens'],
                                           'max_images':spec.get('max_images',0),
                                           'namespace_adapter':spec.get('flatten_namespaces',False),
                                           'terminal_output_events':spec.get('terminal_output_events',False),
                                           'student_instruction_condition': 'native-tool-aligned-v1' if spec.get('align_tool_instructions') else 'original',
                                           'compaction_mode':'local' if spec.get('local_compaction') else 'provider_default',
                                           'cap_enforcement': 'Recorded Responses transport adapter',
                                           'resource_enforcement_requested': 1})
        env.update(OPENAI_API_KEY=token, OPENAI_BASE_URL=f'http://{address}:{port}/v1',
                   RCB_LIMITS_ENFORCE='1')
        # The agent receives only a scoped token for this workflow's gateway.
        env.pop('NVIDIA_INFERENCE_API_KEY', None)
        for index, row in enumerate(spec['tasks']):
            if row['task'] in completed_tasks:
                continue
            job = f'{index:02d}-' + spec.get('agent_slug', 'astra')
            # Container UIDs share a Lustre quota. Keep native agent/verifier
            # writes on the node; archive them as the owning workflow afterward.
            jobs_root = (runtime.ROOT if spec.get('local_episode_outputs') else runtime.ARCHIVE) / 'jobs'
            gateway_root = runtime.ARCHIVE / 'requests' / job
            gateway_root.mkdir(parents=True, exist_ok=True)
            gateway_log = (runtime.ARCHIVE / 'logs' / (job + '-gateway.log')).open('w')
            gateway = subprocess.Popen(
                [sys.executable, '/tmp/efe-budget-gateway.py', '--root', str(gateway_root),
                 '--port', str(port), '--model', spec['request_model'],
                 '--max-output-tokens', str(spec['max_output_tokens']),
                 '--max-images',str(spec.get('max_images',0)),
                 '--upstream', spec.get('upstream', 'https://inference-api.nvidia.com/v1')]
                + (['--flatten-namespaces'] if spec.get('flatten_namespaces') else [])
                + (['--terminal-output-events'] if spec.get('terminal_output_events') else [])
                + (['--align-tool-instructions'] if spec.get('align_tool_instructions') else [])
                + (['--developer-role-adapter'] if spec.get('developer_role_adapter') else [])
                + (['--instruction-suffix-file', spec['instruction_suffix_file']] if spec.get('instruction_suffix_file') else []),
                env={**env, 'NVIDIA_INFERENCE_API_KEY': provider_key, 'EFE_GATEWAY_TOKEN': token},
                stdout=gateway_log, stderr=subprocess.STDOUT)
            for _ in range(60):
                if gateway.poll() is not None:
                    raise RuntimeError('Bounded Responses gateway failed at startup')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('Bounded Responses gateway did not become ready')
            progress('native_episode', f'{label} {index + 1}/{len(spec["tasks"])}: {row["task"]} ({row["partition"]})',
                     current_task=row['task'])
            command = ['harbor', 'run', '-d', 'embodied-first-exam@all', '-p', row['path'],
                       '-a', 'codex', '-m', spec['harbor_model'], '-e', 'podman', '-y',
                       '-k', '1', '-n', '1', '--max-retries', '0', '--job-name', job,
                       '-o', str(jobs_root), '--allow-agent-host', address,
                       '--ak', 'version=' + spec['codex_version'], '--ak', 'reasoning_effort=high',
                       '--ak', 'reasoning_summary=none', '--ak', 'web_search=disabled',
                       '--ak', 'config=' + str(config)]
            receipt = runtime.run('episode_' + job, command, cwd=reference, env=env,
                                  timeout=int(row['cutoff_seconds']) + 2400, required=False)
            if spec.get('local_episode_outputs') and (jobs_root / job).exists():
                shutil.copytree(jobs_root / job, runtime.ARCHIVE / 'jobs' / job, dirs_exist_ok=True)
            gateway.terminate()
            gateway.wait(timeout=20)
            gateway = None
            gateway_log.close()
            gateway_log = None
            candidates = list((runtime.ARCHIVE / 'jobs' / job).glob('*/result.json'))
            parsed = [json.loads(p.read_text()) for p in candidates]
            trial = runtime.find_trial_result(parsed, row['task'])
            rewards = (trial or {}).get('verifier_result') or {}
            rewards = rewards.get('rewards') or {}
            calls = [json.loads(p.read_text()) for p in gateway_root.glob('*.json')]
            benchmark_started = benchmark_started or bool(calls)
            errors = [c for c in calls if c.get('http_status', 0) >= 400 or c.get('error')]
            # A model that uses up its task budget has a valid failed outcome
            # when the native grader finalized it and transport stayed healthy.
            ordinary_deadline = ((trial or {}).get('exception_info') or {}).get('exception_type') == 'AgentTimeoutError'
            ordinary_deadline &= bool(calls) and not errors and rewards.get('finalized') == 1 and rewards.get('success') == 0
            ordinary_deadline &= not any(rewards.get(k) for k in ('no_verdict','missing_log','sim_error','sim_crashed_on_call'))
            valid = bool(trial) and (not trial.get('exception_info') or ordinary_deadline) and receipt['returncode'] == 0
            valid &= rewards.get('success') in (0, 1) and rewards.get('resource_enforced') == 1
            valid &= not any(rewards.get(k) for k in ('infra_error', 'grader_error', 'instance_mismatch',
                                                      'resource_monitor_missing', 'resource_switch_error'))
            valid &= bool(calls) and any(c.get('http_status') == 200 for c in calls)
            record = {'task': row['task'], 'suite': row['suite'], 'partition': row['partition'],
                      'job': job, 'score': rewards.get('success'),
                      'retry_of':spec.get('retry_of',{}).get(row['task']),
                      'protocol_version':spec.get('protocol_version','native-cap16k-v1'),
                      'status': 'graded' if valid else 'infrastructure_failure',
                      'resource_metrics': rewards, 'returncode': receipt['returncode'],
                      'trial_result_sha256': hashlib.sha256(json.dumps(trial, sort_keys=True).encode()).hexdigest(),
                      'model_requests': len(calls), 'model_errors': len(errors),
                      'model_error_examples': [c.get('error') for c in errors[:2]],
                      'recorded_caps': sorted({c['max_output_tokens'] for c in calls}),
                      'returned_models': sorted({c.get('returned_model') for c in calls if c.get('returned_model')}),
                      'video_count': len(list((runtime.ARCHIVE / 'jobs' / job).rglob('*.mp4'))),
                      'archive': str(runtime.ARCHIVE / 'jobs' / job),
                      'error_tail': receipt['log_tail'] if not valid else None,
                      'exception': {k:v for k,v in ((trial or {}).get('exception_info') or {}).items()
                                    if k in {'exception_type','occurred_at'}}, 'completed_utc': runtime.now()}
            results['trials'].append(public_value(record, (provider_key, token)))
            runtime.write('results.json', results)
            if not calls and (trial or {}).get('exception_info'):
                raise RuntimeError('Agent setup/execution failed before any model request; inspect the recorded exception before retrying')
            if not calls or not any(c.get('http_status') == 200 for c in calls):
                raise RuntimeError('Native agent did not complete a successful API request; stopping before further episodes')
            if not valid:
                raise RuntimeError('Native episode infrastructure/resource verification failed; inspect the recorded trial before expansion')
        progress('native_pilot_collected', f'{label} frozen episodes collected. Successful train episodes require provenance/scene admission before SFT.')
    except Exception as exc:
        results['setup_error'] = public_value(str(exc), (provider_key, token))
        progress('native_pilot_needs_review', results['setup_error'])
    finally:
        results['completed_utc'] = runtime.now()
        runtime.write('results.json', results)
        runtime.write('complete.json', {'completed_utc': runtime.now(), 'completed_trials': len(results['trials']),
                                        'planned_trials': len(spec['tasks']), 'error': results.get('setup_error')})
        if gateway is not None:
            gateway.terminate()
        if gateway_log is not None:
            gateway_log.close()
        for _ in range(60):
            if (runtime.ROOT / 'collected').exists():
                break
            time.sleep(5)
        if service is not None:
            service.terminate()
        if service_log is not None:
            service_log.close()


if __name__ == '__main__':
    main()
