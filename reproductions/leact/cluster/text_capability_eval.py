"""Frozen text-Qwen checkpoint screen, derived from the verified EFE v4 evaluator.

The EFE evaluator remains unchanged so its future checkpoint protocol still
matches its completed base run. This protocol supports pinned HF snapshot links
and does not confuse lost arithmetic/reasoning ability with an HTTP failure.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path('/tmp/efe-capability')
SPEC = Path('/tmp/efe-capability-spec.json')
ARCHIVE = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    for directory in [ROOT, ARCHIVE]:
        temp = directory / (name + '.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temp.replace(directory / name)


def progress(phase, detail, **extra):
    value = {'phase': phase, 'detail': detail, 'updated_utc': datetime.now(timezone.utc).isoformat(),
             'workflow': os.environ['EFE_WORKFLOW_ID'], **extra}
    write('progress.json', value)
    print(json.dumps(value), flush=True)


def run(name, command, *, env=None, timeout=900, cwd=None):
    with (ARCHIVE / (name + '.log')).open('w') as log:
        subprocess.run(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def container_runtime(spec):
    progress('scorer_setup', 'Preparing a pinned, network-disabled code evaluator.')
    run('apt', ['bash', '-c', 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq podman uidmap slirp4netns ca-certificates'], timeout=600)
    archive = ROOT / 'podman.tar.gz'
    with urllib.request.urlopen(spec['podman']['url'], timeout=180) as inp, archive.open('wb') as out:
        while chunk := inp.read(1024 * 1024): out.write(chunk)
    if sha(archive) != spec['podman']['sha256']:
        raise ValueError('Podman digest mismatch')
    with tarfile.open(archive) as tar: tar.extractall(ROOT / 'podman', filter='data')
    binary = next((ROOT / 'podman').glob('*/usr/local/bin/podman'))
    prefix = binary.parents[3]
    config = ROOT / 'containers.conf'
    config.write_text('[engine]\nhelper_binaries_dir = [' + json.dumps(str(prefix / 'usr/local/lib/podman')) +
                      ']\nconmon_path = [' + json.dumps(str(next(prefix.rglob('conmon')))) +
                      ']\ncgroup_manager="cgroupfs"\nevents_logger="file"\n[engine.runtimes]\ncrun=[' +
                      json.dumps(str(binary.parent / 'crun')) + ']\n')
    storage = ROOT / 'storage.conf'
    storage.write_text('[storage]\ndriver="vfs"\nrunroot="/tmp/efe-capability/run"\ngraphroot="/tmp/efe-capability/containers"\n')
    return {**os.environ, 'PATH': str(binary.parent) + ':' + os.environ['PATH'],
            'CONTAINERS_CONF': str(config), 'CONTAINERS_STORAGE_CONF': str(storage)}


def scorer_image(spec, bundle, env):
    cache = Path(spec['shared_root']) / 'scorer-cache' / spec['protocol_sha256']
    cache.mkdir(parents=True, exist_ok=True)
    image = 'localhost/efe-capability-scorer:v1'
    with (cache / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (cache / 'image.json').exists():
            meta = json.loads((cache / 'image.json').read_text())
            if sha(cache / 'scorer.oci') != meta['archive_sha256']:
                raise ValueError('Saved evaluator image checksum mismatch')
            run('load-scorer', ['podman', 'load', '-i', str(cache / 'scorer.oci')], env=env)
        else:
            run('build-scorer', ['podman', 'build', '--isolation=chroot', '--network=host',
                                '-f', '/tmp/efe-capability-Dockerfile', '-t', image, str(bundle)], env=env, timeout=1200)
            run('save-scorer', ['podman', 'save', '--format=oci-archive', '-o', str(cache / 'scorer.oci'), image], env=env)
            meta = {'archive_sha256': sha(cache / 'scorer.oci'), 'image': image,
                    'inspect': json.loads(subprocess.check_output(['podman', 'image', 'inspect', image], env=env))}
            temp = cache / 'image.json.tmp'
            temp.write_text(json.dumps(meta)); temp.replace(cache / 'image.json')
    write('scorer-image.json', meta)
    return image


def sandbox(image, *, mounts=()):
    command = ['podman', 'run', '--rm', '--network=none', '--cgroups=disabled', '--cap-drop=ALL',
               '--security-opt=no-new-privileges', '--user=65534:65534', '--read-only',
               '--tmpfs=/tmp:rw,nosuid,nodev,size=4g', '--ulimit=nofile=2048:2048']
    for source, target, mode in mounts:
        command.extend(['-v', str(source) + ':' + target + ':' + mode])
    return command + [image]


def verify_checkpoint(path, manifest):
    manifest = json.loads(Path(manifest).read_text())
    for name, row in manifest['files'].items():
        file = path / name
        if Path(name).name != name:
            raise ValueError('Checkpoint manifest contains a nonlocal filename')
        if str(file.resolve()) != row['resolved_path']:
            raise ValueError('Checkpoint resolved path differs from the frozen manifest')
        if file.stat().st_size != row['bytes'] or sha(file) != row['sha256']:
            raise ValueError('Checkpoint mismatch: ' + name)
    weights = {k: v['sha256'] for k, v in manifest['files'].items() if k.endswith('.safetensors')}
    if not weights:
        raise ValueError('A complete full-parameter safetensors checkpoint is required')
    return hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()


def driver_environment():
    """Prefer the host-matching CUDA driver over this image's older compat driver."""
    kernel = Path('/proc/driver/nvidia/version').read_text()
    match = re.search(r'\b(\d{3}\.\d+(?:\.\d+)?)\b', kernel)
    if not match:
        raise RuntimeError('Cannot identify the allocated node NVIDIA kernel driver')
    version = match.group(1)
    candidates = [Path(root) / ('libcuda.so.' + version) for root in
                  ['/usr/lib/x86_64-linux-gnu', '/usr/local/nvidia/lib64']]
    library = next((p for p in candidates if p.is_file()), None)
    if library is None:
        raise RuntimeError('No CUDA userspace library matches the host kernel driver ' + version)
    # apt/ldconfig during container setup can change the default library cache.
    # Preload the exact host library only in the model server and its workers.
    env = {**os.environ, 'LD_PRELOAD': str(library), 'CUDA_VISIBLE_DEVICES': '0,1,2,3'}
    probe = subprocess.run([sys.executable, '-c',
                           'import torch,json;torch.cuda.init();print(json.dumps({"devices":torch.cuda.device_count(),"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0)}))'],
                           env=env, text=True, capture_output=True, timeout=120)
    write('cuda-preflight.json', {'kernel_driver': version, 'library': str(library),
          'returncode': probe.returncode, 'stdout': probe.stdout[-1500:], 'stderr': probe.stderr[-1500:]})
    if probe.returncode:
        raise RuntimeError('Host-matched CUDA initialization failed; inspect cuda-preflight.json')
    return env


def generate_one(row, settings, token):
    started = time.monotonic()
    # Request seeds are stable per item and do not depend on concurrency/completion order.
    seed = int.from_bytes(hashlib.sha256((str(settings['seed']) + '/' + row['id']).encode()).digest()[:4], 'big') % (2**31)
    body = {**settings, 'seed': seed, 'model': 'efe-retention-student',
            'messages': [{'role': 'user', 'content': row['prompt']}]}
    replica = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:4], 'big') % 4
    request = urllib.request.Request(f'http://127.0.0.1:{18890 + replica}/v1/chat/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
    # A transport failure stops this run; do not replace valid wrong answers.
    with urllib.request.urlopen(request, timeout=1200) as response:
        data = json.load(response)
    if len(data['choices']) != 1:
        raise ValueError('Expected one answer per question')
    choice = data['choices'][0]
    if choice['finish_reason'] not in {'stop', 'length'}:
        raise ValueError('Unexpected generation termination: ' + str(choice['finish_reason']))
    message = choice['message']
    # vLLM deepseek_r1 accepts Qwen3-VL's template-provided opening think tag.
    final = message.get('content') or ''
    # Repeated delimiters can be generated by a finetuned checkpoint, and can
    # also be legitimate text in a programming answer. They are model outputs,
    # not HTTP failures. Keep the final field verbatim and count this condition.
    residual_delimiters = '</think>' in final or '<think>' in final
    key = hashlib.sha256(row['id'].encode()).hexdigest()
    with gzip.open(ARCHIVE / 'raw' / (key + '.json.gz'), 'wt') as out:
        json.dump({'id': row['id'], 'request': body, 'response': data}, out)
    return {'id': row['id'], 'benchmark': row['benchmark'], 'status': 'completed', 'seed': seed, 'replica': replica,
            'final': final, 'residual_reasoning_delimiters': residual_delimiters,
            'finish_reason': choice['finish_reason'], 'output_tokens': data['usage']['completion_tokens'],
            'input_tokens': data['usage']['prompt_tokens'], 'elapsed_seconds': time.monotonic() - started,
            'reasoning_characters': len(message.get('reasoning_content') or message.get('reasoning') or ''),
            'raw_sha256': sha(ARCHIVE / 'raw' / (key + '.json.gz'))}


def main():
    global ARCHIVE
    spec = json.loads(SPEC.read_text())
    ROOT.mkdir(exist_ok=True)
    ARCHIVE = Path(spec['shared_root']) / 'runs' / os.environ['EFE_WORKFLOW_ID']
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / 'raw').mkdir(exist_ok=True)
    write('spec.json', spec)
    servers = []
    logs = []
    try:
        archive = Path(spec['bundle_path'])
        if sha(archive) != spec['bundle_sha256']:
            raise ValueError('Frozen evaluation bundle checksum mismatch')
        bundle = ROOT / 'bundle'
        with tarfile.open(archive) as tar: tar.extractall(bundle, filter='data')
        plan = json.loads((bundle / 'suite.json').read_text())
        for name, digest in plan['file_sha256'].items():
            if sha(bundle / name) != digest: raise ValueError('Frozen suite checksum mismatch: ' + name)
        # The scorer entry point is an OSMO-mounted immutable source file.
        (bundle / 'score.py').write_bytes(Path('/tmp/efe-capability-score.py').read_bytes())
        env = container_runtime(spec)
        scorer = scorer_image(spec, bundle, env)
        smoke = (
            'from evalplus.eval import untrusted_check;'
            'from instruction_following_eval import evaluation_lib;'
            'from evalplus.sanitize import sanitize;'
            'import nltk;nltk.data.load("tokenizers/punkt/english.pickle");'
            'good=untrusted_check("humaneval","def f(x): return x",[[1]],"f",[1],0,[.001]);'
            'bad=untrusted_check("humaneval","def f(x): return x+1",[[1]],"f",[1],0,[.001]);'
            'assert good[0]=="pass" and bad[0]!="pass";print("known-good/known-bad scoring passed")')
        run('scorer-smoke', sandbox(scorer) + ['python', '-c', smoke], env=env, timeout=180)
        progress('verifying_weights', 'Evaluator smoke checks passed; verifying the exact checkpoint weights.')
        driver_env = driver_environment()
        model = Path(spec['checkpoint']['path'])
        weights_sha = verify_checkpoint(model, spec['checkpoint']['manifest'])
        if spec['checkpoint']['id'] != 'qwen-base':
            verify_checkpoint(Path(spec['base_path']), spec['base_manifest'])
        write('checkpoint.json', {**spec['checkpoint'], 'weights_sha256': weights_sha})
        token = secrets.token_urlsafe(32)
        commands = []
        # Four identical one-GPU replicas avoid cross-GPU NCCL initialization.
        # Each question always routes to the same replica at every training stage.
        for replica in range(4):
            command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model),
                       '--tokenizer', spec['base_path'], '--served-model-name', 'efe-retention-student',
                       '--host', '127.0.0.1', '--port', str(18890 + replica), '--tensor-parallel-size', '1',
                       '--max-model-len', '32768', '--max-num-seqs', '8', '--max-num-batched-tokens', '8192',
                       '--gpu-memory-utilization', '0.85', '--dtype', 'bfloat16',
                       '--reasoning-parser', 'deepseek_r1', '--generation-config', spec['base_path']]
            commands.append(command)
            log = (ARCHIVE / f'vllm-{replica}.log').open('w'); logs.append(log)
            servers.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                           env={**driver_env, 'CUDA_VISIBLE_DEVICES': str(replica), 'VLLM_PORT': str(29890 + replica),
                                'VLLM_API_KEY': token, 'OMP_NUM_THREADS': '2'}))
        write('serving.json', {'commands': commands, 'protocol_sha256': spec['protocol_sha256'],
                               'generation': plan['generation'], 'concurrency': 16, 'replicas': 4,
                               'tensor_parallel_per_replica': 1, 'dtype': 'bfloat16'})
        progress('model_loading', 'Loading four independent Qwen replicas for the frozen 985-question screen.')
        ready = set()
        for _ in range(180):
            for replica, server in enumerate(servers):
                if server.poll() is not None: raise RuntimeError(f'vLLM replica {replica} failed; see vllm-{replica}.log')
                if replica in ready: continue
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{18890 + replica}/health', timeout=5): ready.add(replica)
                except Exception: pass
            if len(ready) == 4: break
            time.sleep(5)
        else: raise TimeoutError('Model startup exceeded 15 minutes')
        prompts = [json.loads(x) for x in (bundle / 'inputs/prompts.jsonl').read_text().splitlines()]
        # Validate the chat/reasoning transport on a synthetic, unscored question first.
        probe = generate_one({'id': 'transport-probe', 'benchmark': 'probe', 'prompt': 'What is 2 + 2? Give the answer.'}, plan['generation'], token)
        if spec['checkpoint']['id'] == spec['comparison_base'] and (
                not probe['final'].strip() or '4' not in probe['final'] or not probe['reasoning_characters']):
            raise ValueError('Synthetic final/reasoning separation preflight failed')
        # A finetuned model may have lost even this ability. Record its answer;
        # only the untouched baseline must establish parser/transport behavior.
        write('generation-preflight.json', probe)
        completed = []
        generation_file = ARCHIVE / 'generations.jsonl'
        if generation_file.exists():
            completed = [json.loads(x) for x in generation_file.read_text().splitlines()]
        done = {r['id'] for r in completed}
        started = time.monotonic()
        progress('generating', 'Generating fixed-budget answers; no scores available yet.', completed=len(completed), total=len(prompts))
        with generation_file.open('a') as out, ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(generate_one, row, plan['generation'], token): row for row in prompts if row['id'] not in done}
            try:
                for future in as_completed(futures):
                    row = future.result()
                    out.write(json.dumps(row) + '\n'); out.flush(); completed.append(row)
                    if len(completed) % 10 == 0 or len(completed) == len(prompts):
                        progress('generating', f'{len(completed)}/{len(prompts)} questions answered; grading follows.',
                                 completed=len(completed), total=len(prompts), generation_seconds=time.monotonic() - started,
                                 truncated=sum(r['finish_reason'] == 'length' for r in completed))
            except Exception:
                for future in futures: future.cancel()
                raise
        for server in servers: os.killpg(server.pid, signal.SIGTERM)
        for server in servers: server.wait(timeout=60)
        servers = []
        for log in logs: log.close()
        grading = ARCHIVE / 'grading'; grading.mkdir(exist_ok=True); grading.chmod(0o777)
        progress('grading', 'All answers saved; running expanded coding tests and official instruction checks.',
                 completed=len(completed), total=len(prompts))
        command = sandbox(scorer, mounts=[(bundle / 'inputs', '/inputs', 'ro'),
                                         (generation_file, '/generations.jsonl', 'ro'), (grading, '/output', 'rw')])
        command += ['python', '/opt/score.py', '--inputs', '/inputs', '--generations', '/generations.jsonl',
                    '--output', '/output', '--suite-sha256', spec['protocol_sha256']]
        run('grading', command, env=env, timeout=2400)
        result = json.loads((grading / 'results.json').read_text())
        result.update(checkpoint=spec['checkpoint']['id'], weights_sha256=weights_sha,
                      workflow=os.environ['EFE_WORKFLOW_ID'], generation=plan['generation'],
                      model=spec['evaluation_model'], family=spec['family'], comparison_base=spec['comparison_base'],
                      scorer_image_sha256=json.loads((ARCHIVE / 'scorer-image.json').read_text())['archive_sha256'])
        write('results.json', result)
        progress('completed', 'All three capability benchmarks graded. Compare future checkpoints using this frozen protocol.',
                 completed=len(completed), total=len(prompts))
        write('complete.json', {'success': True, 'updated_utc': datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        progress('needs_review', str(exc)[-1500:])
        write('complete.json', {'success': False, 'error': str(exc)[-1500:]})
        raise
    finally:
        for server in servers:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(server.pid, signal.SIGKILL)
        for log in logs: log.close()
        # Summary collection is bounded; raw outputs also persist on shared storage.
        for _ in range(60):
            if (ROOT / 'collected').exists(): break
            time.sleep(5)


if __name__ == '__main__':
    main()
