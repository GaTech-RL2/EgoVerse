"""Install the pinned upstream PEAR runtime; this worker performs no RL updates."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    lane=Path(__file__).resolve().parent
    spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['remote_root']); root.mkdir(parents=True,exist_ok=True)
    attempt=root/'runtime-attempts'/os.environ['EFE_WORKFLOW_ID']
    attempt.mkdir(parents=True,exist_ok=True)

    def record(phase,**extra):
        value={'phase':phase,'updated_utc':now(),'workflow':os.environ['EFE_WORKFLOW_ID'],
               'primary_rl_updates':0,**extra}
        target=attempt/'status.json'; temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(target)
        print('PEAR_SCALED_RUNTIME '+json.dumps(value),flush=True)
        return value

    try:
        record('runtime_start')
        previous=sorted((root/'runtime-attempts').glob('*/install.log'),key=lambda p:p.stat().st_mtime)[-2:]
        if previous:
            record('previous_install_diagnostic',logs={str(p.parent.name):p.read_text(errors='replace')[-3000:] for p in previous})
        required=['curl','getent','tar','sha256sum','git','perl','gcc','g++']
        missing=[name for name in required if shutil.which(name) is None]
        if missing:
            record('installing_system_dependencies',missing=missing)
            subprocess.run(['apt-get','update','-qq'],check=True,timeout=300)
            subprocess.run(['apt-get','install','-y','-qq','curl','git','ca-certificates','perl',
                            'libc-bin','build-essential'],check=True,timeout=600)
        archive=attempt/'pear-source.tar.gz'
        with urllib.request.urlopen(spec['source_url'],timeout=120) as response,archive.open('wb') as target:
            shutil.copyfileobj(response,target)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=spec['source_sha256']:
            raise ValueError('Upstream PEAR archive checksum mismatch')
        source=root/'sources'/('pear-'+spec['pear_revision'])
        with (root/'runtime-install.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            source.mkdir(parents=True,exist_ok=True)
            if not (source/'archive-sha256.txt').exists():
                with tarfile.open(archive) as stream:
                    prefix='pear-'+spec['pear_revision']+'/'
                    if any(not (m.name+'/' if m.isdir() else m.name).startswith(prefix) for m in stream.getmembers()):
                        raise ValueError('Unexpected source archive prefix')
                    stream.extractall(root/'sources',filter='data')
                (source/'archive-sha256.txt').write_text(spec['source_sha256']+'\n')
            if (source/'archive-sha256.txt').read_text().strip()!=spec['source_sha256']:
                raise ValueError('An incompatible PEAR source already occupies the runtime directory')
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,compute_cap',
                                         '--format=csv,noheader'],text=True,timeout=30)
            record('installing_pinned_runtime',gpu=gpu.strip(),source=str(source))
            env={**os.environ,'WORK_DIR':str(root/'runtime'),'CUDA_ARCH':'9.0','MAX_JOBS':'16',
                 'UV_CACHE_DIR':'/tmp/efe-pear-uv-cache','PYTHONUNBUFFERED':'1'}
            with (attempt/'install.log').open('a') as log:
                subprocess.run(['bash','scripts/install.sh','verl'],cwd=source,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
            python=root/'runtime/envs/verl/bin/python'
            probe='''import json, torch, transformers, vllm
from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels
MODEL_TYPES = Qwen3_5TrainingKernels.MODEL_TYPES
assert torch.cuda.device_count()==8
assert 'qwen3_5_moe' in MODEL_TYPES and 'qwen3_5' in MODEL_TYPES
x=torch.arange(16,device='cuda'); assert x.sum().item()==120
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,
 'vllm':vllm.__version__,'model_types':MODEL_TYPES,'gpu_count':torch.cuda.device_count()}))
'''
            result=subprocess.run([str(python),'-c',probe],cwd=source,env=env,
                                  capture_output=True,text=True,timeout=180,check=True)
            (attempt/'probe.log').write_text(result.stdout+result.stderr)
            details=json.loads(result.stdout.strip().splitlines()[-1])
            value=record('runtime_ready',success=True,details=details,
                model_replay_validated=False,environment_bridge_validated=False,
                dense_gpu_backend_supported=True,moe_gpu_backend_supported=False,
                scope='Upstream runtime installed and CUDA import probe passed. No policy qualification or RL update.')
            (root/'runtime-ready.json').write_text(json.dumps(value,indent=2)+'\n')
    except Exception as exc:
        log_path=attempt/'install.log'
        record('runtime_failed',success=False,error=repr(exc),traceback=traceback.format_exc(),
               install_tail=log_path.read_text(errors='replace')[-4000:] if log_path.exists() else '')
        raise


if __name__=='__main__':
    main()
