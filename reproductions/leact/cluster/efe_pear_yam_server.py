"""Persistent native YAM Kit workers on the same shared volume as the model.

Each allocated GPU owns exactly one task and one live simulator. Reset IDs are
selected only from the independently generated and simulator-checked bank.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import threading
import time

import yaml


def unpack(path,expected,target):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(8<<20):digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError('Pinned native runtime or reset archive changed')
    target.mkdir()
    with tarfile.open(path)as stream:stream.extractall(target,filter='data')


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    transport=Path(spec['transport_root']);transport.mkdir(mode=0o700,parents=True,exist_ok=True)
    if transport.stat().st_mode&0o077:raise ValueError('Native YAM transport root must be private')
    source=root/'source';unpack(spec['source_archive_path'],spec['source_archive_sha256'],source)
    bank=root/'reset-bank';unpack(spec['generated_reset_archive'],spec['generated_reset_archive_sha256'],bank)
    release=spec['assets_release'];assets=Path(release['assets_root'])
    if hashlib.sha256((assets/release['release_manifest']).read_bytes()).hexdigest()!=release['release_manifest_sha256']:
        raise ValueError('Native asset release changed')
    (source/'assets').symlink_to(assets,target_is_directory=True)
    image_assets=Path('/opt/gr00t-sim/source/assets')
    if not image_assets.exists()and not image_assets.is_symlink():image_assets.symlink_to(assets,target_is_directory=True)
    if image_assets.resolve()!=assets.resolve():raise ValueError('Native source and image use different asset releases')
    for name in ['yamlab','IsaacLab-Arena']:
        dependency=Path('/opt/gr00t-sim/source')/name
        if not dependency.is_dir():raise ValueError('Pinned native editable dependency is missing')
        (source/name).symlink_to(dependency,target_is_directory=True)
    from efe_pear_arena_compat import qualify_and_apply
    (root/'arena-compatibility.json').write_text(json.dumps(qualify_and_apply())+'\n')
    devices=[(int(line.split(',')[0]),line.split(',')[1].strip())for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()]
    if len(devices)!=len(spec['tasks']):raise ValueError('Each persistent YAM task requires one exclusive allocated GPU')
    processes=[];lock=threading.Lock();finished=Path(spec['finished']);failed=threading.Event()
    def run(row,device):
        renderer,gpu=device;task=root/row['env_name'];task.mkdir()
        manifest_path=bank/(row['env_name']+'.json');reset=spec['generated_reset_tasks'][row['env_name']]
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=reset['manifest_sha256']:
            raise ValueError('Independent native training reset bank changed')
        manifest=json.loads(manifest_path.read_text());graph=yaml.safe_load((source/row['graphs'][0]['path']).read_text())
        graph.setdefault('extensions',{})['tgen_reset_layouts']={'manifest_path':str(manifest_path),'task':manifest['task'],'source':'saved'}
        graph_path=task/'training-graph.yaml';graph_path.write_text(yaml.safe_dump(graph,sort_keys=False))
        experiment={'shared':{'environment':{'type':str(graph_path),'enable_cameras':True},
            'policy':{'type':'efe_pear_yam_session_policy.YamSessionPolicy',
                'transport_root':spec['transport_root'],'task':row['env_name'],
                'training_reset_manifest':str(manifest_path),'training_reset_manifest_sha256':reset['manifest_sha256'],
                'output':str(task/'sessions'),'worker_id':os.environ['EFE_WORKFLOW_ID']+'-'+row['env_name']},
            'rollout_limit':{'num_steps':100000000},'environment_builder':{'seed':18471}},
            'runs':{'training':{'environment_builder':{'num_envs':1}}}}
        config=task/'experiment.yaml';config.write_text(yaml.safe_dump(experiment,sort_keys=False))
        env={**os.environ,'PYTHONPATH':str(lane)+':'+str(source)+':'+os.environ.get('PYTHONPATH',''),
            'CUDA_VISIBLE_DEVICES':gpu,'OMNI_KIT_ALLOW_ROOT':'1','NVIDIA_DRIVER_CAPABILITIES':'all',
            'RP_JHH_EXPECTED_RENDERER_INDEX':str(renderer),'GR00T_SIM_ROOT':str(source),
            'GR00T_FULL_STATE':'0','PYTHONUNBUFFERED':'1'}
        with(task/'native.log').open('w')as log:
            process=subprocess.Popen(['/opt/gr00t-sim/venv/bin/python','-u',str(lane/'efe_pear_yam_probe_entry.py'),
                str(source/'gr00t_sim/launch_experiment.py'),'--headless','--enable_cameras',
                '--experiment_config',str(config),'--device','cuda:0',
                f'--kit_args=--/renderer/activeGpu={renderer} --/physics/cudaDevice=0 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false'],
                cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:processes.append(process)
            while process.poll()is None:
                head=finished.with_name('head-heartbeat.json')
                lost=head.exists()and time.time()-head.stat().st_mtime>300
                if finished.exists()or failed.is_set()or lost:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    if failed.is_set()or lost:raise RuntimeError('A peer native YAM worker or model head failed')
                    return
                time.sleep(2)
            if not finished.exists():
                failed.set();raise RuntimeError('Native YAM worker exited before training completed')
    try:
        with ThreadPoolExecutor(max_workers=len(devices))as executor:
            futures=[executor.submit(run,row,device)for row,device in zip(spec['tasks'],devices,strict=True)]
            for future in futures:future.result()
    finally:
        for process in processes:
            if process.poll()is None:os.killpg(process.pid,signal.SIGTERM)


if __name__=='__main__':main()
