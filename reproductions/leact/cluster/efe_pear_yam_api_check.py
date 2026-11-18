"""Probe the pinned native YAM catalog with its existing Arena launcher."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import yaml


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    def record(phase,**fields):
        value={'phase':phase,'primary_rl_updates':0,**fields};(root/'status.json').write_text(json.dumps(value)+'\n')
        print('PEAR_NATIVE_YAM_CHECK '+json.dumps(value),flush=True)
    gpu_names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if not gpu_names or any('H100'in name or'A100'in name for name in gpu_names):
        raise RuntimeError('Native Isaac RGB qualification requires an RTX-capable GPU; H100/A100 are unsupported for rendering')
    record('native_yam_source_staging')
    archive=Path(spec['source_archive_path']).read_bytes()
    if hashlib.sha256(archive).hexdigest()!=spec['source_archive_sha256']:raise ValueError('Native source archive changed')
    source=root/'source';source.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive))as stream:stream.extractall(source,filter='data')
    release=spec['assets_release'];assets=Path(release['assets_root'])
    if hashlib.sha256((assets/release['release_manifest']).read_bytes()).hexdigest()!=release['release_manifest_sha256']:
        raise ValueError('Native asset release manifest differs from the pinned release')
    (source/'assets').symlink_to(assets,target_is_directory=True)
    # The image's editable embodiment dependency embeds its build-time asset
    # root. Resolve that path to the same pinned, read-only asset release as our
    # source overlay. Preserve the native canonical-geometry equality guard.
    image_assets=Path('/opt/gr00t-sim/source/assets')
    if not image_assets.exists()and not image_assets.is_symlink():
        image_assets.symlink_to(assets,target_is_directory=True)
    if image_assets.resolve()!=assets.resolve():
        raise ValueError('The image embodiment and source overlay use different asset releases')
    python=Path('/opt/gr00t-sim/venv/bin/python')
    if not python.is_file():raise RuntimeError('Qualified native image Python is absent')
    probe=subprocess.run([str(python),'-c',
        'import importlib.metadata,json,isaaclab_arena;print(json.dumps({"arena_source":isaaclab_arena.__file__,"isaaclab":importlib.metadata.version("isaaclab")}))'],
        capture_output=True,text=True,timeout=120)
    (root/'runtime-probe.log').write_text(probe.stdout+probe.stderr)
    if probe.returncode:raise RuntimeError('Native Arena/Isaac runtime probe failed: '+probe.stderr[-1500:])
    record('native_yam_api_checks',runtime=probe.stdout.strip()[-1500:])
    def check(row,gpu,renderer_index):
        task=root/row['env_name'];task.mkdir();result=task/'api-result.json'
        experiment={'shared':{'environment':{'type':str(source/row['graphs'][0]['path'])},
            'policy':{'type':'efe_pear_yam_api_policy.YamApiProbePolicy','report_path':str(result)},
            'rollout_limit':{'num_episodes':1},'environment_builder':{'seed':18471}},
            'runs':{'api_qualification':{'environment_builder':{'num_envs':1}}}}
        config=task/'experiment.yaml';config.write_text(yaml.safe_dump(experiment,sort_keys=False))
        env={**os.environ,'PYTHONPATH':str(lane)+':'+str(source)+':'+os.environ.get('PYTHONPATH',''),
             'CUDA_VISIBLE_DEVICES':gpu,'OMNI_KIT_ALLOW_ROOT':'1','NVIDIA_DRIVER_CAPABILITIES':'all',
             'RP_JHH_EXPECTED_RENDERER_INDEX':str(renderer_index),
             'GR00T_SIM_ROOT':str(source),'GR00T_FULL_STATE':'0','PYTHONUNBUFFERED':'1'}
        with (task/'native.log').open('w')as log:
            try:
                completed=subprocess.run([str(python),'-u',str(lane/'efe_pear_yam_probe_entry.py'),
                    str(source/'gr00t_sim/launch_experiment.py'),
                    '--headless','--enable_cameras','--experiment_config',str(config),'--device','cuda:0',
                    f'--kit_args=--/renderer/activeGpu={renderer_index} --/physics/cudaDevice=0 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false'],cwd=source,env=env,
                    stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                value=json.loads(result.read_text())if result.exists()else{'api_reset_action_observation_passed':False}
                value.update(env_name=row['env_name'],returncode=completed.returncode,
                    success=completed.returncode==0 and value.get('api_reset_action_observation_passed')is True)
            except Exception as exc:value={'env_name':row['env_name'],'success':False,'error':repr(exc)}
        if not value['success']:value['log_tail']=(task/'native.log').read_text(errors='replace')[-2500:]
        return value
    devices=[(int(line.split(',')[0]),line.split(',')[1].strip())for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()]
    # One sequential lane per GPU: a Kit application never shares a GPU with
    # another application, including during long asset loads.
    results={}
    def lane_work(index):
        renderer,gpu=devices[index]
        return [check(row,gpu,renderer)for row in spec['tasks'][index::len(devices)]]
    with ThreadPoolExecutor(max_workers=len(devices))as pool:
        for future in as_completed([pool.submit(lane_work,i)for i in range(len(devices))]):
            for value in future.result():results[value['env_name']]=value
            (root/'checks.json').write_text(json.dumps(results,indent=2)+'\n')
            record('native_yam_api_checks',checked=len(results),total=len(spec['tasks']),passed=sum(v['success']for v in results.values()))
    record('native_yam_api_checks_completed',checked=len(results),total=len(spec['tasks']),passed=sum(v['success']for v in results.values()),
        training_admitted=False,independent_training_resets_validated=False,
        all_catalog_tasks_checked=len(spec['tasks'])==spec.get('catalog_task_count',35))
    if not all(v['success']for v in results.values()):raise RuntimeError('Some native YAM API probes failed')


if __name__=='__main__':main()
