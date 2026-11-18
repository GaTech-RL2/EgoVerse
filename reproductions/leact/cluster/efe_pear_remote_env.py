"""Model-side PEAR process proxies for independently allocated native workers."""
import os
import json
from pathlib import Path
import sys

from pear.sim.env import EpisodeLost,StatefulEnv
from efe_pear_file_rpc import FileEpisodeClient,RemoteEpisodeLost
from efe_pear_sim_process import SpawnedSimulatorProcess


class RemoteNativeEnv(StatefulEnv):
    SCHEMA={};CAMERAS=();ENVIRONMENT_PROMPT=''
    START_TIMEOUT_S=180.;OPEN_TIMEOUT_S=1800.;STEP_TIMEOUT_S=1800.
    SERVER_RESOURCES={'num_cpus':1,'resources':{'native_proxy':1}}
    LOG_KEYS=('native_trial_path','action_contract_sha256','requested_control_steps','executed_control_steps',
              'interface_context_version','interface_context_sha256')
    KIND='efe'

    @classmethod
    def setup_env(cls,context):
        return [SpawnedSimulatorProcess('native-proxy',cls,sys.executable,[],dict(os.environ),
            Path(__file__).resolve().parent,context.log_dir)],1

    @classmethod
    def server_n(cls,config,cluster_resources):
        count=int(config.get('server_n',0))
        if count<1:raise ValueError('A native proxy requires an explicit positive server count')
        return count

    @classmethod
    def task_name(cls,env_args):return env_args['task']

    def __init__(self,**env_args):
        if self.KIND=='native_yam':kind='native_yam/'+env_args['task']
        else:
            config=json.loads(Path(os.environ['EFE_PEAR_NATIVE_CONFIG']).read_text())
            membership=json.loads(Path(config['membership_file']).read_text())
            item=membership['training_tasks'][env_args['task']]
            if item['qualification']!='passed':raise ValueError('Remote native task is not admitted')
            kind='efe/'+item['row']['suite']
        self.client=FileEpisodeClient(os.environ['EFE_PEAR_REMOTE_ROOT'],kind,env_args,timeout=self.OPEN_TIMEOUT_S)

    def open(self):
        try:return self.client.open()
        except RemoteEpisodeLost as exc:raise EpisodeLost(str(exc))from exc

    def step(self,action):
        try:return self.client.step(action)
        except RemoteEpisodeLost as exc:raise EpisodeLost(str(exc))from exc

    def close(self):self.client.close()


class NativeYamEnv(RemoteNativeEnv):
    KIND='native_yam'
    OPEN_TIMEOUT_S=14400.
    SERVER_RESOURCES={'num_cpus':1,'resources':{'native_yam_proxy':1}}
    LOG_KEYS=RemoteNativeEnv.LOG_KEYS+('scene_sha256','native_termination_terms')
