"""Model-side PEAR process proxies for independently allocated native workers."""
import os
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
              'interface_context_version','interface_context_sha256','scene_sha256','native_termination_terms')
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
        kind=self.KIND+('/'+env_args['task']if self.KIND=='native_yam'else'')
        self.client=FileEpisodeClient(os.environ['EFE_PEAR_REMOTE_ROOT'],kind,env_args)

    def open(self):
        try:return self.client.open()
        except RemoteEpisodeLost as exc:raise EpisodeLost(str(exc))from exc

    def step(self,action):
        try:return self.client.step(action)
        except RemoteEpisodeLost as exc:raise EpisodeLost(str(exc))from exc

    def close(self):self.client.close()


class NativeYamEnv(RemoteNativeEnv):
    KIND='native_yam'
    SERVER_RESOURCES={'num_cpus':1,'resources':{'native_yam_proxy':1}}
