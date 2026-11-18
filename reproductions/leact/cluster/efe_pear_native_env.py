"""PEAR stateful simulator and tool using the existing EFE constrained interface."""
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from pear.sim.env import EpisodeLost, StatefulEnv
from efe_pear_sim_process import SpawnedSimulatorProcess as SimulatorProcess

from efe_pear_native_episode import NativeEfeEpisode
from efe_pear_bridge import HarborEpisode, NativeInfrastructureError, image_blocks, observation_payload
from efe_pear_context import VERSION as INTERFACE_CONTEXT_VERSION
from efe_pear_observation_check import validate_frames


class NativeEfeEnv(NativeEfeEpisode,StatefulEnv):
    SERVER_RESOURCES={'num_cpus':1,'resources':{'efe':1}}
    START_TIMEOUT_S=1200.
    OPEN_TIMEOUT_S=1800.
    STEP_TIMEOUT_S=1800.
    SCHEMA={}
    CAMERAS=()
    ENVIRONMENT_PROMPT=''
    LOG_KEYS=('native_trial_path','action_contract_sha256','requested_control_steps','executed_control_steps',
              'interface_context_version','interface_context_sha256')

    @classmethod
    def server_n(cls,config,cluster_resources):
        count=int(config.get('server_n',cluster_resources.get('efe',0)))
        if count<1:raise ValueError('Native simulator capacity is missing')
        return count

    @classmethod
    def setup_env(cls,context):
        process=SimulatorProcess('efe',cls,sys.executable,[],dict(os.environ),
                                 Path(__file__).resolve().parent,context.log_dir)
        return [process],1

    @classmethod
    def task_name(cls,env_args):return env_args['task']

    def open(self):
        try:return super().open()
        except NativeInfrastructureError as exc:raise EpisodeLost(str(exc))from exc

    def step(self,action):
        try:return super().step(action)
        except NativeInfrastructureError as exc:raise EpisodeLost(str(exc))from exc


def __getattr__(name):
    # Keep older tool configurations readable without importing training libraries
    # into simulator children. Tool lookup is model-side only.
    if name == 'NativeEfeTool':
        from efe_pear_native_tool import NativeEfeTool
        return NativeEfeTool
    raise AttributeError(name)
