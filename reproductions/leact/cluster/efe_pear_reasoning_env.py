"""PEAR stateful tool for qualified, independently generated reasoning scenes."""
import json
import hashlib
import inspect
import os
from pathlib import Path
import sys

from pear.sim.env import StatefulEnv
from efe_pear_sim_process import SpawnedSimulatorProcess as SimulatorProcess

from efe_pear_reasoning_scenes import SceneEpisode


class ReasoningEnv(StatefulEnv):
    EVALUATION_ONLY=False
    SERVER_RESOURCES={'num_cpus':1,'resources':{'reasoning':1}}
    START_TIMEOUT_S=180.;OPEN_TIMEOUT_S=180.;STEP_TIMEOUT_S=120.
    SCHEMA={};CAMERAS=();ENVIRONMENT_PROMPT=''
    LOG_KEYS=('scene_sha256','reasoning_family','view_actions','generator_version')

    @classmethod
    def server_n(cls,config,cluster_resources):
        count=int(config.get('server_n',cluster_resources.get('reasoning',0)))
        if count<1:raise ValueError('Reasoning simulator capacity is missing')
        return count

    @classmethod
    def setup_env(cls,context):
        env={**os.environ,'MUJOCO_GL':'osmesa','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        return [SimulatorProcess('reasoning',cls,sys.executable,[],env,
                                 Path(__file__).resolve().parent,context.log_dir)],1

    @classmethod
    def task_name(cls,env_args):return 'procedural_reasoning/'+env_args['family']

    def __init__(self,family,seed,partition='train'):
        from efe_pear_reasoning_scenes import VERSION
        if partition!=('development'if self.EVALUATION_ONLY else'train'):
            raise ValueError('Training and development scenes require separate environment classes')
        manifest=json.loads(Path(os.environ['EFE_PEAR_REASONING_MEMBERSHIP']).read_text())
        key=f'{family}/{partition}/{seed}';item=manifest['scenes'][key]
        if (item.get('qualification')!='passed'or manifest.get('generator_version')!=VERSION or
            manifest.get('generator_sha256')!=hashlib.sha256(Path(inspect.getfile(SceneEpisode)).read_bytes()).hexdigest()):
            raise ValueError('Reasoning reset has not passed generator/reward qualification')
        self.episode=SceneEpisode(family,seed,partition);self.last_action=None
        if self.episode.scene_sha256!=item['scene_sha256']:
            self.episode.close();raise ValueError('Qualified reasoning scene changed')

    def _result(self,observation,reward=0.,done=False):
        from efe_pear_reasoning_scenes import VERSION
        return {'images':observation['images']if observation else None,
            'state':observation['public']if observation else{},
            'instruction':self.episode.question,'actions':[]if self.last_action is None else[self.last_action],
            'rewards':[reward],'done':done,'executed':int(self.last_action is not None),
            'outcome':'done'if done else'observed','scene_sha256':self.episode.scene_sha256,
            'reasoning_family':self.episode.family,'view_actions':self.episode.view_actions,
            'generator_version':VERSION}

    def open(self):return self._result(self.episode.observation())

    def step(self,action):
        self.last_action=action
        obs,reward,done=self.episode.step(action)
        return self._result(obs,reward,done)

    def close(self):self.episode.close()


class ReasoningDevelopmentEnv(ReasoningEnv):
    EVALUATION_ONLY=True


def __getattr__(name):
    # Keep older tool configurations readable without importing training libraries
    # into simulator children. Tool lookup is model-side only.
    if name == 'ReasoningTool':
        from efe_pear_reasoning_tool import ReasoningTool
        return ReasoningTool
    raise AttributeError(name)
