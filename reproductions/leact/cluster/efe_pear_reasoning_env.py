"""PEAR stateful tool for qualified, independently generated reasoning scenes."""
import json
import hashlib
import inspect
import os
from pathlib import Path
import sys
import time

from pear.agent_loop.protocol import InvalidAction
from pear.envs.tool import EnvironmentTool
from pear.sim.env import StatefulEnv
from pear.sim.process import SimulatorProcess

from efe_pear_reasoning_scenes import SceneEpisode,checked_point


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


class ReasoningTool(EnvironmentTool):
    def system_prompt(self,result):
        return ('Solve the visual reasoning task using the supplied images, instruction, public camera guide '
            'and answer_schema. You may reason before answering. End each response with exactly one json code '
            'block. Return the typed answer in answer_schema; use normalized image coordinates with u increasing '
            'right and v down. If permitted, inspect a camera with {"action_type":"view","view":"overhead"} '
            'or another listed view before answering. Include a short summary. Every final answer is checked by '
            'the scene verifier. No code execution or other actions are available.')

    async def step(self,episode,action):
        if not isinstance(action,dict):raise InvalidAction('The action must be one JSON object')
        try:
            if action.get('action_type')=='view':
                if episode.env_args.get('family')!='active_perception_memory':raise ValueError('This task does not permit camera actions')
                if sum(isinstance(a,dict)and a.get('action_type')=='view'for a in episode.actions)>=3:
                    raise ValueError('The camera-action budget is exhausted')
                if action.get('view')not in {'front','side','overhead'}:raise ValueError('Unknown camera view')
            elif 'bbox'in action:
                b=action['bbox']
                if not isinstance(b,list)or len(b)!=4:raise ValueError('A box needs four coordinates')
                checked_point(b[:2]);checked_point(b[2:])
                if b[0]>=b[2]or b[1]>=b[3]:raise ValueError('Bounding box coordinates are inverted')
            elif 'points'in action:
                if not isinstance(action['points'],list)or type(action.get('count'))is not int or action['count']<0:
                    raise ValueError('Point sets require a list of points and a nonnegative integer count')
                for p in action['points']:checked_point(p)
            else:checked_point(action['point'])
        except (ValueError,KeyError,TypeError)as exc:raise InvalidAction(str(exc))from exc
        before,start=len(episode.actions),time.monotonic()
        result=await self.client.step(episode.uid,action)
        self._log_call(episode,before,action,result,time.monotonic()-start)
        return self._observation(result),result

    def _observation(self,result):
        if result['images']is None:return None
        public=result['state'];blocks=[{'type':'text','text':json.dumps(public)+'\nEnd with one json code block.'}]
        for name,image in zip(public['sensor_order'],result['images'],strict=True):
            blocks.extend([{'type':'text','text':'Current camera: '+name},{'type':'image','image':image}])
        return blocks
