"""Model-side PEAR tool; kept out of native simulator child imports."""
import json
import time
import os
from pathlib import Path
import re
from pear.agent_loop.protocol import InvalidAction
from pear.envs.tool import EnvironmentTool
from efe_action_loop import SYSTEM, parse_action
from efe_pear_native_finalization import finalize_episode


class NativeEfeTool(EnvironmentTool):
    def __init__(self,*args,**kwargs):
        self.contracts={}
        super().__init__(*args,**kwargs)

    async def open(self,episode):
        content,result=await super().open(episode)
        self.contracts[episode.uid]=result['action_contract']
        return content,result

    def system_prompt(self,result):
        # The action schema and controller are unchanged; PEAR stores the model's
        # reasoning separately and its parser expects the final fenced JSON.
        return SYSTEM.replace('Return only one JSON object with action_type',
            'End your response with one json code block containing action_type')+(
            '\nGround your reasoning in the action and observation guide supplied with each observation, '
            'including its coordinate frames, units and named sensor layout.')

    def _observation(self,result):return result['public_blocks']

    def _log_call(self,episode,steps_before,action,result,duration_s):
        super()._log_call(episode,steps_before,action,result,duration_s)
        if os.environ.get('PEAR_TRACE_FRAMES')!='1'or not result.get('images'):return
        # Small copies of every delivered view support behavior review without
        # duplicating native full-resolution videos in shared storage.
        try:
            if not re.fullmatch(r'[A-Za-z0-9_-]+',episode.uid):raise ValueError('Unexpected episode identity')
            index=steps_before+int(action is not None)
            directory=self.log_path.parent/'frames'/episode.uid/f'{index:05d}'
            directory.mkdir(parents=True,exist_ok=True)
            cameras=result['action_contract']['cameras'];frames=[]
            for camera,image in zip(cameras,result['images'],strict=True):
                name=re.sub(r'[^A-Za-z0-9_.-]','_',camera)+'.jpg'
                view=image.convert('RGB').copy();view.thumbnail((320,240));view.save(directory/name,quality=65)
                frames.append({'camera':camera,'file':name,'size':list(view.size)})
            (directory/'frames.json').write_text(json.dumps({'uid':episode.uid,'env':episode.env_args,
                'action':action,'frames':frames,'native_trial_path':result.get('native_trial_path')})+'\n')
        except Exception as exc:
            print('PEAR_FRAME_CAPTURE_ERROR '+json.dumps({'error_type':type(exc).__name__}),flush=True)

    async def step(self,episode,action):
        try:parse_action(json.dumps(action),contract=self.contracts[episode.uid],allow_stop=True)
        except (ValueError,TypeError,KeyError) as exc:raise InvalidAction(str(exc)) from exc
        before,start=len(episode.actions),time.monotonic()
        result=await self.client.step(episode.uid,action)
        self._log_call(episode,before,action,result,time.monotonic()-start)
        return self._observation(result),result

    async def close(self,episode):
        try:
            if episode.uid in self.contracts and not episode.done:
                receipt=await finalize_episode(self.client,episode)
                if receipt:
                    path=self.log_path.parent/('native_finalizations_'+str(os.getpid())+'.jsonl')
                    with path.open('a')as stream:stream.write(json.dumps(receipt)+'\n')
        finally:
            try:await super().close(episode)
            finally:self.contracts.pop(episode.uid,None)
