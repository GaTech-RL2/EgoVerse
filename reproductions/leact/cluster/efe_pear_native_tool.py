"""Model-side PEAR tool; kept out of native simulator child imports."""
import json
import time
from pear.agent_loop.protocol import InvalidAction
from pear.envs.tool import EnvironmentTool
from efe_action_loop import SYSTEM, parse_action


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

    async def step(self,episode,action):
        try:parse_action(json.dumps(action),contract=self.contracts[episode.uid],allow_stop=True)
        except (ValueError,TypeError,KeyError) as exc:raise InvalidAction(str(exc)) from exc
        before,start=len(episode.actions),time.monotonic()
        result=await self.client.step(episode.uid,action)
        self._log_call(episode,before,action,result,time.monotonic()-start)
        return self._observation(result),result

    async def close(self,episode):
        try:await super().close(episode)
        finally:self.contracts.pop(episode.uid,None)
