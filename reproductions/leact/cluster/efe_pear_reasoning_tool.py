"""Model-side PEAR tool; kept out of native simulator child imports."""
import json
import time
from pear.agent_loop.protocol import InvalidAction
from pear.envs.tool import EnvironmentTool
from efe_pear_reasoning_scenes import checked_point


class ReasoningTool(EnvironmentTool):
    def system_prompt(self,result):
        schema=result['state']['answer_schema'];kind=schema['answer_type']
        example={'answer_type':kind,'summary':'A short explanation grounded in the image.'}
        if kind=='bounding_box':example['bbox']=[.1,.2,.3,.4]
        elif kind=='point_set_count':example.update(points=[[.2,.3],[.6,.7]],count=2)
        else:example['point']=[.5,.5]
        return ('Solve the visual reasoning task using the supplied images, instruction, public camera guide '
            'and answer_schema. You may reason before answering. End each response with exactly one json code '
            'block. Copy the answer_type field exactly from answer_schema and return its other required fields '
            'at the top level. Do not nest the action under answer. Use normalized image coordinates with u increasing '
            'right and v down. If permitted, inspect a camera with {"action_type":"view","view":"overhead"} '
            'or another listed view before answering. Include a short summary. Every final answer is checked by '
            'the scene verifier. No code execution or other actions are available.\n'
            'Format example only; choose your own coordinates/count from the actual image:\n```json\n'+
            json.dumps(example)+'\n```')

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
