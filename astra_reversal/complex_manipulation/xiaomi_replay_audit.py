"""Short native-only replay diagnostics; these are not policy success trials."""

import argparse
import copy
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from astra_reversal.records import digest
from .worker import write_json
from .xiaomi_eval import Recorder
from .xiaomi_selector_eval import ReplayEnvironment, integration_state, restore_anchor, fingerprint


def numeric_state(obj, depth=0):
    """Record small controller/buffer values without traversing simulator objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size < 4096 and np.issubdtype(obj.dtype, np.number) else None
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or type(obj) in (str, bool, int, float):
        return obj
    if depth > 3:
        return None
    if isinstance(obj, dict):
        result = {str(k): numeric_state(v, depth+1) for k,v in obj.items()}
        return {k:v for k,v in result.items() if v is not None}
    if isinstance(obj, (tuple, list)):
        return [numeric_state(v, depth+1) for v in obj] if len(obj) < 100 else None
    if obj.__class__.__module__.startswith('robosuite.controllers') or 'Buffer' in obj.__class__.__name__:
        return numeric_state(vars(obj), depth+1)
    return None


def controller_snapshot(raw):
    return [numeric_state({'composite': robot.composite_controller,
             'parts': robot.composite_controller.part_controllers,
             **{k:v for k,v in vars(robot).items() if 'recent' in k or 'buffer' in k}}) for robot in raw.robots]


class AuditEnvironment(ReplayEnvironment):
    def __init__(self, *args):
        super().__init__(*args)
        self.states = []
        self.controllers = []
        self.random_states = []

    def snapshot_auxiliary(self):
        raw = self.env.unwrapped.env
        self.controllers.append(controller_snapshot(raw))
        self.random_states.append(dict(environment=digest(raw.rng.bit_generator.state),
                                       numpy=digest(np.random.get_state()),python=digest(random.getstate())))

    def reset(self, *, seed):
        obs, info = super().reset(seed=seed)
        self.states.append(integration_state(self.env.unwrapped.env))
        self.snapshot_auxiliary()
        return obs, info

    def step(self, action):
        value = super().step(action)
        self.states.append(integration_state(self.env.unwrapped.env))
        if self.recorder.steps % 16 == 0:
            self.snapshot_auxiliary()
        return value


class AuditClient:
    def __init__(self, engine, record, png_delay=False):
        self.engine, self.record, self.png_delay = engine, record, png_delay
        self.actions, self.states = [], []

    def infer(self, states, images, instruction):
        import torch
        from .language_teacher import png_wire
        if self.png_delay and self.record.queries == 0:
            for key in images:
                png_wire(images[key][-1])
            time.sleep(2)
        data = self.engine.inputs(states,images,instruction)
        seed = 20262009+self.record.queries
        started = time.perf_counter()
        first = self.engine.normalized(data,seed=seed)
        second = self.engine.normalized(data,seed=seed)
        actions = self.engine.processor.decode_action(first.cpu(),robot_type='robocasa365')[0,:,:12].float().numpy()
        self.actions.append(actions.copy())
        self.states.append(np.array(states,copy=True))
        self.record.policy_seconds += time.perf_counter()-started
        self.record.queries += 1
        row = dict(query=self.record.queries, control_step=self.record.steps, seed=seed,
            input_observations_sha256=digest({'states':states,'images':images,'instruction':instruction}),
            camera_sha256={k:digest(v) for k,v in images.items()}, state_sha256=digest(states),
            actions_sha256=digest(actions), duplicated_inference_bitwise_equal=torch.equal(first,second),
            duplicated_inference_max_abs=float((first-second).float().abs().max()))
        with (self.record.directory/'policy_queries.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    parent = Path('/opt/astra-results/replay_parent')
    with gzip.open(parent/'anchor_model.xml.gz','rt') as f:
        xml = f.read()
    with np.load(parent/'anchor_state.npz',allow_pickle=False) as f:
        state = f['integration_state']
    rng = json.loads((parent/'anchor_rng.json').read_text())
    def tuples(v):
        return tuple(tuples(x) for x in v) if isinstance(v,list) else v
    anchor = dict(xml=xml,integration_state=state,environment_rng=rng['environment_rng'],
        numpy_rng=(rng['numpy_rng'][0],np.asarray(rng['numpy_rng'][1],np.uint32),*rng['numpy_rng'][2:]),
        python_rng=tuples(rng['python_rng']))
    expected = json.loads((parent/'anchor.json').read_text())
    sys.path.insert(0,'/opt/astra-xiaomi/sources/xiaomi/eval_robocasa365')
    import entry
    import torch
    import gymnasium as gym
    import robocasa  # noqa:F401
    from transformers import AutoModel, AutoProcessor
    from robocasa.utils.env_utils import convert_action
    from .xiaomi_interventions import NativeEngine
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained('/opt/astra-xiaomi/weights',trust_remote_code=True,use_fast=False)
    model = AutoModel.from_pretrained('/opt/astra-xiaomi/weights',trust_remote_code=True,
        attn_implementation='flash_attention_2',dtype=torch.bfloat16).cuda().to(torch.bfloat16)
    builder = entry.EvalClient.__new__(entry.EvalClient)
    builder.crop_ratio = .95
    engine = NativeEngine(model,processor,builder._build_messages)
    counts = dict(constructor_setup_resets=0,anchor_selection_resets=0,diagnostic_restore_resets=0,
                  policy_restore_resets=0,diagnostic_control_steps=0,diagnostic_chunks=0,astra_jobs=0,
                  policy_SR_denominator_contribution=0)
    rows, traces, env = [], {}, None
    task, seed = protocol['replay_task'],protocol['replay_seed']
    def save():
        write_json(args.output/'summary.json',dict(purpose='native_replay_diagnostic',counts=counts,runs=rows))
    try:
        for name, fresh, png in (('same1',True,False),('same2',False,False),('same_png',False,True),
                                  ('fresh1',True,False),('fresh2',True,False)):
            if fresh:
                if env is not None:env.close()
                np.random.seed(2)
                env = gym.make('robocasa/'+task,split='pretrain',seed=2,disable_env_checker=True)
                counts['constructor_setup_resets'] += 1
                env.reset(seed=seed)
                counts['anchor_selection_resets'] += 1
                # Match the two reset-only warmups preceding the parent trial.
                for _ in range(2):
                    restore_anchor(env,anchor)
                    counts['diagnostic_restore_resets'] += 1
            folder = args.output/name
            folder.mkdir()
            record = Recorder(folder,'diagnostic',task,128)
            wrapped = AuditEnvironment(env,record,anchor,expected,counts)
            client = AuditClient(engine,record,png)
            class GymProxy:
                def make(self,*a,**kw):return wrapped
            settings = entry.parse_args([])
            settings.seed, settings.split, settings.num_trials, settings.save_videos = seed,'pretrain',1,False
            result = entry.evaluate_task(task,0,settings,client,GymProxy(),lambda _:128,convert_action,folder,
                episode_indices=[0],show_progress=False,write_task_stats=False)
            row = record.finish(result['episodes'][0])
            row.update(name=name,fresh_constructor=fresh,png_delay=png,counts_as_policy_success_trial=False)
            rows.append(row)
            counts['diagnostic_control_steps'] += record.steps
            counts['diagnostic_chunks'] += record.queries
            data = dict(actions=np.asarray(client.actions),policy_states=np.asarray(client.states),
                        integration_states=np.asarray(wrapped.states))
            np.savez_compressed(folder/'trace.npz',**data)
            write_json(folder/'controllers.json',wrapped.controllers)
            write_json(folder/'rng_trace.json',wrapped.random_states)
            traces[name] = data
            save()
        comparisons = []
        for left,right in (('same1','same2'),('same1','same_png'),('fresh1','fresh2'),('same1','fresh1')):
            row = dict(left=left,right=right)
            for key in traces[left]:
                a,b = traces[left][key],traces[right][key]
                diff = np.max(np.abs(a-b),axis=tuple(range(1,a.ndim)))
                row[key] = dict(bitwise_equal=np.array_equal(a,b),max_abs=float(diff.max()),
                    first_difference_index=next((i for i,v in enumerate(diff) if v!=0),None))
            comparisons.append(row)
        write_json(args.output/'comparisons.json',comparisons)
        write_json(args.output/'completed.json',dict(complete=True,policy_trials=0,diagnostic_episodes=len(rows),counts=counts))
    finally:
        if env is not None:env.close()
        save()


if __name__ == '__main__':
    main()
