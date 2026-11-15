"""JSON pipe to an isolated, exactly pinned simulator Python environment."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys


class Simulator:
    def __init__(self, python, reference):
        self.process = subprocess.Popen([str(python), '-u', '-m', __name__, '--reference', str(reference)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=dict(os.environ))
        self.metadata = self.request('metadata')

    def request(self, command, **kwargs):
        self.process.stdin.write(json.dumps(dict(command=command, **kwargs)) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Simulator worker exited unexpectedly')
            if line.startswith('BUILDER_SIM '):
                value = json.loads(line[len('BUILDER_SIM '):])
                if 'error' in value:
                    raise RuntimeError(value['error'])
                return value

    def close(self):
        if self.process.poll() is None:
            self.request('close')
            self.process.stdin.close()
            self.process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference))
    import numpy as np
    from builderbench.creative_cube_env import CreativeCubeEnv
    from builderbench.creative_cube_language_env import CreativeCubeLanguageWrapper, ENV_SYSTEM_PROMPT
    from .collect import outcome
    from ..data import fingerprint
    metadata = dict(versions={n:version(n) for n in ['mujoco', 'dm-control', 'gymnasium', 'numpy', 'scipy']},
                    system_prompt_sha256=fingerprint(ENV_SYSTEM_PROMPT))
    if args.replay:
        examples = json.loads(args.replay.read_text())
        for example in examples:
            random.seed(example['seed']); np.random.seed(example['seed'])
            env = CreativeCubeLanguageWrapper(CreativeCubeEnv(example['task']))
            try:
                observation, info = env.reset(seed=example['seed'])
                for step in example['steps']:
                    if observation != step['observation']:
                        raise ValueError('Pinned simulator replay public observation differs')
                    observation, _, _, _, info = env.step(step['action'], render=False)
                if outcome(env, info) != example['outcome']:
                    raise ValueError('Pinned simulator replay final outcome differs')
            finally:
                env.close()
        print(json.dumps(dict(passed=True, episodes=len(examples), **metadata)), flush=True)
        return
    env = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request['command']
            if command == 'metadata':
                value = metadata
            elif command == 'reset':
                if env is not None:
                    env.close()
                random.seed(request['seed']); np.random.seed(request['seed'])
                env = CreativeCubeLanguageWrapper(CreativeCubeEnv(request['task']))
                observation, info = env.reset(seed=request['seed'])
                value = dict(observation=observation, outcome=outcome(env, info), cubes=env.unwrapped._num_cubes)
            elif command == 'validate':
                action, valid = env.check_action_validity(json.dumps(request['action']))
                value = dict(valid=bool(valid) and action == request['action'])
            elif command == 'step':
                observation, reward, terminated, truncated, info = env.step(request['action'], render=False)
                value = dict(observation=observation, reward=float(reward), terminated=bool(terminated),
                    truncated=bool(truncated), simulator_steps=int(info['action_steps']), outcome=outcome(env, info))
            elif command == 'close':
                if env is not None:
                    env.close()
                print('BUILDER_SIM ' + json.dumps({'closed':True}), flush=True)
                return
            else:
                raise ValueError('Unknown simulator command')
        except Exception as error:
            value = dict(error=repr(error))
        print('BUILDER_SIM ' + json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
