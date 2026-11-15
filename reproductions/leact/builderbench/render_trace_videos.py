"""Render validated simulator replays of saved actions; never call a model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def render(reference, episode, output):
    sys.path.insert(0,str(reference.resolve()))
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from builderbench.creative_cube_env import CreativeCubeEnv
    from builderbench.creative_cube_language_env import CreativeCubeLanguageWrapper
    result=json.loads((episode/'result.json').read_text())
    steps=[json.loads(p.read_text()) for p in sorted((episode/'steps').glob('*.json'))]
    random.seed(result['seed']);np.random.seed(result['seed'])
    env=CreativeCubeLanguageWrapper(CreativeCubeEnv(result['task'],width=512,height=384))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+'.rendering.mp4')
    stride=4;fps=1/(env.unwrapped._control_timestep*stride)
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','512x384','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','27','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    timings=[];frames=0
    try:
        try: font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',12)
        except OSError: font=ImageFont.load_default()
        def emit(frame,index,sim_time):
            nonlocal frames
            image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,512,42),fill='#172436')
            draw.text((10,5),result['task']+' | '+output.stem,font=font,fill='white')
            action_label=f'Replayed action {index+1}' if index>=0 else 'No action executed'
            draw.text((10,22),f'{action_label} | simulation {sim_time:.2f}s',font=font,fill='#d5e4e8')
            encoder.stdin.write(image.tobytes());frames+=1
        observation,info=env.reset(seed=result['seed'])
        initial=env.unwrapped.render()
        for index,step in enumerate(steps):
            assert observation==step['observation'], f'Observation before action {index} differs'
            start=frames/fps
            observation,reward,terminated,truncated,info=env.step(step['action'],render=True,video_stride=stride)
            assert observation==step['next_observation'], f'Observation after action {index} differs'
            flags=[bool(info[f'privileged/block_{i}_success'][0]) for i in range(env.unwrapped._num_cubes) if env.unwrapped._goal_masks[i]]
            effector=bool(info['privileged/effector_success'][0])
            current=dict(cube_progress=sum(flags)/len(flags),per_target_success=flags,effector_success=effector,success=all(flags) and effector)
            assert current==step['outcome'], f'Outcome after action {index} differs'
            for frame in info['observation_history']:
                if frame['rendered_frame'] is not None:emit(frame['rendered_frame'],index,float(frame['time']))
            timings.append(dict(index=index,start=start,end=frames/fps,source_observation_sha256=hashlib.sha256(step['observation'].encode()).hexdigest()))
        if not frames:
            call=json.loads(next((episode/'calls').glob('*.json')).read_text())
            expected=call['request']['prompt'].rsplit('Current observation:\n',1)[1].split('\n\nChoose the next action',1)[0]
            assert observation==expected
            flags=[bool(info[f'privileged/block_{i}_success'][0]) for i in range(env.unwrapped._num_cubes) if env.unwrapped._goal_masks[i]]
            assert sum(flags)/len(flags)==result['cube_progress']
            for _ in range(round(fps)):emit(initial,-1,float(info['time'][0]))
        if steps:
            assert current['success']==result['success'] and current['cube_progress']==result['cube_progress']
        encoder.stdin.close();assert encoder.wait(timeout=30)==0
        temporary.replace(output)
        record=dict(created_utc=datetime.now(timezone.utc).isoformat(),path=str(output),task=result['task'],seed=result['seed'],
            source=str(episode),result_sha256=digest(episode/'result.json'),video_sha256=digest(output),
            bytes=output.stat().st_size,fps=fps,frames=frames,seconds=frames/fps,actions=timings,
            replay_verified=True,checked='Exact public observations and outcome after every executed action; final result matches',
            type='Post-hoc simulator replay of logged actions, not a recording of live inference',
            timing='Approximately real-time simulator motion; model inference latency omitted',
            versions={n:version(n) for n in ['mujoco','dm-control','gymnasium','numpy','scipy']})
        output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ['path','seconds','actions','bytes','replay_verified']}),flush=True)
    finally:
        if encoder.poll() is None:encoder.kill();encoder.wait()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.reference,a.episode,a.output)
