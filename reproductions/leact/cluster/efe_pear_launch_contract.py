"""Fail before creating GPU replicas when a distributed launch is incomplete."""
from pathlib import Path


def validate_driver_files(config):
    env=config['ray_kwargs']['ray_init'].get('runtime_env',{}).get('env_vars',{})
    if any(not isinstance(k,str)or not isinstance(v,str)for k,v in env.items()):
        raise TypeError('Every Ray runtime environment variable must be a string')
    data=config['data']
    paths=[data['custom_cls']['path'],config['actor_rollout_ref']['rollout']['multi_turn']['tool_config_path'],
           config['actor_rollout_ref']['rollout']['agent']['agent_loop_config_path']]
    for field in ['train_files','val_files']:
        files=data[field];paths.extend([files]if isinstance(files,str)else files)
    for value in paths:
        path=Path(value)
        if not path.is_absolute():raise ValueError('Distributed worker file paths must be absolute: '+str(path))
        if not path.is_file():raise FileNotFoundError('Required distributed worker file is missing: '+str(path))
