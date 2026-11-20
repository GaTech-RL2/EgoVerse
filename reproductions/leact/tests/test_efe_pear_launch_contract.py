import copy

import pytest

from reproductions.leact.cluster.efe_pear_launch_contract import validate_driver_files


@pytest.fixture
def config(tmp_path):
    path=tmp_path/'present';path.write_text('present')
    return {'ray_kwargs':{'ray_init':{'runtime_env':{'env_vars':{'PEAR_PRIMARY_TRAINING':'0'}}}},
        'data':{'custom_cls':{'path':str(path)},'train_files':[str(path)],'val_files':[str(path)]},
        'actor_rollout_ref':{'rollout':{'multi_turn':{'tool_config_path':str(path)},
                                      'agent':{'agent_loop_config_path':str(path)}}}}


def test_valid_distributed_files_and_string_flags_pass(config):
    validate_driver_files(config)


def test_numeric_environment_flag_is_rejected_before_ray(config):
    config['ray_kwargs']['ray_init']['runtime_env']['env_vars']['PEAR_PRIMARY_TRAINING']=0
    with pytest.raises(TypeError,match='string'):validate_driver_files(config)


def test_relative_worker_module_is_rejected_before_gpu_replicas_start(config):
    config['data']['custom_cls']['path']='pear/agent_loop/dataset.py'
    with pytest.raises(ValueError,match='absolute'):validate_driver_files(config)
