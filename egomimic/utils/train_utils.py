"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""

import os
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

from egomimic.utils.dataset import PlaydataSequenceDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

from rldb.utils import RLDBDataset

import egomimic
from egomimic.algo import RolloutPolicy


def get_exp_dir(config, auto_remove_exp_dir=False, rank=0):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.

    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    time_str = f"{config.experiment.description}"

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(egomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    # if os.path.exists(base_output_dir):
    #     if not auto_remove_exp_dir:
    #         ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
    #     else:
    #         ans = "y"
    #     if ans == "y":
    #         print("REMOVING")
    #         shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    if rank == 0:
        os.makedirs(output_dir)
        os.makedirs(log_dir)
        os.makedirs(video_dir)

    return log_dir, output_dir, video_dir, time_str


def load_data_for_training(config, obs_keys, type, dataset_path=None):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (PlaydataSequenceDataset instance): train dataset object
        valid_dataset (PlaydataSequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert (
            config.experiment.validate
        ), "specified validation filter key {}, but config.experiment.validate is not set".format(
            valid_filter_by_attribute
        )

    # load the dataset into memory
    if config.experiment.validate:
        # assert (train_filter_by_attribute is not None) and (
        #     valid_filter_by_attribute is not None
        # ), (
        #     "did not specify filter keys corresponding to train and valid split in dataset"
        #     " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        # )
        # train_demo_keys = FileUtils.get_demos_for_filter_key(
        #     hdf5_path=dataset_path,
        #     filter_key=train_filter_by_attribute,
        # )
        # valid_demo_keys = FileUtils.get_demos_for_filter_key(
        #     hdf5_path=dataset_path,
        #     filter_key=valid_filter_by_attribute,
        # )
        # assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), (
        #     "training demonstrations overlap with " "validation demonstrations!"
        # )
        train_dataset = dataset_factory(
            config,
            obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            dataset_path=dataset_path,
            type=type
        )
        valid_dataset = dataset_factory(
            config,
            obs_keys,
            filter_by_attribute=valid_filter_by_attribute,
            dataset_path=dataset_path,
            type=type
        )
    else:
        train_dataset = dataset_factory(
            config,
            obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            dataset_path=dataset_path,
            type=type
        )
        valid_dataset = None

    valid_dataset.goal_obs_gap = [
        config.algo.playdata.eval_goal_gap,
        config.algo.playdata.eval_goal_gap,
    ]

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, type, filter_by_attribute=None, dataset_path=None):
    """
    Create a PlaydataSequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the PlaydataSequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (PlaydataSequenceDataset instance): dataset object
    """
    if "hdf5" in dataset_path:
        if dataset_path is None:
            dataset_path = config.train.data

        ds_kwargs = dict(
            hdf5_path=dataset_path,
            obs_keys=obs_keys,
            dataset_keys=config.train.dataset_keys,
            goal_obs_gap=config.algo.playdata.goal_image_range,
            load_next_obs=config.train.hdf5_load_next_obs,  # whether to load next observations (s') from dataset
            frame_stack=config.train.frame_stack,
            seq_length=config.train.seq_length,
            pad_frame_stack=config.train.pad_frame_stack,
            pad_seq_length=config.train.pad_seq_length,
            get_pad_mask=True,
            goal_mode=config.train.goal_mode,
            hdf5_cache_mode=config.train.hdf5_cache_mode,
            hdf5_use_swmr=config.train.hdf5_use_swmr,
            hdf5_normalize_obs=config.train.hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute,
            type=type,
            ac_key=config.train.ac_key,
            prestacked_actions=config.train.prestacked_actions,
            hdf5_normalize_actions=config.train.hdf5_normalize_actions
        )
        dataset = PlaydataSequenceDataset(**ds_kwargs)    

        return dataset
    else:
        dataset = RLDBDataset(dataset_path)
        return dataset


def run_rollout(
    policy,
    env,
    horizon,
    use_goals=False,
    render=False,
    video_writer=None,
    video_skip=5,
    terminate_on_success=False,
):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    # if use_goals:
    #     # retrieve goal from the environment
    #     goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.0
    success = {k: False for k in env.is_success()}  # success metrics

    try:
        for step_i in range(horizon):

            # get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)
                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def rollout_with_stats(
    policy,
    envs,
    horizon,
    use_goals=False,
    num_episodes=None,
    render=False,
    video_dir=None,
    video_path=None,
    epoch=None,
    video_skip=5,
    terminate_on_success=False,
    verbose=False,
):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout

    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...)
            averaged across all rollouts

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (
        video_dir is None
    ), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = {k: video_path for k in envs}
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = {k: video_writer for k in envs}
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        video_paths = {
            k: os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs
        }
        video_writers = {k: imageio.get_writer(video_paths[k], fps=20) for k in envs}

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print(
            "rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
                env.name,
                horizon,
                use_goals,
                num_episodes,
            )
        )
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print(
                    "Episode {}, horizon={}, num_success={}".format(
                        ep_i + 1, horizon, num_success
                    )
                )
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict(
            (k, [rollout_logs[i][k] for i in range(len(rollout_logs))])
            for k in rollout_logs[0]
        )
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = (
            np.sum(rollout_logs["time"]) / 60.0
        )  # total time taken for rollouts in minutes
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths
