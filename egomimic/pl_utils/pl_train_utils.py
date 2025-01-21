import json
import os
import robomimic.utils.obs_utils as ObsUtils
import torch
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

from egomimic.configs import config_factory
from egomimic.algo import algo_factory
from egomimic.utils.train_utils import get_exp_dir, load_data_for_training
import copy
from egomimic.utils.egomimicUtils import nds
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_data_utils import (
    DataModuleWrapper,
    DualDataModuleWrapper,
    get_dual_data_module,
    get_data_module,
    json_to_config,
)
import signal
import json
from datetime import timedelta
import pickle
import time
import egomimic

from collections import OrderedDict

class PreemptionHandler(Callback):
    def __init__(self):
        super().__init__()
        signal.signal(signal.SIGUSR1, self.handle_preemption)
        signal.signal(signal.SIGUSR2, self.handle_preemption)
        self.trainer_ref = None

    def setup(self, trainer, pl_module, stage):
        # Store a reference to the trainer when it's initialized
        self.trainer_ref = trainer

    def handle_preemption(self, signum, frame):
        if self.trainer_ref is not None:
            if self.trainer_ref.global_rank == 0:
                print("Preemption signal received. Saving checkpoint.")
                print("root dir: ", self.trainer_ref.default_root_dir)
                path = os.path.join(self.trainer_ref.default_root_dir, f"models/{int(time.time())}.ckpt")
                self.trainer_ref.save_checkpoint(path)

                # make symlink from last.ckpt to this checkpoint
                last_path = os.path.join(self.trainer_ref.default_root_dir, 'models/last.ckpt')
                abs_last_path = os.path.join(egomimic.__path__[0], last_path)
                abs_path = os.path.join(egomimic.__path__[0], path)
                print(f"linking {abs_path} to {abs_last_path}")
                # os.system(f"ln -sf {abs_path} {abs_last_path}")
                if os.path.exists(abs_last_path) or os.path.islink(abs_last_path):
                    os.remove(abs_last_path)
                os.symlink(abs_path, abs_last_path)

                # Optionally, terminate the process
                # os._exit(143)
                os.system(f"scontrol requeue {os.environ['SLURM_JOB_ID']}")
            else:
                print("Exiting on non rank 0 processes")

        else:
            print("Trainer reference is not set. Cannot save checkpoint.")


def init_dataset(config, dataset_path, type, alternate_valid_path=None):
    # load basic metadata from training file
    # print("\n============= Loaded Environment Metadata =============")
    # env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    # breakpoint()
    # shape_meta = FileUtils.get_shape_metadata_from_dataset(
    #     dataset_path=dataset_path,
    #     all_obs_keys=config.all_obs_keys,
    #     verbose=True,
    #     ac_key=config.train.ac_key,
    # )
    shape_meta = {'ac_dim': 7, 'all_shapes': OrderedDict([('front_img_1', [3, 480, 640]), ('joint_positions', [7]), ('right_wrist_img', [3, 480, 640])]), 'all_obs_keys': ['front_img_1', 'joint_positions', 'right_wrist_img'], 'use_images': True, 'use_depths': False}

    # if type(env_meta) is list:
    #     env_metas = env_meta
    # else:
    #     env_metas = [env_meta]
    # create environment

    # load training data
    trainset, validset = load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], dataset_path=dataset_path, type=type
    )

    if alternate_valid_path is not None:
        _, validset = load_data_for_training(
            config,
            obs_keys=shape_meta["all_obs_keys"],
            dataset_path=alternate_valid_path,
            type=None
        )
        #setting type to None here, bc this type isn't necessarily same as the main DS type, so norm stats would be incorrect

    # load training data
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    return trainset, validset, shape_meta


def eval(config, ckpt_path, type):
    resume_dir = os.path.dirname(os.path.dirname(ckpt_path))
    video_dir = os.path.join(resume_dir, "eval_videos")

    dataset_path = os.path.expanduser(config.train.data)
    ObsUtils.initialize_obs_utils_with_config(config)
    trainset, validset, shape_meta = init_dataset(config, dataset_path, type)

    train_sampler = trainset.get_dataset_sampler()
    valid_sampler = validset.get_dataset_sampler()

    datamodule = get_data_module(
        trainset, validset, train_sampler, valid_sampler, config
    )
    model = ModelWrapper.load_from_checkpoint(ckpt_path, datamodule=datamodule, config_json=json.dumps(config))
    # model=ModelWrapper(model, datamodule)
    step_log = model.custom_eval(video_dir)
    # write step_log to file

    with open(os.path.join(video_dir, "step_log.txt"), "w") as f:
        for k, v in step_log.items():
            f.write(f"{k}: {v}\n")

    print(step_log)


def train(config, ckpt_path=None):
    """
    Train a model using the algorithm.
    """
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'

    RANK = int(os.environ["SLURM_PROCID"])
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.train.seed, workers=True)

    if ckpt_path is not None:
        exp_dir = os.path.dirname(os.path.dirname(ckpt_path))

        log_dir, ckpt_dir, video_dir = (
            os.path.join(exp_dir, "logs"),
            os.path.join(exp_dir, "models"),
            os.path.join(exp_dir, "videos"),
        )
    else:
        print("\n============= New Training Run with Config =============")
        print(config)
        print("")
        log_dir, ckpt_dir, video_dir, time_str = get_exp_dir(config, rank=RANK)
        base_output_dir = os.path.join(config.train.output_dir, config.experiment.name)
        exp_dir = os.path.join(base_output_dir, time_str)

    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, "log.txt"))
    #     sys.stdout = logger
    #     sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    dataset_path_2 = (
        None if config.train.data_2 is None else os.path.expanduser(config.train.data_2)
    )
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))
    if dataset_path_2 and not os.path.exists(dataset_path_2):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path_2))

    trainset, validset, shape_meta = init_dataset(
        config, dataset_path, config.train.data_type, config.train.alternate_val,
    )
    if config.train.hdf5_normalize_obs:
        print("Normalization stats for dataset 1: ", trainset.get_obs_normalization_stats())
        with open(os.path.join(log_dir, "..", "ds1_norm_stats.pkl"), "wb") as pickle_file:
            pickle.dump(trainset.get_obs_normalization_stats(), pickle_file)


    if dataset_path_2:
        config_2 = copy.deepcopy(config)
        # TODO: currently hardcoding the obs key for the second dataset
        config_2.observation.modalities.obs.rgb = (
            config_2.observation_hand.modalities.obs.rgb
        )
        config_2.observation.modalities.obs.low_dim = (
            config_2.observation_hand.modalities.obs.low_dim
        )
        config_2.train.dataset_keys = config_2.train.dataset_keys_hand
        config_2.train.ac_key = config_2.train.ac_key_hand
        config_2.train.seq_length = config_2.train.seq_length_hand
        config_2.train.hdf5_filter_key = config_2.train.hdf5_2_filter_key
        trainset_2, validset_2, _ = init_dataset(config_2, dataset_path_2, type=config.train.data_2_type)
        if config.train.hdf5_normalize_obs:
            print("Normalization stats for dataset 2: ", trainset_2.get_obs_normalization_stats())
            with open(os.path.join(log_dir, "..", "ds2_norm_stats.pkl"), "wb") as pickle_file:
                pickle.dump(trainset_2.get_obs_normalization_stats(), pickle_file)

    # save the config as a json file
    if RANK == 0:
        with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
            json.dump(config, outfile, indent=4)

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    # if config.train.hdf5_normalize_obs:
    #     obs_normalization_stats = trainset.get_obs_normalization_stats()

    loggers = (
        []
        if config.experiment.logging.wandb_proj_name is None
        else [
            WandbLogger(
                project=config.experiment.logging.wandb_proj_name,
                sync_tensorboard=True,
                name=config.experiment.description,
                config=config,
                save_dir=log_dir,
            )
        ]
    )

    # breakpoint()
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=config.experiment.save.every_n_epochs,
            dirpath=ckpt_dir,
            save_on_train_epoch_end=True,
            filename="model_epoch_{epoch}",
            save_top_k=-1,
            save_last="link"
        ),
        PreemptionHandler()
        # ModelCheckpoint(
        #     dirpath=ckpt_dir,
        #     save_on_train_epoch_end=True,
        #     filename="model_epoch_{epoch}_{final_mse_avg:.1f}",
        #     save_top_k=3,
        #     monitor="final_mse_avg",
        #     mode="min",
        # )
    ]
    # if config.train.use_swa:
    #     callbacks.append(
    #         StochasticWeightAveraging(swa_lrs=config.algo.optim_params.policy.learning_rate.initial)
    #     )
    trainer = Trainer(
        max_epochs=config.train.num_epochs,
        limit_train_batches=config.experiment.epoch_every_n_steps,
        accelerator="gpu",
        devices=config.train.gpus_per_node,
        num_nodes=config.train.num_nodes,
        logger=loggers,
        default_root_dir=exp_dir,
        callbacks=callbacks,
        fast_dev_run=config.train.fast_dev_run,
        # val_check_interval=config.experiment.validation_epoch_every_n_steps,
        check_val_every_n_epoch=config.experiment.validation_freq,
        # gradient_clip_algorithm="norm",
        # gradient_clip_val=config.train.max_grad_norm,
        # precision=16 if config.train.amp_enabled else 32,
        precision=32,
        reload_dataloaders_every_n_epochs=0,
        use_distributed_sampler=True,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            process_group_backend="nccl",
            timeout=timedelta(minutes=60)
        ),
        # strategy="ddp_find_unused_parameters_true",
        profiler="simple",
        # profiler=AdvancedProfiler(dirpath=".", filename="perf_logs")
        # if args.profiler != "none"
        # else None,
    )
    print(f"NCCL WAIT TIME: {os.environ.get('NCCL_BLOCKING_WAIT', None)}")
    print(f"NCCL WAIT TIME: {os.environ.get('TORCH_NCCL_BLOCKING_WAIT', None)}")

    train_sampler = trainset.get_dataset_sampler()
    valid_sampler = validset.get_dataset_sampler()

    if dataset_path_2 is not None:
        datamodule = get_dual_data_module(
            trainset,
            trainset_2,
            validset,
            validset_2,
            train_sampler,
            valid_sampler,
            config,
        )
    else:
        datamodule = get_data_module(
            trainset, validset, train_sampler, valid_sampler, config
        )

    # dict is picklable, so pass that to model, then create robomimic config inside model
    dataset_path = os.path.expanduser(config.train.data)
    # shape_meta = FileUtils.get_shape_metadata_from_dataset(
    #     dataset_path=dataset_path,
    #     all_obs_keys=config.all_obs_keys,
    #     verbose=True,
    #     ac_key=config.train.ac_key,
    # )
    shape_meta = {'ac_dim': 7, 'all_shapes': OrderedDict([('front_img_1', [3, 480, 640]), ('joint_positions', [7]), ('right_wrist_img', [3, 480, 640])]), 'all_obs_keys': ['front_img_1', 'joint_positions', 'right_wrist_img'], 'use_images': True, 'use_depths': False}

    model = ModelWrapper(config.dump(), shape_meta, datamodule)
    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
