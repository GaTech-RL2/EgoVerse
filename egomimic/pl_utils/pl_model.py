import os
from collections import OrderedDict

import numpy as np
import psutil
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch
from lightning import LightningModule
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo.algo import PolicyAlgo
import egomimic.utils.val_utils as ValUtils
from egomimic.utils.egomimicUtils import nds
from egomimic.pl_utils.pl_data_utils import DualDataModuleWrapper, json_to_config
from egomimic.algo import algo_factory
import robomimic.utils.file_utils as FileUtils
from egomimic.configs import config_factory
import json
from typing import Any, Dict, Tuple
import torchvision.io as tvio
from lightning.pytorch.utilities import rank_zero_only

class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    def __init__(self, robomimic_model, optimizer, scheduler, train_cfg, datamodule=None):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])

        self.model = robomimic_model
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except:
            pass
        self.step_log_all_train = []
        self.step_log_all_valid = []

        self.datamodule = datamodule
        self.dual_dl = isinstance(datamodule, DualDataModuleWrapper)

        self.train_cfg = train_cfg

        self.val_image_buffer, self.val_counter = [], 0
        # TODO __init__ should take the config, and init the model here.  Then save_hyperparameters will just save the config rather than the model
    
    def root_dir(self):
        return self.trainer.default_root_dir

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    def training_step(self, batch, batch_idx):
        print("INSIDE ON TRAIN STEP")

        if self.datamodule is None:
            raise ValueError("Cannot train if datamodule is none, make sure to pass hydra.utils.instantiate(cfg.model, data_module=datamodule)")
        DUAL_DL = isinstance(batch, list)
        # plt.imsave("debug/front_img_1.png", batch[0]["obs"]["front_img_1"][0, 0].cpu().numpy())

        # full_batch = batch
        # batch = full_batch[0]
        self.train()
        loss_dicts = []
        if not DUAL_DL:
            batch = [batch]
        ac_keys = (
            [
                self.model.ac_key,
                self.model.ac_key_hand,
            ]
            if self.dual_dl
            else [self.model.ac_key]
        )
        norm_dicts = (
            [
                self.datamodule.train_dataset1.get_obs_normalization_stats(),
                self.datamodule.train_dataset2.get_obs_normalization_stats()
            ]
            if self.dual_dl
            else [self.datamodule.dataset1.get_obs_normalization_stats()]
        )
        for batch, ac_key, norm_dict in zip(batch, ac_keys, norm_dicts):
            batch = self.model.process_batch_for_training(batch)
            # batch = self.model.postprocess_batch_for_training(batch, norm_dict, normalize_actions=self.train_cfg.normalize_actions)
            predictions = self.model.forward_training(batch)
            losses = self.model._compute_losses(predictions, batch)
            loss_dicts.append(losses)

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        info = {}
        info["losses"] = TensorUtils.detach(losses)
        self.step_log_all_train.append(self.model.log_info(info))

        # count=0
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         count += 1
        #         # print(name)
        # print("Unused params: ", count)
        # print(self.global_step, self.model.global_config.experiment.epoch_every_n_steps)
        # breakpoint()
        return losses["action_loss"]

    @rank_zero_only
    def on_validation_start(self):
        print("INSIDE ON VALIDATION START")
        # make the video directory for this epoch
        os.makedirs(os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"), exist_ok=True)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        """
        Run a validation step on the batch, and save that batch of images into the val_image_buffer.  Once the buffer hits 1000 images, save that as a 30fps video using torchvision.io.write_video.  
        """
        print("INSIDE VAL STEP")

        batch = self.model.process_batch_for_training(batch)
        metrics, images = self.model.forward_eval_logging(batch)
        
        self.val_image_buffer.extend(torch.from_numpy(images))
        if len(self.val_image_buffer) >= 1000:
            frames = torch.stack(self.val_image_buffer)
            path = os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}/validation_video_{self.val_counter}.mp4")
            tvio.write_video(path, frames, fps=30)
            self.val_image_buffer.clear()
        
            self.val_counter += 1
        
        print("Validation Step: ", metrics)
        
        self.log_dict(metrics)
    
    @rank_zero_only
    def on_validation_end(self):
        if len(self.val_image_buffer) != 0:
            frames = torch.stack(self.val_image_buffer)
            path = os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}/validation_video_{self.val_counter}.mp4")
            tvio.write_video(path, frames, fps=30)
        
        self.val_counter = 0
        self.val_image_buffer = []
        

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": "val/loss",
                #     "interval": "epoch",
                #     "frequency": 1,
                # },
            }
        return {"optimizer": optimizer}

    def custom_eval(self, video_dir):
        self.eval()
        self.zero_grad()
        with torch.no_grad():
            valid_step_log = ValUtils.evaluate_high_level_policy(
                self.model,
                self.datamodule.val_dataloader_1(),
                video_dir,
                ac_key=self.model.ac_key,
            )  # save vid only once every video_freq epochs

        return valid_step_log

    def on_train_epoch_start(self):
        # flatten and take the mean of the metrics
        log = {}
        for i in range(len(self.step_log_all_train)):
            for k in self.step_log_all_train[i]:
                if k not in log:
                    log[k] = []
                log[k].append(self.step_log_all_train[i][k])
        log_all = dict((k, float(np.mean(v))) for k, v in log.items())
        for i, param_group in enumerate(self.optimizers().param_groups):
            log_all[f"Optimizer/param_group_{i}_lr"] = param_group["lr"]
        for k, v in log_all.items():
            self.log("Train/" + k, v, sync_dist=True)
        self.step_log_all_train = []

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / int(1e9)
        print("\nEpoch {} Memory Usage: {} GB\n".format(self.current_epoch, mem_usage))
        # self.log('epoch', self.trainer.current_epoch)
        self.log("System/RAM Usage (GB)", mem_usage, sync_dist=True)

        return super().on_train_epoch_start()
