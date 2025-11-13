import os
from collections import OrderedDict

import numpy as np
import psutil
import egomimic.utils.tensor_utils as TensorUtils
import torch
from lightning import LightningModule
from egomimic.utils.egomimicUtils import nds
from egomimic.pl_utils.pl_data_utils import DualDataModuleWrapper, RLDBModule
from typing import Any, Dict
import torchvision.io as tvio
from lightning.pytorch.utilities import rank_zero_only
from egomimic.rldb.utils import get_embodiment


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    def __init__(self, robomimic_model, optimizer, scheduler):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters()

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

        self.val_image_buffer, self.val_counter = {}, {}
        # TODO __init__ should take the config, and init the model here.  Then save_hyperparameters will just save the config rather than the model

    def root_dir(self):
        return self.trainer.default_root_dir

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        self.train()
        loss_dicts = []
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        losses = self.model.compute_losses(predictions, batch)
        loss_dicts.append(losses)

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        info = {}
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        info["policy_grad_norms"] = grad_norm.item()
        info["losses"] = TensorUtils.detach(losses)
        self.step_log_all_train.append(self.model.log_info(info))

        return losses["action_loss"]

    @rank_zero_only
    def on_validation_start(self):
        # make the video directory for this epoch
        os.makedirs(
            os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
            exist_ok=True,
        )

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Run a validation step on the batch, and save that batch of images into the val_image_buffer.  Once the buffer hits 1000 images, save that as a 30fps video using torchvision.io.write_video.
        """
        batch = self.model.process_batch_for_training(batch)
        metrics, images_dict = self.model.forward_eval_logging(batch)

        ## images is now a dict
        for key, images in images_dict.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if key not in self.val_image_buffer or self.val_image_buffer[key] is None:
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self.val_image_buffer[key].extend(torch.from_numpy(images))
            if len(self.val_image_buffer[key]) >= 1000:
                frames = torch.stack(self.val_image_buffer[key])
                path = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")
                self.val_image_buffer[key].clear()
                self.val_counter[key] += 1

        self.log_dict(metrics)

    @rank_zero_only
    def on_validation_end(self):
        for key, buffer in self.val_image_buffer.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")

            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

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
                "lr_scheduler": scheduler,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": "val/loss",
                #     "interval": "epoch",
                #     "frequency": 1,
                # },
            }
        return {"optimizer": optimizer}

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
