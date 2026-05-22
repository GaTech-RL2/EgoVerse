import os

import torch
import torch.nn.functional as F
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment


class EvalTokenizerVideo(Eval):
    """Validation evaluator for the OAT tokenizer.

    Each step: encodes GT actions → decodes → draws GT (green) and reconstruction
    (red) on the ego image. Also logs reconstruction MSE.
    """

    def __init__(self):
        super().__init__()
        self.trainer = None
        self.model = None
        self.val_image_buffer = {}
        self.val_counter = {}
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": 400,
            "check_val_every_n_epoch": 1,
            "profiler": "simple",
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    def on_validation_start(self):
        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        for embodiment_id, buffer in self.val_image_buffer.items():
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(embodiment_id)),
            )
            os.makedirs(out_dir, exist_ok=True)
            if len(buffer) > 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    out_dir,
                    f"validation_video_{self.val_counter[embodiment_id]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")
            self.val_counter[embodiment_id] = 0
            self.val_image_buffer[embodiment_id] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        model = self.model  # OATTokenizerTrainer

        # Run encode → decode once for all embodiments
        recons_dict = model.forward_eval(batch)

        metrics = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = model.ac_keys_by_id[embodiment_id]
            recons_key = f"{embodiment_name}_{ac_key}"

            actions = _batch[ac_key]  # normalized GT actions
            recons = recons_dict[recons_key]  # normalized reconstructions

            metrics[f"{embodiment_name}_reconst_mse"] = F.mse_loss(recons, actions)

            if model.viz_func is None or embodiment_name not in model.viz_func:
                continue

            # Unnormalize GT batch (actions + images)
            unnorm_batch = model.data_schematic.unnormalize_data(_batch, embodiment_id)

            # Unnormalize reconstructed actions
            unnorm_recons = model.data_schematic.unnormalize_data(
                {ac_key: recons}, embodiment_id
            )[ac_key]

            # viz_gt_preds expects {embodiment_name_ac_key: pred_actions}
            predictions = {recons_key: unnorm_recons}
            ims = model.viz_func[embodiment_name](predictions, unnorm_batch)
            # ims: (B, H, W, 3) numpy uint8

            if embodiment_id not in self.val_image_buffer:
                self.val_image_buffer[embodiment_id] = []
                self.val_counter[embodiment_id] = 0

            self.val_image_buffer[embodiment_id].extend(torch.from_numpy(ims))

            if len(self.val_image_buffer[embodiment_id]) >= 1000:
                out_dir = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(embodiment_id)),
                )
                os.makedirs(out_dir, exist_ok=True)
                frames = torch.stack(self.val_image_buffer[embodiment_id])
                path = os.path.join(
                    out_dir,
                    f"validation_video_{self.val_counter[embodiment_id]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")
                self.val_image_buffer[embodiment_id].clear()
                self.val_counter[embodiment_id] += 1

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
