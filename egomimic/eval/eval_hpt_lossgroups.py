import os

import torch
import torchvision.io as tvio
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_hpt import HPTEvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.utils.egomimicUtils import (
    frechet_gaussian_over_time,
    reverse_kl_from_samples,
)


class HPTEvalVideoLossGroups(HPTEvalVideo):
    """
    HPT evaluator that breaks metrics down by loss group (e.g. task / dataset split).

    Per-group metrics are logged under Valid/<group>_<key>_* and macro-averaged
    totals under Valid/<embodiment>_<key>_*. Videos are written into per-group
    subdirectories via tuple (group_name, embodiment_id) image keys.

    Point at this class via the evaluator config:
        _target_: egomimic.eval.eval_hpt_lossgroups.HPTEvalVideoLossGroups
    """

    def _key_subdir(self, key):
        if isinstance(key, tuple):
            group_name, embodiment_id = key
            return os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                group_name,
                str(get_embodiment(embodiment_id)),
            )
        return os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(key)),
        )

    def on_validation_end(self):
        for key, buffer in self.val_image_buffer.items():
            subdir = self._key_subdir(key)
            os.makedirs(subdir, exist_ok=True)
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                tvio.write_video(
                    os.path.join(
                        subdir, f"validation_video_{self.val_counter[key]}.mp4"
                    ),
                    frames,
                    fps=30,
                    video_codec="h264",
                )
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics, images_dict = self.compute_metrics_and_viz(batch)
        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        for key, images in images_dict.items():
            subdir = self._key_subdir(key)
            os.makedirs(subdir, exist_ok=True)
            if key not in self.val_image_buffer or self.val_image_buffer[key] is None:
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self.val_image_buffer[key].extend(torch.from_numpy(images))
            if len(self.val_image_buffer[key]) >= 1000:
                frames = torch.stack(self.val_image_buffer[key])
                tvio.write_video(
                    os.path.join(
                        subdir, f"validation_video_{self.val_counter[key]}.mp4"
                    ),
                    frames,
                    fps=30,
                    video_codec="h264",
                )
                self.val_image_buffer[key].clear()
                self.val_counter[key] += 1
        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()

        for embodiment_id, _batch in batch.items():
            _batch = algo.data_schematic.unnormalize_data(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]

            batch_size = _batch[ac_key].shape[0]
            loss_group_labels = algo._loss_group_labels(
                _batch.get("loss_group") if algo.loss_group_mode else None, batch_size
            )
            unique_groups = list(dict.fromkeys(loss_group_labels))
            indices_by_group = {
                g: [i for i, lbl in enumerate(loss_group_labels) if lbl == g]
                for g in unique_groups
            }

            if f"{embodiment_name}_{ac_key}" in preds and ac_key != algo.shared_ac_key:
                group_paired_mses, group_final_mses = [], []
                group_fd_avgs, group_fd_mins, group_fd_maxs = [], [], []
                for group_name, indices in indices_by_group.items():
                    safe_group = str(group_name).replace("/", "_")
                    pred_g = preds[f"{embodiment_name}_{ac_key}"][indices]
                    gt_g = _batch[ac_key][indices]
                    paired_mse = mse(pred_g.cpu(), gt_g.cpu())
                    final_mse = mse(pred_g[:, -1].cpu(), gt_g[:, -1].cpu())
                    fd = frechet_gaussian_over_time(pred_g, gt_g)
                    if algo.loss_group_mode:
                        metrics[f"Valid/{safe_group}_{ac_key}_paired_mse_avg"] = (
                            paired_mse
                        )
                        metrics[f"Valid/{safe_group}_{ac_key}_final_mse_avg"] = (
                            final_mse
                        )
                        metrics[f"Valid/{safe_group}_{ac_key}_frechet_gauss_avg"] = (
                            fd.mean().item()
                        )
                        metrics[f"Valid/{safe_group}_{ac_key}_frechet_gauss_min"] = (
                            fd.min().item()
                        )
                        metrics[f"Valid/{safe_group}_{ac_key}_frechet_gauss_max"] = (
                            fd.max().item()
                        )
                    group_paired_mses.append(float(paired_mse))
                    group_final_mses.append(float(final_mse))
                    group_fd_avgs.append(fd.mean().item())
                    group_fd_mins.append(fd.min().item())
                    group_fd_maxs.append(fd.max().item())
                n = len(unique_groups)
                metrics[f"Valid/{embodiment_name}_{ac_key}_paired_mse_avg"] = (
                    sum(group_paired_mses) / n
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_final_mse_avg"] = (
                    sum(group_final_mses) / n
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_avg"] = (
                    sum(group_fd_avgs) / n
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_min"] = (
                    sum(group_fd_mins) / n
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_max"] = (
                    sum(group_fd_maxs) / n
                )

            if embodiment_name in algo.auxiliary_ac_keys:
                for aux_key in algo.auxiliary_ac_keys[embodiment_name]:
                    pred_key = f"{embodiment_name}_{aux_key}"
                    if pred_key in preds:
                        group_paired_mses, group_final_mses = [], []
                        group_fd_avgs, group_fd_mins, group_fd_maxs = [], [], []
                        for group_name, indices in indices_by_group.items():
                            safe_group = str(group_name).replace("/", "_")
                            pred_g = preds[pred_key][indices]
                            gt_g = _batch[aux_key][indices]
                            paired_mse = mse(pred_g.cpu(), gt_g.cpu())
                            final_mse = mse(pred_g[:, -1].cpu(), gt_g[:, -1].cpu())
                            fd = frechet_gaussian_over_time(pred_g, gt_g)
                            if algo.loss_group_mode:
                                metrics[
                                    f"Valid/{safe_group}_{aux_key}_paired_mse_avg"
                                ] = paired_mse
                                metrics[
                                    f"Valid/{safe_group}_{aux_key}_final_mse_avg"
                                ] = final_mse
                                metrics[
                                    f"Valid/{safe_group}_{aux_key}_frechet_gauss_avg"
                                ] = fd.mean().item()
                                metrics[
                                    f"Valid/{safe_group}_{aux_key}_frechet_gauss_min"
                                ] = fd.min().item()
                                metrics[
                                    f"Valid/{safe_group}_{aux_key}_frechet_gauss_max"
                                ] = fd.max().item()
                            group_paired_mses.append(float(paired_mse))
                            group_final_mses.append(float(final_mse))
                            group_fd_avgs.append(fd.mean().item())
                            group_fd_mins.append(fd.min().item())
                            group_fd_maxs.append(fd.max().item())
                        n = len(unique_groups)
                        metrics[f"Valid/{pred_key}_paired_mse_avg"] = (
                            sum(group_paired_mses) / n
                        )
                        metrics[f"Valid/{pred_key}_final_mse_avg"] = (
                            sum(group_final_mses) / n
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = (
                            sum(group_fd_avgs) / n
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_min"] = (
                            sum(group_fd_mins) / n
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_max"] = (
                            sum(group_fd_maxs) / n
                        )

            if (
                algo.shared_ac_key
                and f"{embodiment_name}_{algo.shared_ac_key}" in preds
            ):
                pred_key = f"{embodiment_name}_{algo.shared_ac_key}"
                group_paired_mses, group_final_mses = [], []
                group_fd_avgs, group_fd_mins, group_fd_maxs = [], [], []
                for group_name, indices in indices_by_group.items():
                    safe_group = str(group_name).replace("/", "_")
                    pred_g = preds[pred_key][indices]
                    gt_g = _batch[algo.shared_ac_key][indices]
                    paired_mse = mse(pred_g.cpu(), gt_g.cpu())
                    final_mse = mse(pred_g[:, -1].cpu(), gt_g[:, -1].cpu())
                    fd = frechet_gaussian_over_time(pred_g, gt_g)
                    if algo.loss_group_mode:
                        metrics[
                            f"Valid/{safe_group}_{algo.shared_ac_key}_paired_mse_avg"
                        ] = paired_mse
                        metrics[
                            f"Valid/{safe_group}_{algo.shared_ac_key}_final_mse_avg"
                        ] = final_mse
                        metrics[
                            f"Valid/{safe_group}_{algo.shared_ac_key}_frechet_gauss_avg"
                        ] = fd.mean().item()
                        metrics[
                            f"Valid/{safe_group}_{algo.shared_ac_key}_frechet_gauss_min"
                        ] = fd.min().item()
                        metrics[
                            f"Valid/{safe_group}_{algo.shared_ac_key}_frechet_gauss_max"
                        ] = fd.max().item()
                    group_paired_mses.append(float(paired_mse))
                    group_final_mses.append(float(final_mse))
                    group_fd_avgs.append(fd.mean().item())
                    group_fd_mins.append(fd.min().item())
                    group_fd_maxs.append(fd.max().item())
                n = len(unique_groups)
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = sum(group_paired_mses) / n
                metrics[f"Valid/{pred_key}_final_mse_avg"] = sum(group_final_mses) / n
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = sum(group_fd_avgs) / n
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = sum(group_fd_mins) / n
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = sum(group_fd_maxs) / n

            if algo.rkl_samples and algo.rkl_samples > 1:
                hpt_batch = {
                    "domain": embodiment_name,
                    "data": algo._robomimic_to_hpt_data(
                        batch[embodiment_id],
                        algo.camera_keys[embodiment_id],
                        algo.proprio_keys[embodiment_id],
                        algo.lang_keys[embodiment_id],
                        ac_key,
                        algo.auxiliary_ac_keys.get(embodiment_name, []),
                    ),
                }
                rkl_targets = []
                if (
                    f"{embodiment_name}_{ac_key}" in preds
                    and ac_key != algo.shared_ac_key
                ):
                    rkl_targets.append(
                        (
                            f"{embodiment_name}_{ac_key}",
                            _batch[ac_key].to(algo.device),
                            embodiment_name,
                        )
                    )
                if embodiment_name in algo.auxiliary_ac_keys:
                    for aux_key in algo.auxiliary_ac_keys[embodiment_name]:
                        aux_pred_key = f"{embodiment_name}_{aux_key}"
                        if aux_pred_key in preds:
                            rkl_targets.append(
                                (aux_pred_key, _batch[aux_key].to(algo.device), aux_key)
                            )
                if algo.shared_ac_key:
                    shared_pred_key = f"{embodiment_name}_{algo.shared_ac_key}"
                    if shared_pred_key in preds:
                        rkl_targets.append(
                            (
                                shared_pred_key,
                                _batch[algo.shared_ac_key].to(algo.device),
                                "shared",
                            )
                        )
                M = int(algo.rkl_samples)
                for pred_key_name, gt_tensor, head_key in rkl_targets:
                    samples = self._collect_policy_samples(
                        hpt_batch, ref=gt_tensor, key_name=head_key, M=M
                    )
                    rkl = reverse_kl_from_samples(samples, gt_tensor)
                    metrics[f"Valid/{pred_key_name}_reverse_kl_M{M}"] = rkl.item()

            ims = self._visualize_preds(preds, _batch)
            if algo.loss_group_mode:
                for group_name, indices in indices_by_group.items():
                    safe_group = str(group_name).replace("/", "_")
                    images_dict[(safe_group, embodiment_id)] = ims[indices]
            else:
                images_dict[embodiment_id] = ims

        return metrics, images_dict
