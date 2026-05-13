import copy

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.egomimicUtils import (
    frechet_gaussian_over_time,
    reverse_kl_from_samples,
)


class HPTEvalVideo(EvalVideo):
    """
    Eval class for HPT models. Per embodiment, computes:
      - val loss (BC loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE + Frechet over time for the main / shared / auxiliary heads
      - paired/final MSE in cam frame on the main ``ac_key``, when a
        ``transform_lists`` entry is configured
      - optional Reverse KL from samples
    The revert transform is applied once and reused by both the cam-frame MSE
    and the viz video.
    """

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss_embodiments = 0
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]

            loss_key = f"{embodiment_name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_val)
                total_loss = total_loss + loss_val
                n_loss_embodiments += 1

            if f"{embodiment_name}_{ac_key}" in preds and ac_key != algo.shared_ac_key:
                metrics[f"Valid/{embodiment_name}_{ac_key}_paired_mse_avg"] = mse(
                    preds[f"{embodiment_name}_{ac_key}"].cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_final_mse_avg"] = mse(
                    preds[f"{embodiment_name}_{ac_key}"][:, -1].cpu(),
                    _batch[ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[f"{embodiment_name}_{ac_key}"], _batch[ac_key]
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_avg"] = (
                    fd.mean().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_min"] = (
                    fd.min().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_max"] = (
                    fd.max().item()
                )

            if embodiment_name in algo.auxiliary_ac_keys:
                for aux_key in algo.auxiliary_ac_keys[embodiment_name]:
                    pred_key = f"{embodiment_name}_{aux_key}"
                    if pred_key in preds:
                        metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                            preds[pred_key].cpu(), _batch[aux_key].cpu()
                        )
                        metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                            preds[pred_key][:, -1].cpu(), _batch[aux_key][:, -1].cpu()
                        )
                        fd = frechet_gaussian_over_time(
                            preds[pred_key], _batch[aux_key]
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = (
                            fd.mean().item()
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                        metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

            if (
                algo.shared_ac_key
                and f"{embodiment_name}_{algo.shared_ac_key}" in preds
            ):
                pred_key = f"{embodiment_name}_{algo.shared_ac_key}"
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[algo.shared_ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(),
                    _batch[algo.shared_ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[pred_key], _batch[algo.shared_ac_key]
                )
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

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
                                (
                                    aux_pred_key,
                                    _batch[aux_key].to(algo.device),
                                    aux_key,
                                )
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

            transform_list = self.transform_lists.get(embodiment_name)
            main_pred_key = f"{embodiment_name}_{ac_key}"
            gt_batch_viz = _batch
            preds_for_viz = preds
            if transform_list is not None and main_pred_key in preds:
                pred_batch = copy.deepcopy(_batch)
                pred_batch[ac_key] = preds[main_pred_key]
                gt_t = Embodiment.apply_transform(_batch, transform_list)
                pred_t = Embodiment.apply_transform(pred_batch, transform_list)
                # apply_transform drops keys whose shape[0] != batch_size
                # (e.g. ``embodiment``, ``annotations``). Merge to preserve them.
                gt_batch_viz = {**_batch, **gt_t}
                pred_batch_viz = {**_batch, **pred_t}

                # ``.contiguous()`` because ``apply_transform`` returns CPU tensors,
                # so ``.cpu()`` here is a no-op and ``[:, -1]`` leaves a non-contiguous
                # view that torchmetrics' MSE doesn't accept.
                metrics[f"Valid/{main_pred_key}_cam_paired_mse_avg"] = mse(
                    pred_batch_viz[ac_key].cpu().contiguous(),
                    gt_batch_viz[ac_key].cpu().contiguous(),
                )
                metrics[f"Valid/{main_pred_key}_cam_final_mse_avg"] = mse(
                    pred_batch_viz[ac_key][:, -1].cpu().contiguous(),
                    gt_batch_viz[ac_key][:, -1].cpu().contiguous(),
                )

                preds_for_viz = dict(preds)
                preds_for_viz[main_pred_key] = pred_batch_viz[ac_key]

            ims = self._visualize_preds(preds_for_viz, gt_batch_viz)
            images_dict[embodiment_id] = ims

        if total_loss is not None and n_loss_embodiments > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss_embodiments

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        return self.viz_func[embodiment_name](predictions, batch)

    @torch.no_grad()
    def _collect_policy_samples(self, hpt_batch, ref, key_name, M):
        """Collect policy samples for Reverse KL."""
        algo = self.model
        B, T, D = ref.shape
        samples = []
        was_training = algo.nets.training
        algo.nets.eval()
        for _ in range(M):
            out = algo.nets["policy"].forward(
                hpt_batch["domain"], algo._clone_batch(hpt_batch["data"])
            )
            if key_name in out:
                pred = out[key_name]
            else:
                pred = out[hpt_batch["domain"]]

            pred = pred[:, :T, :D]
            samples.append(pred.unsqueeze(0))
        if was_training:
            algo.nets.train()
        return torch.cat(samples, dim=0)
