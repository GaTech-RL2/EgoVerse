import numpy as np
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.utils.viz_utils import draw_actions


class ACTEvalVideo(EvalVideo):
    """
    Eval class for ACT models. Computes paired/final MSE for the action key
    and draws predicted/ground-truth trajectories on the visualization image.
    """

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)
        # ground truth normalized; unnormalize for direct comparison.
        batch = algo.data_schematic.unnormalize_data(batch, algo.embodiment_id)

        metrics = {}
        mse = MeanSquaredError()
        for ac_key in algo.data_schematic.keys_of_type("action_keys"):
            if len(preds[ac_key].shape) != 3:
                raise ValueError("predictions should be (B, Seq, D)")
            metrics[f"Valid/{ac_key}_paired_mse_avg"] = mse(
                preds[ac_key].cpu(), batch[ac_key].cpu()
            )
            metrics[f"Valid/{ac_key}_final_mse_avg"] = mse(
                preds[ac_key][:, -1].cpu(), batch[ac_key][:, -1].cpu()
            )

        ims = {algo.embodiment_id: self._visualize_preds(preds, batch)}
        return metrics, ims

    def _visualize_preds(self, predictions, batch):
        algo = self.model
        ims = (
            batch[algo.data_schematic.viz_img_key()[algo.embodiment_id]]
            .cpu()
            .numpy()
            .transpose((0, 2, 3, 1))
            * 255
        ).astype(np.uint8)
        preds = predictions[algo.data_schematic.action_keys()[0]]
        gt = batch[algo.data_schematic.action_keys()[0]]

        for b in range(ims.shape[0]):
            if preds.shape[-1] == 7 or preds.shape[-1] == 14:
                ac_type = "joints"
            elif preds.shape[-1] == 3 or preds.shape[-1] == 6:
                ac_type = "xyz"
            else:
                raise ValueError(f"Unknown action type with shape {preds.shape}")

            arm = "right" if preds.shape[-1] == 7 or preds.shape[-1] == 3 else "both"
            ims[b] = draw_actions(
                ims[b],
                ac_type,
                "Purples",
                preds[b].cpu().numpy(),
                algo.camera_transforms.extrinsics,
                algo.camera_transforms.intrinsics,
                arm=arm,
            )

            ims[b] = draw_actions(
                ims[b],
                ac_type,
                "Greens",
                gt[b].cpu().numpy(),
                algo.camera_transforms.extrinsics,
                algo.camera_transforms.intrinsics,
                arm=arm,
            )

        return ims
