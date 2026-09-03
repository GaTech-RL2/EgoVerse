"""PIEvalVideo plus a palm-origin cam-frame metric for KEYPOINT-action pi runs.

The keypoint runs are scored in cam frame over all 21 MANO keypoints per hand
(``..._actions_keypoints_cam_paired_mse_avg``); the cartesian runs are scored on
the palm-origin position only (``..._actions_cartesian_cam_xyz_paired_mse_avg``).
To put the two on the same target, this evaluator reverts the predicted (and
GT) keypoint chunks to head/cam frame exactly as PIEvalVideo does, then derives
the palm origin from the 21 reverted keypoints with the SAME definition the
mecka conversion used to build ``ee_pose``
(``mecka_to_zarr.compute_hand_pose_xyzquat``: centroid of the wrist and the four
MCPs, MANO indices 0, 5, 9, 13, 17) and reports its paired / final MSE in m² per
coordinate — directly comparable to the cartesian runs' ``cam_xyz`` metric.

Sanity: ``..._palmdef_check_step0_dist_cm`` is the distance between the GT
palm centroid at chunk step 0 and the head-frame proprio ``ee_pose`` position in
the same batch; near zero confirms the palm definition matches the data.

The palm trajectories (pred / gt / proprio ee) are dumped to
``<save_dir>/palm_cam_trajectories.npz`` for offline analysis.
"""

import copy
import os

import numpy as np
import torch

from egomimic.eval.eval_pi import PIEvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment

# mecka_to_zarr.compute_hand_pose_xyzquat: wrist + index/middle/ring/pinky MCPs
MECKA_PALM_IDX = (0, 5, 9, 13, 17)
# aria_utils.MANO_PALM_IDX: pinky excluded
ARIA_PALM_IDX = (0, 5, 9, 13)


class PIEvalKeypointPalm(PIEvalVideo):
    def __init__(self, *args, palm_idx=MECKA_PALM_IDX, save_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.palm_idx = [int(i) for i in palm_idx]
        self.save_dir = save_dir
        self._dump = {"pred": [], "gt": [], "ee": []}

    def compute_metrics_and_viz(self, batch, do_viz=True):
        metrics, images = super().compute_metrics_and_viz(batch, do_viz=do_viz)
        algo = self.model
        preds = algo.forward_eval(batch)
        for embodiment_id, _batch in batch.items():
            _b = algo.norm_stats.unnormalize(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{name}_{ac_key}"
            tl = self.transform_lists.get(name)
            if tl is None or pred_key not in preds:
                continue
            pb = copy.deepcopy(_b)
            pb[ac_key] = preds[pred_key]
            gt = torch.as_tensor(Embodiment.apply_transform(_b, tl)[ac_key]).float().cpu()
            pr = torch.as_tensor(Embodiment.apply_transform(pb, tl)[ac_key]).float().cpu()
            B, T, D = gt.shape
            if D != 126:  # not a keypoint action -> nothing to derive
                continue
            gk = gt.view(B, T, 2, 21, 3)
            pk = pr.view(B, T, 2, 21, 3)
            gp = gk[:, :, :, self.palm_idx].mean(3)  # (B, T, 2 hands, xyz) head frame, metres
            pp = pk[:, :, :, self.palm_idx].mean(3)
            err = (pp - gp) ** 2
            metrics[f"Valid/{pred_key}_palm_cam_paired_mse_avg"] = err.mean()
            metrics[f"Valid/{pred_key}_palm_cam_final_mse_avg"] = err[:, -1].mean()
            metrics[f"Valid/{pred_key}_palm_cam_rmse_cm"] = err.mean().sqrt() * 100.0
            for h, hn in enumerate(("left", "right")):
                metrics[f"Valid/{pred_key}_palm_cam_paired_mse_{hn}"] = err[:, :, h].mean()
            # wrist-joint-only variant (MANO 0) for reference
            werr = (pk[:, :, :, 0] - gk[:, :, :, 0]) ** 2
            metrics[f"Valid/{pred_key}_wristjoint_cam_paired_mse_avg"] = werr.mean()
            ee = _b.get("observations.state.ee_pose")
            if ee is not None:
                ee = torch.as_tensor(ee).float().cpu()
                # padded 20-D layout per arm: xyz(3) rot6d(6) grip(1) -> xyz at 0:3 and 10:13
                ee_xyz = torch.stack([ee[:, 0:3], ee[:, 10:13]], dim=1)  # (B, 2, 3)
                dist = (gp[:, 0] - ee_xyz).norm(dim=-1)  # (B, 2) metres
                metrics[f"Valid/{pred_key}_palmdef_check_step0_dist_cm"] = dist.mean() * 100.0
                self._dump["ee"].append(ee_xyz.numpy())
            self._dump["pred"].append(pp.numpy())
            self._dump["gt"].append(gp.numpy())
        return metrics, images

    def on_validation_end(self):
        super().on_validation_end()
        if self.save_dir and self._dump["pred"]:
            os.makedirs(self.save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.save_dir, "palm_cam_trajectories.npz"),
                pred=np.concatenate(self._dump["pred"]),
                gt=np.concatenate(self._dump["gt"]),
                ee=(np.concatenate(self._dump["ee"]) if self._dump["ee"] else np.zeros(0)),
                palm_idx=np.array(self.palm_idx),
            )
            print(f"[palm-eval] saved {self.save_dir}/palm_cam_trajectories.npz", flush=True)
