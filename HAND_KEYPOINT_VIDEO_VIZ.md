# Hand-keypoint validation video visualization

This branch contains the evaluator-side visualization used for the validation
video at:

```text
/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse_launch/logs/mecka/cotrain_140d_300M_shared_wsd_val50_ckpt50_2026-08-26_18-12-28/0/videos/epoch_1149/HUMAN_BIMANUAL/validation_video_0.mp4
```

The visualization is selected with:

```text
evaluator=eval_hpt_140d_cross_viz
```

For `human_bimanual` batches, the evaluator slices `actions_140d` into the
138-D MANO keypoint layout and calls `Human.viz_gt_preds` in
`keypoints_traj_gripper_tag` mode. `EvalVideo.on_validation_end` writes the
result as `validation_video_<n>.mp4` at 30 FPS.

The visualization code does not infer keypoints from an MP4 by itself; it
renders model predictions and ground truth while the validation evaluator has
the corresponding batch data available.
