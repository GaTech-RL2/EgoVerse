# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=motion_diversity_multi_scene_test \
#     logger.wandb.project=everse_offline_eval_motion_diversity_multi_scene \
#     train=False \
#     eval=True


python egomimic/test_dataset.py \
    --config-name=train.yaml \
    data=motion_diversity_multi_scene_4_15 \
    logger.wandb.project=everse_offline_eval_motion_diversity_multi_scene \