# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=motion_diversity/motion_diversity_multi_scene_4_15 \
#     logger.wandb.project=everse_motion_diversity_multi_scene_fold_clothes \
#     name=eval-fold-clothes-motion-diversity \
#     description=operator-4-time-15 \
#     train=false \
#     validate=true \
#     ckpt_path=/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/operator-4-time-15_2026-01-21_10-15-31/everse_motion_diversity_multi_scene_fold_clothes/fold-clothes_operator-4-time-15_2026-01-21_10-15-31/checkpoints/epoch_1999.ckpt


python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=mixed_diversity/mixed_eval \
    logger.wandb.project=everse_mixed_diversity_fold_clothes \
    name=eval-fold-clothes-mixed-diversity \
    description=4-8-7_5 \
    train=false \
    validate=true \
    ckpt_path=/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/mixed_diversity/mixed-diversity-4-8-7-5_2026-01-23_02-20-18/checkpoints/last.ckpt
