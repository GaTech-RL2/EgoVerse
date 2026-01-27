# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=test_cotrain \
#     trainer=debug \
#     trainer.limit_val_batches=30 \
#     model=hpt_cotrain_flow_shared_head \
#     logger.wandb.project=everse_diversity_debug \

# python egomimic/trainHydra.py \
#     --config-name=train.yaml \
#     data=test_mixed \
#     trainer=debug \
#     logger.wandb.project=everse_diversity_debug \
#     ckpt_path=/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold_clothes/scenes-16-time-60_2026-01-23_22-22-22/checkpoints/last.ckpt

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=scene_diversity_cotrain/scene_diversity_cotrain_1_3_75 \
    logger.wandb.project=everse_scenes_diversity_fold_clothes_cotrain \
    name=fold-clothes-cotrain-2 \
    trainer.limit_val_batches=30 \
    model=hpt_cotrain_flow_shared_head \
    description=scenes-1-time-3_75-cotrain \
    ckpt_path=/coc/cedarp-dxu345-0/bli678/EgoVerse/logs/fold-clothes-cotrain/scenes-1-time-3_75-cotrain_2026-01-23_22-30-19/checkpoints/last.ckpt
