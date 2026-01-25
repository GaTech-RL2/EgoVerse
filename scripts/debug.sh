python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=test_cotrain \
    trainer=debug \
    trainer.limit_val_batches=30 \
    model=hpt_cotrain_flow_shared_head \
    logger.wandb.project=everse_diversity_debug \


# python egomimic/trainHydra.py --config-name=train.yaml data=test_mixed trainer=debug 
python egomimic/trainHydra.py --config-name=train.yaml data=test_cotrain trainer=debug trainer.limit_val_batches=30 model=hpt_cotrain_flow_shared_head logger.wandb.project=everse_diversity_debug 