python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=eva_bc_s3 \
    model=hpt_bc_flow_eva \
    trainer=debug \
    logger=debug \
    name=debug \
    description=robot_bc \

python egomimic/trainHydra.py \
    --config-name=train.yaml \
    data=cotrain_s3 \
    model=hpt_cotrain_flow_shared_head \
    trainer=debug \
    logger=debug \
    name=debug \
    description=robot_bc \


python egomimic/trainHydra.py --config-name=train.yaml data=eva_bc_s3 model=hpt_bc_flow_eva trainer=debug logger=debug name=debug description=robot_bc 