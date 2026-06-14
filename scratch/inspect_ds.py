import egomimic.utils.hydra_resolvers  # noqa
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CD = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs"
with initialize_config_dir(version_base=None, config_dir=CD):
    cfg = compose(
        config_name="train_zarr_cartesian",
        overrides=[
            "model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist",
            "data=tsimulation",
        ],
    )
print("train_datasets keys:", list(cfg.data.train_datasets.keys()))
for k in cfg.data.train_datasets:
    node = cfg.data.train_datasets[k]
    print(k, "->", list(node.keys()))
    if "resolver" in node:
        print("   resolver keys:", list(node.resolver.keys()))
    print("   _target_:", OmegaConf.select(node, "_target_"))
