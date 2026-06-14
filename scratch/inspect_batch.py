import sys
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")
import egomimic.utils.hydra_resolvers  # noqa
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CD = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs"
NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
KM = "egomimic.rldb.embodiment.pushshapes.get_keymap_eval"
with initialize_config_dir(version_base=None, config_dir=CD):
    cfg = compose(
        config_name="train_zarr_cartesian",
        overrides=[
            "model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist",
            "data=tsimulation",
            "norm_stats.norm_mode=minmax",
        ],
    )
OmegaConf.set_struct(cfg, False)
for split in ("train_datasets", "valid_datasets"):
    node = cfg.data[split]["pushshapes_sim"]
    node.resolver.folder_path = NC3
    node.resolver.key_map._target_ = KM
cfg.data.train_dataloader_params.pushshapes_sim.batch_size = 16
cfg.data.valid_dataloader_params.pushshapes_sim.batch_size = 16
OmegaConf.set_struct(cfg, True)

train_datasets = {dn: hydra.utils.instantiate(cfg.data.train_datasets[dn]) for dn in cfg.data.train_datasets}
valid_datasets = {dn: hydra.utils.instantiate(cfg.data.valid_datasets[dn]) for dn in cfg.data.valid_datasets}
dm = hydra.utils.instantiate(cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets)
dl = dm.train_dataloader()
b = next(iter(dl))
print("type:", type(b))
if isinstance(b, (list, tuple)):
    print("len:", len(b))
    for i, el in enumerate(b):
        print(f"  [{i}] type={type(el)}", "keys=", list(el.keys()) if hasattr(el, "keys") else "n/a")
elif hasattr(b, "keys"):
    print("keys:", list(b.keys()))
    for k in b:
        v = b[k]
        print(f"  {k}: type={type(v)}", "keys=", list(v.keys()) if hasattr(v, "keys") else "")
