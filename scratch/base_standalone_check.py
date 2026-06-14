from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

cd = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs"
for b in ["_tsim_packed_base", "_tsim_perframe_base", "_pickplace_qwen_base"]:
    with initialize_config_dir(version_base=None, config_dir=cd):
        try:
            cfg = compose(config_name="train_zarr_cartesian", overrides=[f"data={b}"])
            OmegaConf.to_container(cfg.data, resolve=True)
            print(b, "STANDALONE_COMPOSE_OK")
        except Exception as e:
            print(b, "standalone-fails:", repr(e).splitlines()[0][:90])
