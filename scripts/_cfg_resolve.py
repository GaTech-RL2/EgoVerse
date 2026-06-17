"""Resolve a single Hydra group config to its merged YAML (structure only).

Used to verify config-consolidation refactors preserve the resolved config.
Composes the real primary config (train_zarr_cartesian) with one group
override and dumps that group's subtree with resolve=False so interpolations
stay literal (identical before/after a structural refactor).

Usage: python scripts/_cfg_resolve.py <group> <name>
  e.g. python scripts/_cfg_resolve.py model dfot_pushshapes_pixel
"""
import os
import socket
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# eval resolver is registered by trainHydra; register here too so compose of
# any default-list interpolation that needs it does not blow up.
OmegaConf.register_new_resolver("eval", eval, replace=True)


def main() -> None:
    if socket.gethostname().startswith("sky1"):
        sys.exit("refusing to run on login node sky1")
    group, name = sys.argv[1], sys.argv[2]
    cfg_dir = os.path.abspath("egomimic/hydra_configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"{group}={name}"],
        )
    sub = cfg[group]
    print(OmegaConf.to_yaml(sub, resolve=False, sort_keys=True))


if __name__ == "__main__":
    main()
