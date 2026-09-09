"""Every arc experiment's semantic_blocks must partition its own token width.

evaluator/eval_planar_v2.yaml hardcodes the width-5 planar partition
([x,y] [cos,sin] [grip]). Any experiment whose token is a different width must
override it, because energy_score validates that the blocks cover every channel
exactly once and raises otherwise.

That raise happens when the evaluator first runs -- in phase 2, after the
dataset stage and the whole norm-stats pass -- so a width mismatch costs ~25
minutes of cluster time per run. This test costs a few seconds.
"""

import glob
import os

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "egomimic/hydra_configs")
PATTERNS = ("planar_v2_usocket_arc_velocity_*.yaml",
            "planar_v2_usocket_arc_duration_*.yaml")
EXPERIMENTS = sorted(
    os.path.basename(p)[:-5]
    for pat in PATTERNS
    for p in glob.glob(os.path.join(CFG, "experiment/pusht", pat))
)


def _episode(length=90):
    rng = np.random.default_rng(0)
    theta = np.cumsum(rng.normal(0.05, 0.02, length))
    xy = np.cumsum(rng.normal(0, 3, (length, 2)), axis=0)
    actions = np.zeros((length, 5), dtype=np.float32)
    actions[:, :2] = xy
    actions[:, 2] = np.cos(theta)
    actions[:, 3] = np.sin(theta)
    return actions


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_semantic_blocks_cover_the_token_width(experiment):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        cfg = compose(config_name="train_zarr_cartesian",
                      overrides=[f"+experiment=pusht/{experiment}"])
    node = cfg.data.train_datasets.pushshapes_sim_u_socket.resolver.transform_list
    token = instantiate(node, _convert_="all")[0].transform(
        {"actions": _episode()}
    )["actions"]
    width = token.shape[-1]
    blocks = [list(b) for b in cfg.evaluator.semantic_blocks]
    covered = [i for start, end in blocks for i in range(start, end)]
    assert len(covered) == width and set(covered) == set(range(width)), (
        f"{experiment}: token width {width} but semantic_blocks {blocks} "
        f"cover {sorted(set(covered))}"
    )
