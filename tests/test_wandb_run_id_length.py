"""Every experiment's wandb run id must fit wandb's 128-character Name limit.

logger/wandb.yaml builds the id as ``${name}_${description}_${now:...}``, and
wandb rejects anything longer at ``init()`` -- which happens in phase 2, AFTER
the dataset stage and the whole norm-stats pass. A config that is eight
characters too long therefore costs ~20 minutes of cluster time before it says
so. This has now bitten this project twice, so it is a test rather than a note.
"""

import re
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs/experiment"
TIMESTAMP_LEN = len("2026-09-08_16-51-11")  # ${now:%Y-%m-%d_%H-%M-%S}
WANDB_NAME_LIMIT = 128


def _experiments():
    for path in sorted(CONFIGS.rglob("*.yaml")):
        text = path.read_text()
        name = re.search(r"^name: (.+)$", text, re.M)
        desc = re.search(r"^description: (.+)$", text, re.M)
        if name and desc:
            yield path, name.group(1).strip(), desc.group(1).strip()


@pytest.mark.parametrize("path,name,description", list(_experiments()),
                         ids=lambda v: v.stem if isinstance(v, Path) else None)
def test_wandb_run_id_fits(path, name, description):
    if "${" in name or "${" in description:
        # The id contains an unresolved interpolation, so its final length is
        # not knowable without composing the config. Skip rather than measure
        # the literal "${...}" text, which would be a false failure.
        pytest.skip(f"{path.name}: name/description interpolates at compose time")
    total = len(name) + len(description) + TIMESTAMP_LEN + 2  # two underscores
    assert total <= WANDB_NAME_LIMIT, (
        f"{path.name}: wandb id would be {total} chars (limit "
        f"{WANDB_NAME_LIMIT}); shorten name or description"
    )
