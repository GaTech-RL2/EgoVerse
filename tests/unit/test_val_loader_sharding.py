import json
import os
import subprocess
import sys
from pathlib import Path


def test_single_val_head_is_not_sharded_under_ddp(tmp_path):
    """With one val head every rank must still see every val frame: the overlay
    video is written at the source fps and the metric budget is not scaled by
    the world size, both of which assume no DistributedSampler on any val
    loader."""
    script = Path(__file__).resolve().parents[1] / "fixtures" / "ddp_val_unsharded.py"
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    seen = {
        int(p.stem.removeprefix("rank")): json.loads(p.read_text())["seen"]
        for p in tmp_path.glob("rank*.json")
    }
    assert set(seen) == {0, 1}
    assert seen[0] == seen[1] == [float(i) for i in range(8)]
