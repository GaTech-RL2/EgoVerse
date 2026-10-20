"""Vendor MCAP validator on real episodes.

    EGOVERSE_TEST_ABC_DIR=<dir of ABC-130k episode_*/>   pytest --integration tests/integration/test_mcap_abc.py
    EGOVERSE_TEST_MCAP_DIR=<dir of spec episode_*/>      pytest --integration tests/integration/test_mcap_abc.py

ABC-130k predates the spec (own robot types, no /tf_static or full episode info), so its
episodes must be read end to end and reported, not pass. Episodes delivered to the spec
(e.g. Aria recordings converted to it) must pass.
"""

import os
from pathlib import Path

import pytest

from egomimic.rldb.mcap import spec
from egomimic.rldb.mcap.validate import validate_episode


def _episodes(var: str) -> list[Path]:
    root = os.environ.get(var)
    if not root:
        pytest.skip(f"set {var} to a directory of episode_*/ dirs")
    eps = sorted(p for p in Path(root).glob("episode_*") if p.is_dir())
    if not eps:
        pytest.skip(f"no episode_*/ dirs under {root}")
    return eps


def test_abc_episodes_are_read_and_reported():
    for ep in _episodes("EGOVERSE_TEST_ABC_DIR"):
        r = validate_episode(ep)
        unreadable = [p for p in r.problems if p.message.startswith("cannot read")]
        assert not unreadable, f"{ep}: {unreadable}"
        assert r.problems, f"{ep}: ABC predates the spec, expected problems"
        print(f"{ep.name}: {len(r.problems)} problems")
        for p in r.problems:
            print(f"  {p}")


def test_spec_deliveries_pass():
    for ep in _episodes("EGOVERSE_TEST_MCAP_DIR"):
        r = validate_episode(ep)
        assert r.problems == [], f"{ep}:\n" + "\n".join(map(str, r.problems))
        assert r.recording_id, ep
        assert (ep / spec.EPISODE_FILE).exists()
