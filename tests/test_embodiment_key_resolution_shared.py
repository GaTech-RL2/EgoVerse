"""Permanent equality guard for dedup collapse c5.

Collapse c5 hoisted three byte-identical methods that used to be copy-pasted
across the policy algos into the shared base ``Algo``:

  * ``_resolve_embodiment_keys`` — the per-embodiment key-resolution loop
    (formerly duplicated in PackedAlgoBase ``packed_base.py``, DFoT
    ``diffusion/algo.py``, and WindowedBC ``bc.py``).
  * ``_build_obs`` — assembles the obs dict from proprio/lang/camera keys
    (formerly duplicated in PackedAlgoBase and DFoT; WindowedBC inherited
    PackedAlgoBase's).
  * ``log_info`` — surfaces ``action_loss`` as ``Loss`` plus every loss
    under its own key (formerly duplicated in PackedAlgoBase and DFoT).

This test pins two invariants so a future edit can't silently re-diverge:

  1. **Import-identity** — ``PackedAlgoBase`` / ``DFoT`` / ``WindowedBC`` all resolve the
     SAME function object from ``Algo`` (no subclass re-introduces a private
     override of these three methods).
  2. **Behavioural equality on a fixed norm_stats fixture** — driving the
     shared resolver from a minimal stand-in of each class produces byte-equal
     ``embodiment_ids`` / ``proprio_keys`` / ``lang_keys`` / ``camera_keys`` /
     ``resolved_ac_keys`` dicts, and ``_build_obs`` selects the identical obs
     keys.
"""

from __future__ import annotations

import torch

from egomimic.algo.algo import Algo
from egomimic.algo.packed_base import PackedAlgoBase
from egomimic.algo.bc import WindowedBC
from egomimic.algo.diffusion.algo import DFoT
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

# pushshapes_sim -> EMBODIMENT.PUSHSHAPES_SIM.value (== 15; the emb id used by
# the dedup-baseline smoke manifest).
_EMB_NAME = "pushshapes_sim"
_EMB_ID = get_embodiment_id(_EMB_NAME)
_AC_KEY = "actions_cartesian"


def _make_norm_stats() -> MultiDataset:
    """A fixed norm_stats fixture exercising every key type the resolver
    branches on: an action key (must match ``ac_keys``), a decoy action key
    (must be skipped), two proprio keys, one lang key, two camera keys."""
    ns = MultiDataset(state={}, norm_mode="minmax")
    ns.key_types = {
        _EMB_ID: {
            _AC_KEY: "action_keys",
            "actions_joint": "action_keys",  # decoy — not selected (!= ac_key)
            "observations.proprio": "proprio_keys",
            "observations.ee_pose": "proprio_keys",
            "observations.lang": "lang_keys",
            "observations.images.front_img_1": "camera_keys",
            "observations.images.wrist_img": "camera_keys",
        }
    }
    return ns


def _resolve_on(cls) -> dict:
    """Drive ``Algo._resolve_embodiment_keys`` from a minimal stand-in of
    ``cls`` (bypassing the heavy real ``__init__``) and return the resolved
    dicts. Only the two attributes the resolver reads are set."""
    obj = object.__new__(cls)
    obj.domains = [_EMB_NAME]
    obj.ac_keys = {_EMB_NAME: _AC_KEY}
    obj._resolve_embodiment_keys(_make_norm_stats())
    return {
        "embodiment_ids": dict(obj.embodiment_ids),
        "proprio_keys": dict(obj.proprio_keys),
        "lang_keys": dict(obj.lang_keys),
        "camera_keys": dict(obj.camera_keys),
        "resolved_ac_keys": dict(obj.resolved_ac_keys),
    }


def test_resolver_is_single_shared_function():
    # All three policy algos resolve the SAME function object from Algo —
    # i.e. none re-introduces a private copy of the shared helpers.
    for cls in (PackedAlgoBase, WindowedBC, DFoT):
        assert cls._resolve_embodiment_keys is Algo._resolve_embodiment_keys, cls
        assert cls._build_obs is Algo._build_obs, cls
        assert cls.log_info is Algo.log_info, cls


def test_resolver_produces_identical_keysets_across_algos():
    ref = _resolve_on(PackedAlgoBase)

    # The fixture must actually exercise every branch (else the test is vacuous).
    assert ref["embodiment_ids"] == {_EMB_NAME: _EMB_ID}
    assert ref["resolved_ac_keys"] == {_EMB_ID: _AC_KEY}  # decoy excluded
    assert ref["proprio_keys"][_EMB_ID] == [
        "observations.proprio",
        "observations.ee_pose",
    ]
    assert ref["lang_keys"][_EMB_ID] == ["observations.lang"]
    assert ref["camera_keys"][_EMB_ID] == [
        "observations.images.front_img_1",
        "observations.images.wrist_img",
    ]

    # PackedAlgoBase == WindowedBC == DFoT, exactly.
    for cls in (WindowedBC, DFoT):
        assert _resolve_on(cls) == ref, cls


def test_build_obs_selects_identical_obs_across_algos():
    # _build_obs reads proprio+lang+camera keys; "extra" is dropped, a missing
    # camera key is skipped.
    batch = {
        "observations.proprio": torch.zeros(2, 3),
        "observations.ee_pose": torch.zeros(2, 6),
        "observations.lang": torch.zeros(2, 8),
        "observations.images.front_img_1": torch.zeros(2, 3, 4, 4),
        # "observations.images.wrist_img" deliberately absent
        "extra": torch.zeros(2, 1),
    }
    expected_keys = {
        "observations.proprio",
        "observations.ee_pose",
        "observations.lang",
        "observations.images.front_img_1",
    }

    results = []
    for cls in (PackedAlgoBase, WindowedBC, DFoT):
        obj = object.__new__(cls)
        obj.domains = [_EMB_NAME]
        obj.ac_keys = {_EMB_NAME: _AC_KEY}
        obj._resolve_embodiment_keys(_make_norm_stats())
        obs = obj._build_obs(batch, _EMB_ID)
        assert set(obs.keys()) == expected_keys, cls
        results.append(set(obs.keys()))

    assert results[0] == results[1] == results[2]
