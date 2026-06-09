"""Verify bank-frame normalization wiring before a RICL training run (CPU).

For a handful of (bank episode, frame) pairs taken from a retrieval cache, this
checks that :class:`egomimic.ricl.data.ZarrBankFrameProvider` configured with
``norm_stats`` returns exactly ``to32(normalize(raw))`` (the query-path order:
MultiDataset normalizes, then PI applies the converter — see ``pi.py``), and
reports the post-normalization value ranges (quantile-normalized values should
be overwhelmingly inside [-1, 1]).

Usage (D0):
  python -m egomimic.ricl.scripts.check_bank_norm \\
      --cache ricl_artifacts/cache_d0 \\
      --stats ricl_artifacts/stats_d0/norm_stats/norm_stats.json \\
      --bank-zarr-root '/storage/project/r-dxu345-0/agao81/pick_place/{hash}' \\
      --bank-embodiment eva_bimanual
"""

from __future__ import annotations

import argparse

import numpy as np

from egomimic.ricl.data import (
    ZarrBankFrameProvider,
    load_bank_norm_stats,
    normalize_with_stats,
)
from egomimic.ricl.retrieval import RetrievalCache
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _embodiment_pipeline(name: str):
    """(keymap, transform_list, converter) for a bank embodiment, matching the
    cotrain_pi_ricl_* configs."""
    if "eva" in name:
        from egomimic.rldb.embodiment.eva import Eva
        from egomimic.utils.action_utils import RobotBimanualCartesianEuler

        return (
            Eva.get_keymap(keymap_mode="cartesian_pi", annotation_key="annotations"),
            Eva.get_transform_list(mode="cartesian"),
            RobotBimanualCartesianEuler(),
        )
    if "aria" in name:
        from egomimic.rldb.embodiment.human import Aria
        from egomimic.utils.action_utils import HumanBimanualCartesianEuler

        return (
            Aria.get_keymap(keymap_mode="cartesian_pi", annotation_key="annotations"),
            Aria.get_transform_list(mode="cartesian"),
            HumanBimanualCartesianEuler(),
        )
    raise SystemExit(f"unknown bank embodiment {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stats", required=True, help="norm_stats.json (file or dir)")
    ap.add_argument("--bank-zarr-root", required=True, help="'{hash}' template")
    ap.add_argument("--bank-embodiment", required=True)
    ap.add_argument("--norm-mode", default="quantile")
    ap.add_argument("--action-horizon", type=int, default=1)
    ap.add_argument("--num-queries", type=int, default=3)
    args = ap.parse_args()

    emb_id = get_embodiment_id(args.bank_embodiment)
    stats = load_bank_norm_stats(args.stats, emb_id)
    keymap, transform_list, converter = _embodiment_pipeline(args.bank_embodiment)
    common = dict(
        resolve_store=lambda h: args.bank_zarr_root.format(hash=h),
        keymap=keymap,
        transform_list=transform_list,
        action_horizon=args.action_horizon,
    )
    provider = ZarrBankFrameProvider(
        converter=converter, norm_stats=stats, norm_mode=args.norm_mode, **common
    )
    raw_provider = ZarrBankFrameProvider(converter=None, norm_stats=None, **common)

    cache = RetrievalCache.load(args.cache)
    state_key = provider.state_key
    action_key = provider.action_key

    checked, in_range = 0, []
    for qh in cache.query_hashes[: args.num_queries]:
        bh, bf, _ = cache.neighbors(qh, 0)
        for h, fi in zip(bh.tolist(), bf.tolist()):
            if not h:
                continue
            got = provider(h, int(fi))
            raw = raw_provider(h, int(fi))

            want_state = normalize_with_stats(
                raw["state"], stats[state_key], args.norm_mode
            )
            import torch

            want_action = converter.to32(
                torch.as_tensor(
                    normalize_with_stats(
                        raw["action"], stats[action_key], args.norm_mode
                    ),
                    dtype=torch.float32,
                )
            )
            want_action = (
                want_action.detach().cpu().numpy().reshape(got["action"].shape)
            )

            np.testing.assert_allclose(got["state"], want_state, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(got["action"], want_action, rtol=1e-5, atol=1e-6)
            vals = np.concatenate([got["state"].ravel(), got["action"].ravel()])
            in_range.append(np.mean(np.abs(vals) <= 1.0))
            checked += 1

    if checked == 0:
        raise SystemExit("cache produced no valid neighbors to check")
    print(
        f"OK: {checked} bank frames match to32(normalize(raw)); "
        f"{100 * float(np.mean(in_range)):.1f}% of normalized values in [-1, 1] "
        "(quantile-1/99 normalization => expect ~98%; far lower means the "
        "stats file does not match this bank's representation)"
    )


if __name__ == "__main__":
    main()
