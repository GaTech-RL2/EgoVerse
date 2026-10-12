"""Bigger val sets for re-scoring the 30-min opscale checkpoints (2026-10-09).

seen:   per private operator, its 5 held-out episodes + 15 seeded picks among its clean
        fold_tshirt episodes that no 30-min arm trained on (the 30-min picks are nested,
        so an operator's untrained set is the same in every arm).
unseen: every SQL mecka folding_clothes operator with a t-shirt in the description,
        up to 3 seeded t-shirt episodes each (fold_clothes has no descriptions).

  python -m egomimic.scripts.mecka_process.mono.make_bigval
"""

import json

import numpy as np
import pandas as pd

from egomimic.scripts.mecka_process.mono.make_configs import (
    HERE,
    REPO,
    SQL_PKL,
    private_rows,
)

N_SEEN_EXTRA, N_UNSEEN_PER_OP, SEED = 15, 3, 0

picks = json.load(open(HERE / "opscale_picks.json"))
p30 = json.load(open(HERE / "opscale_picks_30m.json"))
rows = private_rows()
rng = np.random.default_rng(SEED)

seen = {}
for op in picks["order"]:
    trained = set(p30["train"][op])
    held = picks["held_out"][op]
    pool = rows[
        (rows.user_id == op) & rows.clean & ~rows.episode_hash.isin(trained | set(held))
    ]
    extra = [
        str(h)
        for h in rng.choice(sorted(pool.episode_hash), N_SEEN_EXTRA, replace=False)
    ]
    seen[op] = held + extra

df = pd.read_pickle(SQL_PKL)
bl = set(
    (REPO / "egomimic/resources/blocklists/mecka_bad_egomotion.txt").read_text().split()
)
f = df[
    (df.lab.str.lower() == "mecka")
    & (df.task == "folding_clothes")
    & ~df.is_deleted.fillna(False).astype(bool)
    & df.zarr_processed_path.notna()
    & ~df.episode_hash.isin(bl)
    & df.task_description.fillna("").str.contains(r"t-?shirt|tee\b", case=False)
]
assert not set(f.operator) & set(rows.user_id)
unseen = {}
for op, g in sorted(f.groupby("operator")):
    hs = sorted(g.episode_hash)
    unseen[op] = [
        str(h) for h in rng.choice(hs, min(N_UNSEEN_PER_OP, len(hs)), replace=False)
    ]

out = {
    "seed": SEED,
    "seen": seen,
    "unseen_tshirt": unseen,
    "unseen_old": picks["unseen"],
}
json.dump(out, open(HERE / "bigval" / "bigval_picks.json", "w"), indent=1)
print({op[:8]: len(v) for op, v in seen.items()})
print("unseen_tshirt ops", len(unseen), "eps", sum(map(len, unseen.values())))
