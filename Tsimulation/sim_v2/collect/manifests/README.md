# ChainGripper obstacle initialization bank

`chain_obstacle_seed_bank_v1.json` contains 32 deterministic initializations
for every obstacle level from 1 through 30. Each entry is specific to its
level's geometry and is accepted only when:

- the T start, goal, and open ChainGripper are physically valid;
- the ChainGripper can approach the T without a wall between them;
- the direct T-to-goal swept path intersects a physical obstacle; and
- start/goal distance and spatial diversity thresholds are met.

Levels 5 and 6 additionally keep both full rotated T silhouettes at least 200
units from the arena corner where the diagonal obstacle emerges. The bank
serializes this level-specific policy, and the silhouette plot renders its
quarter-circle boundary.

Levels 25 and 26 use two inward-facing obstacle sides at opposing arena
corners. Because the arena walls complete sealed corner pockets, their policy
forbids both full start and goal silhouettes from entering those pockets. The
silhouette plot shades these rectangular exclusions. Their valid routes are
sparse, so generation uses and records a deterministic 200,000-seed search
budget while preserving the global acceptance criteria and balanced selection.
The search budget is provenance, not a runtime constraint or part of the
per-level bank identity; an explicit `--seed-limit` remains a hard override.

The playground verifies the level, geometry hash, seed, resolved poses, pusher
angle, and ChainGripper joint angle before collection. It refuses a stale bank
after obstacle geometry changes.

Regenerate the bank and contact sheet from the repository root:

```bash
source .venv/bin/activate
python -m Tsimulation.sim_v2.examples.curate_obstacle_inits \
  --output Tsimulation/sim_v2/collect/manifests/chain_obstacle_seed_bank_v1.json \
  --plot "$HOME/Downloads/chain_obstacle_level_specific_inits_30x32.png" \
  --silhouette-plot "$HOME/Downloads/chain_obstacle_T_silhouettes_30x32.png" \
  --levels 1-30 \
  --per-level 32
```

Collect one level (replace `01` in both places together):

```bash
source .venv/bin/activate
python -m Tsimulation.sim_v2.examples.playground \
  --obstacles 1 \
  --seeds-file Tsimulation/sim_v2/collect/manifests/chain_obstacle_seed_bank_v1.json \
  --output "$HOME/Documents/PushShapes Chain Gripper Obstacles V2/level_01" \
  --image-size 96 \
  --per-agent 32 \
  --auto
```

`R` discards and retries the current initialization. A successful committed
episode advances to the next uncollected seed. Relaunching the same command
records the full manifest SHA for provenance and resumes only from episodes
whose sampler revision, geometry hash, level-bank SHA, level, seed, and entry
index match the active bank. The per-level SHA lets later edits to another level
leave approved/collected levels valid. Random or stale episodes cannot satisfy
the curated collection target.
