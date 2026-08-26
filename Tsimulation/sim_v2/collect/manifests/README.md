# ChainGripper obstacle initialization bank

`chain_obstacle_seed_bank_v1.json` contains 32 deterministic initializations
for every obstacle level from 1 through 30. Each entry is specific to its
level's geometry and is accepted only when:

- the T start, goal, and open ChainGripper are physically valid;
- the ChainGripper can approach the T without a wall between them;
- the direct T-to-goal swept path intersects a physical obstacle; and
- start/goal distance and spatial diversity thresholds are met.

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
resumes only from episodes whose saved manifest SHA, sampler revision, geometry
hash, level, seed, and entry index match the active bank. Random or stale
episodes cannot satisfy the curated collection target.
