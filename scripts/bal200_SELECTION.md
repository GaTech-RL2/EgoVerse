# new_circle_3_bal200 — Balanced 200-episode subset

## Method

- **Source**: `/coc/flash7/paphiwetsa3/datasets/new_circle_3` (953 episodes).
- **Pool**: restricted to **obstacle_level == 0** (766 episodes). Levels 1–12 (187 eps) excluded.
  - obstacle_level read from each episode's `zarr.json` attrs `task_description.env_args.obstacle_level`; it matches the filename `obsN` token for every episode.
- **Selection**: farthest-point / **k-center greedy** on the per-episode 4-d vector
  `[Tstart_x, Tstart_y, goal_x, goal_y]`, each dim normalized to [0,1] by the obs0-pool per-dim min/max.
  - **Tstart** = object (T) pose at frame 0 = `observations.state[0, 2:4]` (state layout `[agent_x, agent_y, obj_x, obj_y, obj_theta]`, confirmed in `egomimic/eval/eval_sim.py:_state_to_init`).
  - **goal** = `goal_pose[0, 0:2]`.
- **Seed**: 0 (deterministic). Greedy starts from the episode nearest the distribution centroid in normalized space: `episode_T_circle_obs0_ontarget_000017.zarr`.
- **norm mins** = [62.83542239604325, 65.78984192842445, 62.67600184394976, 61.356056468283306]
- **norm maxs** = [446.7630886213069, 448.38656239909346, 450.76094912992386, 450.5049808272626]

## Evidence: 4x4 occupancy grids (counts per bin) over the 512x512 world

### FULL (953)  (n=953)

T-start (x,y) 4x4 occupancy [rows=y-bin top->bottom, cols=x-bin]:
```
  21   69   42   32
  52   96  104   61
  61   89   99   48
  16   73   74   16
```
  min=16 max=104 mean=59.6 CV=0.474 empty_bins=0/16

Goal (x,y) 4x4 occupancy:
```
  26   32   54   50
  68  110   86   48
  68   91  115   56
  24   56   39   30
```
  min=24 max=115 mean=59.6 CV=0.463 empty_bins=0/16

### NAIVE first200  (n=200)

T-start (x,y) 4x4 occupancy [rows=y-bin top->bottom, cols=x-bin]:
```
   3   12   10    6
   4   30   22   20
  11   20   19   10
   3   13   14    3
```
  min=3 max=30 mean=12.5 CV=0.618 empty_bins=0/16

Goal (x,y) 4x4 occupancy:
```
   6    3   14    9
  11   26   20   10
  12   22   23   13
   6   11    7    7
```
  min=3 max=26 mean=12.5 CV=0.529 empty_bins=0/16

### BALANCED 200  (n=200)

T-start (x,y) 4x4 occupancy [rows=y-bin top->bottom, cols=x-bin]:
```
   3   14   10    6
  12   21   20    8
  16   19   23   13
   3   15   14    3
```
  min=3 max=23 mean=12.5 CV=0.508 empty_bins=0/16

Goal (x,y) 4x4 occupancy:
```
   7    5   12    9
  12   25   22   11
  11   19   17   10
   8   13   12    7
```
  min=5 max=25 mean=12.5 CV=0.434 empty_bins=0/16

## Selected episodes (200)

- episode_T_circle_obs0_000000.zarr
- episode_T_circle_obs0_000002.zarr
- episode_T_circle_obs0_000003.zarr
- episode_T_circle_obs0_000007.zarr
- episode_T_circle_obs0_000008.zarr
- episode_T_circle_obs0_000009.zarr
- episode_T_circle_obs0_000010.zarr
- episode_T_circle_obs0_000011.zarr
- episode_T_circle_obs0_000012.zarr
- episode_T_circle_obs0_000013.zarr
- episode_T_circle_obs0_000019.zarr
- episode_T_circle_obs0_000020.zarr
- episode_T_circle_obs0_000021.zarr
- episode_T_circle_obs0_000025.zarr
- episode_T_circle_obs0_000028.zarr
- episode_T_circle_obs0_000031.zarr
- episode_T_circle_obs0_000032.zarr
- episode_T_circle_obs0_000033.zarr
- episode_T_circle_obs0_000035.zarr
- episode_T_circle_obs0_000039.zarr
- episode_T_circle_obs0_000040.zarr
- episode_T_circle_obs0_000047.zarr
- episode_T_circle_obs0_000049.zarr
- episode_T_circle_obs0_000051.zarr
- episode_T_circle_obs0_000054.zarr
- episode_T_circle_obs0_000056.zarr
- episode_T_circle_obs0_000059.zarr
- episode_T_circle_obs0_000065.zarr
- episode_T_circle_obs0_000069.zarr
- episode_T_circle_obs0_000071.zarr
- episode_T_circle_obs0_000073.zarr
- episode_T_circle_obs0_000083.zarr
- episode_T_circle_obs0_000087.zarr
- episode_T_circle_obs0_000091.zarr
- episode_T_circle_obs0_000093.zarr
- episode_T_circle_obs0_000094.zarr
- episode_T_circle_obs0_000096.zarr
- episode_T_circle_obs0_000097.zarr
- episode_T_circle_obs0_000098.zarr
- episode_T_circle_obs0_000099.zarr
- episode_T_circle_obs0_000100.zarr
- episode_T_circle_obs0_000102.zarr
- episode_T_circle_obs0_000103.zarr
- episode_T_circle_obs0_000106.zarr
- episode_T_circle_obs0_000107.zarr
- episode_T_circle_obs0_000109.zarr
- episode_T_circle_obs0_000112.zarr
- episode_T_circle_obs0_000114.zarr
- episode_T_circle_obs0_000115.zarr
- episode_T_circle_obs0_000117.zarr
- episode_T_circle_obs0_000118.zarr
- episode_T_circle_obs0_000119.zarr
- episode_T_circle_obs0_000120.zarr
- episode_T_circle_obs0_000121.zarr
- episode_T_circle_obs0_000127.zarr
- episode_T_circle_obs0_000131.zarr
- episode_T_circle_obs0_000133.zarr
- episode_T_circle_obs0_000134.zarr
- episode_T_circle_obs0_000136.zarr
- episode_T_circle_obs0_000139.zarr
- episode_T_circle_obs0_000140.zarr
- episode_T_circle_obs0_000141.zarr
- episode_T_circle_obs0_000144.zarr
- episode_T_circle_obs0_000146.zarr
- episode_T_circle_obs0_000148.zarr
- episode_T_circle_obs0_000155.zarr
- episode_T_circle_obs0_000156.zarr
- episode_T_circle_obs0_000164.zarr
- episode_T_circle_obs0_000165.zarr
- episode_T_circle_obs0_000166.zarr
- episode_T_circle_obs0_000168.zarr
- episode_T_circle_obs0_000169.zarr
- episode_T_circle_obs0_000170.zarr
- episode_T_circle_obs0_000175.zarr
- episode_T_circle_obs0_000177.zarr
- episode_T_circle_obs0_000180.zarr
- episode_T_circle_obs0_000184.zarr
- episode_T_circle_obs0_000185.zarr
- episode_T_circle_obs0_000187.zarr
- episode_T_circle_obs0_000189.zarr
- episode_T_circle_obs0_000190.zarr
- episode_T_circle_obs0_000192.zarr
- episode_T_circle_obs0_000193.zarr
- episode_T_circle_obs0_000194.zarr
- episode_T_circle_obs0_000199.zarr
- episode_T_circle_obs0_000200.zarr
- episode_T_circle_obs0_000201.zarr
- episode_T_circle_obs0_000203.zarr
- episode_T_circle_obs0_000204.zarr
- episode_T_circle_obs0_000205.zarr
- episode_T_circle_obs0_000207.zarr
- episode_T_circle_obs0_000208.zarr
- episode_T_circle_obs0_000215.zarr
- episode_T_circle_obs0_000216.zarr
- episode_T_circle_obs0_000217.zarr
- episode_T_circle_obs0_000218.zarr
- episode_T_circle_obs0_000221.zarr
- episode_T_circle_obs0_000223.zarr
- episode_T_circle_obs0_000227.zarr
- episode_T_circle_obs0_000230.zarr
- episode_T_circle_obs0_000232.zarr
- episode_T_circle_obs0_000233.zarr
- episode_T_circle_obs0_000234.zarr
- episode_T_circle_obs0_000239.zarr
- episode_T_circle_obs0_000241.zarr
- episode_T_circle_obs0_000242.zarr
- episode_T_circle_obs0_000243.zarr
- episode_T_circle_obs0_000244.zarr
- episode_T_circle_obs0_000245.zarr
- episode_T_circle_obs0_000247.zarr
- episode_T_circle_obs0_000248.zarr
- episode_T_circle_obs0_000249.zarr
- episode_T_circle_obs0_000251.zarr
- episode_T_circle_obs0_000252.zarr
- episode_T_circle_obs0_000255.zarr
- episode_T_circle_obs0_000269.zarr
- episode_T_circle_obs0_000275.zarr
- episode_T_circle_obs0_000281.zarr
- episode_T_circle_obs0_000295.zarr
- episode_T_circle_obs0_000319.zarr
- episode_T_circle_obs0_000321.zarr
- episode_T_circle_obs0_000338.zarr
- episode_T_circle_obs0_000339.zarr
- episode_T_circle_obs0_000340.zarr
- episode_T_circle_obs0_000344.zarr
- episode_T_circle_obs0_000345.zarr
- episode_T_circle_obs0_000348.zarr
- episode_T_circle_obs0_000351.zarr
- episode_T_circle_obs0_000366.zarr
- episode_T_circle_obs0_000371.zarr
- episode_T_circle_obs0_000375.zarr
- episode_T_circle_obs0_000384.zarr
- episode_T_circle_obs0_000389.zarr
- episode_T_circle_obs0_000392.zarr
- episode_T_circle_obs0_000398.zarr
- episode_T_circle_obs0_000403.zarr
- episode_T_circle_obs0_000405.zarr
- episode_T_circle_obs0_000406.zarr
- episode_T_circle_obs0_000411.zarr
- episode_T_circle_obs0_000416.zarr
- episode_T_circle_obs0_000418.zarr
- episode_T_circle_obs0_000421.zarr
- episode_T_circle_obs0_000434.zarr
- episode_T_circle_obs0_000435.zarr
- episode_T_circle_obs0_000442.zarr
- episode_T_circle_obs0_000446.zarr
- episode_T_circle_obs0_000450.zarr
- episode_T_circle_obs0_000454.zarr
- episode_T_circle_obs0_000455.zarr
- episode_T_circle_obs0_000459.zarr
- episode_T_circle_obs0_000461.zarr
- episode_T_circle_obs0_000464.zarr
- episode_T_circle_obs0_000467.zarr
- episode_T_circle_obs0_000470.zarr
- episode_T_circle_obs0_000472.zarr
- episode_T_circle_obs0_000473.zarr
- episode_T_circle_obs0_000480.zarr
- episode_T_circle_obs0_000483.zarr
- episode_T_circle_obs0_000492.zarr
- episode_T_circle_obs0_000495.zarr
- episode_T_circle_obs0_000496.zarr
- episode_T_circle_obs0_000500.zarr
- episode_T_circle_obs0_000502.zarr
- episode_T_circle_obs0_000505.zarr
- episode_T_circle_obs0_000506.zarr
- episode_T_circle_obs0_000508.zarr
- episode_T_circle_obs0_000509.zarr
- episode_T_circle_obs0_000513.zarr
- episode_T_circle_obs0_000514.zarr
- episode_T_circle_obs0_000516.zarr
- episode_T_circle_obs0_000518.zarr
- episode_T_circle_obs0_000520.zarr
- episode_T_circle_obs0_000524.zarr
- episode_T_circle_obs0_000525.zarr
- episode_T_circle_obs0_000557.zarr
- episode_T_circle_obs0_000571.zarr
- episode_T_circle_obs0_000590.zarr
- episode_T_circle_obs0_000702.zarr
- episode_T_circle_obs0_ontarget_000009.zarr
- episode_T_circle_obs0_ontarget_000010.zarr
- episode_T_circle_obs0_ontarget_000013.zarr
- episode_T_circle_obs0_ontarget_000016.zarr
- episode_T_circle_obs0_ontarget_000017.zarr
- episode_T_circle_obs0_ontarget_000019.zarr
- episode_T_circle_obs0_ontarget_000020.zarr
- episode_T_circle_obs0_ontarget_000021.zarr
- episode_T_circle_obs0_ontarget_000022.zarr
- episode_T_circle_obs0_ontarget_000023.zarr
- episode_T_circle_obs0_ontarget_000026.zarr
- episode_T_circle_obs0_ontarget_000027.zarr
- episode_T_circle_obs0_ontarget_000029.zarr
- episode_T_circle_obs0_ontarget_000037.zarr
- episode_T_circle_obs0_ontarget_000038.zarr
- episode_T_circle_obs0_ontarget_000039.zarr
- episode_T_circle_obs0_ontarget_000040.zarr
- episode_T_circle_obs0_ontarget_000042.zarr
- episode_T_circle_obs0_ontarget_000043.zarr
- episode_T_circle_obs0_ontarget_000044.zarr
- episode_T_circle_obs0_ontarget_000045.zarr
- episode_T_circle_obs0_ontarget_000046.zarr
## Note on method choice (k-center greedy vs. joint occupancy)

The full obs0 pool (766 eps) is ALREADY near-uniform in the joint 4-d space:
a 2-bin/dim joint-16 histogram of [Tstart_x,Tstart_y,goal_x,goal_y] has every
cell at 45-50 episodes (joint-16 CV = 0.028). The dataset was evidently
generated to cover (Tstart x goal) jointly.

k-center greedy (the method used here, as specified) maximizes coverage of the
support extremes/corners, which IMPROVES the marginal 4x4 grids vs the naive
first-200 (T-start CV 0.618 -> 0.508, goal 0.529 -> 0.434, no empty bins) and
guarantees corner coverage. As a side effect it slightly concentrates the joint
occupancy (joint-16 CV 0.200, cell counts 9-18) relative to the already-flat
pool. A pure count-equalizing stratified pick would yield joint-16 CV ~0.040
(cells 12-13) but WORSE marginals; it is recorded here as an alternative for
future variants. k-center was used because it is the requested farthest-point
method and gives the strongest marginal-grid evenness + extreme coverage, which
is what the balance evidence below shows.
