# Running EgoVerse arc-tokenizer cotrains on OSMO (8× L40S)

`arc_cotrain_l40s.yaml` replaces the PACE + submitit launch path with a single
OSMO workflow: 1 node × 8 L40S, Lightning DDP inside one container.

## Why the PACE commands can't run as-is

Every path in them is a Georgia Tech PACE path. Verified by exec'ing into a
live OSMO workflow — `/storage/project/r-dxu345-0` does not exist there. So the
repo, the `.venv`, the dataset, and both `norm_stats.json` files all have to be
reconstructed on OSMO.

## Picking the pool

```bash
osmo profile list                     # which pools you can use
osmo pool list --mode free            # Quota Free / Total Free per pool
osmo resource list -p groot-l40s-03   # per-node GPU capacity
```

Effective availability is `min(Quota Free, Total Free)`.

**Use `groot-l40s-03`.** It is the only L40S pool with **8 GPUs per node**
(289 quota free / 138 physically free at time of writing). `groot-l40s-01`
nodes are 4 GPU/node, so they cannot host 8-way single-node DDP.

If `Quota Free` is 0 but `Total Free` > 0, submit with `--priority LOW` — it
bypasses quota onto idle capacity, at the cost of being preemptible.

## Dataset autopull

`data/arc_tests_cotrain.yaml` and `data/arc_tests_cotrain_arctok.yaml` use
`S3EpisodeResolver`, which resolves the episode set from the SQL episode table
and syncs it from R2 into `${paths.dataset_dir}` itself
(`S3EpisodeResolver.resolve` → `sync_from_filters` → s5cmd). Nothing needs to
be pre-staged; on a machine that already has the episodes,
`_episode_already_present` makes it a no-op.

Both configs filter to `lab='rl2'`, `task='fold_clothes'`, split by embodiment
— **290 eva_bimanual + 572 human_bimanual** (862 episodes, ~155 GB). That
matches the population the hand-staged PACE `arc_tests` folder held, and the
eva filter is the same one `human_mecka_eva_rl2_fold_clothes.yaml` already
uses.

Because the resolver reads the SQL table, the container needs AWS credentials
for Secrets Manager (`create_default_engine` → DB password) as well as the R2
keys. The entry script gets both by running the repo's own
`egomimic/utils/aws/setup_secret.sh`, which writes `SECRETS_ARN` and `R2_*`
into `~/.egoverse_env`.

## One-time credential setup

```bash
# AWS key for Secrets Manager — setup_secret.sh pulls R2 keys + DB creds with it
osmo credential set egoverse-aws --type GENERIC \
  --payload aws_access_key_id=AKIA... \
            aws_secret_access_key=... \
            aws_region=us-east-2

# GitHub PAT with read access to GaTech-RL2/EgoVerse
osmo credential set egoverse-github --type GENERIC --payload github_token=ghp_...

# wandb — the key MUST have access to whatever you pass as wandb_entity
osmo credential set egoverse-wandb --type GENERIC --payload wandb_api_key=...
```

Verify with `osmo credential list`. `credential set` does not overwrite — it
fails with `duplicate key value violates unique constraint`, so
`osmo credential delete <name>` first when rotating one.

Check which entities a wandb key can actually reach before using it:

```bash
curl -s -H "Content-Type: application/json" -u "api:$WANDB_API_KEY" \
  -d '{"query":"{viewer{username entity teams(first:50){edges{node{name}}}}}"}' \
  https://api.wandb.ai/graphql
```

`logger/wandb.yaml` hardcodes `entity: rl2-group`; the workflow overrides it
with `logger.wandb.entity={{wandb_entity}}`. A mismatch between key and entity
fails at `wandb.init()` in phase 2 — after the dataset pull and norm stats,
~2.5h in — which is why the entry script preflights entity access up front.

## Submitting

`job_name` becomes the OSMO workflow name, so pick something that identifies
the variant — both runs otherwise land under one name and are only
distinguishable by the trailing counter. Convention: `arc-<variant>-<hardware>`.

```bash
# Baseline cartesian cotrain (no-bounds, const lr 3e-4)
osmo workflow submit osmo/arc_cotrain_l40s.yaml --pool groot-l40s-03 --set \
  branch=arc-length-nv \
  job_name=arc-baseline-cart-8xl40s \
  wandb_entity=rl2-group \
  num_gpu=8 batch_size=4 num_workers=4 \
  run_name=arc_tests_cotrain \
  run_desc=full_8xl40s_constlr3e4_nobounds \
  data_cfg=arc_tests_cotrain \
  model_cfg=hpt_cotrain_mecka_flow_shared_head

# Arc-tok D20/M15 cotrain (no-bounds, const lr 3e-4)
osmo workflow submit osmo/arc_cotrain_l40s.yaml --pool groot-l40s-03 --set \
  branch=arc-length-nv \
  job_name=arc-tok-d20m15-8xl40s \
  wandb_entity=rl2-group \
  num_gpu=8 batch_size=4 num_workers=4 \
  run_name=arc_tests_cotrain_arctok \
  run_desc=D20_M15_8xl40s_constlr3e4_nobounds \
  data_cfg=arc_tests_cotrain_arctok \
  model_cfg=hpt_cotrain_mecka_flow_shared_head_arc_D20_M15
```

Workflow names are immutable after submit, and `osmo workflow tag` returns 403
for this profile — so get `job_name` right at submit time.

The workflow clones `$branch` from GitHub, so config changes must be pushed
before submitting.

Monitor with `osmo workflow query <id> --format-type json` and
`osmo workflow logs <id> -n 10000`.

## What changed vs the PACE commands

| PACE | OSMO |
| --- | --- |
| `-m` + `hydra/launcher=submitit_pace_l40s` | dropped; Lightning DDP runs the ranks in-container |
| `launch_params.gpus_per_node=2` | `=8` (still feeds `trainer/ddp.yaml`'s `devices`) |
| `batch_size: 32` (config default) | `4` on both train and valid dataloaders, per embodiment |
| `num_workers: 6` | `4`, so 8×2×4=64 workers fits the 100 CPU request |
| `precomputed_norm_path=<PACE json>` | computed once on 1 GPU in phase 1, consumed in phase 2 |
| `paths.dataset_dir=/storage/.../arc_tests` | `/workspace/arc_tests`, autopulled by the data config |
| `LocalFolderEpisodeResolver` | `S3EpisodeResolver` + per-leaf `DatasetFilter` |
| `source .venv/bin/activate` | `git clone` + `uv sync --frozen` at runtime |

Dropping `-m` also means Hydra writes to `hydra.run.dir`
(`./logs/<name>/<description>_<timestamp>`) instead of the multirun sweep dir,
so output paths no longer have the `/0/` job-number segment.

`bounds_check: false` (no-bounds) and `lr: 3e-4` + `scheduler: null` (constant
LR) are already set in the data and model configs on this branch — the PACE
commands did not override them either.

## Two things worth knowing

- **Norm stats are computed in two phases on purpose.** `trainHydra.py:127`
  computes them with no rank guard, and Lightning's `ddp` strategy re-executes
  the script in every rank — so an 8-GPU run with no precomputed path would
  compute stats 8× over 862 episodes and race on the cache write. Phase 1 does
  it once on 1 GPU; phase 2 loads the JSON. Phase 1 also serializes the dataset
  download into a single process.

- **Global batch size drops.** Per-GPU 32 on 2 GPUs was a global batch of 64
  per embodiment; per-GPU 4 on 8 GPUs is 32. At a fixed lr of 3e-4 that is a
  different optimization regime than the earlier PACE runs. Both new runs get
  the same treatment, so the baseline-vs-arc-tok comparison stays internally
  valid, but it is not directly comparable to those. Use `batch_size=8` to hold
  the global batch at 64.
