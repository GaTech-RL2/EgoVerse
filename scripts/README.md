# scripts/

## Eval rollout for `tsim` (PushShapes) — saved MP4s + local pygame viewer

Two scripts:

- **`eval_rollout.py`** — runs open-loop evaluation rollouts of a trained
  policy in `PushShapesEnv`, in parallel headless workers. For each
  episode it writes an MP4 of the env render plus a per-step `.npz`
  sidecar with state / action / reward / coverage, then aggregates
  metrics.
- **`rollout_viewer.py`** — a pygame viewer that watches the eval's
  output dir and plays each episode back as it lands, styled to match
  `Tsimulation/visualize_episode.py`. No streaming, no SSH tunnels — it
  just reads files.

### Rollout semantics (important)

The H-Net policy is behavior-cloning with **full-episode action chunking**:
it predicts the entire `action_horizon` action chunk from the *initial*
observation. So the rollout is **open-loop** — reset env, query the policy
once on `obs_0`, then replay every predicted action through `env.step` with
no re-query. This matches exactly how the model is trained and validated.
(There is no closed-loop `get_action(obs)` interface in this codebase.)

"Scenario" for this env = `(object_shape ∈ T/U/Z, pusher_shape ∈
circle/stick, obstacle_level ∈ 0-3)`, set via `--object/--pusher/--obstacles`.

### Run the rollout

```bash
python scripts/eval_rollout.py \
    --policy-path /path/to/model.ckpt \
    --num-episodes 100 --num-workers 8 \
    --object T --pusher circle --obstacles 0 \
    --output-dir ./eval_runs/my_eval
```

`--policy-path` accepts any Lightning `ModelWrapper` checkpoint (HNet / HPT /
variants) — `config_tree` + `norm_stats_state` are read out of the
checkpoint, so no separate config is needed. Omit `--policy-path` (or pass
`none`) to use the **random-policy fallback**, which exercises the whole
harness (env, MP4, sidecar, metrics, determinism) without a trained net —
useful for a smoke test; metrics are meaningless in that mode.

Harness self-test (no checkpoint needed):

```bash
python scripts/eval_rollout.py --num-episodes 3 --num-workers 2 \
    --max-steps 40
```

### Watch the rollout live

Point the viewer at the same `--output-dir` you passed to the rollout. It
can start before, during, or after the rollout — episodes appear as their
`.npz` sidecars land in `videos/`:

```bash
python scripts/rollout_viewer.py --output-dir ./eval_runs/my_eval
```

The viewer needs a display, so run it wherever you have one — locally on
your laptop if the eval's output dir is on a shared/NFS path, or on the
compute node via X11 / VS Code Remote-Desktop. No port forwarding involved
at any step.

Hotkeys:

| Key | Action |
|-----|--------|
| `SPACE` | Play / pause |
| `←` / `→` | Step one frame |
| `↑` / `↓` | Faster / slower (×1.5 per press) |
| `R` | Restart current episode |
| `N` / `P` | Next / previous episode |
| `A` | Toggle auto-advance to next episode |
| `Q` / `Esc` | Quit |

Episodes are played in `episode_idx` order (sorted filenames) even if
workers finish out of order. Auto-advance is on by default — when the
current episode ends and a later one is ready, the viewer jumps to it.

### Outputs (`--output-dir`)

```
eval_runs/<timestamp>/
  episodes.jsonl         # one line per episode, SORTED by episode_idx
  summary.json           # aggregates + CLI args + git SHA + timestamp
  config_resolved.yaml   # resolved args actually used
  videos/
    episode_0000.mp4     # env render (512×512)
    episode_0000.npz     # per-step state — viewer reads this
    episode_0001.mp4
    episode_0001.npz
    ...
  rollout.log
```

The `.npz` sidecar is written **after** the MP4 is finalized, so its
presence is what the viewer treats as the "episode is ready" sentinel —
you never see half-written files. Each sidecar contains parallel arrays of
length T (`pusher_obs`, `object_obs`, `actions`, `rewards`, `coverages`),
the constant `goal_pose`, and a `metadata_json` blob with episode-level
summary fields.

`episodes.jsonl` is sorted by `episode_idx` (not completion order) and
per-episode seeding is `seed + episode_idx`, so a fixed `--seed` +
`--policy-path` produces byte-identical `episodes.jsonl` regardless of
`--num-workers`.

### Metrics

This env exposes coverage-IoU and a single object (no per-block distance /
collisions), so the metric set is adapted:

- **Per-episode** (`episodes.jsonl`): `success` (coverage ≥ 0.95 at any
  step), `episode_length`, `episode_return`, `final_coverage`,
  `max_coverage`, `final_distance_to_goal` and `min_distance_to_goal`
  (object↔goal centroid px), `time_to_first_success`, `video_path`,
  `state_path`, `error` (traceback string or null).
- **Aggregate** (`summary.json`): success rate + 95% Wilson CI; mean /
  median / stderr of return, length, final distance; wall-clock,
  episodes/sec; CLI args; git SHA.

The run exits nonzero if more than 5% of episodes errored.
