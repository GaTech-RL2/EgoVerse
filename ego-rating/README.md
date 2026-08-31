# ego-rating

A **pairwise, multi-axis Glicko-2** ranking app for episode quality, with the
comparison UI ported from the fpl preference-collection app (dark theme, A/B
slots, synced scrubbers, per-axis pills). A rater is shown two clips whose
annotations are semantically similar and rates them **A / Equal / B on a fixed
set of axes** (defined in `config.yaml`); each axis is an **independent
Glicko-2** rating system (rating + rating-deviation + volatility) derived from
the comparison log, with **Equal counted as a draw**. Only spans whose
`task_description` embeddings are at least a cosine threshold apart (default
**0.8**, via `all-mpnet-base-v2`) are ever pitted against each other.

The output of the app is the **reward-model training dataset**
(`/export/reward_dataset.jsonl`): one record per episode with a normalized
per-axis score on a **1–10 scale**. Axis scores combine into a **weighted
total score** — the weights are adjusted live with sliders on the leaderboard
(persisted server-side) — and the cumulative episode ranking by total score is
exported separately (`/export/total_ranking.jsonl`). See
[Exports](#exports--the-reward-model-dataset-format).

Pairing is **two-phase** (validated against the active-learning literature —
ASAP / BALD / TrueSkill match-quality / Chatbot-Arena): the rater's first few
comparisons are **most-similar-first** (warm-up), then it pivots to **active
learning**, choosing the pair that maximises expected information gain,
approximated by `(rd_a + rd_b) · p·(1−p)` averaged across the axes — high
uncertainty (rating deviation) × maximal outcome entropy (closest rating ⇒
p≈0.5). The queue can be filtered by **scene** or **operator**.

Spans are defined entirely in `config.yaml` — they are never created or edited
through the UI. The config selects episodes the **same way they're fed to a
model** (a `DatasetFilter`, exactly like `egomimic/hydra_configs/data/*.yaml`):
the backend resolves the filter against `app.episodes`, turns every matching
episode into one rating span, and the queue streams through all of them.

## Quick start

```bash
cd ego-rating
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

On startup the backend creates `data/ego_rating.db` (migrating a pre-axes DB
forward automatically), reads `config.yaml`, resolves its `dataset` filter
against `app.episodes`, and upserts one span per matching episode. The default
config pulls from the DB, so it needs DB/R2 creds (see
[Config](#config-configyaml)); for an offline run use the `spans:` form.
Easiest is to run inside the repo's `emimic` venv, which already has all deps.

## Videos: streamed, never downloaded

Each span's `video` field is resolved to a **streamable URL** by the backend at
`GET /video/{span_id}` — the server never downloads the whole clip. The field
accepts (checked in this order):

| `video` value | Resolves to |
| --- | --- |
| `videos/clip.mp4` (local path) | the range-capable `/videos` static mount |
| `696da439fd6a4da2c4f27354` (24-hex `episode_hash`) | MongoDB `mecka-ai.episodes._id` → `video_1` storage key → **presigned R2 URL** |
| `r2://rldb/raw_v2/mecka/<ts>.mp4` (or `s3://…`) | **presigned R2 URL** |
| `https://…` | used as-is |

The `<video>` element points at `/video/{span_id}`; the backend proxies the
presigned R2 URL, forwarding HTTP **Range**, so `#t=start,end` seeks transfer
only the watched bytes — nothing is pre-downloaded server- or client-side.
Presigned URLs are short-lived (1 h default) and re-signed on demand. The
**synced scrubbers** under the two clips step both videos to the same
proportional position, so corresponding phases of the two episodes line up.

**Credentials** for remote refs (same vars / `~/.egoverse_env` as the main repo):

```
R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
MONGODB_URI=mongodb+srv://...        # only needed for episode_hash refs
```

Local clips need no credentials. `python scripts/make_sample_videos.py`
generates browser-playable demo clips for the offline `spans:` form.

## Config (`config.yaml`)

```yaml
annotation: "Which clip performs the described task better?"

axes:                            # the FIXED set of rating axes (see below)
  - name: annotation_accuracy
    description: "Which episode's annotation more accurately describes what actually happens?"
    weight: 1.0                  # seeds the total-score weight (slider-adjustable)
  - name: efficiency
    description: "Which episode achieves the task with less wasted time and motion?"
    weight: 1.0
  - name: minimal_head_motion
    description: "Which episode has steadier, less erratic head/camera motion?"
    weight: 1.0
  - name: consistency
    description: "Which episode executes the task more consistently?"
    weight: 1.0
  - name: reachability
    description: "Which episode keeps hand poses within a plausible robot workspace?"
    weight: 1.0
  - name: grasp_transferability
    description: "Which episode uses grasps a parallel-jaw gripper could replicate?"
    weight: 1.0

similarity:                      # which spans may be compared
  threshold: 0.8                 # min cosine sim of task_description embeddings
  backend: "modal"               # modal | local | tfidf | auto
  model: "sentence-transformers/all-mpnet-base-v2"

glicko:                          # Glicko-2 rating params (per axis)
  initial: 1500
  rd: 350                        # initial rating deviation (uncertainty)
  vol: 0.06                      # initial volatility
  tau: 0.3                       # system constant (low: clip quality is stable)

pairing:                         # two-phase active-learning scheduler
  warmup_comparisons: 10         # rater's first N comparisons are most-similar-first
  epsilon: 0.1                   # active phase: fraction of random-exploration pairs
  prefer_diverse: true           # different-operator/scene pairs first (same-operator
                                 # pairs tend to be Equal on every axis)

dataset:
  episode_ids_file: egomimic/hydra_configs/data/extra/fold_clothes_all.json
  fps: 30                        # clip end = num_frames / fps
  shuffle: true
```

**Axes are fixed for the collection.** Every saved comparison must rate all of
them (or skip the pair entirely), each axis accumulates its own independent
Glicko-2 ranking, and each axis becomes one 1–10 score field in the exported
reward dataset. Recorded outcomes are keyed by axis *name* — don't rename axes
mid-collection. `weight` only **seeds** an axis's contribution to the total
score; the live value comes from the leaderboard sliders (persisted in the DB,
never clobbered by a config reload).

Episode selection accepts any combination of:

| key | meaning |
| --- | --- |
| `episode_ids_file: <path>` | explicit episode-hash allowlist (JSON list), e.g. the split-viz `extra/fold_clothes_all.json`; bounds the query directly |
| `episode_ids: [...]` | inline episode-hash allowlist |
| `filters.filter_lambdas` | the model-config predicate list; `row['col'] in [...]` / `==,!=,<,<=,>,>=` are pushed to SQL, anything else runs in Python on the bounded subset |
| `where: {col: val \| [vals]}` | structured filter compiled straight to SQL |
| `data_config: <path>` | reuse an existing model data config's `filter_lambdas` verbatim |
| `sql_where: "<raw SQL>"` | escape hatch, AND-ed into every filter group |

> Each filter group must contain at least one SQL-pushable predicate (or a
> `where` / `sql_where` / `limit`) so the 200k-row episode table is never loaded
> whole. Selection runs in Postgres (`WHERE … ORDER BY random() LIMIT n`).

Requires `DATABASE_URL` (episode table), `MONGODB_URI` (episode → video), and
`R2_*` (presign) in the env or `~/.egoverse_env`.

**Offline / no-DB:** provide an explicit `spans:` list *instead of* `dataset:`
(mutually exclusive) — each `video` may be a local path, `r2://` key, `https://`
URL, or `episode_hash`.

Edit `config.yaml`, then **Admin → Reload config** (or `POST /reload-config`) to
re-resolve and sync into the DB.

### Subtask annotations (the legacy DB)

Each clip's annotation context includes **subtask segments** — a list of
`{label, start_seconds, end_seconds}` (e.g. *fold shorts* 0–28s → *move shorts
on stack* 28–37s). These live only in the **legacy episode DB** (an older AWS
RDS), not the new selection DB, so they're pulled by `episode_hash` at
span-resolve time and cached in the SQLite `spans.segments` column (serving
never needs a live legacy connection). Resolution order (`db._enrich_segments`):

1. segments already on the span (offline `spans:` config may hand-provide them);
2. the **live legacy DB** — via `backend/legacy_db.py`, using a *separate* creds
   file `~/.egoverse_env_old` (`SECRETS_ARN` → AWS Secrets Manager → RDS; region
   is read from the ARN so it's immune to R2's `AWS_DEFAULT_REGION=auto`);
3. the bundled **cache** `backend/segments_cache.json`.

**Live on Modal (no cache).** The legacy RDS is a *public* endpoint, so give
Modal a direct connection string (this bypasses AWS Secrets Manager — no AWS
creds needed) and segments are fetched live at pool-resolve time:

```bash
modal secret create egoverse-legacy \
  LEGACY_DATABASE_URL='postgresql://appuser:<password>@lowuse-pg-east2.cdc8824mase4.us-east-2.rds.amazonaws.com:5432/appdb?sslmode=require'
```

`modal_app.py` attaches this secret by default (`LEGACY_DB_SECRET`); with it in
place the bundled cache is never consulted. (Resolve `<password>` from the
legacy secret once: it's the `password` field of the `SECRETS_ARN` value in
`~/.egoverse_env_old`.)

**Cache fallback (no legacy creds).** Set `LEGACY_DB_SECRET = None` in
`modal_app.py` to rely on the bundled `backend/segments_cache.json` instead;
regenerate it whenever the pool's task set changes:

```bash
python scripts/dump_segments.py                     # folding_clothes (default)
python scripts/dump_segments.py --task dishwashing  # other task(s)
```

Either way it's best-effort — with no source, clips just show "no subtask
annotations" and the app runs normally. **Local runs always use the live legacy
DB** (via `~/.egoverse_env_old`), so no cache is involved there.

## Comparability, per-axis Glicko-2 & active learning

**Comparability.** Two spans are only pitted against each other when their
`task_description` embeddings have cosine similarity ≥ `similarity.threshold`.
That defines a graph whose **connected components are the comparability groups**
(`Grp` on the leaderboard); ratings — and the exported scores — are only
comparable within a group, since comparisons never cross the threshold.
`/compare` rejects a rated cross-group pair (400) to keep this invariant.

**Per-axis Glicko-2.** Each (span, axis) carries `rating`, `rd` (rating
deviation = uncertainty), and `vol` (volatility). It's **derived** from the
comparison log by replaying that axis's outcomes in timestamp order, one
comparison = a one-game rating period (standard online Glicko-2). `a`/`b` are
win/loss; **`equal` is a draw (score 0.5 for both)**; skipped pairs are ignored
(but never re-served to that rater). τ is low (0.3) because clip quality is
stable.

**Active learning (two-phase pairing).** The rater's first `warmup_comparisons`
comparisons are **most-similar-first** (seed the ratings). After that it pivots
to **active** mode and picks, among eligible unseen pairs, the one maximising

```
score = mean over axes of (rd_a + rd_b) · p·(1 − p)     # p = Glicko-2 expected outcome
```

— an expected-information-gain approximation combining **uncertainty** (the RD
sum) with **outcome entropy** (`p(1−p)` peaks at p=0.5 ⇔ closest rating).
ε-greedy (`pairing.epsilon`) adds exploration so no span is starved. The
compare screen shows the current **phase** and **pair similarity**.

**Embeddings run on Modal** (`similarity.backend: modal`); deploy once with
`modal deploy backend/modal_embed.py`. Backends: `modal` (default), `local`
(in-process sentence-transformers), `tfidf` (lexical, no torch), `auto`
(modal → tfidf). **Any failure falls back to TF-IDF**, so the app never hangs
on embeddings.

## Screens

- **#compare** — two clips side by side (Slot A blue / Slot B purple) with
  **synced scrubbers** (hover either bar to step both clips to the same
  proportional position), a shared **playback-speed** control (1×–3×, both
  clips), and one A / Equal / B pill row per configured axis. Under each clip:
  the episode's full annotation context — description, **subtask segments**
  (a clickable `{label, start–end}` timeline; click one to jump the clip
  there), objects, and scene / environment text — the reference for
  `annotation_accuracy`. All axes need a choice before saving. **Skip is "not
  now", not "never"**: skipped pairs return once the unseen ones run out, and
  **↶ Undo** takes back your latest submission (rated or skipped) with your
  prior choices prefilled. With `pairing.prefer_diverse`, same-operator pairs
  are only served after cross-operator pairs are exhausted. Keys:
  <kbd>↑</kbd>/<kbd>↓</kbd> pick an axis, <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd>
  = A/Equal/B (auto-advances), <kbd>Enter</kbd> save, <kbd>S</kbd> skip,
  <kbd>⌫</kbd> undo.
- **#leaderboard** — spans ranked per axis (pill selector) by Glicko-2 rating
  with ± RD, 1–10 score, games, W–L–D, win%; the **Total** pill ranks by the
  weighted total score with one score column per axis. **Weight sliders**
  re-rank the table instantly and persist to the server (debounced), so the
  export always matches the screen. Filter by scene/operator.
  **⬇ Total ranking** / **⬇ Reward dataset** download the exports.
- **#admin** — raw comparison log (per-axis outcome chips), Reload config,
  **⬇ Preference log** export.

The `Rater` field in the header is the active `rater_id` (no auth this pass).

## Exports — the reward-model dataset format

### `GET /export/reward_dataset.jsonl` (the primary output)

One JSON line **per episode**, over the full span pool (no filter). This is the
training set for a reward model that regresses *episode video → per-axis
quality score*:

```json
{
  "episode_hash": "696da439fd6a4da2c4f27354",
  "video": "696da439fd6a4da2c4f27354",
  "start": 0.0,
  "end": 14.2,
  "task_description": "fold the shirt neatly",
  "scene": "lab_bench_1",
  "operator": "alice",
  "group": 0,
  "group_size": 12,
  "axes": {
    "annotation_accuracy": {"score": 6.72, "rating": 1662.3, "rd": 290.3,
                            "games": 3, "wins": 2, "losses": 0, "draws": 1},
    "efficiency":          {"score": 5.5,  "rating": 1500.0, "rd": 290.3,
                            "games": 3, "wins": 1, "losses": 1, "draws": 1},
    "minimal_head_motion": {"...": "..."},
    "consistency":         {"...": "..."}
  },
  "total_score": 6.11,
  "weights": {"annotation_accuracy": 2.0, "efficiency": 1.0,
              "minimal_head_motion": 1.0, "consistency": 0.5},
  "num_comparisons": 3
}
```

- **`axes.<name>.score`** — the training target, on a **1–10 scale**:
  `1 + 9 × P(beat the group average)`, where P is the Glicko-2 expected win
  probability of this episode against its comparability-group average (group
  mean rating / mean RD), RD-deflated. **5.5 = average for its group**,
  monotone in rating, and only ever **group-relative** (scores are not
  comparable across `group` values, matching the invariant that comparisons
  never cross a group).
- **`total_score`** — the weighted mean of the axis scores under `weights`
  (the current slider values; all-zero weights degrade to the plain mean).
- **`rating` / `rd` / `games`** — kept so the trainer can filter low-evidence
  episodes (`games == 0` ⇒ prior, `score = 5.5`) or weight the loss by
  confidence (e.g. `1/rd²`).
- **`video`** — same ref the app streams (episode_hash / r2:// / path); resolve
  it to pixels the same way `backend/s3_resolver.py` does.

### `GET /export/total_ranking.jsonl` (the cumulative episode ranking)

One JSON line **per episode, best first**, ranked by the weighted total score
under the current slider weights:

```json
{
  "rank": 1,
  "episode_hash": "696da439fd6a4da2c4f27354",
  "video": "696da439fd6a4da2c4f27354",
  "task_description": "fold the shirt neatly",
  "scene": "lab_bench_1",
  "operator": "alice",
  "group": 0,
  "group_size": 12,
  "total_score": 6.11,
  "weights": {"annotation_accuracy": 2.0, "efficiency": 1.0,
              "minimal_head_motion": 1.0, "consistency": 0.5},
  "axis_scores": {"annotation_accuracy": 6.72, "efficiency": 5.5,
                  "minimal_head_motion": 5.9, "consistency": 5.1},
  "num_comparisons": 3
}
```

Ties on `total_score` rank the episode with more games first (more evidence).

### `GET /export/preferences.jsonl` (the raw pairwise log)

One JSON line **per rated comparison** (skips excluded) — the source data the
scores derive from, and directly usable for Bradley–Terry / preference-loss
training:

```json
{
  "comparison_id": 17,
  "episode_a": "696da439...", "episode_b": "696d0d52...",
  "video_a": "696da439...",   "video_b": "696d0d52...",
  "task_a": "fold the shirt neatly", "task_b": "fold the shirt neatly",
  "labels": {"annotation_accuracy": "a", "efficiency": "equal",
             "minimal_head_motion": "b", "consistency": "equal"},
  "rater_id": 1,
  "ts": "2026-07-03 19:34:08"
}
```

`labels.<axis>` ∈ `"a" | "b" | "equal"`. (Comparisons migrated from the
pre-axes schema appear with a single `"_legacy"` label — filter them out for
training.)

## Persistence & multiple raters

Everything already persists in `data/ego_rating.db` (SQLite): the episode pool,
the full comparison log (the source of truth — ratings are re-derived from it),
and the axis weights. Stopping and restarting the server loses nothing; back up
by copying that one file.

Multiple people can rank simultaneously against one server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Each person opens the same URL and sets their own **Rater** number in the
header (there's no auth — agree on who is which number). The pair queue is
per-rater (nobody sees a pair twice, different raters *can* judge the same
pair — that's good signal), while ratings, weights, and exports pool everyone's
comparisons. Connections use WAL journaling with a busy timeout, which
comfortably handles a handful of concurrent raters; if you outgrow SQLite the
schema is deliberately Postgres-shaped.

For a long-lived deployment, run it under tmux/systemd on a lab box (raters
need network reach — LAN, VPN, or tailscale), and note the server needs the
DB/R2 creds (`~/.egoverse_env`) since videos stream through it.

### Deploy on Modal (permanent URL, zero babysitting)

```bash
cd ego-rating
modal deploy backend/modal_embed.py   # once, if not already deployed
modal deploy modal_app.py             # prints https://<workspace>--ego-rating-web.modal.run
```

`modal_app.py` serves the same FastAPI app as a Modal ASGI web endpoint: the
SQLite DB lives on the `ego-rating-data` Volume (persists across restarts and
redeploys), the `egoverse-*` secrets provide the DB/Mongo/R2 creds, and
`max_containers=1` keeps a single SQLite writer while concurrent raters share
the container. `config.yaml` is baked into the image, so axis/dataset changes
mean redeploying — the pinned pool on the volume survives a redeploy unless the
selection block changed. The URL is public-but-unguessable; see the module
docstring for enabling proxy auth.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/config` | annotation, **axes**, scenes/operators, similarity threshold |
| GET | `/next-pair?rater_id=&scene=&operator=` | next pair `{a, b, similarity, phase}`, or `null` |
| POST | `/compare` | `{span_a, span_b, rater_id, ratings: {axis: a\|b\|equal}}` or `{…, skip: true}` — record a comparison |
| GET | `/progress?rater_id=&scene=&operator=` | `{done, total}` — rated (not skipped) eligible pairs for the rater |
| DELETE | `/comparisons/last?rater_id=` | undo the rater's most recent submission; returns the pair + prior ratings |
| GET | `/annotation/{span_id}` | annotation context: label + objects/scene/environment from the episode's Mongo doc |
| GET | `/leaderboard?scene=&operator=` | per-axis Glicko-2 ranking (`rows[*].axes.<name>` + `total_score`), current `weights` |
| PUT | `/weights` | `{weights: {axis: w ≥ 0}}` — adjust the total-score axis weights (persisted) |
| GET | `/export/reward_dataset.jsonl` | **the reward-model training dataset** (download) |
| GET | `/export/total_ranking.jsonl` | **cumulative episode ranking by weighted total score** (download) |
| GET | `/export/preferences.jsonl` | raw per-axis pairwise preference log (download) |
| GET | `/video/{span_id}` | streams the clip through this server (Range-forwarded) |
| GET | `/raw-comparisons` | full comparison log incl. per-axis outcomes (admin screen) |
| POST | `/reload-config` | re-read config.yaml, re-resolve spans (authoritative), re-apply params |

## Notes

- The comparison log (`comparisons` + `axis_ratings`) is the source of truth;
  Glicko-2 ratings and exported scores are recomputed on demand, never stored.
- A rater never sees the same pair twice (rated or skipped); `/compare` rejects
  rated cross-group pairs and partial axis sets (400).
- The resolved episode pool is **pinned**: restarts and `POST /reload-config`
  keep the persisted pool unless the config's `dataset`/`spans` block actually
  changed (annotation/axes/param edits apply without touching the pool). This
  matters with `shuffle`+`limit`, where re-resolving would draw a different
  random sample. **Admin → ↺ Re-sample pool** (`/reload-config?force=true`)
  explicitly draws a fresh sample.
- When the pool *does* change, the sync is **authoritative**: spans no longer
  selected (and their comparisons) are deleted, so a narrowed filter leaves no
  orphans.
- Pre-axes databases are migrated automatically on startup: skips are kept as
  skips, old single-winner outcomes are preserved under the reserved `_legacy`
  axis (seen-pair semantics intact, no effect on the configured axes' ratings).
- SQLite for dev; the schema maps cleanly to Postgres later.
