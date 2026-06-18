# Human↔Robot Episode Pairing — findings & how to reproduce

Context for a future agent picking up the `ryanco/in-context-learning` work. Goal:
find paired **human (aria_bimanual)** and **robot (eva_bimanual)** `pick_place`
episodes with the same scene/objects/task, for side-by-side eval / in-context
learning. This doc records what the data actually looks like (much of it
non-obvious) and how the scripts here produce the pairs.

## TL;DR
- **No human↔robot pairing exists in the data model.** Embodiments are pulled &
  filtered independently. Pairs must be constructed.
- The DB `scene`/`objects` columns are **useless for matching**: `objects` is
  literal `"{None}"`/`"{NOne}"`, `scene` has only 2 values
  (`pick_place_diverse`, `pick_place_div_scene`). Match on **language
  annotations** instead.
- **Two tiers of pairs** (both emitted to `human_robot_pairs.json`):
  1. **True same-scene pairs** = the `alignment data set 1/2` captures (the only
     co-located human+robot recordings). Small: 15 eva + 9 aria.
  2. **Similar-task pairs** = language object-set matching over the bulk. Best
     pairs share ~4–5 objects; **no identical scenes exist** in the bulk data.

## Where the data lives & how to reach it
- **Episode DB**: Postgres `app.episodes`, via `egomimic/utils/aws/aws_sql.py`
  (`create_default_engine()`, `episode_table_to_df()`). Creds: `SECRETS_ARN`
  (boto3 SecretsManager) or `~/.egoverse_env`. Row schema = `TableRow`
  (`aws_sql.py`): `episode_hash` (a UTC timestamp `YYYY-MM-DD-HH-MM-SS-ffffff`),
  `embodiment`, `task`, `task_description`, `scene`, `objects`, `num_frames`,
  `zarr_processed_path` (→ `s3://rldb/processed_v3/<emb>/<hash>.zarr`), `segments`
  (empty for pick_place), `is_deleted`, …
- **Object store** = Cloudflare **R2** bucket `rldb` (NOT AWS S3).
  - ⚠️ Plain `aws s3 ls s3://rldb/...` → **AccessDenied** (hits the AWS endpoint).
    Use `get_boto3_s3_client()` in `egomimic/utils/aws/aws_data_utils.py` (reads
    `R2_ENDPOINT_URL` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` from
    `~/.egoverse_env`), or the aws CLI with `--endpoint-url "$R2_ENDPOINT_URL"`
    and R2 creds exported as `AWS_*` (see `language_process/pull_csv.sh`).
  - Top-level prefixes: `processed_v3/` (zarrs; annotations baked in),
    **`processed_annotations/`** (one `<hash>_annotations.json` per episode),
    `processed_v2/`, `raw_v2/`.
- **`SCALE_API_KEY` is NOT needed** to read annotations — only the
  filter/download pipeline (`rldb/filters.py`, `language_process/`) calls Scale.
  The finished annotations already live in `processed_annotations/` and inside
  the zarrs.

## The pick_place episode landscape (live as of 2026-06)
- **382 pick_place episodes**: 250 eva + 132 aria (after dropping `is_deleted`).
- `task_description` encodes capture sub-protocols and is the useful match key.
  eva and aria label sets are **disjoint** except for `alignment data set N`:
  - eva: `set 2 data collection day 1/2`, `... base data`, `pick and place on
    variety of objects`, `... large task variations`, `... diversity ...`, …
  - aria: `set N base data`, `set N object gen eval`, `set N motion gen eval`,
    `... color correct`, …

## Annotations (`processed_annotations/<hash>_annotations.json`)
- A **flat JSON list** of records `{"text": str, "start_idx": int, "end_idx": int}`
  (the converted/final form — same as the zarr `annotations` array). Half-open
  frame spans `[start_idx, end_idx)`.
- **Dense via paraphrase**: each action segment → ~**13** paraphrases sharing one
  span (1 base instruction from `language_process/prompt.txt` + 12 augmentations
  from `augment_prompt.txt`). The **base instruction is the first record** of a
  span and is imperative-first (`Pick up the X ...` / `Put the X ...`).
- **Coverage: 334 annotated episodes = 228 eva + 106 aria** (all cross-ref to a
  live DB row). eva `alignment` IS annotated (6+6); **aria `alignment` is NOT**.
- **Structural asymmetry that drives the matching choice**:
  - eva episodes ≈ **20** segments (short focused demos, ~10 pick/put pairs).
  - aria episodes ≈ **250** segments (long free-form *play* sessions where a human
    repeatedly rearranges a **fixed set of objects**). So an aria episode is a
    *scene* (object set), not a single task → match by **object-set containment**
    (is the short eva demo's object set ⊆ the aria scene?), not symmetric Jaccard.

## Tier 1 — true pairs: the `alignment data set` captures
| set | eva | aria | capture |
|----|----|----|----|
| set 1 | 6 | 6 | eva + 4 aria interleaved same session **2026-04-14**; 2 aria recollected 2026-04-26 |
| set 2 | 9 | 3 | eva 2026-04-14; all 3 aria recollected 2026-04-26 |

These are the **only** genuinely co-located human↔robot episodes. Pair **set 1 by
capture time** (the 2026-04-14 timestamps interleave eva→aria→eva). aria-alignment
has no language, so they won't appear in Tier 2.

## Tier 2 — similar-task pairs (language object matching)
Method (in `pair_episodes_by_language.py`): parse the manipulated object of each
segment's base instruction → normalize with a hand-built synonym map → per-episode
object set → match each aria scene to eva demos by **containment** (tie-break: #
shared, then Jaccard). Best-match overlap per aria scene:

| max shared objects w/ any eva demo | # aria scenes |
|---|---|
| 5 | 2 |
| 4 | 17 |
| 3 | 53 |
| none (≥3) | 34 |

**Conclusion: no identical scenes in the bulk data** — aria & eva were collected
independently from a shared ~50-object vocabulary, so they overlap partially but
never exactly. Tier-2 pairs are "same objects / similar task," good enough for
qualitative eval, not pixel-aligned scenes.

### Object-name normalization (learned from the corpus — extend as needed)
Annotators describe the same physical object many ways, **including across
embodiments**: aria says `stuffed toy`, eva says `stuffed animal`. Current map
(`canon()`): `stuffed toy|stuffed animal|teddy bear|"X toy"` → `plush`;
`coffee cup|mug` → `cup` (keyed by color); `juice pouch|pack|bag` → `juice`;
`bag of chips|…` → `chips`; `canned good(s)` → `cannedgoods`;
`screw driver`→`screwdriver`; `green cabbage`→`cabbage`; `stainless`→`silver`;
strip trailing `right side up` and size words (`small/big/large/left/right`).
Bare color-less tokens (`cup`, `bowl`, `plush`, …) are dropped before matching.

## How to run
```bash
source emimic/bin/activate            # per AGENTS.md
# 1) audit DB columns / coverage to choose a strategy (read-only)
python -m egomimic.scripts.inspect_episode_metadata
# 2) build the pairs (downloads annotations to ~/.cache/egoverse_annotations)
python -m egomimic.scripts.pair_episodes_by_language \
    --out-json egomimic/scripts/human_robot_pairs.json
```
Outputs `human_robot_pairs.json` (`tiers.true_pairs_alignment`,
`tiers.similar_task_language_pairs`, `summary.best_match_object_overlap_histogram`).
Cache also holds readable digests under `~/.cache/egoverse_annotations/digests/`
(`*_bimanual.txt` = full action sequences; `*_objsets.txt` = object set per episode).

## Open questions / next steps
- **Tie-break the 53 three-shared matches** using action *order* from the full
  digests (`digests/*_bimanual.txt`), not just the object set.
- **Visual / latent NN** for true cross-modal pairing — pairs the `alignment
  set-2` cross-day episodes and the bulk where language is ambiguous. The cKDTree
  latent-KNN machinery on `elmo/inspector-knn-tree` (`eval_latent`,
  `PILatentEvalVideo`) is built for exactly this; consider indexing aria episodes
  and querying with eva (or annotation embeddings via the Qwen3-Embedding stem).
- If true same-scene pairs at scale are needed, either **annotate aria-alignment**
  episodes or define a co-located capture protocol and populate DB `scene`/
  `objects` for real (currently only Mecka episodes carry real `scene_id`/`objects`).
