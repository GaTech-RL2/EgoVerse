# EgoVerse: Egocentric Data for Robot Learning from Around the World
![EgoVerse](./assets/egoverse.png)
This repository contains the data processing, training and evaluation code for EgoVerse.

This fork (`hackathon/diversity-dashboard`) adds **EgoSpectrum**: a visual interaction diversity score for egocentric demos, plus an equal-update HPT training protocol that tests whether that score changes learning. Live writeup: [egospectrum-dashboard.vercel.app](https://egospectrum-dashboard.vercel.app/).

---

## EgoSpectrum

Are you training on 100 distinct experiences, or repeated footage?

Robot datasets can look balanced on task labels and still be the same demo twice. EgoSpectrum turns **visual interaction diversity** into a number — no captions, no LLM-as-a-judge — then trains comparable HPT policies on subsets that differ only in which clips they keep. The dashboard is the public writeup; this section is the same argument with the math and protocol attached.

### Representation

Each episode is a single 512-D vector in CLIP image space.

1. Decode the preview MP4 and take **8 frames** at `linspace(0, T-1, 8)`.
2. Run frozen **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) on Modal T4s. Text tower is unused.
3. L2-normalize each frame feature, mean-pool, L2-normalize again. That mean is the episode.

```
x_i = normalize( (1/8) Σ_t normalize( CLIP_image(frame_{i,t}) ) ) ∈ R^{512}
```

Cosine similarity is then a dot product. The same embedding is used for selection and for scoring. PCA to 2-D is computed afterward for the dashboard map only; it is not an input to either step.

Implementations: [`modal_embed.py`](./modal_embed.py) (mixed 400-clip corpus, version `clip_v1_8frames`) and [`modal_embed_foldclothes_train.py`](./modal_embed_foldclothes_train.py) (fold-clothes train pool, version `foldclothes_train_clip_v1_8frames`).

### Scoring: coverage and repetition

A subset \(S\) of size \(k\) is judged against the full corpus \(C\) (\(|C|=n\)) by two cosine statistics.

**Coverage.** For every corpus episode, take similarity to its nearest selected neighbor, then average:

```
coverage(S) = (1/n) Σ_{i ∈ C} max_{j ∈ S} ⟨x_i, x_j⟩
```

High coverage means the subset still stands in for the population. Once \(k/n\) is large, almost any fair subset covers well. Random can even win this term by keeping a lot of typical footage.

**Repetition.** Mean pairwise similarity inside the subset (upper triangle, no diagonal):

```
repetition(S) = (2 / (k(k-1))) Σ_{a < b, a,b ∈ S} ⟨x_a, x_b⟩
```

Low repetition means fewer lookalikes in the clips you would actually train on. This is the term EgoSpectrum is trying to move.

Percentiles (`nearest_similarity_p10/p50/p90`, `pairwise_similarity_p95`) are logged for inspection. They are not in the score.

### Visual Interaction Diversity Score

Raw cosine deltas on this data are small — ego videos already sit in a tight CLIP blob (hands, rooms, objects, same camera). The published number is indexed so a typical random subset of the same size lands near **50**.

Let \(\bar{c}\) and \(\bar{r}\) be the mean coverage and repetition of **20** uniform random subsets of size \(k\) (seeds `0..19`):

```
Δ_c = (coverage(S) − c̄) / c̄
Δ_r = (r̄ − repetition(S)) / r̄
score(S) = clip( 50 + 500 · (0.5 Δ_c + 0.5 Δ_r), 0, 100 )
```

Equal weight on “represent the population” and “don’t keep clones.” A single random draw can land a hair above 50; that is expected. The seed-42 random subset shown on the dashboard is one such draw, not the 20-trial mean.

Code: [`analyze_diversity.py`](./analyze_diversity.py) (mixed corpus) and [`score_foldclothes_curation_variants.py`](./score_foldclothes_curation_variants.py) (fold-clothes). Same formula, different \(n\) and \(k\).

### Selection: farthest-first, not one distance per video

EgoSpectrum does not score a video in isolation. It builds a subset by greedy farthest-first in the normalized CLIP sphere.

Mixed-corpus version ([`analyze_diversity.py`](./analyze_diversity.py)): start from a seed-42 random episode. Repeatedly add the corpus point whose **max cosine to the set so far is smallest** (least covered so far). After each add, update

```
best_sim[i] ← max(best_sim[i], ⟨x_i, x_new⟩)
```

and pick `argmin(best_sim)` next. That is the standard greedy 1-center / farthest-first traversal of a cosine k-center heuristic.

Fold-clothes version ([`make_foldclothes_curation_variants.py`](./make_foldclothes_curation_variants.py)) is the same idea **within each task**: start at the first episode by `episode_hash` sort, then repeatedly add the point with largest `min (1 − ⟨x, x_chosen⟩)` to the set so far, 129 times per task. Task counts stay balanced by construction; visual clones inside a label do not.

### How to read the headline: similar coverage is not a tie

Mixed EgoVerse, 400 clips → two 100-clip subsets, 25 task types in the corpus.

| Subset | Score | Coverage | Repetition |
|---|---:|---:|---:|
| EgoSpectrum (farthest-first) | **62.8** | 0.9210 | 0.7442 |
| Random (seed 42) | 50.3 | 0.9278 | 0.7897 |

Coverage retained: \(0.9210 / 0.9278 = 99.3\%\). Lookalikes down: \((0.7897 - 0.7442) / 0.7897 = 5.8\%\). Both subsets still hit **24 of 25** task labels.

These ego videos already look related to CLIP. Once you keep 100 of 400, almost any fair subset covers the population. Random even covers a hair more, because it keeps typical footage. EgoSpectrum is not trying to cover more. It is trying to cover about as much with fewer repeats: keep the unusual angles, drop the near-duplicates. **62.8 vs 50.3 is that trade — same coverage, less wasted footage.**

The dashboard PCA map is a picture of the same 512-D cloud. Gray = corpus, warm = random, lime = EgoSpectrum. It is not the scoring engine.

### Why visual, not “within-task,” diversity

A common curation move is within-task diversity: keep 129 fold-shirt clips, 129 fold-jeans clips, and call the set diverse. That only proves the **labels** are balanced. It does not tell you whether those 129 shirts are the same table, the same fold, the same camera angle.

We treat **visual diversification** as the thing you can actually measure. CLIP sees the interaction, not the task string. Two subsets can have identical task counts and still waste the budget on lookalikes. That is a tangible difference, and it is why we train a policy at all: if visual spread is real, held-out action loss should move; if it is not, the score said so before the GPU hours.

### Fold-clothes: same metric, a real training corpus

Stress test: one activity family, six garment tasks, frozen EgoVerse split.

| Split | Episodes | Manifest | SHA-256 |
|---|---:|---|---|
| Train pool | 1,304 | `artifacts/foldclothes-v1/manifests/train.csv` | `c574843dc0f35c91bebdc84973aff789a9eda4837ab8ae2c71a3d12cde597da8` |
| Val (frozen) | 163 | `artifacts/foldclothes-v1/manifests/val.csv` | `4e41651a2f09f4b98c506dca9780809da5786d6186250e0cc3e866e76511ff3c` |
| Test (frozen) | 164 | `artifacts/foldclothes-v1/manifests/test.csv` | `5f7781b3d2c7bd5f784a843d0bebbc63bc2c62e31486ffe0113c29d6d8462d35` |

80/10/10 on the candidate pool, seed 42. Curation is **train-only**. Val and test are never touched. `train.csv` is byte-identical to `train_embedding_manifest.csv`. There are no val/test embedding manifests; embeddings exist only to select training episodes.

Three 774-episode variants (129 per source task, seed 42):

| Run | Selector | Manifest SHA-256 | Score | Coverage | Repetition |
|---|---|---|---:|---:|---:|
| `random-774` | `DataFrame.sample(n=129, random_state=42)` within task | `9e47194b6b62c4e6ac5bca4f3d276b502bbe2565834cc1395ba9978ca66c8c08` | 50.3 | 0.9793 | 0.8076 |
| `duration-balanced-774` | Sort by `num_frames`, split into tertiles, sample 43/43/43 per task | `60513d2d188424d27402f77fcf72345938d12dd9ebac9173bb542c189ce6ada3` | 50.3 | 0.9786 | 0.8071 |
| `diversity-774` | Within-task farthest-first in CLIP | `dd438636443971b6fe3d6220c7db663296c90ccc91cfb95fd07f3916879b356a` | **52.1** | 0.9824 | 0.8044 |

Source tasks: `fold_black_t-shirt`, `fold_blue_jeans`, `fold_clothes`, `fold_laundry`, `fold_shirt`, `fold_white_shirt`. All three variants have the same label histogram. Duration-balanced covers **length**, not vision. Scored against the 1,304 train pool with the same metric (random-774-sized 20-trial baseline ≈ 50; baseline coverage 0.9808, baseline repetition 0.8098).

Farthest-first still wins, but the gap is small because every clip is already garment folding. 774/1304 already covers the population; random can match coverage by keeping typical folds. EgoSpectrum’s edge is fewer lookalikes, not a bigger map. The score is doing its job: it reports a redundant pool instead of inventing a large win. `full-1304` is hashed and not trained in v1.

### From score to policy: equal-update HPT

Task-balanced subsets can still be visual clones. The only way to find out if visual diversification changes learning is to train. We run EgoVerse’s HPT “copy the hands” policy on each frozen 774-clip subset. Same model, same budget, only the 774 clips change.

**Model.** `hpt_bc_flow_human` — HPT flow-matching, human bimanual cartesian, image/action stride 3 (`Human.get_keymap` / `get_transform_list`, `keymap_mode: cartesian`). Not π0.5.

**Data.** [`ManifestEpisodeResolver`](./egomimic/rldb/zarr/manifest_resolver.py) reads `episode_hash` + `zarr_processed_path` from a CSV and syncs those zarrs from R2. No SQL. Env vars `FOLDCLOTHES_TRAIN_MANIFEST` and `FOLDCLOTHES_VAL_MANIFEST` select the variant. Val is always the frozen 163-episode `val.csv` (or `smoke_val.csv` for plumbing).

**Equal update budget.** Every 774-episode run gets the same optimizer trajectory, not the same number of epochs:

| Knob | Value |
|---|---|
| Seed | 42 |
| Batch size | 16 (train and val) |
| `max_steps` | **2000** (`max_epochs: -1`) |
| Precision | bf16 |
| Val interval | every 500 steps, `limit_val_batches: 40` |
| Checkpoint | `Valid/action_loss`, `mode: min`, `save_top_k: 1`, every 500 steps |
| Test | once, on the chosen checkpoint |
| Scheduler | `T_max: 2000` |
| Norm stats | quantile, `sample_frac: 0.2`, `reject_outliers: true` |

A weaker result cannot be blamed on fewer gradient steps. Full 1,304-episode training is out of scope; the question is whether the 774-clip selection changes held-out action loss.

**Eval.** [`HPTLossEval`](./egomimic/eval/eval_hpt_loss.py) runs `forward_training` / `compute_losses` and logs `Valid/action_loss` only. No videos. Metrics are written to `val_metrics.json` because Lightning eval mode does not keep the CSV logger.

**Compute.** [`modal_train_foldclothes.py`](./modal_train_foldclothes.py), app `egoverse-foldclothes-hpt`, volume `egoverse-hackathon-data`, secret `egoverse-r2`. Sync listed zarrs with s5cmd, then train `random-774` → `duration-balanced-774` → `diversity-774` sequentially. `modal deploy` + spawn so the job survives a laptop disconnect.

### Status

A 4-step smoke run on 24 train / 6 val episodes completed on Modal (`val_action_loss ≈ 139.41`, 132.4s). That is a plumbing check, not a comparison. The three full equal-budget jobs are in flight. Do not treat any number as a policy win until [`artifacts/foldclothes-v1/training_results.json`](./artifacts/foldclothes-v1/training_results.json) has finished `runs`.

### Reproduce

Embeddings and training run on Modal (volume `egoverse-hackathon-data`, secret `egoverse-r2`).

```bash
# Mixed 400-clip corpus → CLIP embeddings, then the 100-subset score
modal run modal_embed.py
python analyze_diversity.py

# Fold-clothes train-pool embeddings
modal run modal_embed_foldclothes_train.py

# Build the three 774-episode manifests (seed 42, 129 / task)
python make_foldclothes_curation_variants.py

# Score them against the 1,304 train pool
python score_foldclothes_curation_variants.py

# HPT smoke (24/6 episodes, 4 steps)
modal run modal_train_foldclothes.py --smoke

# Full three-run job (survives laptop disconnect)
modal deploy modal_train_foldclothes.py
```

Local training: `python egomimic/trainHydra.py --config-name=train_foldclothes_hpt` with `FOLDCLOTHES_TRAIN_MANIFEST` / `FOLDCLOTHES_VAL_MANIFEST` pointing at the CSVs.

Frozen hashes: [`artifacts/foldclothes-v1/manifests/SPLIT_SHA256SUMS.txt`](./artifacts/foldclothes-v1/manifests/SPLIT_SHA256SUMS.txt) and [`artifacts/foldclothes-v1/curation_experiment_plan.md`](./artifacts/foldclothes-v1/curation_experiment_plan.md). Talk track: [`artifacts/foldclothes-v1/PRESENTATION.md`](./artifacts/foldclothes-v1/PRESENTATION.md).

---

## Change Log
### EgoSpectrum visual diversity + equal-update HPT [08/15/2026]
- Added a CLIP-based Visual Interaction Diversity Score and farthest-first subset selection (`analyze_diversity.py`).
- Frozen fold-clothes-v1 splits (1,304 / 163 / 164) and three 774-episode curation variants (random, duration-balanced, diversity).
- Wired manifest-backed HPT training on Modal with an equal 2,000-step budget and action-loss eval. Smoke passed; full runs pending.
- Dashboard: [egospectrum-dashboard.vercel.app](https://egospectrum-dashboard.vercel.app/).

### Mandatory Camera Intrinsics + Human Embodiment Collapse [07/08/2026]
- Camera **intrinsics are now MANDATORY** in every episode's `zarr.json`, stored as a `{camera_key: 3×4 K}` dict (single-camera = one entry, e.g. `{"front_1": K}`). `ZarrWriter.create_and_write` raises if it is missing or not a non-empty dict. `extrinsics` (robots) is now strictly `None` or a non-empty dict.
- **Embodiments collapsed**: all human demonstration data is a single `human_*` embodiment (`human_right_arm`/`human_left_arm`/`human_bimanual`, ids 1–3); the robot Eva is `eva_*` (ids 4–6). Vendor labels (`aria_*`, `mecka_*`, `scale_*`, `lightwheel_*`) are **removed** at the embodiment level — the data source now lives only in the SQL `lab` field. Conversion scripts write `human_*`.
- **Aria processing — EE/wrist orientation fixed**: aria EE-pose and wrist-pose orientations are now correct and **consistent across both hands** — the left and right hand share the same canonical `T_ROT_CAM` convention, aligned with the robot (Eva) tool frame, so human and robot EE orientations line up for joint training / visualization. Re-process aria data to pick up the corrected orientations.
- **⚠️ Action required — RE-DOWNLOAD your data.** The embodiment string is stored inside each episode's `zarr.json`, and the reader looks it up with **no alias fallback** (`get_embodiment_id`): a locally cached episode written *before* the collapse still says `aria_bimanual` / `scale_bimanual` / `mecka_bimanual`, and loading it now **hard-crashes with `KeyError: 'ARIA_BIMANUAL'`**. Delete any local cache and re-pull the reprocessed episodes (they carry `human_*` + the mandatory intrinsics). Data *producers*: re-process and re-upload so `zarr.json` includes intrinsics — see [CONTRIBUTING_DATA.md](./CONTRIBUTING_DATA.md). Mecka and Scale will be asked to add intrinsics to their exports.

### Mecka Data Reprocessing [04/01/2026]
Mecka removed some poorer quality episodes and replaced them with higher quality alternatives.

### Scale Data Reprocessing [05/03/2026]
The Scale dataset was fully reprocessed on 05/03/2026. All active Scale episodes now use newly generated episode hashes, Zarr paths, and preview MP4 paths. If you previously referenced Scale episode hashes from an older export or intermediate processing run, refresh from the SQL episode table before downloading or training. Old Scale hashes should be treated as stale and should not be mixed with the current active dataset.

---

## Structure
- [``egomimic/trainHydra.py``](./egomimic/trainHydra.py): Main training script, powered by Pytorch Lightning and Hydra (DDP supported)
- [``egomimic/hydra_configs``](./egomimic/hydra_configs): Train configs for each algorithm
- [``egomimic/algo``](./egomimic/algo): Algorithm code: ACT, EgoMimic (HPT based), Pi
- [``egomimic/scripts/aloha_process``](./egomimic/scripts/aloha_process/): Process raw aloha hdf5 to zarr/lerobot
- [``egomimic/scripts/aria_process``](./egomimic/scripts/aria_process/): Process aria vrs to zarr/lerobot

## Installation

### UV (Recommended)

if uv not installed
```
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/path/to/flash/storage" sh
```

```
git clone git@github.com:GaTech-RL2/EgoVerse.git
cd EgoVerse
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv run pre-commit install
```

### Conda
```
git clone --recursive git@github.com:GaTech-RL2/EgoVerse.git
cd EgoVerse
conda env create -f environment.yaml
conda activate emimic
pip install -e .
pre-commit install
```

### AWS Configure
Download the AWS cli
```
 curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
 unzip awscliv2.zip
 ./aws/install -i ~/aws-cli -b ~/bin
```

Set up your AWS keys to access our cloud storage
```
aws configure
AccessKeyId: AKIAYDKH4BNCAYHE5NG2
SecretAccessKey: rGjT6NSh55YiB9MC9EyNGpVy8qcaTn4i19OmkhRW
Default region name: us-east-2
Default output format:
./egomimic/utils/aws/setup_secret.sh
```
`setup_secret.sh` will allow your current env to download data from cloudflare.


### Other Settings
Set `git config --global submodule.recurse true` if you want `git pull` to automatically update the submodule as well.
Set your wandb project in ``egomimic/hydra_configs/logger/wandb.yaml``

## Submitit modification
For the integrated hydra submitit plugin to work, make the following modification...

`/path/to/your/venv/emimic/lib/python3.11/site-packages/hydra_plugins/hydra_submitit_launcher/submitit_launcher.py`

Change line 144 to
```
        jobs = executor.map_array(self, *zip(*job_params))

        return [asyncLauncher() for j in jobs]

class asyncLauncher:
    def __init__(self):
        self.return_value = 0
```

I wanted to package this change nicely, but the hydra package is built very weirdly.

## Quick Start Guide
### Data Visualization
Visit https://partners.mecka.ai/egoverse to view our entire dataset in the web!

To visualize data programatically see [``zarr_data_viz.ipynb``](./egomimic/scripts/tutorials/zarr_data_viz.ipynb)

To programatically view the SQL table of all episodes + metadata see [``sql_tutorial.ipynb``](./egomimic/scripts/tutorials//sql_tutorial.ipynb)

#### Interactive Dataset Browser

`latent_inspector.py` also ships a local web app for browsing a **folder of per-episode zarrs** — scrub any episode, overlay the recorded actions (cartesian trajectory / orientation axes / MANO keypoints), and toggle language annotations. Frames are rendered server-side using each episode's `zarr.json` camera intrinsics, so it works for any embodiment (the overlay is drawn by that episode's embodiment class, e.g. `Human`/`Eva`; human poses are projected in the head frame).

```bash
python egomimic/scripts/data_visualization/latent_inspector.py \
    --dataset-path /path/to/folder_of_zarrs \
    --host 127.0.0.1 --port 8050
# then open http://localhost:8050
```

`--dataset-path` is a directory of `<episode>.zarr` stores (each with `images.front_1`, `left/right.obs_ee_pose`, `obs_head_pose`, optional `*.obs_keypoints` and `annotations`, and `intrinsics` in its `zarr.json`). In the browser: pick an episode (searchable by filename or annotation text), scrub the frame slider or press ▶ to play, choose an overlay (None / Cartesian / Orientation / Keypoints), and toggle annotations on/off.

To browse data on a remote machine, run the app there and forward the port — `ssh -L 8050:<node>:8050 <host>` — then open `http://localhost:8050` (rendering locally is far more responsive than over the tunnel).

### Data Downloading
While our training pipeline automatically downloads data, you can manually download data via [``sync_s3.py``](./egomimic/scripts/data_download/sync_s3.py)

For example, to download all our flagship Aria fold clothes data...
```
python egomimic/scripts/data_download/sync_s3.py \
     --local-dir <local directory> \
     --filters aria-fold-clothes
```

### Training
Basic training run (robot BC)...
``` bash
python egomimic/trainHydra.py --config-name=train_zarr_cartesian
```
For full instructions on training see [``training.md``](./training.md)

### Converting your own data
See [``embodiment_tutorial.ipynb``](./egomimic/scripts/tutorials/embodiment_tutorial.ipynb) as reference to write a conversion script for your own data.
