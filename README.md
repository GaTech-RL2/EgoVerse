# EgoVerse: Egocentric Data for Robot Learning from Around the World
![EgoVerse](./assets/egoverse.png)
This repository contains the data processing, training and evaluation code for EgoVerse.

This fork (`hackathon/diversity-dashboard`) adds **EgoSpectrum**: a visual interaction diversity score for egocentric demos, plus an equal-update HPT training protocol that tests whether that score changes learning. Live writeup: [egospectrum-dashboard.vercel.app](https://egospectrum-dashboard.vercel.app/).

---

## EgoSpectrum

Robot datasets can look balanced on task labels and still be the same demo twice. EgoSpectrum measures **visual** diversity — no captions, no LLM-as-a-judge — then trains comparable HPT policies on subsets that differ only in which clips they keep.

### Method

1. Sample 8 evenly spaced egocentric frames per episode.
2. Embed them with frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) on Modal.
3. Mean-pool and L2-normalize to a 512-D episode vector.
4. Select a subset with greedy farthest-first in that space (seed 42).
5. Score the subset on two quantities, computed in the original 512-D space (PCA is only a map):
   - **Coverage** — mean similarity of every corpus clip to its nearest selected neighbor. Higher means the subset still stands in for the population.
   - **Repetition** — mean pairwise similarity inside the subset. Lower means fewer lookalikes.
6. Index the score so a typical random subset of the same size lands near **50**:

```
score = 50 + 500 * (0.5 * coverage_delta + 0.5 * repetition_delta)
```

`coverage_delta` and `repetition_delta` are relative to the mean of 20 random subsets. Random can score a hair above 50 on a single draw; that is expected.

This is **not** within-task label balance. A set can keep 129 fold-shirt clips and still be the same table, the same fold, and the same camera angle. CLIP sees the interaction, not the task string.

### Headline result: mixed EgoVerse, 400 → 100

`analyze_diversity.py` compares two 100-episode subsets of a 400-clip mixed corpus (25 task types).

| Subset | Score | Coverage | Repetition |
|---|---:|---:|---:|
| EgoSpectrum (farthest-first) | **62.8** | 0.9210 | 0.7442 |
| Random (seed 42) | 50.3 | 0.9278 | 0.7897 |

Coverage is essentially tied (EgoSpectrum keeps 99.3% of random coverage). Both subsets still hit 24 of 25 task labels. The win is **5.8% fewer lookalikes** in the 100 clips you would actually train on. Similar coverage is not a tie.

### Training experiment: fold-clothes, 1,304 → 774

Fold-clothes is the stress test: six garment tasks, one activity family, so visual clones are the default.

| Split | Episodes | Manifest | SHA-256 |
|---|---:|---|---|
| Train pool | 1,304 | `artifacts/foldclothes-v1/manifests/train.csv` | `c574843dc0f35c91bebdc84973aff789a9eda4837ab8ae2c71a3d12cde597da8` |
| Val (frozen) | 163 | `artifacts/foldclothes-v1/manifests/val.csv` | `4e41651a2f09f4b98c506dca9780809da5786d6186250e0cc3e866e76511ff3c` |
| Test (frozen) | 164 | `artifacts/foldclothes-v1/manifests/test.csv` | `5f7781b3d2c7bd5f784a843d0bebbc63bc2c62e31486ffe0113c29d6d8462d35` |

Curation is **train-only**. Val and test never change. `train.csv` is byte-identical to `train_embedding_manifest.csv`.

Three 774-episode variants, 129 episodes per source task, seed 42:

| Run | How it is chosen | Manifest SHA-256 | Score | Coverage | Repetition |
|---|---|---|---:|---:|---:|
| `random-774` | Seeded random within task | `9e47194b6b62c4e6ac5bca4f3d276b502bbe2565834cc1395ba9978ca66c8c08` | 50.3 | 0.9793 | 0.8076 |
| `duration-balanced-774` | Seeded duration tertiles within task | `60513d2d188424d27402f77fcf72345938d12dd9ebac9173bb542c189ce6ada3` | 50.3 | 0.9786 | 0.8071 |
| `diversity-774` | Farthest-first in CLIP space within task | `dd438636443971b6fe3d6220c7db663296c90ccc91cfb95fd07f3916879b356a` | **52.1** | 0.9824 | 0.8044 |

All three sets are label-balanced. Duration-balanced covers length, not vision. EgoSpectrum still wins, but the gap is small because every clip is already garment folding. That is the point: the score reports a redundant pool instead of inventing a large map. Full 1,304-episode training is recorded as `full-1304` and is out of scope for v1.

### Equal-update HPT protocol

The diversity number is not the efficacy test. Held-out action loss is. We train EgoVerse’s HPT “copy the hands” policy (`hpt_bc_flow_human`) on each 774-clip set with everything else locked.

Held fixed:

- Human bimanual cartesian, stride 3
- Seed 42, batch size 16, AdamW, bf16
- **2,000 optimizer steps** for every run (`max_steps: 2000`, `max_epochs: -1`)
- Val every 500 steps, `limit_val_batches: 40`
- Frozen val (163) and test (164) manifests
- Checkpoint only by `Valid/action_loss`
- Test evaluated once on the chosen checkpoint
- Manifest-backed loader: `egomimic/rldb/zarr/manifest_resolver.py` (CSV, no SQL)

Allowed to change: which 774 train episodes are in the manifest.

Eval writes `val_metrics.json` via `egomimic/eval/eval_hpt_loss.py` (action loss only, no videos).

### Status

A 4-step smoke run on 24 train / 6 val episodes completed on Modal (`val_action_loss ≈ 139`, ~132s). That is plumbing, not a comparison. The three full equal-budget jobs are launched from `modal_train_foldclothes.py` against R2 zarrs. Do not treat any number as a policy win until `artifacts/foldclothes-v1/training_results.json` has finished runs.

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

Local training uses `FOLDCLOTHES_TRAIN_MANIFEST` and `FOLDCLOTHES_VAL_MANIFEST` with `egomimic/hydra_configs/train_foldclothes_hpt.yaml`.

### Key files

| Path | Role |
|---|---|
| [analyze_diversity.py](./analyze_diversity.py) | Mixed-corpus farthest-first + Visual Interaction Diversity Score |
| [modal_embed.py](./modal_embed.py) | CLIP embeddings for the 400-clip headline set |
| [modal_embed_foldclothes_train.py](./modal_embed_foldclothes_train.py) | CLIP embeddings for the 1,304 train pool |
| [make_foldclothes_curation_variants.py](./make_foldclothes_curation_variants.py) | random / duration / diversity 774 manifests |
| [score_foldclothes_curation_variants.py](./score_foldclothes_curation_variants.py) | Same metric on the three training subsets |
| [modal_train_foldclothes.py](./modal_train_foldclothes.py) | Modal HPT trainer, smoke + three equal-budget runs |
| [egomimic/hydra_configs/train_foldclothes_hpt.yaml](./egomimic/hydra_configs/train_foldclothes_hpt.yaml) | Hydra entry (HPT, 2000 steps, CSV logger) |
| [egomimic/rldb/zarr/manifest_resolver.py](./egomimic/rldb/zarr/manifest_resolver.py) | Episode-hash CSV resolver |
| [egomimic/eval/eval_hpt_loss.py](./egomimic/eval/eval_hpt_loss.py) | Val/test action-loss eval |
| [artifacts/foldclothes-v1/curation_experiment_plan.md](./artifacts/foldclothes-v1/curation_experiment_plan.md) | Frozen hashes and invariants |
| [artifacts/foldclothes-v1/PRESENTATION.md](./artifacts/foldclothes-v1/PRESENTATION.md) | 5-minute talk track |

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
