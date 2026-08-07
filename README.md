# EgoVerse: Egocentric Data for Robot Learning from Around the World
![EgoVerse](./assets/egoverse.png)
This repository contains the data processing, training and evaluation code for EgoVerse.

---

## Change Log
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
