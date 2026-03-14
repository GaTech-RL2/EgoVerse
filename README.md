# EgoVerse: Egocentric Data for Robot Learning from Around the World
![EgoVerse](./assets/egoverse.png)
This repository contains the data processing, training and evaluation code for EgoVerse.

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
pip install projectaria-tools'[all]'
pip install -e external/lerobot
pip install -e .
pre-commit install
```

### AWS Configure
```
aws configure
<fill in credentials simar sent>
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
python egomimic/trainHydra.py --config-name=train_zarr
```
For full instructions on training see [``training.md``](./training.md)

### Converting your own data
See [``embodiment_tutorial.ipynb``](./egomimic/scripts/tutorials/embodiment_tutorial.ipynb) as reference to write a conversion script for your own data.
