# EgoVerse: Egocentric Data for Robot Learning from Around the World
![EgoVerse](./assets/egoverse.png)
This repository contains the data processing and training code for EgoVerse and a refactored pipeline for training multi-embodiment BC policies and rolling them out.

---

## Structure
- [``egomimic/scripts/aloha_process``](./egomimic/scripts/aloha_process/): Process raw aloha style data into a robomimic style hdf5 or compressed efficient RLDB parquet files, compatible for training here.
- [``egomimic/scripts/aria_process``](./egomimic/scripts/aria_process/): Process human embodiment data from Aria Glasses into a robomimic style hdf5, or compressed efficient RLDB parquet files.
- [``egomimic/algo``](./egomimic/algo): Algorithm code for EgoMimic, ACT and HPT
- [``egomimic/hydra_configs``](./egomimic/hydra_configs): Train configs for each algorithm
- [``egomimic/trainHydra.py``](./egomimic/trainHydra.py): Main training script, powered by Pytorch Lightning and Hydra (DDP enabled)
- [``data_processing.md``](./data_processing.md): Instructions to process your own data, both Aria Human data and teleoperated robot data.
- [``egomimic/evaluation``](./egomimic/evaluation/): Evaluation scripts
- [``data_upload.md``](./data_upload.md): Instructions to upload data (any type) to S3 bucket

## Installation

# UV

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

# Conda
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

Set `git config --global submodule.recurse true` if you want `git pull` to automatically update the submodule as well.
Set your wandb project in ``egomimic/hydra_configs/logger/wandb.yaml``

## Quick Start
### Processing your own data for training
![Data Streams](./assets/train_data.png)
See [``data_processing.md``](./data_processing.md)

## Hydra Comands
### Quick start
If you want to quickly train a robot BC policy using the default Hydra configuration on Eva robot data, simply run:

`python egomimic/trainHydra.py --config-name train_zarr.yaml`

Important:
The default EVA BC config pulls data from S3 to a local scratch directory where the path is set to my local path
Before running, open:

`EgoVerse/egomimic/hydra_configs/data/eva_bc_s3.yaml`

and modify:

`temp_root: /path/to/where/you/want/s3/data/stored`

to point to a local disk location with sufficient space for caching the downloaded dataset.

### Additional Options
Debug (run on a compute node )
`python egomimic/trainHydra.py trainer=debug logger=debug`

Submitit (Run this on slurm)
`python egomimic/trainHydra.py -m launch_params.gpus_per_node=<gpus per node> launch_params.nodes=<nodes> name=<name> description=<>`

Eval (add your own rollout class in [``egomimic/evaluation``](./egomimic/evaluation/) and update [``egomimic/hydra_configs/train.yaml``](./egomimic/hydra_configs/train.yaml))
`python egomimic/trainHydra.py train=false eval=true`

## Add your embodiment
See [``model.md``](./model.md)

## Submitit modification
Tip: after you launch via submitit, you'll notice that the command won't finish executing.  If you want it to end the command after you launch a job, edit the following file

`/path/to/your/miniconda3/envs/emimic/lib/python3.10/site-packages/hydra_plugins/hydra_submitit_launcher/submitit_launcher.py`

Change line 144 to
```
        jobs = executor.map_array(self, *zip(*job_params))

        return [asyncLauncher() for j in jobs]

class asyncLauncher:
    def __init__(self):
        self.return_value = 0
```

I wanted to package this change nicely, but the hydra package is built very weirdly.  I tried to pip install -e . locally but the plugins package doesn't install correctly.  I'll try to PR this change into the main repo
