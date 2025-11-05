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
- [``egoverse.md``](./egoverse.md): Instructions to upload data to S3 bucket
- [``vrs_upload.md``](./vrs_upload.md): Instructions to use a GUI to add metadata, rename and upload VRS files in a directory to S3 bucket

## Installation

# Conda
```
git clone --recursive git@github.com:GaTech-RL2/EgoVerse.git
cd EgoVerse
conda env create -f environment.yaml
conda activate emimic
pip install projectaria-tools'[all]'==1.5.7
pip install -e external/rldb
pip install -e external/rldb/external/lerobot
pip install -e .
```
# UV

if uv not installed
```
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/path/to/flash/storage" sh
```

```
uv venv emimic --python 3.10
source emimic/bin/activate
git clone git@github.com:GaTech-RL2/EgoVerse.git
cd EgoVerse
uv pip install -r requirements.txt
uv pip install -e external/lerobot
uv pip install -e .
```

Set `git config --global submodule.recurse true` if you want `git pull` to automatically update the submodule as well.
Set your wandb project in ``egomimic/hydra_configs/logger/wandb.yaml``

## Quick Start
### Processing your own data for training
![Data Streams](./assets/train_data.png)
See [``data_processing.md``](./data_processing.md)

### Pulling processed data for training from AWS
See [``training_aws.md``](./training_aws.md)

## Hydra Comands
`python egomimic/trainHydra.py`

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
