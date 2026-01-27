# EgoVerse Dataset Setup and Training Guide

This guide provides step-by-step instructions for setting up the dataset and training a model in the **EgoVerse** repository.

---

## 1. Setting Up the Data Directory

Start by navigating out of the **EgoVerse** repository and creating a `data` directory:

```bash
cd ..
mkdir data
cd data
```

---

## 2. Downloading the Processed Data

Download the processed dataset from AWS S3:

```bash
aws s3 cp s3://rldb/processed/{processed_data_directory}/ {my_processed_data_directory} --recursive
```

Replace `{processed_data_directory}` with the name of the dataset you want to download, and `{my_processed_data_directory}` with your desired local directory name.

---

## 3. Modifying Configuration Files

Once the dataset is downloaded, navigate back to the **EgoVerse** repository:

```bash
cd ..
cd EgoVerse
```

### **Modify `multi-data.yaml`**
Open the configuration file located at:  
📌 **`hydra_configs/data/multi_data.yaml`**

Update the following segments to match your local dataset path:

```yaml
train_datasets:
  dataset1:
    _target_: rldb.utils.RLDBDataset
    repo_id: "egoverse/smallShirtFold"
    root: "{path/to/data/my_processed_data_directory/lerobot}"
    local_files_only: true
    mode: "train"

valid_datasets:
  dataset1:
    _target_: rldb.utils.RLDBDataset
    repo_id: "egoverse/smallShirtFold"
    root: "{path/to/data/my_processed_data_directory/lerobot}"
    local_files_only: true
    mode: "valid"
```
🔹 Replace `{path/to/data/my_processed_data_directory/lerobot}` with the actual path to the lerobot folder in your downloaded dataset.

---

### **Modify `train.yaml`**
Open the configuration file located at:  
📌 **`hydra_configs/train.yaml`**

Modify the **data schematic** section as follows:

```yaml
data_schematic: # Dynamically fill in these shapes from the dataset
  _target_: rldb.utils.DataSchematic
  schematic_dict:
    aria_bimanual:
      front_img_1:
        key_type: camera_keys
        lerobot_key: observations.images.front_img_1
      ee_pose:
        key_type: proprio_keys
        lerobot_key: observations.state.ee_pose
      actions_cartesian:
        key_type: action_keys
        lerobot_key: actions_cartesian
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
  viz_img_key:
    aria_bimanual:
      front_img_1
```

---

## 4. Launch Training

### **Activate Your Environment**
Before running training, activate your **Conda** or **UV** environment:

```bash
conda activate <your_env_name>  # If using Conda
```
or
```bash
uv venv <your_env_name> && source <your_env_name>/bin/activate  # If using UV
```

### **Run Training on a GPU Node**
Execute the training script:

```bash
python trainHydra.py
```
---

