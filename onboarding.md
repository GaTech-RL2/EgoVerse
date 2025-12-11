# Onboarding to EgoVerse

Welcome to **EgoVerse**! This onboarding guide will walk you through the steps to get started using **Project Aria Glasses** for robot learning within our codebase.

Please make sure to follow the setup instructions in the main [README](readme.md) before proceeding.

---

## Step 0: Setup AWS, Repo, and Credentials

> **Status:** 🔧 _Pending_  

---

## Step 1: Human Data (Aria Glasses)

1. Use your **Aria Glasses** to collect **3–5 `.vrs` files**.
2. Follow instructions in [`data_processing.md`](data_processing.md) to convert `.vrs` files into our training format.

---

## Step 2: Robot Data

1. Refer to the sample conversion script:  
   [`egomimic/scripts/aloha_process/aloha_to_lerobot.py`](egomimic/scripts/aloha_process/aloha_to_lerobot.py)
2. Write your own conversion script to convert your robot's native data format (e.g., `.hdf5`) to our training format.

After completing Steps 1 and 2, visualize your data using:  
[`egomimic/scripts/data_visualization.ipynb`](egomimic/scripts/data_visualization.ipynb)

---

## Step 3: Robot BC Training and Evaluation

1. Follow [`model.md`](model.md) to configure:
   - `egomimic/hydra_configs/model/hpt_robot.py`
   - `egomimic/hydra_configs/data/multi_data.yaml`
2. Launch training using the **Hydra commands** from the [main README](readme.md).
3. Implement your own evaluation script (e.g., `eval_{your_robot}.py`) following the structure of:  
   [`egomimic/scripts/evaluation/eval_eve.py`](egomimic/scripts/evaluation/eval_eve.py)

---

## Step 4: Co-Training with Human and Robot Data

For the **Object in Bowl** task:

1. Collect ~1 hour of robot demonstrations and process it as before.
2. Use [`training_aws.md`](training_aws.md) to:
   - Pull processed human data from AWS for the task.
3. Override the `@visualize_preds` function in:  
   [`egomimic/algo/hpt.py`](egomimic/algo/hpt.py)  
   to enable robot-specific validation visualization.
4. Follow [`model.md`](model.md) to configure co-training via:  
   `egomimic/hydra_configs/hpt.yaml`
5. Run evaluations to compare:
   - **Robot-only BC**
   - **Co-trained (Human + Robot)** models

---

## 🎉 You're All Set!

By completing these steps, you should be well-equipped to navigate and contribute to the **EgoVerse** codebase. Welcome aboard!
