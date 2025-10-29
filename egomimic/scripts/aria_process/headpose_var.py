import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor

def headpose_var(csv_path):
    """Compute variance of head poses from CSV files in the given directory.
    Args:
        csv_path (str): Path to the directory containing subdirectories with CSV files.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Variance of translations and Euler angles (yaw, pitch, roll).
    """
    os.chdir(csv_path)
    data_paths = [f for f in os.listdir(".") if os.path.isdir(f)]
    csv_paths = [os.path.join(path, "slam/closed_loop_trajectory.csv") for path in data_paths]
    csv_paths = [p for p in csv_paths if os.path.exists(p)]

    if len(csv_paths) == 0:
        print("No valid data found.")
        return None, None

    all_translations = []
    all_rotations = []

    def load_csv(path):
        df = pd.read_csv(path)
        translation = df.iloc[:, 3:6].values
        rotation = df.iloc[:, 6:10].values
        return translation, rotation

    # Parallel loading
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, (translation, rotation) in enumerate(executor.map(load_csv, csv_paths)):
            all_translations.append(translation)
            all_rotations.append(rotation)
            if (i + 1) % 10 == 0 or (i + 1) == len(csv_paths):
                print(f"Processed {i+1}/{len(csv_paths)} paths")

    all_translations = torch.tensor(np.concatenate(all_translations, axis=0)).float().cuda()
    all_rotations = torch.tensor(np.concatenate(all_rotations, axis=0)).float().cuda()

    translation_var = torch.var(all_translations, axis=0)

    euler = R.from_quat(all_rotations.cpu().numpy()).as_euler('zyx', degrees=True)
    euler_var = torch.var(torch.tensor(euler).cuda(), axis=0)

    return translation_var.cpu().numpy(), euler_var.cpu().numpy()
