import os
import subprocess
import boto3
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter.filedialog import askdirectory
import pandas as pd
from pathlib import Path
import tempfile

# AWS S3 setup
s3 = boto3.client("s3")
bucket_name = "rldb"
task_map_key = "raw/task_map.csv"

# Download existing task_map.csv (if any)
task_map_path = Path(tempfile.gettempdir()) / "task_map.csv"
try:
    s3.download_file(bucket_name, task_map_key, str(task_map_path))
    task_map_df = pd.read_csv(task_map_path)
except s3.exceptions.ClientError as e:
    task_map_df = pd.DataFrame(columns=["task", "arm"])  # Create if doesn't exist

# Root window for simpledialog prompts
tk_root = tk.Tk()
tk_root.withdraw()

# Ask user for required information
task_name = simpledialog.askstring("Task Name", "Enter task name (e.g., pour_milk):")
if not task_name:
    raise ValueError("Task name is required.")
s3_base_path = f"s3://{bucket_name}/raw/{task_name}/"

# Handle task_map: load arm if known, else prompt and update map
if task_name in task_map_df["task"].values:
    arm = task_map_df.loc[task_map_df["task"] == task_name, "arm"].iloc[0]
    print(f"Found arm '{arm}' for task '{task_name}' in task_map.csv.")
else:
    arm = simpledialog.askstring(
        "Arm Type",
        f"Task '{task_name}' not in task_map.csv. Enter arm type (e.g., left, right):",
    )
    if not arm:
        raise ValueError("Arm type is required for new tasks.")
    new_row = pd.DataFrame([{"task": task_name, "arm": arm}])
    task_map_df = pd.concat([task_map_df, new_row], ignore_index=True)
    task_map_df.to_csv(task_map_path, index=False)
    s3.upload_file(str(task_map_path), bucket_name, task_map_key)
    print(f"Updated task_map.csv with task '{task_name}' and arm '{arm}'.")

# Ask user for folder
folder_path = askdirectory(title="Select folder containing .vrs and .vrs.json files")
if not folder_path:
    raise ValueError("Folder selection is required.")
folder_path = Path(folder_path)

# Ask for custom metadata fields
custom_keys_str = simpledialog.askstring(
    "Custom Metadata", "Enter additional metadata keys (comma-separated):"
)
custom_keys = [k.strip() for k in custom_keys_str.split(",")] if custom_keys_str else []

# Default metadata fields
default_keys = ["collector", "lab", "scene", "recording_number"]
all_keys = default_keys + custom_keys

# List to track files for deferred upload
upload_queue = []

# Process each .vrs file
vrs_files = sorted(folder_path.glob("*.vrs"))
for vrs_file in vrs_files:
    json_file = vrs_file.with_suffix(".vrs.json")
    if not json_file.exists():
        print(f"Missing .vrs.json for {vrs_file}, skipping.")
        continue

    # Launch vrsplayer
    player_proc = subprocess.Popen(["vrsplayer", str(vrs_file)])

    # GUI for metadata input
    metadata_window = tk.Tk()
    metadata_window.title(f"Metadata for {vrs_file.name}")
    entries = {}
    submitted_metadata = {}

    def submit_and_close():
        for key in all_keys:
            submitted_metadata[key] = entries[key].get()
        metadata_window.quit()

    for idx, key in enumerate(all_keys):
        tk.Label(metadata_window, text=key).grid(row=idx, column=0)
        e = tk.Entry(metadata_window, width=40)
        e.grid(row=idx, column=1)
        entries[key] = e

    tk.Button(metadata_window, text="Submit", command=submit_and_close).grid(
        row=len(all_keys), columnspan=2
    )

    metadata_window.mainloop()
    metadata_window.destroy()
    player_proc.terminate()

    metadata = submitted_metadata
    metadata["arm"] = arm
    metadata["task"] = task_name
    lab = metadata["lab"]
    scene = metadata["scene"]
    rec = metadata["recording_number"]
    new_name_base = f"{task_name}_{lab}_{scene}_recording_{rec}"

    s3_vrs_key = f"raw/{task_name}/{new_name_base}.vrs"
    s3_json_key = f"raw/{task_name}/{new_name_base}.vrs.json"
    s3_meta_key = f"raw/{task_name}/{new_name_base}_meta.csv"

    # os.rename(vrs_file, renamed_vrs)
    # print(f"Renamed {vrs_file} to {renamed_vrs}")
    # os.rename(json_file, renamed_json)
    # print(f"Renamed {json_file} to {renamed_json}")
    metadata_csv = folder_path / f"{new_name_base}_meta.csv"
    pd.DataFrame([metadata]).to_csv(metadata_csv, index=False)
    print(f"Saved metadata to {metadata_csv}")

    # instead of: upload_queue.extend([renamed_vrs, renamed_json, metadata_csv])
    upload_queue.append({"original_path": str(vrs_file), "s3_key": s3_vrs_key})
    upload_queue.append({"original_path": str(json_file), "s3_key": s3_json_key})
    upload_queue.append({"original_path": str(metadata_csv), "s3_key": s3_meta_key})


# Upload to S3
print("\nUploading all processed files to S3...")
for item in upload_queue:
    print(
        f"Uploading {Path(item['original_path']).name} to s3://{bucket_name}/{item['s3_key']}"
    )
    s3.upload_file(item["original_path"], bucket_name, item["s3_key"])

messagebox.showinfo("Done", "All VRS files processed and uploaded.")
