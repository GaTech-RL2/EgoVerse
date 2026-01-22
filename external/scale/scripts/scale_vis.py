# IMPORTS
from egomimic.rldb.utils import *
import torch
import numpy as np
from egomimic.utils.egomimicUtils import CameraTransforms, draw_actions
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
import torchvision.io as io
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scale_root = "/nethome/agao81/flash/EgoVerse/external/scale/scripts/dataset"
scale_repo_id = "scale/dataset"

episodes = [0]
dataset = RLDBDataset(
    repo_id=scale_repo_id,
    root=scale_root,
    local_files_only= True,
    episodes = episodes,
    mode="sample",
    tolerance_s=2e-2,
    )

image_key = "observations.images.front_img_1"
actions_key = "actions_ee_cartesian_cam"

print(dataset.embodiment)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
camera_transforms = CameraTransforms(intrinsics_key="scale", extrinsics_key="x5Nov18_3")

from egomimic.utils.egomimicUtils import *
def draw_actions_both(im,color, actions, intrinsics):
    #print(actions.shape)
    actions_left = actions[:,:3]
    actions_right = actions[:,6:9]
    #print(actions_left.shape)
    #print(actions_right.shape)
    actions_drawable_left = cam_frame_to_cam_pixels(actions_left, intrinsics)
    actions_drawable_right = cam_frame_to_cam_pixels(actions_right, intrinsics)
    im = draw_dot_on_frame(im, actions_drawable_left, show=False, palette=color)
    im = draw_dot_on_frame(im, actions_drawable_right, show=False, palette=color)
    return im


def visualize_actions(ims, actions, extrinsics, intrinsics, arm="both"):
    for b in range(ims.shape[0]):
        if actions.shape[-1] == 7 or actions.shape[-1] == 14:
            ac_type = "joints"
            ims[b] = draw_actions(
                ims[b], ac_type, "Purples", actions[b], extrinsics, intrinsics, arm=arm
            )
        elif actions.shape[-1] == 3 or actions.shape[-1] == 6:
            ac_type = "xyz"
            ims[b] = draw_actions(
                ims[b], ac_type, "Purples", actions[b], extrinsics, intrinsics, arm=arm
            )
        elif actions.shape[-1] == 12:
            ac_type = "xyz"
            ims[b] = draw_actions_both(
                ims[b], "Purples", actions[b], intrinsics
            )

        else:
            raise ValueError(f"Unknown action type with shape {actions.shape}")
     

    return ims

save_dir = "./visualization/"
os.makedirs(save_dir, exist_ok=True)

#num_batches = 10
num_batches = 10  # Check more batches to see variation
every_n_batches = 10  # Sample every 10th frame for more visible difference
cur_batch = 0
print(len(data_loader))
prevdata = None
for i, data in enumerate(data_loader):
    if i % every_n_batches != 0:
        continue

    print(data["annotations"])
    print(f"\nBatch {i} (frame {i}):")
    print(f"  Images equal to previous: {torch.equal(data[image_key], prevdata[image_key]) if prevdata is not None else 'First batch'}")
    
    # Check raw image statistics
    img_tensor = data[image_key][0]  # First image in batch
    print(f"  Raw image stats - min: {img_tensor.min():.4f}, max: {img_tensor.max():.4f}, mean: {img_tensor.mean():.4f}")
    print(f"  First pixel [0,0]: {img_tensor[:, 0, 0]}")  # Print first pixel RGB values
    
    prevdata = data
    if cur_batch > num_batches:
        break
    ims = (data[image_key].permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    
    # Check processed image before visualization
    print(f"  Processed image shape: {ims.shape}, first pixel: {ims[0, 0, 0, :]}")
    
    actions = data[actions_key].cpu().numpy()
    #print(actions_key)
    #print(actions[:10, :])
    #print(actions[:, :3].shape)
    ims_viz = visualize_actions(ims, actions[:, :], camera_transforms.extrinsics, camera_transforms.intrinsics)
    
    # Check visualized image
    print(f"  After viz first pixel: {ims_viz[0, 0, 0, :]}")
    
    #print(ims_viz.shape)
    for j, im in enumerate(ims_viz):
        img_tensor = torch.from_numpy(im).permute(2, 0, 1)
        save_path = os.path.join(save_dir, f"image_{i}_{j}.png")
        io.write_png(img_tensor, save_path)

    print(f"Saved batch {i} images to {save_dir}")
    cur_batch += 1