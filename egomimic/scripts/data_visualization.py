# IMPORTS
from egomimic.rldb.utils import *
import torch
import numpy as np
from egomimic.utils.egomimicUtils import CameraTransforms, draw_actions, cam_frame_to_base_frame, base_frame_to_cam_frame
import torchvision.io as io
import os

# Load dataset
root = "/home/robot/robot_ws/lerobot_data/lerobot_test"
repo_id = "arx_test_3"

episodes = [0, 1]
dataset = RLDBDataset(repo_id=repo_id, root=root, local_files_only=True, episodes=episodes, mode="sample")

print(dataset.meta.info["features"])

image_key = "observations.images.front_img_1"
actions_key = "actions_cartesian"

print(dataset.embodiment)

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False)

camera_transforms = CameraTransforms(intrinsics_key="base", extrinsics_key="x5Nov18_3")

def visualize_actions(ims, actions, extrinsics, intrinsics, arm="both"):
    for b in range(ims.shape[0]):
        if actions.shape[-1] == 7 or actions.shape[-1] == 14:
            ac_type = "joints"
        elif actions.shape[-1] == 3 or actions.shape[-1] == 6:
            ac_type = "xyz"
        else:
            raise ValueError(f"Unknown action type with shape {actions.shape}")

        # arm = "right" if actions.shape[-1] == 7 or actions.shape[-1] == 3 else "both"
        ims[b] = draw_actions(ims[b], ac_type, "Purples", actions[b], extrinsics, intrinsics, arm=arm)

    return ims

save_dir = "./visualization/"
os.makedirs(save_dir, exist_ok=True)

num_batches = 4
every_n_batches = 50 
cur_batch = 0
for i, data in enumerate(data_loader):

    if cur_batch > num_batches:
        break
    if i % every_n_batches != 0:
        continue
    ims = (data[image_key].permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    actions = data[actions_key].cpu().numpy()
    base_actions = cam_frame_to_base_frame(actions.squeeze(), camera_transforms.extrinsics['left'])
    actions = actions[..., :3]

    ims_viz = visualize_actions(ims, actions, camera_transforms.extrinsics, camera_transforms.intrinsics, arm="left")

    for j, im in enumerate(ims_viz):
        img_tensor = torch.from_numpy(im).permute(2, 0, 1)
        save_path = os.path.join(save_dir, f"image_{cur_batch}_{j}.png")
        io.write_png(img_tensor, save_path)
        # save base_actions to txt
        with open(os.path.join(save_dir, f"base_actions_{cur_batch}_{j}.txt"), "w") as f:
            np.savetxt(f, base_actions.squeeze()) # base action N, 6

    print(f"Saved batch {cur_batch} images to {save_dir}")
    cur_batch += 1