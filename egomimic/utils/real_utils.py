import types
import numpy as np
import torch
from egomimic.utils.egomimicUtils import (
    ee_pose_to_cam_pixels,
    draw_dot_on_frame,
)
from einops import rearrange
import cv2
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import os


def get_image(ts, camera_names, device):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0).to(device)
    return curr_image


def render_traj(img, cam2base, K, trajs, colors):
    for i in range(len(trajs)):
        # NOTE: dividing by 2 bc image is 320x240 but camera is 640x480
        traj_pixels = ee_pose_to_cam_pixels(np.array(trajs[i]), cam2base, K) / 2
        img = draw_dot_on_frame(
            img, traj_pixels, palette=colors[i], dot_size=2, show=False
        )

    return img


def render_trajs_batch(img_data, traj_dict, cam2base, K, colors):
    """
    img_data: (B, numImages(high, wrist), C, H, W)
    cam2base (iextrinsics matrix): (4, 4)
    K (intrinsics): (3, 4)
    """
    b = img_data.shape[0]

    # left_pred_eef_trs = trajs["left_pred_eef"][..., :3, 3]
    # left_gt_eef_trs = trajs["left_gt_eef"][..., :3, 3]
    # right_pred_eef_trs = trajs["right_pred_eef"][..., :3, 3]
    # right_gt_eef_trs = trajs["right_gt_eef"][..., :3, 3]

    # guide_traj = trajs["guide_traj"]

    rend_imgs = []
    trajs = [traj_dict[k] for (k, v) in colors.items()]
    colors = [v for (k, v) in colors.items()]

    for i in range(b):
        # print("initial traj: ", trajs[i])
        img = np.ascontiguousarray(
            np.array(img_data[i, 0] * 255, dtype="uint8").transpose(1, 2, 0)
        )
        img = render_traj(
            img,
            cam2base=cam2base,
            K=K,
            trajs=[traj[i] for traj in trajs],
            colors=colors,
        )
        rend_imgs.append(img)

    return rend_imgs, trajs


def save_images(images, viz_dir):
    """
    images: list of images with batch size [(B, H, W, C), (B, H, W, C)]
    """

    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)
    for rend_img_index, rend_img in enumerate(images):
        # B = rend_img.shape[0]
        # for b in range(B):
        rend_img = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(viz_dir, f"img_{rend_img_index}.png"), rend_img)
        rend_img_index += 1


def resize_curr_img(curr_image):
    """
    curr_image: (B, 2, 3, 480, 640)
    """
    orig_shape = list(curr_image.shape)
    curr_image = curr_image.view((-1, 3, 480, 640))
    curr_image = TVF.resize(
        curr_image, size=(240, 320), interpolation=TVF.InterpolationMode.BILINEAR
    )
    orig_shape[-2] = 240
    orig_shape[-1] = 320
    curr_image = curr_image.view(orig_shape)

    return curr_image


def plot_joint_pos(ax, joint_pos, linestyle="solid"):
    """
    ax: matplotlib axis
    joint_pos: (T, 14)
    """
    pallette = plt.get_cmap("tab10")
    pallette = pallette.colors
    # breakpoint()

    for i, color in zip(range(7, 14), pallette):
        ax.plot(joint_pos[0, :, i], color=color, linestyle=linestyle)

    return ax


def make_fake_env(init_node=True, arm_left=False, arm_right=True):
    return FakeEnv()


class FakeEnv:
    def __init__(self):
        pass

    def reset(self):
        return self.step(None)

    def step(self, action):
        ts = types.SimpleNamespace()
        # make random image of size (480, 640, 3) with values between 0 and 255
        ts.observation = {
            "images": {
                "cam_high": np.random.randint(0, 255, (480, 640, 3)),
                "cam_right_wrist": np.random.randint(0, 255, (480, 640, 3)),
            },
            "qpos": np.random.rand(14),
        }

        return ts
