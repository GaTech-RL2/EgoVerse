# An example of converting from RGBD to point cloud
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from rpl_vision_utils.k4a.k4a_interface import K4aInterface

vis = o3d.visualization.Visualizer()
vis.create_window()
vis_pcd = o3d.geometry.PointCloud()

k4a_interface = K4aInterface()

k4a_interface.start()

frame_count = 0
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
while True:

    capture = k4a_interface.get_last_obs()
    if capture is None:
        continue

    # Flip channel order to make sure the color rendering is correct
    # in open3d visualizer
    color_img = capture["color"][:, :, ::-1]

    # Change contiguous array, otherwise the constructed Image object
    # is not correct.
    color_img = np.ascontiguousarray()

    # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    depth_img = capture["depth"]
    intrinsics = k4a_interface.get_depth_intrinsics()

    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=depth_img.shape[0],
        height=depth_img.shape[1],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
    )

    color_img = o3d.geometry.Image(color_img)
    depth_img = o3d.geometry.Image(depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, depth_trunc=2.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics)
    pcd.transform(flip_transform)
    vis_pcd.points = pcd.points
    vis_pcd.colors = pcd.colors
    if frame_count == 0:
        vis.add_geometry(vis_pcd)
    vis.update_geometry(vis_pcd)
    vis.poll_events()
    vis.update_renderer()
    frame_count += 1
    print(frame_count)
    time.sleep(0.05)
