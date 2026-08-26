import zarr, numpy as np
z = zarr.open("/coc/flash7/paphiwetsa3/datasets/circle_3000/episode_T_circle_obs0_000000.zarr", mode="r")
print("actions[0:3]\n", np.asarray(z["actions"][0:3]))
print("pusher_cmd_pose[0:3]\n", np.asarray(z["observations.pusher_cmd_pose"][0:3]))
print("state[0:2]\n", np.asarray(z["observations.state"][0:2]))
print("goal_pose[0:2]\n", np.asarray(z["goal_pose"][0:2]))
print("reward[0:3].ravel()", np.asarray(z["reward"][0:3]).ravel())
img0 = z["observations.images.front_img_1"][0]
print("img0 type", type(img0), "len" , len(img0) if hasattr(img0,'__len__') else None)
