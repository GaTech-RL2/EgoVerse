from projectaria_tools.core import data_provider
import cv2, os

vrs_dir = "/media/rl2-bonjour/egoverse/Operator3"
vrs_name = "fold_clothes_scene1_operator3_8.vrs"

vrs_path = os.path.join(vrs_dir, vrs_name)
provider = data_provider.create_vrs_data_provider(vrs_path)

# Inspect streams
print(provider.get_all_streams())

# ---- save dir based on vrs name (without .vrs) ----
vrs_stem = os.path.splitext(vrs_name)[0]
save_dir = f"/home/rl2-bonjour/code/EgoVerse/temp/{vrs_stem}"
os.makedirs(save_dir, exist_ok=True)

# ---- extract frames ----
sid = provider.get_stream_id_from_label("camera-rgb")

num = provider.get_num_data(sid)
print("num frames:", num)

# for i in range(num):
i = 2500
image_data, record = provider.get_image_data_by_index(sid, i)
img = image_data.to_numpy_array()  # RGB
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(save_dir, f"{i:06d}.png"), img_bgr)
