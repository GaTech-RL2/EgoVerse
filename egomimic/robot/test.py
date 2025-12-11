import numpy as np
from scipy.spatial.transform import Rotation as R

rot = R.random()
rpy = rot.as_euler("XYZ", degrees=False)
ypr = rot.as_euler("ZYX", degrees=False)

mat1 = R.from_euler("ZYX", rpy, degrees=False)
mat2 = R.from_euler("XYZ", ypr, degrees=False)

breakpoint()
