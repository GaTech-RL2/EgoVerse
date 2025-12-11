import os

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D

workdir = "./"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 1, 0.8, aruco_dict)
imboard = board.draw((2000, 2000))
cv2.imwrite(workdir + "chessboard.png", imboard)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(imboard, cmap=mpl.cm.gray, interpolation="nearest")
ax.axis("off")
plt.show()
