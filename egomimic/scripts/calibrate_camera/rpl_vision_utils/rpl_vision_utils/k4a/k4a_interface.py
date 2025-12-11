import json
import threading
from typing import Any, Optional, Tuple

import k4a_module
import numpy as np
import pyk4a
from easydict import EasyDict
from pyk4a import Config, PyK4A
from pyk4a.calibration import CalibrationType

from rpl_vision_utils.threading.threading_utils import Worker


def get_k4a_intrinsics_param(K_matrix: np.ndarray):
    """
    Args:
       K_matrix (np.ndarray): Numpy matrix of camera intrinsics

    Return:
       intrinsics_params (dict): a dictionary of intrinsics parameters, namely fx, fy, cx, cy
    """
    return {
        "fx": K_matrix[0][0],
        "fy": K_matrix[1][1],
        "cx": K_matrix[0][2],
        "cy": K_matrix[1][2],
    }


# def get_k4a_distortion_param(distortion_coeff: np.ndarray):
#     """
#     Args:
#        distortion_coeff (np.ndarray): Numpy array of the distortion coeff, in the ordre of [cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, p2, p1]

#     """


class CameraWorker(Worker):
    def __init__(
        self,
        config: Optional[Config] = None,
        device_id=0,
        thread_safe: bool = True,
        fps: int = 30,
    ):
        self.config = config
        self._device_id = device_id
        self.thread_safe = thread_safe
        self.last_obs = None
        self.fps = fps
        self.calibration = {
            "color": {"intrinsics": None, "distortion": None},
            "depth": {"intrinsics": None, "distortion": None},
        }
        super().__init__()

    def run(self) -> None:
        # print("Start run")
        camera = PyK4A(
            config=self.config, device_id=self._device_id, thread_safe=self.thread_safe
        )
        camera.start()

        # Intrinsics matrix takes the form:
        # [[fx, 0,  cx],
        #  [0,  fy, cy],
        #  [0,  0,   1]]
        color_K_matrix = camera.calibration.get_camera_matrix(CalibrationType.COLOR)
        depth_K_matrix = camera.calibration.get_camera_matrix(CalibrationType.DEPTH)

        self.calibration["color"]["intrinsics"] = color_K_matrix
        self.calibration["depth"]["intrinsics"] = depth_K_matrix

        # [k1, k2, p1, p2, [k3, [k4, k5, k6]] ]
        self.calibration["color"][
            "distortion"
        ] = camera.calibration.get_distortion_coefficients(
            pyk4a.calibration.CalibrationType.COLOR
        )

        print(camera.calibration.get_camera_matrix(CalibrationType.COLOR))
        while not self._halt:
            capture = camera.get_capture()
            assert capture.depth is not None
            self._count += 1
            self.last_obs = capture

        camera.stop()
        del camera
        # print("Stop run")

    def get_intrinsics(self, key, mode=None):
        assert key in ["color", "depth"]
        if mode == "dict":
            return get_k4a_intrinsics_param(self.calibration[key]["intrinsics"])
        else:
            return self.calibration[key]["intrinsics"]

    def get_distortion(self, key):
        assert key in ["color"]
        return self.calibration[key]["distortion"]


class K4aInterface:
    """
    This is the Python Interface for getting images from Kinect Azure.
    Args:

    """

    def __init__(
        self,
        device_id=0,
        color_cfg: dict = EasyDict(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
        ),
        depth_cfg: dict = EasyDict(depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED),
    ):
        camera_config = Config(
            color_resolution=color_cfg.color_resolution,
            color_format=color_cfg.color_format,
            depth_mode=depth_cfg.depth_mode,
            synchronized_images_only=True,
        )
        self.camera = CameraWorker(
            config=camera_config, device_id=device_id, thread_safe=False
        )
        self.is_recording = False

        self.enable_color = True
        self.enable_depth = True

    def start(self):
        self.camera.start()

    def get_last_obs(self):
        """
        Get last observation from camera
        """
        if self.camera.last_obs is None:
            return None
        else:
            last_obs = {}
            if self.enable_color:
                last_obs["color"] = self.camera.last_obs.color[:, :, :3]
            if self.enable_depth:
                last_obs["depth"] = self.camera.last_obs.transformed_depth
            return last_obs

    def close(self):
        self.camera.halt()
        self.camera.join()

    def get_depth_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("depth", mode=mode)
        return intrinsics

    def get_color_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("color", mode=mode)
        return intrinsics

    def get_color_distortion(self):
        distortion = self.camera.get_distortion("color")
        return distortion
