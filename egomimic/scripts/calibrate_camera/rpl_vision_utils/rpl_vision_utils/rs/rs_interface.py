import pyrealsense2 as rs
import numpy as np
import cv2
import time
from easydict import EasyDict

from rpl_vision_utils.threading.threading_utils import Worker


def get_rs_intrinsics_param(K_matrix: np.ndarray):
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


class RSCameraWorker(Worker):
    def __init__(
        self,
        camera_config: EasyDict = {},
        # device: int = 0,
        thread_safe: bool = True,
    ):
        # try:
        self.pipeline = rs.pipeline()

        self.config = rs.config()
        self.config.enable_device(camera_config.serial_number)

        # rs.config.enable_device_from_file(config, args.input)
        # Configure the pipeline to stream the depth stream

        self.enable_color = camera_config.enable_color
        self.enable_depth = camera_config.enable_depth
        if camera_config.enable_color:
            self.config.enable_stream(
                rs.stream.color,
                camera_config.color_cfg.img_w,
                camera_config.color_cfg.img_h,
                camera_config.color_cfg.img_format,
                camera_config.color_cfg.fps,
            )

        if camera_config.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                camera_config.depth_cfg.img_w,
                camera_config.depth_cfg.img_h,
                camera_config.depth_cfg.img_format,
                camera_config.depth_cfg.fps,
            )

        # Start streaming from file
        # profile = self.pipeline.start(self.config)
        # sensor_dep = profile.get_device().first_depth_sensor()
        # sensor_dep.set_option(rs.option.enable_auto_exposure, 1)
        self.pipeline.start(self.config)

        # # Create colorizer object (for depth)
        # colorizer = rs.colorizer()

        self.last_obs = None
        self.last_time = time.time()
        self.camera_config = camera_config

        self.calibration = {
            "color": {"intrinsics": None, "distortion": None},
            "depth": {"intrinsics": None, "distortion": None},
        }

        super().__init__()

    def get_intrinsics(self, key, mode=None):
        assert key in ["color", "depth"]
        if mode == "dict":
            return get_rs_intrinsics_param(self.calibration[key]["intrinsics"])
        else:
            return self.calibration[key]["intrinsics"]

    def get_distortion(self, key):
        assert key in ["color"]
        return self.calibration[key]["distortion"]

    def get_filters(self):
        """
        Filters for depth images.
        """
        filters = []

        # NOTE: we just use default realsense-viewer parameters below

        # # decimation
        # decimate = rs.decimation_filter(
        #     magnitude=2.,
        # )
        # filters.append(decimate)

        # threshold
        threshold = rs.threshold_filter(
            min_dist=0.1,
            max_dist=4.0,
        )
        filters.append(threshold)

        # depth2disparity
        depth2disparity = rs.disparity_transform(
            transform_to_disparity=True,
        )
        filters.append(depth2disparity)

        # spatial
        spatial = rs.spatial_filter(
            smooth_alpha=0.5,
            smooth_delta=20.0,
            magnitude=2.0,
            hole_fill=0.0,
        )
        filters.append(spatial)

        # temporal
        temporal = rs.temporal_filter(
            smooth_alpha=0.4,
            smooth_delta=20.0,
            # below corresponds to "valid in 2 / last 4" - see https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.temporal_filter.html
            persistence_control=3,
        )
        filters.append(temporal)

        # disparity2depth
        disparity2depth = rs.disparity_transform(
            transform_to_disparity=False,
        )
        filters.append(disparity2depth)

        return filters

    def apply_filters(self, frame, filters):
        """
        Apply filters to frame.
        """
        out = frame
        for f in filters:
            out = f.process(out)
        return out

    def run(self) -> None:
        self.last_obs = EasyDict()

        self.profile = self.pipeline.get_active_profile()
        self.color_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.depth_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.depth)
        )
        color_intrinsics = self.color_profile.intrinsics
        depth_intrinsics = self.depth_profile.intrinsics

        color_K_matrix = np.array(
            [
                [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ]
        )
        depth_K_matrix = np.array(
            [
                [depth_intrinsics.fx, 0.0, depth_intrinsics.ppx],
                [0.0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ]
        )

        self.calibration["color"]["intrinsics"] = color_K_matrix
        self.calibration["depth"]["intrinsics"] = depth_K_matrix
        print(color_K_matrix)
        print(depth_K_matrix)

        self.calibration["color"]["distortion"] = np.array(color_intrinsics.coeffs)
        self.calibration["depth"]["distortion"] = np.array(depth_intrinsics.coeffs)

        # color_intrinsics = self.color_profile.get_intrinsics()
        # return {'fx': color_intrinsics.fx,
        #         'fy': color_intrinsics.fy,
        #         'cx': color_intrinsics.ppx,
        #         'cy': color_intrinsics.ppy}, color_intrinsics.width, color_intrinsics.height

        align = rs.align(rs.stream.color)
        use_filtering = True
        if use_filtering:
            filters = self.get_filters()
        while not self._halt:
            frames = self.pipeline.wait_for_frames()
            if frames is None:
                continue
            if self.enable_color:
                self.last_obs["color"] = np.asanyarray(
                    frames.get_color_frame().get_data()
                )
            # if self.enable_depth:
            #     self.last_obs["depth"] = np.asanyarray(frames.get_depth_frame().get_data())
            if self.enable_depth:
                unaligned_frame = frames.get_depth_frame()
                if use_filtering:
                    unaligned_frame = self.apply_filters(
                        unaligned_frame, filters=filters
                    )
                self.last_obs["unaligned_depth"] = np.asanyarray(
                    unaligned_frame.get_data()
                )
                frames = align.process(frames)
                aligned_frame = frames.get_depth_frame()
                if use_filtering:
                    aligned_frame = self.apply_filters(aligned_frame, filters=filters)
                self.last_obs["depth"] = np.asanyarray(aligned_frame.get_data())
            # NOTE: uncomment to measure fps
            # print("fps: {}".format(1. / (time.time() - self.last_time)))
            # self.last_time = time.time()

        self.pipeline.stop()
        del self.pipeline

    def save_img(self, img_name):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite(img_name, color_image)


class RSInterface:
    """ "
    This is the Python Interface for getting images from Realsense D435i.
    """

    def __init__(
        self,
        device_id,
        color_cfg: dict = None,
        depth_cfg: dict = None,
        pc_cfg: dict = None,
        serial_number=None,
    ):
        if color_cfg is not None:
            self.color_cfg = color_cfg
        else:
            self.color_cfg = EasyDict(
                enabled=True, img_w=640, img_h=480, img_format=rs.format.bgr8, fps=30
            )

        if depth_cfg is not None:
            self.depth_cfg = depth_cfg
        else:
            self.depth_cfg = EasyDict(
                enabled=False, img_w=640, img_h=480, img_format=rs.format.z16, fps=30
            )
        self.serial_number = serial_number

        # TODO: Implement getting point clouds
        if pc_cfg is not None:
            self.pc_cfg = pc_cfg
        else:
            self.pc_cfg = EasyDict(enabled=False)

        if not (
            self.color_cfg.enabled or self.depth_cfg.enabled or self.pc_cfg.enabled
        ):
            raise ValueError

        camera_config = EasyDict(
            enable_color=self.color_cfg.enabled,
            enable_depth=self.depth_cfg.enabled,
            enable_pc=self.pc_cfg.enabled,
            color_cfg=self.color_cfg,
            depth_cfg=self.depth_cfg,
            pc_cfg=self.pc_cfg,
            serial_number=self.serial_number,
        )
        self.camera = RSCameraWorker(
            camera_config=camera_config,
            # device_id=device_id,
            thread_safe=False,
        )

    def start(self):
        self.camera.start()

    def get_last_obs(self):
        """
        Get last observation from camera
        """
        if self.camera.last_obs is None or self.camera.last_obs == {}:
            return None
        else:
            self.last_obs = self.camera.last_obs
            return self.last_obs

    def close(self):
        self.camera.halt()

    def get_camera_intrinsics(self):
        return self.camera.get_intrinsics()

    def get_depth_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("depth", mode=mode)
        return intrinsics

    def get_color_intrinsics(self, mode=None):
        intrinsics = self.camera.get_intrinsics("color", mode=mode)
        return intrinsics

    def get_color_distortion(self, mode=None):
        distortion = self.camera.get_distortion("color")
        return distortion
