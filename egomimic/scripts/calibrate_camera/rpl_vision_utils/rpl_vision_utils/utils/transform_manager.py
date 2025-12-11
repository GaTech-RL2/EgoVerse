"""
This is a class wrapper to performan transformation between different frames. Currently, we use pytransform3d for doing this. But this package is better at explaining rather than efficiency. So we might want to change it afterwards.
"""
import numpy as np
import pytransform3d.transform_manager as pt3_tm
import pytransform3d.transformations as pt3_tf


class RPLTransformManager:
    """
    Transform Manager for RPL lab.
    """

    def __init__(self):
        self.tm = pt3_tm.TransformManager()

    def add_transform(self, from_frame: str, to_frame: str, R: np.ndarray, p: np.ndarray):
        """
        from_frame (str): frame name of targets
        to_frame (str): The specified coordinate frame to represent from_frame.
        transformation: 4x4 transformation matrices.
        """
        assert R.shape == (3, 3)
        if p.shape == (3, 1):
            p = p.reshape(
                3,
            )
        if p.shape == (1, 3):
            p = p.transpose().squeeze()
        assert p.shape == (3,)
        transformation = pt3_tf.transform_from(R=R, p=p)
        self.tm.add_transform(from_frame, to_frame, transformation)

    def get_transform(self, from_frame: str, to_frame: str):
        try:
            return self.tm.get_transform(from_frame, to_frame)
        except ValueError:
            return None

    def remove_transform(self, from_frame: str, to_frame: str):
        self.tm.remove_transform(from_frame, to_frame)

    def apply_transform_to_point(self):
        """Apply to 3d points"""
        raise NotImplementedError

    def apply_transform_to_pose(self):
        """Apply to 6DoF poses"""
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError
