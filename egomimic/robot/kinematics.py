"""
Generic kinematics solver.
"""

from pyexpat import model
from typing import List, Tuple, Optional
import numpy as np
from sklearn import base
import pybullet as p
import pybullet_data
import os
import re
import tempfile
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from trac_ik import TracIK

class KinematicsSolver:
    """
    Generic kinematics solver using PyTracik

    Args:
        urdf_path: Path to the URDF file
        base_link_name: Name of the base link in the urdf
        ee_link_name: Name of the link you are solving for
    """

    def __init__(
        self,
        urdf_path: str,
        base_link_name: str,
        eef_link_name: str,
        num_joints: int
    ):
        self.num_joints = num_joints
        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.solver = self.kinematics_solver = TracIK(base_link_name=base_link_name,
                                tip_link_name=eef_link_name,
                                urdf_path=self.urdf_path, )

    

    def _resolve_urdf_path(self, urdf_path: str) -> str:
        """
        Resolve URDF path, handling package:// URIs if present.
        """
        # Make path absolute
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.abspath(urdf_path)

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

        # Directory where the original URDF sits
        model_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(urdf_path)))
        print(f"Model root path is {model_root_dir}")

        try:
            with open(urdf_path, "r", encoding="utf-8") as f:
                urdf_text = f.read()

            if "package://" in urdf_text:
                # Replace every package://<pkg_name>/... with model_root_dir/...
                # This is a cheap fallback when you don't have ROS package paths
                def _replace_pkg(match: re.Match) -> str:
                # match.group() is like "package://X5A/"
                    return model_root_dir + "/"

                urdf_text = re.sub(r"package://[^/]+/", _replace_pkg, urdf_text)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode="w")
                tmp.write(urdf_text)
                tmp.flush()
                tmp.close()
                return tmp.name
        except Exception:
        # If anything goes wrong, just return the original path
             return urdf_path

        return urdf_path
    
    def ik(self, pos_xyz, rot_mat, cur_jnts):
        """
        Inverse kinematics
        
        Args:
            pos_xyz: numpy array of xyz
            rot_mat: 3x3 rotation matrix in numpy
            cur_jnts: numpy array of length num_joints
        
        Return:
            solved_jnts: numpy array of length num_joints
        """
        return self.kinematics_solver.ik(pos_xyz, rot_mat, seed_jnt_values=cur_jnts[:self.num_joints])
    
    def fk(self, jnts):
        """
        Forward Kinematics
        
        Args:
            jnts: numpy array of length num_joints
        
        Return:
            pos: xyz, rot: Scipy rotation matrix
        """
        return self.kinematics_solver.fk(jnts[:self.num_joints])
