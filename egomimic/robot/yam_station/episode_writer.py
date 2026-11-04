"""Demonstration HDF5 writer shared by GELLO collection and rollout episodes."""

from pathlib import Path

import h5py
import numpy as np

VECTOR_KEYS = (
    "observations/joints",
    "observations/joint_positions",
    "observations/eepose",
    "actions/eepose",
    "actions/joints",
    "action",
)


class EpisodeWriter:
    """Append complete synchronized rows; exclusive creation never overwrites a demo."""

    def __init__(self, path, camera_res):
        self.path, self.camera_res = Path(path), dict(camera_res)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.file.attrs["sim"] = False
        self.file.attrs["complete"] = False
        self.frames = 0
        self.file.require_group("observations/images")
        for key in VECTOR_KEYS:
            self.file.create_dataset(
                key, (0, 14), maxshape=(None, 14), dtype="float32", chunks=(1, 14)
            )
        for camera, (height, width) in self.camera_res.items():
            shape = (height, width, 3)
            self.file.create_dataset(
                f"observations/images/{camera}",
                (0, *shape),
                maxshape=(None, *shape),
                dtype="uint8",
                chunks=(1, *shape),
            )
        self.file.flush()

    def append(self, obs, joints, ee_pose):
        values = {
            "observations/joints": obs["joint_positions"],
            "observations/joint_positions": obs["joint_positions"],
            "observations/eepose": obs["ee_poses"],
            "actions/eepose": ee_pose,
            "actions/joints": joints,
            "action": joints,
        }
        for key, value in values.items():
            value = np.asarray(value)
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid 14D recording row: {key}")
        for name, (height, width) in self.camera_res.items():
            image = obs.get(name)
            if (
                image is None
                or image.shape != (height, width, 3)
                or image.dtype != np.uint8
            ):
                raise ValueError(f"Missing or malformed camera frame: {name}")
            # The existing drivers publish BGR; the existing HDF5 files store RGB.
            values[f"observations/images/{name}"] = image[..., ::-1]
        for key, value in values.items():
            dataset = self.file[key]
            dataset.resize(self.frames + 1, axis=0)
            dataset[self.frames] = value
        self.frames += 1
        self.file.flush()

    def close(self, complete=True):
        if self.file is not None:
            self.file.attrs["complete"] = bool(complete)
            self.file.close()
            self.file = None


def next_episode_path(directory, start=0):
    directory = Path(directory)
    episode = int(start)
    while (directory / f"demo_{episode}.hdf5").exists():
        episode += 1
    return directory / f"demo_{episode}.hdf5"
