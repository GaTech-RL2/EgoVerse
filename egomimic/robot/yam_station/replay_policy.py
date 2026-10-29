"""Read-only Zarr joint replay through the same rollout loop as model policies."""

import numpy as np

from egomimic.robot.yam_station.interface import ARM_OFFSET, joint_vector


class ZarrReplayPolicy:
    action_type = "joints"

    def __init__(
        self,
        path,
        keys=None,
        action_key=None,
        joint_action_key=None,
        gripper_action_key=None,
        arm_order=None,
        require_complete=True,
        chunk_size=1,
        start=0,
        stop=None,
    ):
        import zarr

        self.store = zarr.open_group(str(path), mode="r")
        if require_complete and not bool(self.store.attrs.get("complete", True)):
            raise ValueError("Replay requires a completed Zarr episode")
        self.keys, self.action_key = dict(keys or {}), action_key
        if (joint_action_key is None) != (gripper_action_key is None):
            raise ValueError("Split replay requires both joint and gripper action keys")
        split_actions = joint_action_key is not None
        if sum((bool(self.keys), action_key is not None, split_actions)) != 1:
            raise ValueError(
                "Choose per-arm keys, one bimanual action_key, or split Yam action keys"
            )
        self.joint_action_key = joint_action_key
        self.gripper_action_key = gripper_action_key
        if self.keys and not set(self.keys) <= ARM_OFFSET.keys():
            raise ValueError("Replay arm keys must be left/right")
        if split_actions:
            stored_order = self.store.attrs.get("arm_order")
            self.arm_order = tuple(arm_order or stored_order or ())
            if len(self.arm_order) != 2 or set(self.arm_order) != set(ARM_OFFSET):
                raise ValueError(
                    "Split Yam replay requires arm_order containing left and right"
                )
            arrays = [
                self.store[joint_action_key],
                self.store[gripper_action_key],
            ]
        elif action_key is not None:
            self.arm_order = ()
            arrays = [self.store[action_key]]
        else:
            self.arm_order = ()
            arrays = [
                self.store[key] for spec in self.keys.values() for key in spec.values()
            ]
        if not arrays:
            raise ValueError("Replay needs action arrays")
        # EgoVerse Zarr arrays may have chunk padding beyond total_frames.
        length = self.store.attrs.get("total_frames")
        if length is None:
            length = self.store.attrs.get("committed_samples")
        length = int(
            min(array.shape[0] for array in arrays) if length is None else length
        )
        if length <= 0 or any(array.shape[0] < length for array in arrays):
            raise ValueError("Invalid Zarr frame count")
        if action_key and arrays[0].shape[1:] != (14,):
            raise ValueError("Bimanual joint replay requires an (N, 14) array")
        if split_actions and (
            arrays[0].shape[1:] != (2, 6) or arrays[1].shape[1:] not in ((2,), (2, 1))
        ):
            raise ValueError(
                "Split Yam replay requires (N, 2, 6) joints and (N, 2) grippers"
            )
        for arm, spec in self.keys.items():
            if (
                set(spec) != {"joints", "gripper"}
                or self.store[spec["joints"]].shape[1:] != (6,)
                or self.store[spec["gripper"]].shape[1:] not in ((), (1,))
            ):
                raise ValueError(
                    f"Replay requires six joints and one gripper per row: {arm}"
                )
        self.cursor, self.stop, self.chunk_size = (
            int(start),
            length if stop is None else int(stop),
            int(chunk_size),
        )
        if not 0 <= self.cursor < self.stop <= length or self.chunk_size < 1:
            raise ValueError("Invalid replay bounds or chunk size")

    def predict(self, obs):
        if self.cursor >= self.stop:
            raise StopIteration
        end = min(self.stop, self.cursor + self.chunk_size)
        if self.action_key:
            rows = np.asarray(
                self.store[self.action_key][self.cursor : end], dtype=float
            )
        elif self.joint_action_key:
            rows = np.tile(obs["joint_positions"], (end - self.cursor, 1)).astype(float)
            joints = np.asarray(
                self.store[self.joint_action_key][self.cursor : end], dtype=float
            )
            grippers = np.asarray(
                self.store[self.gripper_action_key][self.cursor : end], dtype=float
            ).reshape(-1, 2)
            for index, arm in enumerate(self.arm_order):
                offset = ARM_OFFSET[arm]
                rows[:, offset : offset + 6] = joints[:, index]
                rows[:, offset + 6] = grippers[:, index]
        else:
            # Hold arms missing from the replay at their measured positions.
            rows = np.tile(obs["joint_positions"], (end - self.cursor, 1)).astype(float)
            for arm, spec in self.keys.items():
                offset = ARM_OFFSET[arm]
                rows[:, offset : offset + 6] = self.store[spec["joints"]][
                    self.cursor : end
                ]
                rows[:, offset + 6] = np.asarray(
                    self.store[spec["gripper"]][self.cursor : end]
                ).reshape(-1)
        for row in rows:
            for offset in ARM_OFFSET.values():
                joint_vector(row[offset : offset + 7])
        self.cursor = end
        return rows


class Hdf5ReplayPolicy:
    """Read a completed EgoVerse HDF5 joint demo without modifying it."""

    action_type = "joints"

    def __init__(
        self,
        path,
        action_key="actions/joints",
        require_complete=True,
        chunk_size=1,
        start=0,
        stop=None,
    ):
        import h5py

        self.file = h5py.File(str(path), "r")
        if require_complete and not bool(self.file.attrs.get("complete", True)):
            self.file.close()
            raise ValueError("Replay requires a completed HDF5 episode")
        try:
            self.actions = self.file[action_key]
        except KeyError:
            self.file.close()
            raise ValueError(
                f"HDF5 replay action dataset is missing: {action_key}"
            ) from None
        if (
            self.actions.ndim != 2
            or self.actions.shape[1] != 14
            or not len(self.actions)
        ):
            self.file.close()
            raise ValueError(
                "HDF5 replay requires a nonempty (N, 14) actions/joints dataset"
            )
        self.cursor, self.stop, self.chunk_size = (
            int(start),
            len(self.actions) if stop is None else int(stop),
            int(chunk_size),
        )
        if not 0 <= self.cursor < self.stop <= len(self.actions) or self.chunk_size < 1:
            self.file.close()
            raise ValueError("Invalid HDF5 replay bounds")

    def predict(self, obs):
        if self.cursor >= self.stop:
            raise StopIteration
        end = min(self.stop, self.cursor + self.chunk_size)
        rows = np.asarray(self.actions[self.cursor : end], dtype=float)
        for row in rows:
            for offset in ARM_OFFSET.values():
                joint_vector(row[offset : offset + 7])
        self.cursor = end
        return rows

    def close(self):
        """Release the replay handle so the demo can be inspected or replaced."""
        file, self.file = self.file, None
        self.actions = None
        if file is not None:
            file.close()
