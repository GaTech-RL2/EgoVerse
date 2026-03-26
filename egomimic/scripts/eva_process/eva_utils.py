import h5py
import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import EMBODIMENT

DATASET_KEY_MAPPINGS = {
    "joint_positions": "joint_positions",
    "front_img_1": "front_img_1",
    "right_wrist_img": "right_wrist_img",
    "left_wrist_img": "left_wrist_img",
}


class EvaHD5Extractor:
    @staticmethod
    def process_episode(episode_path, arm):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        arm : str
            String for which arm to add data for
        Returns
        -------
        episode_feats : dict
            dictionary mapping keys in the episode to episode features
            {
                {action_key} :
                observations :
                    images.{camera_key} :
                    state.{state_key} :
            }

            #TODO: Add metadata to be a nested dict

        """
        episode_feats = dict()

        with h5py.File(episode_path, "r") as episode:
            for camera in EvaHD5Extractor.get_cameras(episode):
                images = (
                    torch.from_numpy(episode["observations"]["images"][camera][:])
                    .permute(0, 3, 1, 2)
                    .float()
                )

                images = images.byte().numpy()

                mapped_key = DATASET_KEY_MAPPINGS.get(camera, camera)
                episode_feats[f"images.{mapped_key}"] = images

            for state in EvaHD5Extractor.get_obs_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats[f"obs_{mapped_key}"] = episode["observations"][state][:]

            for state in EvaHD5Extractor.get_cmd_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats[f"cmd_{mapped_key}"] = episode["actions"][state][:]

            num_timesteps = episode_feats["obs_eepose"].shape[0]
            if arm == "right":
                value = EMBODIMENT.EVA_RIGHT_ARM.value
            elif arm == "left":
                value = EMBODIMENT.EVA_LEFT_ARM.value
            else:
                value = EMBODIMENT.EVA_BIMANUAL.value

            episode_feats["metadata.embodiment"] = np.full(
                (num_timesteps, 1), value, dtype=np.int32
            )

        return episode_feats

    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        list of str
            A list of keys corresponding to RGB cameras in the dataset.
        """

        rgb_cameras = [
            key for key in hdf5_data["/observations/images"] if "depth" not in key
        ]
        return rgb_cameras

    @staticmethod
    def get_obs_state(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        states : list of str
            A list of keys corresponding to states in the dataset.
        """

        states = [key for key in hdf5_data["/observations"] if "images" not in key]
        return states

    @staticmethod
    def get_cmd_state(hdf5_data: h5py.File):
        """
        Extracts the list of command state keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        cmd_states : list of str
        """
        states = [key for key in hdf5_data["/actions"]]
        return states
