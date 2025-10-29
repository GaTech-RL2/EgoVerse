import numpy as np
import torch

Z_FRONT_TO_Z_UP = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])

ARIA_SLAM_TO_CENTER_LEN = 0.2 # meters from left RGB cam to center

# wrist, thumb, index, middle, ring, pinky
# https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking
# https://facebookresearch.github.io/projectaria_tools/assets/images/21-keypoints-36dacdda7266325ce379b7e90c5a31b7.png
ARIA_FINGERTIP_INDICES = [5, 0, 1, 2, 3, 4]

ARIA_LEFT_HAND_TO_H1_LEFT_HAND = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

ARIA_RIGHT_HAND_TO_H1_RIGHT_HAND = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

# Constants from human policy
ACTION_STATE_VEC_SIZE = 128
QPOS_INDICES = np.arange(100, 100 + 26)
OUTPUT_LEFT_EEF = np.arange(80, 89)
OUTPUT_RIGHT_EEF = np.arange(30, 39)
OUTPUT_HEAD_EEF = np.arange(0, 9)
NUM_KEYPOINTS_PER_HAND = 6  # (5 fingertip + 1 wrist, though wrist is always 0 in wrist coords)
OUTPUT_LEFT_KEYPOINTS = np.arange(10, 10 + 3 * NUM_KEYPOINTS_PER_HAND)
assert OUTPUT_LEFT_KEYPOINTS[-1] < OUTPUT_RIGHT_EEF[0]
OUTPUT_RIGHT_KEYPOINTS = np.arange(40, 40 + 3 * NUM_KEYPOINTS_PER_HAND)
assert OUTPUT_RIGHT_KEYPOINTS[-1] < OUTPUT_LEFT_EEF[0]

OUTPUT_INDEX = np.concatenate([OUTPUT_LEFT_EEF, OUTPUT_LEFT_KEYPOINTS, OUTPUT_RIGHT_EEF, OUTPUT_RIGHT_KEYPOINTS, OUTPUT_HEAD_EEF])

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

# Transform finger positions from local wrist frame to world frame
def transform_hand_keypoints_local_to_world(points, wrist_mat):
    # Convert to homogeneous coordinates
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # Transform points
    transformed = np.dot(wrist_mat, points_h.T).T
    return transformed[:, :3]  # Return only xyz coordinates

# Transform finger positions from world frame to local wrist frame
def transform_hand_keypoints_world_to_local(points, wrist_mat):
    """
    Transform hand keypoints from world frame to local wrist frame.
    This is the inverse of transform_hand_keypoints_local_to_world.
    
    Args:
        points: numpy array of shape (N, 3) containing world-frame hand keypoints
        wrist_mat: 4x4 transformation matrix from local wrist frame to world frame
    
    Returns:
        numpy array of shape (N, 3) containing local-frame hand keypoints
    """
    # Convert to homogeneous coordinates
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # Apply inverse transformation
    wrist_mat_inv = np.linalg.inv(wrist_mat)
    transformed = np.dot(wrist_mat_inv, points_h.T).T
    return transformed[:, :3]  # Return only xyz coordinates

def fk_cmd_dict2policy(fk_dict, num_timesteps):
    # #! always 128 dims
    # num_timesteps = fk_dict['head_mat'].shape[0]
    # print("num_timesteps", num_timesteps)
    
    left_cmds_matrix = fk_dict['rel_left_wrist_mat'].reshape((-1, 4, 4))  
    right_cmds_matrix = fk_dict['rel_right_wrist_mat'].reshape((-1, 4, 4))  
    head_cmds_matrix = fk_dict['head_mat'].reshape((-1, 4, 4))

    left_rot_matrix = torch.tensor(left_cmds_matrix[:, :3, :3], dtype=torch.float32)  
    right_rot_matrix = torch.tensor(right_cmds_matrix[:, :3, :3], dtype=torch.float32) 
    head_rot_matrix = torch.tensor(head_cmds_matrix[:, :3, :3], dtype=torch.float32)

    left_rot_6d = matrix_to_rotation_6d(left_rot_matrix).numpy()
    right_rot_6d = matrix_to_rotation_6d(right_rot_matrix).numpy()
    head_rot_6d = matrix_to_rotation_6d(head_rot_matrix).numpy()

    left_wrist_action = np.concatenate([left_cmds_matrix[:, 0:3, 3], left_rot_6d], axis=1)
    right_wrist_action = np.concatenate([right_cmds_matrix[:, 0:3, 3], right_rot_6d], axis=1)

    left_hand_action = fk_dict['rel_left_hand_keypoints']
    right_hand_action = fk_dict['rel_right_hand_keypoints']

    head_action = np.concatenate([0 * head_cmds_matrix[:, 0:3, 3], head_rot_6d], axis=1)  # mask the translation to 0

    # [0,128)
    policy_state = np.zeros((num_timesteps, ACTION_STATE_VEC_SIZE))
    # [80, 89)
    policy_state[:, OUTPUT_LEFT_EEF] = left_wrist_action
    #! [10, 28) right gripper, arm
    policy_state[:, OUTPUT_LEFT_KEYPOINTS] = left_hand_action.reshape(num_timesteps, -1)

    # [30, 39)
    policy_state[:, OUTPUT_RIGHT_EEF] = right_wrist_action
    #! [40, 58) right gripper, arm
    policy_state[:, OUTPUT_RIGHT_KEYPOINTS] = right_hand_action.reshape(num_timesteps, -1)
    #! right arm
    policy_state[:, OUTPUT_HEAD_EEF] = head_action

    return policy_state
