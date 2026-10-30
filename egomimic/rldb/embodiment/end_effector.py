"""END_EFFECTOR, torch-free so the Lance schema module can import it."""

from enum import Enum


class END_EFFECTOR(Enum):
    """What acts at the end of each arm; the Lance `end_effector` column holds the value."""

    # A tracked human hand (keypoints).
    HUMAN_HAND = "human_hand"
    PARALLEL_JAW_GRIPPER = "parallel_jaw_gripper"
    # An actuated multi-finger robot hand (joints, FK keypoints, tactile).
    DEX_HAND = "dex_hand"
