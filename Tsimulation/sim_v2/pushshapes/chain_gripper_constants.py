"""Dependency-free ChainGripper geometry and control limits.

Both the physics geometry and the FK/IK control adapter import these values so
that data processing does not need simulator-only dependencies such as pymunk.
"""

CHAIN_GRIPPER_LINK_LEN = 38.0
CHAIN_GRIPPER_LINK_HALF_W = 5.0
CHAIN_GRIPPER_OPEN_ANGLE = 0.12
CHAIN_GRIPPER_CLOSED_ANGLE = 1.45
