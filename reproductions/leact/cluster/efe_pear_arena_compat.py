"""Validate and enable native quaternion AABBs for authored non-quarter yaw.

The pinned Arena already implements exact corner rotation. Its placement
adapter still calls an older quarter-turn-only helper. Keep that helper's
existing behavior where supported; use its native quaternion method otherwise.
No task pose, collision mesh, physics parameter or success predicate changes.
"""
import hashlib
import math
from pathlib import Path
from uuid import uuid4

ORIGINAL_SHA256='4a9b3a2daaf989cbcd0ad3d88d03da0b81dcc62c0ea02a9b72fde1cff27d9726'
PATCHED_SHA256='b5b0a77b868d6e286a1a10588fd6942176e2ce61e5bdb5251c138306b6a9245a'
OLD='''        quarters = quaternion_to_90_deg_z_quarters(initial_pose.rotation_xyzw)
        return bounding_box.rotated_90_around_z(quarters).translated(initial_pose.position_xyz)'''
NEW='''        try:
            quarters = quaternion_to_90_deg_z_quarters(initial_pose.rotation_xyzw)
        except AssertionError:
            return bounding_box.rotated_by_quat(initial_pose.rotation_xyzw).translated(initial_pose.position_xyz)
        return bounding_box.rotated_90_around_z(quarters).translated(initial_pose.position_xyz)'''


def qualify_and_apply(root='/opt/gr00t-sim/source/IsaacLab-Arena'):
    import numpy as np
    import torch
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from itertools import product
    low=(-.2,-.07,-.03);high=(.11,.15,.04)
    box=AxisAlignedBoundingBox(low,high);corners=np.array(list(product(*zip(low,high))))
    maximum_error=0.
    for degrees in [0.,90.,180.,270.,106.5,155.6]:
        angle=math.radians(degrees);quaternion=(0.,0.,math.sin(angle/2),math.cos(angle/2))
        result=box.rotated_by_quat(quaternion)
        rotation=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1.]])
        transformed=corners@rotation.T
        error=max(float(np.max(np.abs(result.min_point.numpy()[0]-transformed.min(axis=0)))),
            float(np.max(np.abs(result.max_point.numpy()[0]-transformed.max(axis=0)))))
        if error>1e-6:raise ValueError('Native quaternion AABB disagrees with independent transformed corners')
        maximum_error=max(maximum_error,error)
        if degrees%90==0:
            previous=box.rotated_90_around_z(int(degrees/90))
            if not torch.allclose(result.min_point,previous.min_point,atol=1e-6,rtol=0)or not torch.allclose(
                result.max_point,previous.max_point,atol=1e-6,rtol=0):raise ValueError('Quarter-turn AABB behavior changed')
    path=Path(root)/'isaaclab_arena/relations/placement_asset.py';raw=path.read_bytes()
    source=raw.decode();before=hashlib.sha256(raw).hexdigest()
    if before==ORIGINAL_SHA256:
        if source.count(OLD)!=1:raise ValueError('Unrecognized pinned Arena placement implementation')
        temporary=path.with_suffix('.'+uuid4().hex+'.tmp')
        temporary.write_text(source.replace(OLD,NEW));temporary.replace(path)
    elif before!=PATCHED_SHA256:raise ValueError('Unrecognized pinned Arena placement implementation')
    return {'scope':'Use native quaternion AABB only where the older quarter-turn adapter rejects authored rotation',
        'original_sha256':ORIGINAL_SHA256,'patched_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'independent_corner_max_error':maximum_error,'quarter_turn_equivalence_passed':True,
        'task_physics_and_reward_unchanged':True}
