"""Native GR00T YAM Cartesian targets through bounded joint-position control.

Uses only the robot's own poses, Jacobians and limits. No object state, grasp
planner, pose writes, joint-state writes or success shaping is involved.
"""
import math

try:
    from .efe_pear_eef import VERSION
except ImportError:
    from efe_pear_eef import VERSION


class YamEefController:
    position_gain = 3.
    rotation_gain = 2.
    def __init__(self, env):
        from gr00t_sim.teleop.yam_controller import YamTeleopController
        self.robot = YamTeleopController(env)

    def reset(self):
        self.robot.reset()

    def contract(self, cameras, control_dt):
        from gr00t_sim.teleop.yam_controller import _as_torch
        layout, lower, upper, groups = [], [], [], []
        for side in ('left', 'right'):
            arm = self.robot._arms[side]
            root = _as_torch(arm.asset.data.root_link_pos_w)[0].detach().cpu().tolist()
            radius = self.robot.workspace_radius
            offset = len(layout)
            layout.extend(side + '_' + name for name in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper'))
            lower.extend([v - radius for v in root] + [-1.] * 4 + [0.])
            upper.extend([v + radius for v in root] + [1.] * 4 + [1.])
            groups.append(list(range(offset + 3, offset + 7)))
        return {'protocol': VERSION, 'control_interface': VERSION, 'control_dt': control_dt,
            'max_repeat': 25, 'cameras': list(cameras),
            'output_keys': ['action_type', 'action', 'repeat', 'summary'],
            'action_types': {'ee_pose': {'dim': 16, 'layout': layout, 'lower': lower, 'upper': upper,
                'frame': 'Native simulator world frame. +Z is up; camera image directions are separate.',
                'units': 'Position in metres; quaternion wxyz; gripper normalized from 0 open to 1 closed.',
                'quaternion_order': 'wxyz', 'quaternion_groups': groups,
                'tool_frame': 'Per-arm grasp-site TCP, link_6 orientation; link-local TCP offset [0,0,0.1347] metres.',
                'gripper': '0 = fully open; 1 = fully closed; intermediate values are supported.',
                'semantics': 'Absolute left and right TCP targets, held for at most repeat native control steps. '
                    'Bounded damped-least-squares IK recomputes joint targets at every step. '
                    'Cartesian position/rotation feedback gains are 3 and 2, with 0.04 m and 0.30 rad error caps '
                    'and a maximum 0.05 rad change per joint per control step. '
                    'A target need not be reachable inside the declared command box. '
                    'Consult measured TCP poses and tracking errors after every chunk.'}}}

    def state(self):
        from gr00t_sim.teleop.yam_controller import _as_torch
        poses = {}
        for side in ('left', 'right'):
            position, quaternion = self.robot.current_tcp_pose(side)
            q = quaternion[0].detach().cpu().tolist()  # native XYZW -> public WXYZ
            grip = float(self.robot.gripper_targets[side][0, 0])
            lo, hi = self.robot._arms[side].gripper_limits
            if not hi>lo:raise ValueError('Native gripper has no controllable opening range')
            arm=self.robot._arms[side]
            measured=float(_as_torch(arm.asset.data.joint_pos)[0,arm.finger_ids].mean())
            poses[side] = {'position_m': position[0].detach().cpu().tolist(),
                'quaternion_wxyz': [q[3], *q[:3]],
                'gripper_target_closed_fraction': (grip - lo) / (hi - lo),
                'gripper_measured_closed_fraction': (measured-lo)/(hi-lo)}
        return {'frame': 'native_world', 'arms': poses}

    def step(self, action):
        import torch
        from gr00t_sim.teleop.yam_controller import (
            _as_torch, _limit_norm, _quat_rotate, _rotation_error,
            damped_joint_delta, tcp_jacobian, TCP_OFFSET)
        robot = self.robot
        for index, side in enumerate(('left', 'right')):
            row = action[8 * index:8 * index + 8]
            arm = robot._arms[side]
            position, quaternion = robot.current_tcp_pose(side)
            target_position = position.new_tensor(row[:3]).unsqueeze(0)
            target_quaternion = quaternion.new_tensor([*row[4:7], row[3]]).unsqueeze(0)
            target_quaternion /= target_quaternion.norm(dim=-1, keepdim=True)
            jacobian = _as_torch(arm.asset.data.body_link_jacobian_w)
            jacobian = jacobian[:, arm.jacobian_index, :, :][:, :, arm.jacobian_columns]
            offset = _quat_rotate(quaternion, position.new_tensor(TCP_OFFSET).expand_as(position))
            delta = damped_joint_delta(tcp_jacobian(jacobian, offset),
                _limit_norm((target_position - position)*self.position_gain, robot.max_position_error),
                _limit_norm(_rotation_error(target_quaternion, quaternion)*self.rotation_gain, robot.max_rotation_error),
                max_delta=robot.max_joint_delta)
            measured = _as_torch(arm.asset.data.joint_pos)[:, arm.joint_ids]
            candidate = measured + delta
            previous = robot.joint_targets[side]
            candidate = previous + (candidate - previous).clamp(-robot.max_joint_delta, robot.max_joint_delta)
            robot.joint_targets[side] = candidate.clamp(arm.limits[..., 0], arm.limits[..., 1])
            lo, hi = arm.gripper_limits
            desired = lo + row[7] * (hi - lo)
            robot.gripper_targets[side] += (desired - robot.gripper_targets[side]).clamp(
                -robot.max_gripper_delta, robot.max_gripper_delta)
        result = robot._action()
        if not torch.isfinite(result).all():
            raise ValueError('YAM EEF controller generated a nonfinite joint target')
        return result

    def feedback(self, action):
        from gr00t_sim.teleop.yam_controller import _rotation_error,_as_torch
        state = self.state()
        errors = {}
        for i, side in enumerate(('left', 'right')):
            row = action[8 * i:8 * i + 8]
            position, quaternion = self.robot.current_tcp_pose(side)
            target = quaternion.new_tensor([*row[4:7], row[3]]).unsqueeze(0)
            target /= target.norm(dim=-1, keepdim=True)
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(row[:3], state['arms'][side]['position_m'])))
            angle = float(_rotation_error(target, quaternion).norm())
            arm=self.robot._arms[side]
            joints=_as_torch(arm.asset.data.joint_pos)[:,arm.joint_ids]
            margins=(joints-arm.limits[...,0]).minimum(arm.limits[...,1]-joints)[0]
            errors[side] = {'position_error_m': distance, 'rotation_error_rad': angle,
                'target_reached': distance <= .01 and angle <= .05,
                'joint_limit_min_margin_rad':float(margins.min()),
                'near_joint_limits':[f'joint{i+1}'for i,v in enumerate(margins.tolist())if v<.02]}
        return {'frame': 'native_world', 'tracking': errors, 'measured_tcp': state,
                'target_reached_is_not_task_success': True}
