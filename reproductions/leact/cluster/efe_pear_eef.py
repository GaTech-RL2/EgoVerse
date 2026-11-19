"""Versioned Cartesian control contracts shared by teacher and student.

Only the action representation changes. Native frames, robot proprioception,
physics and task graders stay with their simulator. Enabling this interface is
explicit; an older joint-space run never acquires it through an environment
default or an updated source file.
"""
import copy
import math


VERSION = 'pear-eef-native-v1'
POSE_SUITES = {'robodojo', 'robolab', 'yamlab', 'dexjoco', 'abc-sim', 'vlabench'}
DELTA_SUITES = {'libero-max', 'metaworldplus', 'robocasa'}
QUATERNION_TOLERANCE = 0.01


def enabled(value):
    if value not in (None, '', VERSION):
        raise ValueError('Unknown Cartesian control interface: ' + str(value))
    return value == VERSION


def validate_quaternions(action, entry):
    for group in entry.get('quaternion_groups', []):
        if (not isinstance(group, list) or len(group) != 4 or
                any(type(i) is not int or not 0 <= i < len(action) for i in group)):
            raise ValueError('Malformed quaternion component declaration')
        norm = math.sqrt(sum(action[i] ** 2 for i in group))
        if abs(norm - 1.) > QUATERNION_TOLERANCE:
            raise ValueError('Each EEF quaternion must have unit norm within 0.01 (wxyz).')


def contract(spec, suite, max_repeat=25):
    """Expose native Cartesian deltas or bounded native absolute pose targets.

    The finite position box is a declared command guard, not a claim that every
    point in it is reachable. Pose tracking must return actual reached errors.
    """
    try:
        from .efe_action_loop import native_contract
    except ImportError:
        from efe_action_loop import native_contract
    if suite in DELTA_SUITES:
        result = native_contract(spec, max_repeat, allowed_action_types=['ee_delta'])
    elif suite in POSE_SUITES:
        public = copy.deepcopy(spec)
        entry = public['action_types']['ee_pose']
        layout = entry['layout']
        dim = entry['dim']
        stride = 23 if suite == 'dexjoco' else 8
        if len(layout) != dim or dim % stride:
            raise ValueError('Unexpected native Cartesian action layout')
        low, high, quaternions = [], [], []
        for offset in range(0, dim, stride):
            # RoboLab's native parser uses a box about the base, at the origin.
            extent = 2. if suite == 'robolab' else 3.
            low.extend([-extent] * 3 + [-1.] * 4)
            high.extend([extent] * 3 + [1.] * 4)
            quaternions.append(list(range(offset + 3, offset + 7)))
            if suite == 'dexjoco':
                for name in layout[offset + 7:offset + stride]:
                    key=name if name in entry['hand_limits']else name.removesuffix('_target')
                    lo, hi = entry['hand_limits'][key]
                    low.append(lo); high.append(hi)
            else:
                low.append(0.); high.append(1.)
        entry.update(lower=low, upper=high, quaternion_groups=quaternions,
                     quaternion_order='wxyz', gripper='0 = fully open; 1 = fully closed'
                     if suite != 'dexjoco' else 'Native Allegro finger joint targets in radians; retain their named order.',
                     semantics='Absolute end-effector targets in the declared native frame. '
                     'repeat is a maximum number of native control steps, not a number of completed waypoints. '
                     'The bounded controller returns a fresh observation even when the target has not been reached. '
                     'Read execution feedback before assuming contact, grasping or arrival.')
        if suite!='dexjoco':
            entry['units']='Position metres; quaternion wxyz; normalized gripper 0 open to 1 closed.'
        # The controller maps the common gripper scalar to the native endpoint.
        if suite == 'yamlab':
            entry['gripper_mapping'] = {'native_open': -0.0475, 'native_closed': 0.}
        elif suite == 'robodojo':
            entry['gripper_mapping'] = {'native_open': 1., 'native_closed': 0.}
        elif suite == 'robolab':
            entry['gripper_mapping'] = {'native_open': 0., 'native_closed': 1.}
        result = native_contract(public, max_repeat, allowed_action_types=['ee_pose'])
    else:
        raise ValueError('Cartesian adapter has not been qualified for ' + suite)
    result['control_interface'] = VERSION
    result['protocol'] = VERSION
    return result


def native_pose(action, entry):
    """Translate declared gripper units; never localize objects or choose goals."""
    result = list(action)
    mapping = entry.get('gripper_mapping')
    if mapping:
        for index in range(7, len(result), 8):
            result[index] = mapping['native_open'] + result[index] * (
                mapping['native_closed'] - mapping['native_open'])
    return result


def public_spec(spec, action_contract):
    """Advertise exactly what is callable, while preserving native frame text."""
    result = copy.deepcopy(spec)
    result['action_types'] = copy.deepcopy(action_contract['action_types'])
    result['control_interface'] = VERSION
    if 'frames' not in result and 'world' in result:
        result['frames'] = copy.deepcopy(result['world'])
    return result


class AbcCartesianRobot:
    """Cartesian actions using ABC-Sim's public, robot-only IK endpoint.

    A single bounded joint interpolation is submitted for the requested chunk.
    No private simulator state or high-level skill is available to this wrapper.
    """
    def __init__(self, robot):
        self.robot = robot
        self.native = robot.spec()
        self.last = None
        kinematics = self.native['kinematics']
        if (kinematics['frame'] != 'world' or kinematics['quaternion_order'] != 'wxyz' or
                any(name not in kinematics['ik_targets'] for name in ['left_tcp', 'right_tcp'])):
            raise ValueError('ABC native Cartesian frames differ from the reviewed public IK contract')

    def spec(self):
        result = copy.deepcopy(self.native)
        result['action_types']['ee_pose'] = {'dim': 16,
            'layout': [side + '_' + name for side in ['left', 'right']
                for name in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper']],
            'frame': self.native['world']['frame'], 'quaternion_order': 'wxyz',
            'tool_frame': 'Native left_tcp and right_tcp, as reported by public proprioception and /kinematics.',
            'units': 'Position metres; quaternion wxyz; gripper 0 open to 1 closed.'}
        return result

    def get_obs(self, **kwargs):
        self.last = self.robot.get_obs(**kwargs)
        return self.last

    def submit(self, kind, rows, **kwargs):
        import numpy as np
        if kind != 'ee_pose' or self.last is None:
            raise ValueError('ABC Cartesian execution requires a prior public observation and ee_pose targets')
        if not 1 <= len(rows) <= 25 or any(row != rows[0] for row in rows):
            raise ValueError('ABC Cartesian submit requires one repeated target and a bounded chunk')
        row = rows[0]; proprio = self.last.meta['proprio']
        state = dict(zip(proprio['layout'], proprio['now'], strict=True))
        joint_names = self.native['kinematics']['joint_names']
        seed = [state[name] for name in joint_names]
        targets = {side + '_tcp': row[i * 8:i * 8 + 7] for i, side in enumerate(['left', 'right'])}
        solution = self.robot.ik(targets, seed=seed)
        q = np.asarray(solution.get('q', seed), dtype=float)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError('Native ABC IK returned an invalid joint solution')
        limits = self.native['action_types']['joint_position']['limits']
        low, high = np.asarray(limits['low']), np.asarray(limits['high'])
        q = np.clip(q, low, high)
        q[6] = 1. - row[7]; q[13] = 1. - row[15]
        current = np.clip(np.asarray(seed), low, high)
        commands = []
        for _ in rows:
            current = current + np.clip(q - current, -.05, .05)
            commands.append(current.tolist())
        before = self.last.meta['control_steps']
        self.last = self.robot.submit('joint_position', commands, **kwargs)
        after = self.last.meta['control_steps']
        if not 0 <= after - before <= len(rows):
            raise ValueError('Native ABC Cartesian control exceeded its step budget')
        measured = dict(zip(self.last.meta['proprio']['layout'], self.last.meta['proprio']['now'], strict=True))
        errors = {}
        for i, side in enumerate(['left', 'right']):
            distance = math.sqrt(sum((row[i * 8 + j] - measured[side + '_tcp_' + axis]) ** 2
                for j, axis in enumerate('xyz')))
            errors[side] = {'position_error_m': distance, 'target_position_reached': distance <= .01}
        self.last.meta['eef_execution'] = {'requested_control_steps': len(rows),
            'executed_control_steps': after - before, 'tracking': errors,
            'native_ik': solution, 'target_reached_is_not_task_success': True}
        return self.last


def wrap_robot(robot, suite):
    return AbcCartesianRobot(robot) if suite == 'abc-sim' else robot
