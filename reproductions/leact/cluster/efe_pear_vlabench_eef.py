"""Install EEF targets on an owned VLABench training-side simulator copy.

The native robot-only IK solver provides joint targets. Each control step still
runs VLABench's original step, journal, scene validation and success predicate.
"""


def install(cls):
    import numpy as np
    original_spec, original_clip = cls.spec, cls.clip
    original_to_sim, original_command, original_step = cls.to_sim, cls.command, cls.step

    def spec(self):
        result = original_spec(self)
        result['action_types']['ee_pose'] = {'dim': 8,
            'layout': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper'],
            'lower': [-3.] * 3 + [-1.] * 4 + [0.],
            'upper': [3.] * 3 + [1.] * 4 + [1.],
            'frame': result['frames'], 'quaternion_order': 'wxyz',
            'quaternion_groups': [[3, 4, 5, 6]],
            'units': 'Position metres; quaternion wxyz; gripper 0 open to 1 closed.',
            'semantics': 'Absolute native TCP goal. Native robot-only IK is recomputed every control step; '
                'arm joint target changes are bounded by 0.05 radians per control step. '
                'An IK failure holds the arm and is returned as feedback, never a fabricated success.'}
        result['action_types']['not_yet'] = [v for v in result['action_types'].get('not_yet', []) if v != 'ee_pose']
        return result

    def clip(self, kind, rows):
        if kind != 'ee_pose':return original_clip(self, kind, rows)
        rows = np.asarray(rows, dtype=float)
        if (not np.isfinite(rows).all() or np.max(np.abs(rows[:, :3])) > 3. or
                np.any(np.abs(np.linalg.norm(rows[:, 3:7], axis=1) - 1.) > .01) or
                np.any((rows[:, 7] < 0) | (rows[:, 7] > 1))):
            raise ValueError('Invalid bounded native EEF target')
        return rows.copy(), []

    def command(self, kind):
        if kind != 'ee_pose':return original_command(self, kind)
        return np.r_[self.s.env.get_ee_pos(), self.s.env.get_ee_quat(), 1. - self.s.targets[7:9].mean() / .04]

    def to_sim(self, kind, row):
        if kind != 'ee_pose':return original_to_sim(self, kind, row)
        before = self.s.state_sha()
        solution = self.s.ik(row[:3], row[3:7] / np.linalg.norm(row[3:7]))
        if self.s.state_sha() != before:
            raise RuntimeError('Native robot IK unexpectedly modified physical simulator state')
        current = np.asarray(self.s.targets).copy()
        joints = np.asarray(solution.get('joint_positions', []), dtype=float)
        valid = bool(solution.get('success') and joints.shape == (7,) and np.isfinite(joints).all())
        if valid:
            measured = np.asarray(self.s.env.robot.get_qpos(self.s.physics))
            # The native model does not enforce Panda's documented limits.
            # Respect both its actuator bounds and the published robot limits.
            if not hasattr(self,'_pear_eef_arm_limits'):
                self._pear_eef_arm_limits=np.asarray(original_spec(self)[
                    'action_types']['joint_position']['panda_joint_limits_rad']).T
            lo, hi = self._pear_eef_arm_limits
            goal = np.clip(joints, np.maximum(lo, self.lower[:7]), np.minimum(hi, self.upper[:7]))
            current[:7] = measured + np.clip(goal - measured, -.05, .05)
        current[7:9] = .04 * (1. - row[7])
        self._pear_eef_feedback = {'native_ik_success': valid, 'target_position_m': row[:3].tolist(),
            'target_quaternion_wxyz': row[3:7].tolist(), 'target_reached_is_not_task_success': True}
        return np.clip(current, self.lower, self.upper)

    def step(self, row):
        done, success, info = original_step(self, row)
        feedback = getattr(self, '_pear_eef_feedback', None)
        if feedback:
            feedback = dict(feedback)
            feedback['position_error_m'] = float(np.linalg.norm(
                self.s.env.get_ee_pos() - np.asarray(feedback['target_position_m'])))
            info = {**info, 'eef_execution': feedback}
        return done, success, info

    cls.spec, cls.clip, cls.command = spec, clip, command
    cls.to_sim, cls.step = to_sim, step


def stage(reference, rows, module_source):
    """Adapt only an isolated run's VLABench source, preserving initial scenes."""
    import hashlib
    from pathlib import Path
    records = [];adapted_files={}
    source = Path(module_source).read_bytes()
    for row in rows:
        if row['suite'] != 'vlabench':continue
        folder = Path(reference) / row['path'] / 'environment/sim'
        path = folder / 'server.py'
        # The EFE task hook materializes shared simulator files only when
        # Harbor creates an episode. Adapt the owned central template before
        # that copy when a sparse task checkout has only its instance files.
        if not path.exists():
            path=Path(reference)/'embodied_first_exam/benchmarks/vlabench/sim/server.py'
        key=str(path)
        if key in adapted_files:
            details,adapted=adapted_files[key]
        else:
            original = path.read_text()
            marker = 'if __name__ == "__main__":'
            if original.count(marker) != 1 or 'PEAR_VLABENCH_EEF_OVERLAY' in original:
                raise ValueError('VLABench EEF overlay needs a fresh, unmodified native server copy')
            inline=source.decode().replace('def install(cls):','def install_pear_eef(cls):',1)
            adapted = original.replace(marker,
                '# PEAR_VLABENCH_EEF_OVERLAY\n'+inline+'\ninstall_pear_eef(VLABench)\n\n' + marker)
            temporary=path.with_name('server.pear-eef.tmp')
            temporary.write_text(adapted);temporary.replace(path)
            details={'overlaid_source':str(path.relative_to(reference)),
                'original_server_sha256': hashlib.sha256(original.encode()).hexdigest(),
                'adapted_server_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
                'adapter_sha256': hashlib.sha256(source).hexdigest(), 'native_instance_modified': False}
            adapted_files[key]=(details,adapted)
        # Native task Dockerfiles copy server.py from a pinned files-only
        # image after COPY . /srv. Override that one service file LAST in the
        # owned build context, preserving the pinned base, instance and grader.
        recipe=folder/'Dockerfile';dockerfile=recipe.read_text()
        if 'PEAR_VLABENCH_EEF_SERVICE_COPY'in dockerfile:
            raise ValueError('The VLABench EEF service override was already staged')
        (folder/'server.pear-eef.py').write_text(adapted)
        updated=dockerfile+'\n# PEAR_VLABENCH_EEF_SERVICE_COPY\nCOPY server.pear-eef.py /srv/server.py\n'
        recipe.write_text(updated)
        records.append({'task':row['task'],**details,'original_dockerfile_sha256':hashlib.sha256(dockerfile.encode()).hexdigest(),
            'adapted_dockerfile_sha256':hashlib.sha256(updated.encode()).hexdigest(),
            'execution_source':'Final owned Dockerfile COPY overrides the original service from the pinned files image.'})
    return records
