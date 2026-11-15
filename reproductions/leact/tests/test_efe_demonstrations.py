"""Synthetic receipts test admission boundaries, not simulator replay quality."""
import unittest

from reproductions.leact.embodied_first_exam.demonstrations import admit
from reproductions.leact.embodied_first_exam.prepare import encode, sha


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        self.split = dict(train_candidates=['train'], official_core_evaluation=['test'], development=['dev'],
                          simulator_holdout_reserve=['reserve'], held_out_runtimes=['sapien','genesis'])
        self.rows = [dict(task='train', suite='robocasa', semantic_task='cabinet', simulator_runtime='robosuite',
                          mode='standard', official_core=False, upstream_digest='a'*64)]
        self.episode = dict(dataset_url='https://example.invalid/fixture', dataset_revision='fixture-rev',
                            dataset_task='cabinet', license_id='fixture-license', simulator_runtime='robosuite')
        for i, key in enumerate(('episode_sha256','native_actions_sha256','public_observations_sha256','scene_identity_sha256','adapter_sha256')):
            self.episode[key] = sha(str(i).encode())
        self.binding = {**self.episode, 'efe_task':'train', 'semantic_task':'cabinet', 'mode':'standard',
                        'reviewed':True, 'split_sha256':sha(encode(self.split))}
        self.audit = dict(complete=True, split_sha256=sha(encode(self.split)), audited_protected_tasks=['test','dev','reserve'],
                          episode_sha256=self.episode['episode_sha256'], scene_identity_sha256=self.episode['scene_identity_sha256'],
                          collisions=[], method='canonical_scene_and_initial_state')
        self.replay = {**self.episode, 'efe_task':'train','upstream_digest':'a'*64, 'success':True,
                       'public_observation_boundary_passed':True,'action_fidelity_passed':True,
                       'input_format':'rgb_proprioception','target_format':'native_action_sequence'}

    def admit(self):
        return admit(self.episode,self.rows,self.split,binding=self.binding,scene_audit=self.audit,replay=self.replay)

    def test_verified_receipts_preserve_native_target(self):
        result=self.admit()
        self.assertEqual(result['native_actions_sha256'],self.episode['native_actions_sha256'])

    def test_held_out_task_binding_is_rejected(self):
        self.binding['efe_task']='test'
        with self.assertRaisesRegex(ValueError,'training task'): self.admit()

    def test_held_out_simulator_is_rejected_even_if_bound_to_train_name(self):
        self.rows[0]['simulator_runtime']='sapien'
        with self.assertRaisesRegex(ValueError,'Held-out simulator'): self.admit()

    def test_matching_suite_but_new_goal_is_not_enough(self):
        self.binding['semantic_task']='stove'
        with self.assertRaisesRegex(ValueError,'goal differs'): self.admit()

    def test_scene_collision_is_rejected(self):
        self.audit['collisions']=['test']
        with self.assertRaisesRegex(ValueError,'collides'): self.admit()

    def test_partial_scene_audit_is_rejected(self):
        self.audit['audited_protected_tasks'].remove('dev')
        with self.assertRaisesRegex(ValueError,'omits'): self.admit()

    def test_replay_of_another_action_sequence_is_rejected(self):
        self.replay['native_actions_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'native_actions'): self.admit()

    def test_hidden_state_and_invented_program_targets_are_rejected(self):
        self.replay['input_format']='privileged_state'
        with self.assertRaisesRegex(ValueError,'Hidden simulator state'): self.admit()
        self.replay['input_format']='rgb_proprioception'
        self.replay['target_format']='invented_expert_program'
        with self.assertRaisesRegex(ValueError,'invented expert code'): self.admit()


if __name__=='__main__':
    unittest.main()
