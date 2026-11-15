"""Leakage and taxonomy checks against the frozen benchmark inventory."""
import json
import gzip
from pathlib import Path
import unittest

from reproductions.leact.embodied_first_exam.generalization import classify, task_identity, validate

FIXTURE = Path(__file__).parent / 'fixtures/efe-generalization-v2.json.gz'


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        # The task-identity projection is versioned; no local run artifacts or
        # simulator installation are needed to check the frozen split.
        with gzip.open(FIXTURE, 'rt') as stream:
            fixture = json.load(stream)
        self.rows = fixture['inventory']
        self.split = fixture['splits']
        self.by_name = {r['task']:r for r in self.rows}

    def test_frozen_split_valid(self):
        validate(self.rows, self.split)

    def test_reserve_simulator_cannot_enter_development(self):
        self.split['development'].append(self.split['simulator_holdout_reserve'].pop())
        with self.assertRaisesRegex(ValueError, 'Held-out simulator'):
            validate(self.rows, self.split)

    def test_core_alternate_mode_cannot_enter_training(self):
        name = next(n for n in self.split['protected_core_all_modes'] if n.endswith('-privileged'))
        self.split['train_candidates'].append(name)
        with self.assertRaisesRegex(ValueError, 'mode twin'):
            validate(self.rows, self.split)

    def test_omitting_a_difficult_core_task_is_rejected(self):
        self.split['official_core_evaluation'].pop()
        with self.assertRaisesRegex(ValueError, 'membership'):
            validate(self.rows, self.split)

    def test_unseen_suite_is_not_necessarily_unseen_simulator(self):
        test = next(r for r in self.rows if r['suite']=='robocasa' and r['official_default'])
        training = [r for r in self.rows if r['suite']=='libero-max' and r['task'] in self.split['train_candidates']]
        self.assertTrue(training)
        self.assertEqual(classify(test, training)['cohort'], 'unseen_suite_seen_simulator')

    def test_sapien_is_held_out_runtime_but_not_held_out_physics(self):
        rows = [r for r in self.split['core_cohorts'] if r['simulator_runtime']=='sapien']
        self.assertTrue(rows)
        self.assertTrue(all(not r['runtime_seen_in_training'] and r['physics_engine_seen_in_training'] for r in rows))

    def test_libero_perturbation_is_not_semantic_goal(self):
        a = dict(suite='libero-max', family='libero_goal/target_relocation', base_task='libero-max-put-mug-target-relocation-i00')
        b = dict(a, base_task='libero-max-turn-stove-target-relocation-i00')
        c = dict(a, family='libero_goal/object_recolor', base_task='libero-max-put-mug-object-recolor-i00')
        self.assertNotEqual(task_identity(a), task_identity(b))
        self.assertEqual(task_identity(a), task_identity(c))


if __name__ == '__main__':
    unittest.main()
