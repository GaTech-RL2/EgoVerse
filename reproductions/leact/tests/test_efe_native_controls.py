import unittest

from reproductions.leact.cluster.efe_native_pilot import control_verdict,find_trial_result


class NativeControlTests(unittest.TestCase):
    def test_opposite_expected_verdicts(self):
        positive={'verifier_result':{'rewards':{'reward':1.0}}}
        negative={'verifier_result':{'rewards':{'reward':0.0}}}
        self.assertEqual(control_verdict('oracle',positive)[0],'passed')
        self.assertEqual(control_verdict('nop',negative)[0],'passed')
        self.assertEqual(control_verdict('nop',positive)[0],'unexpected_verdict')

    def test_infrastructure_failure_is_not_a_valid_negative(self):
        broken={'exception_info':{'exception_type':'RuntimeError'},'verifier_result':{'rewards':{'reward':0.0}}}
        self.assertEqual(control_verdict('nop',broken)[0],'infrastructure_failure')
        missing_monitor={'verifier_result':{'rewards':{'reward':1.0,'resource_monitor_missing':1}}}
        self.assertEqual(control_verdict('oracle',missing_monitor)[0],'infrastructure_failure')

    def test_ungraded_and_student_episodes_cannot_pass_as_controls(self):
        self.assertEqual(control_verdict('nop',{})[0],'infrastructure_failure')
        with self.assertRaises(ValueError):control_verdict('codex',{})

    def test_native_harbor_dataset_namespace_is_recognized(self):
        trial={'task_name':'embodied-first-exam/metaworldplus-reach-i00-standard'}
        self.assertIs(find_trial_result([{'stats':{}},trial],'metaworldplus-reach-i00-standard'),trial)
        self.assertIsNone(find_trial_result([{'task_name':'different-benchmark/metaworldplus-reach-i00-standard'}],'metaworldplus-reach-i00-standard'))

    def test_ambiguous_control_results_are_rejected(self):
        with self.assertRaises(ValueError):
            find_trial_result([{'task_name':'task'},{'task_name':'embodied-first-exam/task'}],'task')


if __name__=='__main__':unittest.main()
