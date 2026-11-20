import unittest
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

from reproductions.leact.cluster.efe_native_pilot import control_verdict,find_trial_result


def test_concurrent_controls_publish_status_without_sharing_staging_files(tmp_path,monkeypatch):
    from reproductions.leact.cluster import efe_native_pilot as native
    monkeypatch.setattr(native,'ROOT',tmp_path/'root')
    monkeypatch.setattr(native,'ARCHIVE',tmp_path/'archive')
    write_text=Path.write_text;barrier=threading.Barrier(4)
    def simultaneous_write(path,*args,**kwargs):
        result=write_text(path,*args,**kwargs)
        if path.name.endswith('.tmp'):barrier.wait(timeout=5)
        return result
    monkeypatch.setattr(Path,'write_text',simultaneous_write)
    with ThreadPoolExecutor(max_workers=4)as pool:
        list(pool.map(lambda i:native.write('progress.json',{'control':i}),range(4)))
    for root in [native.ROOT,native.ARCHIVE]:
        assert json.loads((root/'progress.json').read_text())['control']in range(4)
        assert not list(root.glob('*.tmp'))


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
