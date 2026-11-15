"""Storage recovery must retain final saves and the original training lineage."""
import json
from pathlib import Path
import tempfile
import unittest

from reproductions.leact.cluster.efe_training_state import restart_due, choose_resume, FORMAT


class StorageScheduleTests(unittest.TestCase):
    def test_existing_policy_and_final_only_recovery(self):
        policy = {'origin_step':0, 'restart_interval':4}
        self.assertEqual([s for s in range(13,19) if restart_due(policy,s,12,[18])], [13,16,18])
        recovered = {**policy, 'save_restart_steps':[18]}
        self.assertEqual([s for s in range(13,19) if restart_due(recovered,s,12,[18])], [18])
        self.assertTrue(restart_due(recovered,17,12,[17]))
        self.assertFalse(restart_due(None,18,12,[18]))

    def test_external_committed_restart_remains_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'retained-checkpoint'; source.mkdir()
            (source/'weights').write_bytes(b'verified-test-weight')
            receipt = {'format':FORMAT, 'lineage_sha256':'fixed', 'step':12,
                'files':{'weights':{'bytes':20}}}
            (source/'training-state.json').write_text(json.dumps(receipt))
            states = root/'new-states'; states.mkdir()
            (states/'checkpoint-12').symlink_to(source, target_is_directory=True)
            policy = {'state_directory':str(states), 'origin_step':0, 'target_step':18,
                      'lineage_sha256':'fixed', 'save_restart_steps':[18]}
            chosen = choose_resume({'durable_restart':policy})
            self.assertEqual(chosen['step'],12)
            self.assertEqual(Path(chosen['path']).resolve(),source.resolve())
            self.assertEqual(chosen['format'],FORMAT)


if __name__=='__main__':
    unittest.main()
