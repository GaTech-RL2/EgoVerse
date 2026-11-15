"""Research split regressions: core twins and development families cannot leak."""
import unittest

from reproductions.leact.embodied_first_exam.prepare import (
    base_name, check_partitions, research_split,
)


def row(base, family, mode="standard", *, core=False):
    return dict(task=f"{base}-{mode}", base_task=base, suite="suite", family=family,
                mode=mode, official_core=core, official_default=core and mode == "standard")


class EmbodiedSplitTests(unittest.TestCase):
    def test_core_privileged_twin_cannot_become_training_data(self):
        rows = [row("test-i00", "same", core=True),
                row("test-i00", "same", "privileged", core=True),
                row("other-i00", "other")]
        split = research_split(rows, {"test-i00": "standard"})
        self.assertEqual(len(split["protected_core_all_modes"]), 2)
        self.assertEqual(split["train_candidates"], ["other-i00-standard"])
        split["train_candidates"].append("test-i00-privileged")
        with self.assertRaisesRegex(ValueError, "split boundary"):
            check_partitions(rows, split)

    def test_same_family_instances_stay_together(self):
        rows = [row(f"f{f}-i{i}", f"family{f}") for f in range(10) for i in range(3)]
        split = research_split(rows, {})
        self.assertEqual(len(split["development"]), 6)
        assignments = {}
        for item in rows:
            assignments.setdefault(item["family"], set()).add(item["partition"])
        self.assertTrue(all(len(values) == 1 for values in assignments.values()))

    def test_split_independent_of_inventory_order(self):
        rows = [row(f"task{f}-i00", f"family{f}") for f in range(20)]
        self.assertEqual(research_split(rows, {}), research_split(list(reversed(rows)), {}))

    def test_realtime_suffix_not_misidentified_as_a_different_task(self):
        self.assertEqual(base_name("task-i00-standard-realtime", "standard-realtime"), "task-i00")
        with self.assertRaises(ValueError):
            base_name("task-i00-standard", "privileged")


if __name__ == "__main__":
    unittest.main()
