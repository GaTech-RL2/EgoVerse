"""Reject missing, replicated, reordered-contract, and malformed shard layouts."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cluster'))
from efe_qwen36_restart import adapter_shard_layout


class AdapterShardLayoutTests(unittest.TestCase):
    expected={'layer.lora_A.default.weight':[2,3],'layer.lora_B.default.weight':[3,2]}

    def rows(self,lengths):
        contract=hashlib.sha256(json.dumps(sorted(self.expected.items())).encode()).hexdigest()
        return [{'contract':contract,'lengths':row} for row in lengths]

    def test_empty_ranks_and_parameter_boundary(self):
        names,totals=adapter_shard_layout(self.expected,self.rows([[0,0],[4,0],[2,1],[0,5]]))
        self.assertEqual(names,sorted(self.expected))
        self.assertEqual(totals,[0,4,3,5])

    def test_all_parameters_on_one_rank(self):
        self.assertEqual(adapter_shard_layout(self.expected,self.rows([[0,0],[6,6]]))[1],[0,12])

    def test_missing_or_replicated_elements_fail(self):
        for lengths in ([[2,3],[3,3]],[[6,6],[6,6]],[[3,4],[3,3]]):
            with self.subTest(lengths=lengths),self.assertRaisesRegex(ValueError,'Incomplete or replicated'):
                adapter_shard_layout(self.expected,self.rows(lengths))

    def test_every_rank_must_match_contract_and_dtype_validation(self):
        for change in ({'contract':'other'},{'error':'non-FP32 parameter'}):
            rows=self.rows([[3,3],[3,3]]);rows[1].update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'rank 1'):
                adapter_shard_layout(self.expected,rows)

    def test_invalid_counts_rejected_before_tensor_collective(self):
        for lengths in ([[6,-1],[0,7]],[[6,6,0],[0,0,0]],[[6,6],[False,0]]):
            with self.subTest(lengths=lengths),self.assertRaisesRegex(ValueError,'Invalid adapter shard lengths'):
                adapter_shard_layout(self.expected,self.rows(lengths))
        with self.assertRaisesRegex(ValueError,'No adapter shards'):
            adapter_shard_layout(self.expected,[])


if __name__=='__main__':unittest.main()
