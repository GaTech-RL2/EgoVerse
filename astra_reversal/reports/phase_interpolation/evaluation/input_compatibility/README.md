All eight evaluation workers passed this read-only input compatibility check. Their checkpoint metadata and bank inventory bytes match the completed development runs. Source/payload identity, recorded package versions, L40S hardware and TF32-off settings also match. The protocol changes only the selected seed, from 19 to 29.

The assigned episodes cover exactly 20 unique cases: one reset for each of ten Goal and ten Spatial tasks at seed 29. Each suite's four workers have byte-identical full reset manifests and the prescribed shard assignments. Development references are bound to the final archive audits for recovery workers 0/1 and retained original worker 2.

For all three overlapping development tasks, the checker verified the saved tensor hashes and measured actual differences in both dynamic state and model body poses. Task instructions and BDDL hashes remain unchanged.

| Overlapping task | Changed dynamic-state values | Changed body-position values | Changed body-quaternion values |
| --- | ---: | ---: | ---: |
| Goal task 6 | 8 | 6 | 0 |
| Spatial task 2 | 10 | 4 | 2 |
| Spatial task 8 | 7 | 4 | 2 |

[receipt.json](receipt.json) records the input hashes, exact assignments, reset differences and checker dependencies. [The checker source](source/check_input_compatibility.py.txt) is copied byte-for-byte; [publication_manifest.json](publication_manifest.json) binds the public files. The operational checker tests remain private.

This receipt validates saved inputs before rollout completion. It does not certify evaluation outcomes or replace the full array and feedback audits. Reset comparisons concern saved pre-stabilization arrays; no simulator or model replay occurred. The task compositions remain known, so the new reset seed does not establish zero-shot generalization.

To repeat the check from the repository root in an activated project environment, with the downloaded inputs available:

```sh
PYTHONPATH=. python astra_reversal/reports/phase_interpolation/evaluation/input_compatibility/source/check_input_compatibility.py.txt \
  --deps-root astra_reversal/.deps \
  --development-reports astra_reversal/reports/phase_interpolation/development \
  --output astra_reversal/.deps/interpolation-evaluation-v1/recheck.json
```

The checker refuses to overwrite a receipt. It reads the source metadata without changing live downloads.
