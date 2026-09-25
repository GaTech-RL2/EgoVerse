# Recovery input compatibility

Both recovery-v2 workers use the same protocol, checkpoint, complete reset-manifest bytes, bank-inventory bytes, package versions and assigned episodes as their original interrupted workers. Their checkpoint/packages/bank inventory also match retained original worker 2. The complete frozen plans are equal apart from runtime, whose only changed field is the workflow name.

This read-only check covers immutable inputs before case completion. It makes no claim about recovered outcomes, simulator execution, numerical-array replay or feedback bindings; those require the final development audits. The separate frozen interruption bundle remains unchanged.

`receipt.json` is the exact checker receipt. `source/check_input_compatibility.py.txt` is the exact Python source with only its filename changed for publication. Source and receipt paths contain no absolute local paths, signed URLs or credentials.

Invocation from the repository root after activating emimic:

```sh
PYTHONPATH=. python astra_reversal/.deps/interpolation-development-recovery-v2/check_input_compatibility.py --original astra_reversal/.deps/interpolation-development-v1 --recovery astra_reversal/.deps/interpolation-development-recovery-v2 --output astra_reversal/.deps/interpolation-development-recovery-v2/input_compatibility.json
```

The checker requires the immutable original and recovery metadata at the supplied directories and refuses to overwrite its receipt.
