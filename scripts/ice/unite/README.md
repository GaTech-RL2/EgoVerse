# UNITE ICE contracts

These small UNITE-specific validators sit beside the shared ICE runner. The
single-GPU entrypoint is `../launch_unite.sbatch`; it accepts either H100 or
H200 and binds training to one rank.

## What each tool owns

- `unite_contract.py` validates the clean register-sweep rows and their
  14:1 update ratio, noise start, Dopri5, EMA, checkpoint, and EnergyScore
  contracts.
- `requeue_integration_contract.py` validates the deterministic step 3 to 8,
  requeue, strict step 8 resume, and step 8 to 11 proof trace.
- `fixed_step_requeue_boundary.py` provides the matching exact-step smoke
  boundary.

## Safety boundary

All deployment-specific paths, W&B identity, Python interpreter, source
checkout, and exact source commit are inputs. Missing or ambiguous values fail
closed. Checkpoints use the shared strict Lightning validator and generic
runner; completed-run archival is handled independently by the CPU mirror pool.

Before a production launch, use the live cluster protocol, run its official
preflight, then complete a real optimizer-plus-validation smoke. Passing unit
tests alone does not authorize a training submission.

## Tests

Run the focused contract suite from the repository environment:

```bash
source emimic/bin/activate
python -m pytest -q \
  tests/test_hydra_override.py \
  tests/test_unite_ice_contract.py \
  tests/test_unite_fixed_step_requeue_boundary.py \
  tests/test_unite_register.py
```
