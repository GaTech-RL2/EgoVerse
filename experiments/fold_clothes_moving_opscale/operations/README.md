# Portable experiment operations

Scope is read from the private run root's `production_control.json`; no control or invalid control grants production permission. Explicit `{"allow_training": false}` remains supported for preflight: data readiness, normalization and short GPU evaluation/checkpoint-resume smoke checks. Production incident repair requires all of:

```json
{"schema_version": 1, "allow_training": true, "scope": "production", "user_instruction": "The actual renewed user authorization"}
```

The manager records the actual instruction after user authorization; agents cannot change this file. Optional audit fields, such as an authorization receipt path, grant no permissions by themselves. Invalid controls block autonomous repairs and local worker restarts while observers remain available. Each repair/review/apply sequence binds the exact control checksum; any change, including revocation or scope expansion, stops that attempt for reassessment. A later attempt must read current authorization afresh.

Production scope permits verified operational incident repairs and existing full-state continuation under unchanged experiment gates. The existing pipeline owns initial submission after all readiness, normalization and GPU evaluation gates; agents must not submit initial runs directly. Production source migration remains blocked, as do blind reruns and checkpoint or W&B identity changes. The operations package may be installed from a separately reviewed commit without changing the active training source pin, normalization provenance or recovery_config identities.

`PREFLIGHT_COMPLETE` remains an expected, healthy conductor/monitor stop; it does not mean training completed. Observers keep checking quietly. A new authorization alone is not evidence of completed gates or completed training.

This versioned package contains only reusable code, policies, prompts and schema. Live credentials, private data/splits, run/job identities, incident history, logs and receipts belong outside the repository in the run root. Do not copy those artifacts into Git or a source bundle.

## Install reviewed code

Run `python experiments/fold_clothes_moving_opscale/operations/install_operations.py --run-root "$RUN_ROOT"` to print the file manifest and package SHA without writing runtime state. Review and test that exact package. Disable dispatch and stop/reconcile the operations supervisor, recovery watcher and dispatcher using their recorded process identities; leave experiment jobs and their manifests untouched.

The installer accepts either explicit preflight control or the complete production authorization above, and records its checksum. It never writes authorization or changes training identities. Then run the same helper with `--apply --expected-package-sha REVIEWED_SHA`. It refuses active/expired repair claims, active source migration, enabled dispatch, live operations services or any checksum mismatch. It backs up old code, writes a resumable installation intent, atomically replaces only the explicit code/policy allowlist, and preserves every other runtime file. It never starts services or jobs. An interrupted partial install must be resumed with the same reviewed package while services remain stopped.

## Services and manual ownership

All service commands run from the installed `$RUN_ROOT/operations` directory. Initialize `recovery_controller.py init` using the actual clean repository/full SHA, Python executable, exact conductor/monitor PIDs and allowed H100/H200 GPU type. Start `supervise.py`, `recovery_controller.py watch`, and `agent_dispatcher.py serve` with those identities; each command exposes `--help`. Keep `dispatcher_control.json` disabled until review and an authenticated, restricted Codex CLI smoke pass.

Manual ownership is an object or list in `manual_ownership.json`, with `owner`, `alert_ids`, and `expires_at` (UTC/Unix timestamp or null for an explicit persistent hold). Either an action-group ID or any original member alert ID holds the whole repair. Never delete claim/process history to force a retry. Disable dispatch to stop new attempts; taking manual ownership of an active incident stops its agent but does not cancel Slurm jobs.

The dispatcher invokes the user's configured model through `codex exec --approve-for-me --json --output-schema ... --output-last-message ...`. It never bypasses approvals or starts a model while healthy. One agent runs at a time; claims last 30 minutes and renew every two minutes. Each incident has a one-hour repair/review/apply deadline, a 30-minute cooldown and at most three attempts. Transport timeouts receive at most three renewal attempts, bounded by the last acknowledged lease with a 60-second margin; explicit rejection or uncertain ownership stops the agent.

## Incidents, review and promotion

Array-task, per-cell normalization, log and pipeline symptoms for one action/allocation form one repair incident. Every original alert remains in `member_alerts`. Existing per-symptom attempts and receipts are merged without resetting the repair budget. Independent action/job identities remain separate; generic pipeline blockage cannot consume a duplicate repair budget.

Code fixes require tested candidates, a fresh independent reviewer and a separate apply agent. Fixed results must list the exact reviewed code destinations and matching applied bytes, plus observed evidence. Source fixes use separate clean `fold-moving-repair-*` worktrees beside the pinned repository, never under the run root. Candidate copies/evidence stay in the incident workspace. `promote_preproduction.py` generates a plan after source review, then applies it only with no production registry, terminal recorded jobs, unchanged scientific inputs and exact source/process identities. It archives invalidated readiness/norm/smoke evidence and rebuilds gates. Production source migration is forbidden. Agents cannot edit the dispatcher, controller, promotion safety gate or protected policies.

The controller marks reported fixes `AWAITING_VERIFICATION`; the original alert must clear before resolution. Observer-only health alerts never dispatch a repair of their own recovery controls. Ambiguous launches, stale process identities, expired claims, failed approval review or exhausted budgets stop for review. A launcher skipping an existing entry is not a successful relaunch; generic retries require a reviewed helper and truthful verification.

Tests live under `tests/unit/test_fold_ops_*.py`; run them with ordinary repository pytest. They include synthetic controller/agent execution and never call a model, Slurm or R2.
