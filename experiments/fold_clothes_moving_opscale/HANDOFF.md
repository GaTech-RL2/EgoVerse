# Moving-budget folding experiment handoff

## User's current instruction

Run all preflight checks only. **Do not submit the 36 production training runs.** The user explicitly withdrew the earlier launch instruction. GPU smoke tests, including their bounded training/resume checks, remain part of preflight. Any production launch requires a new user instruction; preserve the run-root `production_control.json` with `allow_training: false`.

## Repository and experiment

The requested starting state is commit `dd87495a7dd7cc71a7865c0c6a738f24c3cbc9b4`. Descendant setup commits add the full moving-budget experiment and corrected data verification. The transfer bundle includes the subsequent setup work; its external handoff receipt records the exact final branch/SHA and bundle prerequisites.

The 36 cells cross budgets 24, 32, 48, 64, 72, 96 hours with 4, 8, 16, 32, 64, 128 operators. Operator prefixes and nested episode prefixes are fixed. The reference RDT has width 1280, depth 18, 20 heads, approximately 300M DiT parameters (402.56M trainable with DINOv3), frozen Qwen, and 144-D wrist-frame actions. Keep training batch 64, validation batch 32 and **800 batches per cohort**, seed 42026, 400000 steps, WSD 7000/323000/70000, bf16 and compile enabled, EMA disabled. Use H100/H200 only; current preflight is configured for one H200.

Validation cohorts are seen_matched (40 episodes), seen_perlevel (3 per training operator), and unseen (400 fixed episodes from 20 held-out operators). Preserve exact reads, holdouts, banned IDs, history/null-token behavior, pinned videos and all reference normalization settings. See `reference_recipe.json`, `GUIDE.md`, generated configs and private design for details.

## Pace state and paths

Run root: `/storage/project/r-dxu345-0/acheluva3/egoverse-runs/fold-clothes-moving-opscale-20261101`.

Private data/design root: the run root's `private/`. Input manifests, exclusions, operator IDs and 36 split files remain private and are deliberately outside Git. Canonical training data is `private/canonical_zarr`; raw downloads are `private/zarr`. Both are already present on Pace. Destination source: `s3://private-mecka/private_opscale/`.

Python: `/storage/project/r-dxu345-0/acheluva3/EgoVerse-graph/.venv/bin/python`. Source that environment before project Python commands; prepend this checkout and `RUNROOT/python-deps` to PYTHONPATH. Existing credentials are provisioned through `~/.egoverse_env`; source with export enabled without printing values. Do not change credentials or inspect unrelated secrets. Shared pretrained caches are under `/storage/project/r-dxu345-0/acheluva3/.cache`.

Do not modify the canonical `EgoVerse-graph` checkout: a separate all-Mecka task owns it. Use the isolated checkout recorded in `run_manifest.json` and `operations/recovery_config.json`. Verify the clean full SHA before any job submission.

## Verified data and resolved issues

Full selected-data verification job **14969140** completed successfully: **12669 episodes, 38271219 retained frames, 240711 destination objects, zero failures**. `readiness.json` and `private/canonical_zarr/_episode_index.json` contain the actual proof. All required arrays and selected objects are present. This proves readiness of this selected cohort, not completion of the separate whole-bucket migration/deletion workflow.

Two destination-only recovered episodes are intentional: `6a942c3d1f68e301d7b3a591` and `6a92af6102944ea8c09c362e`, totaling 38 objects and 478005085 bytes. Keep them. `private/recovery2/report.json` records conversion hashes and frame counts. Never delete or overwrite R2 objects during experiment recovery.

The canonical view corrects only legacy left-wrist orientation and preserves exact all-zero-keypoint/full-positive-1e9 missing-hand sentinels. Provenance version is `left-wrist-local-rz180-v2`; stale v1 views are rejected. Source arrays/frames and raw/R2 objects remain unchanged. Earlier v1 job/view evidence is archived under `operations/sentinel-20261101`.

Normalization array **14974234** failed all 36 tasks because inherited local staging tried to create `/ephemeral` on Pace. The isolated repair adds an explicit `stage_to_local` option (default True) and disables staging only for this experiment's already verified shared canonical data. Selection, metadata, keymaps, transformations and sample values are preserved. The passed data proof is retained unchanged after independent hash review; normalization and GPU smoke require fresh results for the new source commit.

## Continue preflight safely

First read current `pipeline_status.json`, `status.json`, `pipeline.json`, `operations/supervision.json`, `operations/dispatcher_status.json`, `operations/recovery_watch_status.json`, `operations/recovery_state.json`, and `production_control.json`. Use live Slurm to reconcile actual jobs. Historical job IDs above are evidence, not instructions to resubmit or cancel anything.

The conductor performs canonical readiness, 36 split-specific normalization tasks (array concurrency 8), then an H200 smoke test once the small and large operator smoke cells have normalization. The smoke checks real model loading, finite training loss, checkpoint/optimizer/scheduler continuation, all validation cohorts at the 800-batch limit, metrics and actual W&B videos. All 36 normalization reports and smoke evidence are required for final preflight completion.

The conductor must stop at `PREFLIGHT_COMPLETE`; the production launcher independently checks the persisted authorization before submitting anything. Never remove a failed episode, lower the validation limit, change the model or schedule, fabricate passed reports, or bypass readiness/normalization/evaluation checks to make progress.

Do not duplicate active jobs or agents. Submission intents and full process identity (PID, start time, host, command) are authoritative. Account for all array tasks, not only a parent job. Old-job terminal state is required before a retry. Keep logs and failed receipts when superseding work.

## Monitoring and repair system

Reusable supervisor, incident controller, repair dispatcher, prompts, schemas and promotion helper are versioned under `operations/` here. Runtime copies and state live under `RUNROOT/operations`. Install reviewed code while services are quiesced; preserve all runtime state and receipts. The supervisor checks every 120 seconds. Diagnostic agents run only for actionable incidents; code changes require independent review before application. Related array-failure symptoms must share one incident and retry budget.

One agent owns an incident at a time, with a renewable lease, bounded retries/cooldown and durable receipts. A process or submission with ambiguous identity stops automatic action. Scientific changes and production training are forbidden under the current instruction. Recovery-system faults are observer-only to avoid recursive repairs. Local alert files do not themselves send chat messages; a separate desktop task may observe them.

Use `manual_ownership.json` to reserve incidents before manual repair and disable `dispatcher_control.json` for coordinated maintenance. Do not delete claims to force a duplicate worker. See the packaged operations documentation and runtime `RUNBOOK.md` for detailed controls.

## Evidence and transfer

`RUNROOT/operations/normalization-storage-20261101` contains storage-repair tests, real-data probes and independent review. `RUNROOT/operations/sentinel-20261101` contains canonical-view repair evidence. The external handoff receipt records the final tests, clean Git SHA, current jobs and remaining preflight work at transfer time. Do not interpret a clean Git state as proof that all preflight has finished.

Run `pytest tests/unit` before committing source changes. Use CPU allocations and node-local temporary directories for the suite; project dependencies are in the existing environment plus `RUNROOT/python-deps`. Optional Pi tests need canonical `external/openpi/src` and the existing all-Mecka `source_test_overlay`, for tests only. Do not put that test-only overlay into production runtime. Request a GPU for actual training/smoke execution.
