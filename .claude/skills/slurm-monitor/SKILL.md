---
name: slurm-monitor
description: Monitor a set of SLURM job IDs every 10 minutes and report any that fail and cannot recover (non-requeue-able failures). Use when the user asks to watch/monitor SLURM jobs or check if training runs are healthy.
---

# SLURM Job Monitor

Monitor SLURM jobs on a 10-minute timer. Report failures that are terminal (cannot be requeued).

## How to invoke

The user should provide a list of job IDs to monitor, or they are inferred from context.

## Behavior

Each tick you must:

1. Run `squeue --jobs <comma-separated-ids> --format="%i %j %T %R" --noheader 2>/dev/null` to get current state.
2. Run `sacct -j <comma-separated-ids> --format=JobID,JobName,State,ExitCode,Reason --noheader --parsable2 2>/dev/null` to catch completed/failed jobs that have already left the queue.
3. Classify each job:
   - **RUNNING / PENDING / REQUEUED / RESIZING**: healthy, no action.
   - **FAILED / CANCELLED / TIMEOUT / NODE_FAIL** with `Requeue=1` or job was automatically requeued: note it but don't alert.
   - **FAILED / CANCELLED** and NOT requeued (no longer in queue, sacct shows FAILED/CANCELLED with exit code != 0:0): **ALERT** — report job ID, name, state, exit code, and reason.
   - **COMPLETED**: healthy finish, remove from watch list.
4. If all jobs are COMPLETED or terminally failed: stop the loop (do not call ScheduleWakeup).
5. Otherwise: call ScheduleWakeup with delaySeconds=600, reason="checking SLURM job health in 10 min", prompt="<<autonomous-loop-dynamic>>".

## Terminal failure detection logic

A job is considered terminally failed if:
- It no longer appears in `squeue` output (not PENDING/RUNNING/REQUEUED), AND
- `sacct` shows State = FAILED or CANCELLED, AND
- The job has not been requeued (no new sibling job with same name running).

## Reporting format

When alerting, output a clear summary like:

```
ALERT: Job 3151432 (can_sort_mink_eef_first100__train_dexmimicgen_no_mobile_joint_act32)
  State: FAILED
  Exit code: 1:0
  Reason: NonZeroExitCode
```

## Job IDs being monitored (this session)

The 18 jobs submitted for training_first100_eef_cropped × 3 dexmimicgen configs:
3151432 3151433 3151434 3151435 3151436 3151437 3151438 3151439 3151440 3151441 3151442 3151443 3151444 3151445 3151446 3151447 3151448 3151449

If the user provides a different set of job IDs, use those instead.
