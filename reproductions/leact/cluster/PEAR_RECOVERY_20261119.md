# PEAR checkpoint interruption and recovery safeguards — November 19, 2026

The startup32 hero completed four actor updates on 128 usable groups. Its shared
telemetry contains 3,227 completed training episodes and 576 native successes;
only the first three iteration metric records were published. The fourth actor
update returned at 17:23 UTC, before synchronous checkpoint archival.

At 19:37 UTC, the trainer reported that its Ray GCS was unreachable and exited.
The immediate cause of the GCS loss is not established. OSMO subsequently ended
the model allocation at 19:55 UTC through `groot-idle-job-shutdown`, explicitly
reporting GPU quota enforcement against `pretrain-extension`. An automatically
requested replacement was also quota-reclaimed at 20:05 UTC. These were external
allocation terminations, not agent-issued cancellations.

The interrupted archive is incomplete: eight complete model shards and six
complete optimizer shards survive in private object storage. Optimizer ranks six
and seven and part of the TransferQueue remain incomplete multipart uploads.
There is no committed archive manifest and no `checkpoint-latest.json` pointer.
Four completed gradients must not be described as four resumable updates.

Two recovery bugs required changes:

1. A failed allocation with a nonempty historical `start_time` could reach native
   worker preparation before the terminal retry branch. Failed/no-child model
   allocations now leave independently retained native jobs untouched.
2. Model startup defaulted to initialization whenever the checkpoint pointer was
   absent, even if the same run directory already contained optimizer history.
   Startup now refuses that implicit reset. Existing telemetry and artifact
   provenance remain intact.

The primary controller also honors a durable explicit checkpoint-recovery hold:
it maintains the existing standby request without activating a new writer.
The user was asked to choose between a separately labeled recovery from the
surviving step-four weights with a fresh optimizer, or a fresh replicate from
warmstart496 using the original recipe. Neither choice is applied automatically.

Recovery inspection artifacts live under the study's `pear-scaled-v1` directory:
`hourly-reclaimed-primary-telemetry.json`,
`hourly-primary-driver-error-tail.json`,
`hourly-interrupted-checkpoint-object-census.json`, and
`checkpoint-recovery-required.json`. Raw private credentials are excluded.

Validation: 36 focused primary/ownership/recovery tests passed; the complete
research suite passed 765 tests with four skips and eight subtests. The required
root `tests/unit` invocation was attempted but could not collect because this
local environment lacks `lightning`.
