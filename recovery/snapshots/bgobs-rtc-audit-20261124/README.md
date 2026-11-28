# bgobs observation / RTC audit — 2026-11-24

Result: isolated client fixes and offline 12/13/21 tests pass. No robot command,
camera startup, policy-server restart, training mutation, or active-client replacement.

## Findings and fixes

1. The original bg worker captured body state/odometry but read `last_hand_left`
   and `last_hand_right` later, after depth/cloud construction. These are last
   commanded hand targets, not measured hand feedback; the main loop updates
   them while the worker runs. The fixed worker copies them at submission.
2. The original acquired cameras in the worker, after the main-thread body/RTC
   snapshot. The fixed prefetch copies the newest available stereo pair and RGB
   together with body/odometry/hand inputs before scheduling. Camera reads use
   zero freshness-wait; depth/cloud computation remains in the worker.
   Missing snapshotted RGB does not silently get replaced by a later frame.
3. The original applies `smooth_chunk` again to server-frozen rows, breaking the
   frozen-prefix contract. Fixed code filters the chunk then restores the frozen
   prefix unchanged. First unconstrained chunks keep the original filtering.
4. Saved timing fields previously labeled the worker start as snapshot time and
   RGB use-after-depth as RGB read time. Fixed timestamps identify the main-thread
   snapshot and camera-copy completion separately from worker build time.

## Evidence

- `offline-results.json`: original standalone RTC test, period13 -> 51-row tail,
  pending-request exclusion, delta0/1/5/11/12/13/20 slicing, late flags,
  actual extracted worker capture, and actual filter block. Synthetic old frozen
  max change0.98253 versus fixed0 (synthetic action units, not robot error).
  Passed on Skynet and robotpc. Python AST and launcher `bash -n` pass.
- `model-results.json`: actual portable Cart fix FT10k checkpoint, 3 recorded
  fixtures; CPU ordinary and constrained inference finite64x49; old tail offset13,
  freeze12/soft21, frozen-prefix max error0 for every fixture.
  This tests the actual server policy adapter directly, not a websocket round trip.
- CPU4854571 on hal was cancelled after slow initialization with no result;
  bounded retry4854597 excluding hal COMPLETED0. Unique intents retained.
- `recorded-timing.json`: first15 already-saved observations of the existing
  Cloth rollout (14 prefetch). Absolute RGB/stereo timestamp difference median
  37.10ms/max165.16ms; stereo age relative to old worker-build timestamp median
  50.09ms/max788.13ms. This is not a Cart runtime test and sensor timestamps are
  not established as exposure-clock aligned.

## Files on robotpc

Repository: `/home/aloha/RB_Y1_workspace/SEW-Geometric-Teleop/projects/rby1_teleop/`

- `rollout_rby1_policy_real_bgobs_fixed.py`
- `run_rollout_pcd_policy_bgobs_fixed.sh`

The additional launcher calls the additional fixed client; all existing env knobs
are preserved. It has NOT been run. Original `*_bgobs.py`, `*_bgobs.sh`,
shared `utils/rtc_client.py`, server, checkpoint and assets are unchanged.
The existing Cloth bgobs process was observed active and was left alone.
The named fixed files do not affect it or become active automatically.

Audit copy: `/home/aloha/RB_Y1_workspace/bgobs-rtc-audit-20261124/`.
CPU/model evidence: `/coc/flash7/cxin34/bgobs-rtc-audit-20261124/`.
`patch_offline.py` is a count-checked transformation of the captured generated
client. It produces a separate fixed file; it does not edit the upstream generator.
If upstream is regenerated/changed, rebase this patch and repeat the tests.

## Limits

No real-time/hardware validation was performed. Software snapshot consistency
does not synchronize physical exposures with joint sampling. True exposure-time
alignment needs a common clock plus timestamped state/frame history. No such
interpolation or arbitrary new freshness cutoff was introduced here.
Hands retain the original commanded-target convention. Sampling/copy overhead
and end-to-end latency on the live robot have not been measured for the fixed copy.
12 frozen rows only cover the corresponding consumed-row window; delta>12 is
still late, and delta12 is the first non-hard-frozen row. Period13 is an action-row
count, not a guaranteed wall-clock duration. Original late/hold behavior remains.
