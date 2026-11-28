# Drawer diagnostic Atlas export

Canonical workspace: `/coc/flash7/czhang883/warp-data/registry/cxin34-drawer-diagnostic-20261125`.

Pilot: [Open Atlas](http://100.76.236.119:8080/?dataset=cxin34_drawer_1125_diagnostic%40r1_failed_pilot&episode=78040270-b3c6-5ae6-932e-136a2679e2fc).

The user identifies pilot segment15 as a post-task collection tail, not a task segment. Do not use its quality failures as representative task evidence. The annotation is saved in remote user_annotations.json and included by the final publisher. Original QC measurements remain unchanged.

Real FFS stereo depth is projected using recorded device poses and calibration. These are diagnostic 2048/1024/1024 point samples, not the HD2 training crop or virtual-camera filtering recipe. Pre-BS knuckle registration is attempted with its original acceptance result reported, never relaxed into a training pass. Failed registration means approximate scene/robot alignment. Original q, base, actions, RGB and timestamps are unchanged; display-only EEF transforms are recomputed from saved q to avoid stale-FK rendering contradictions.

Pilot146frames: API first/middle/last chunks passed, adapter14, pointcounts2048/1024/1024. Browser actual playback, robot rendering and allthree cloud tabs visually checked. Registrationfailed with17.81cm median, clearly diagnostic only. SourceHDF hashes verified unchanged.

Batch:114recordings/145segments.24existingdepthcaches reused;90missingrecordings get newFFS depth. Unique CPUarray4874455 cap2, A40array4874456 cap2, afteranypublisher4874457. At23:22EST bothCPU andbothGPUworkersRUNNING;2segments built (15,113), no recordederrors. No Drawertraining launched; no existingtraining/jobs changed. Finalpublisher emits completion.json and r2_diagnostic_<count>segments, preserving missing/error lists. Inspect receipts and actualsqueue/sacct before recovery, never blindly resubmit or deleteintent. Failedattempts must be preserved.
