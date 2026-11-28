# Drawer rejected BS example: manifest index 7

Source: `unknown_20261125_004242.vrs`; recording `ea60a177-cea4-4c3d-a623-a053372dc60e`; segment `a3dfa5f8-c243-5670-bf5f-9f87c73b74ec`, 818 frames, head height 1.57 m, no LA.

## Verified finding

At frame 761 (12.6833 seconds), stored right-hand center changes from (1.621351, 0.681661, 0.979693) m to (1.636692, 0.950632, 1.820812) m: 88.3211 cm total, including +84.1119 cm vertically. The hand-displacement p99.9 is 87.0027 cm. 575/818 frames have at least one hand displaced >2 cm. These are BS-before/after differences in stored solver FK, not human tracking displacement.

An isolated CPU replay, instrumenting the geometric IK function in memory, reproduced saved joint positions, base poses, and both stored EEF arrays EXACTLY (max absolute differences zero). No original source, environment, artifact, or training was modified.

At frame 761, right-arm geometric IK returns:
- Before BS shoulder q2 = -29.12 deg; elbow raw -152.19 deg is clipped to -150 deg.
- After BS shoulder q2 raw = 169.59 deg is clipped to +1 deg; elbow raw = 8.50 deg is clipped to +1 deg.
- Shoulder q1 changes -31.44 -> 148.82 deg; shoulder q3 changes 68.41 -> -111.84 deg.

The geometric solver selects a shoulder branch using distance of q1 to its previous value, then clips individual joints. It does not first reject the branch based on all joint bounds, and post-clip wrist position is not guaranteed. The BS stage changes base heading/path and re-solves torso/arms with the existing functional/SEW solver; it is not a hard constraint holding the previous robot TCP fixed. This example confirms invalid branch selection and clipping in the BS re-solve as the immediate mechanism of the very large hand displacement. It does NOT establish this mechanism for every rejected segment.

Before BS already had limit saturation / least-squares IK. Original report limit rows 530/818 -> 203/818; therefore the baseline is not a gold-standard valid target. BS worsens hand preservation in this example despite reducing that count.

## Additional data consistency issue

`convert_human_csv_to_robot_hdf5.py` records EEF from per-frame solver FK, then smooths joint-goal buffers without recomputing EEF. Stored `robot0_joint_pos` and EEF can disagree near branch/angle-wrap jumps. This applies to the meaning of the gate: 88.3 cm refers to stored pre-smoothing FK. A later check should recompute final FK after joint filtering and enforce limits/continuity there too.

The first exploratory independent-FK script incorrectly assumed a common local SE2 origin before/after BS; DO NOT use its base/after-FK numerical estimates. The second analysis fits separate anchors and agrees approximately with the original BS world-base diagnostics, but fitted-anchor residuals are large at pathological frames. Do not use its counterfactual FK values as exact causal evidence. The exact replay and stored world EEF comparison above are the authoritative example evidence.

## Evidence on Skynet

Root: `/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125`
- Original pair: `drawer/segments/a3dfa5f8-c243-5670-bf5f-9f87c73b74ec/retarget/{fix.hdf5,fix_bs.hdf5}`
- Before SHA256: `31f4b4528aa4dd4a7f8cbaf230df40d45f11683f586c1205c8198fb17b9d0c24`
- After SHA256: `8d74470a2bb97abe81c6ee0ce5a992c07a3378bdf0600466e510453584776378`
- `reports/bs-example7-replay-20261125/{ik-trace.json,replay-verification.json,basefix.jsonl,diagnostic_bs.hdf5}`
- `scripts/replay_bs7_instrument.py`, `reports/bs7-replay.log`
- `reports/bs-example7-comparison.png`

Before Atlas: http://100.76.236.119:8080/?dataset=cxin34_drawer_1125_qc%40r1_fix_all145&episode=cba205f6-f246-5abb-b32b-9e561fb981e1

After Atlas: http://100.76.236.119:8080/?dataset=cxin34_drawer_1125_qc%40r2_fix_bs_all145&episode=a452e958-aac6-51b6-8149-ca7e4d9416cb

Inspect around 12.68 seconds. Keep this segment quarantined. Candidate repairs are branch feasibility/continuity before clipping and constrained IK preserving world TCP, followed by final filtered-q FK checks. No repair or training launch was performed in this diagnostic session.
