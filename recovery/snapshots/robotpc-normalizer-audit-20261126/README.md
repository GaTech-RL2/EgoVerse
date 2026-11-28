# Robotpc Cart normalizer audit — Nov26

Read-only SSH to robotpc (rl2-laptop). Latest Cart rollout found: `rollout_logs/cart_B180_msc_cone_bs_FT10k_bgobs_20261126_001423`, 12 saved observations. No robot commands, server requests, policy inference, process changes, or deployment edits.

## Findings

- Cone and ball FT10k bundles use normalizer SHA256 `8b04560000279d363f22eaf9ddb2d9a76ff2960f3aed6e8e55ad914efa78335e`. Actual checkpoint identity matches; checkpoint action_center/action_scale buffers match all49 JSON entries exactly.
- Training campaign/data.py:84–87 uses `(x-center)/scale`. Inference campaign/cart_sim.py:58 uses same proprio formula; lines85/90 normalize physical RTC prefix then denormalize output once. RobotPolicy pins only constant proprio dimensions. No yaw-specific clipping or periodic wrapping in the action normalizer.
- Base action layout is first3 channels dx/dy/dyaw. Client rollout_rby1_policy_real_bgobs.py:1688–1697 accumulates dyaw; BaseDeltaCommandAdapter transforms XY only, preserves dyaw. Recorded rollout has base_delta_frame=start. Thus normalize a per-row yaw increment, not absolute orientation; no evidence here that crossing±pi is the observed failure.
- Yaw center0.02166366577rad, scale0.27304857969rad; sparse_fallback=true, min/max[-0.2513849139,+0.2947112322]. Configuration adapts only action3,8 and state0,5 for target; yaw inherited source normalization. Physical zero normalizes to−0.07933997, NOT0. This affine offset is expected, not proof of clockwise bias. Roundtrip max error1.86e−9rad.
- Scale sensitivity:0.01normalized error ->0.156445deg/row;64same-sign rows ->10.0125deg. This is a numerical amplification risk, not proof of a bad deployment formula or actual-rollout causality. Never substitute a new normalizer at inference alone.
- Stored package reference fixtures (NOT current rollout predictions): cone64-row signed yaw sums−2.21/−4.96/−15.96deg; ball−7.17/−5.65/−13.09deg. Frozen RTC first8rows match exactly in those saved fixtures. Existing robotpc-selftest also reports GPU direct/server parity, not newly rerun here.
- Actual saved12observations: normalized joint maxabs1.6045; outside±1 in dims1,2,15,19 (torso q5/q6, leftarm q19/q23). Finite; quantile exceedance alone is not an error. EEF pose maxabs0.934. Hand input, historical flow RNG/prefix, predicted actions and odometry are not saved in these NPZs, so exact historical replay and causal localization remain unavailable from this recording alone.

Next causal test: capture or recover actual normalized output, denormalized dyaw, executed dyaw and odometry together; compare source/target dyaw distributions and spike timestamps. Keep any corrected-normalizer experiment separate and match training/inference, rather than editing deployed statistics in place.
