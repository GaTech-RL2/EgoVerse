# Real-World Deployment Debug Guide — RBY1 Whole-Body Image Policy

> Field guide for debugging `crop100_2k` / `dino_lora_2k` on the real robot.
> Contract & serve commands: `deployment_plan.md` (§1, §8.0). Architecture/params:
> `policy_model_card.md`. Everything below assumes the offline facts: these policies
> predict the 30 demos to ~0.7-1.0° MAE from dataset obs — **if the robot misbehaves,
> the bug is almost always in obs feeding or action execution, not the policy.**

## 0. Golden rules (violating any one of these reproduces a known failure)

1. **Image**: raw fisheye, SQUARE 224×224, uint8, **BGR**, no rectification/undistortion.
   Live camera at 1280×720 must be cropped square THEN resized — never aspect-squashed.
2. **Proprio**: radians; `robot0_joint_pos` = 26-D robot state `[4:26]` in order
   torso(6), r_arm(7), l_arm(7), head(2). Tolerant to ≤3° noise; NOT tolerant to
   ordering swaps or degrees.
3. **Actions come back un-normalized** — execute directly; never scale/normalize on the
   receiver.
4. **Base `[0:3]` is per-step deltas** — integrate by plain cumsum in the start frame;
   never rotate deltas by the running yaw (known export bug pattern; warps XY, yaw stays right).
5. **Targets are 10 Hz trajectory samples** — interpolate between them at control rate;
   receding horizon K=4-8 of 32, then re-query.
6. Serve from a checkout with the updated `egomimic/algo/hpt.py` + `hpt_nets.py`
   (branch `rby1_aria_policy`) — crop100's eval-time proprio clamp lives in code.
7. `dino_lora_2k` must ALWAYS receive real proprio (zeros break it: 0.122 MAE).
   `crop100_2k` accepts real proprio AND zeros.

## 1. Pre-flight (5 min, before any motion)

```python
ws = connect("ws://SERVER:8000"); meta = unpack(ws.recv())
assert meta["embodiment"] == "rby1" and meta["action_dim"] == 49 and meta["action_horizon"] == 32
assert set(meta["camera_keys"]) == {"front_img_1"}
# proprio_keys must contain robot0_joint_pos, hand_left_qpos, hand_right_qpos
```
Then one-frame smoke: send any obs (even zeros + a real image) → expect finite
(1,32,49); per-step base deltas |Δ|≲0.1; joint targets within limits. NaN/huge values
here = wrong ckpt/code pairing, not a robot problem.

## 2. The isolation ladder (run in order; do not skip)

Each rung swaps ONE thing from dataset(GT) to live. PASS criteria in brackets.

| rung | image | proprio | execute? | isolates | PASS |
|---|---|---|---|---|---|
| 0 | GT | GT | no | ckpt+server+receiver plumbing | pred≈GT, MAE ≲0.02 rad |
| 1 | **live** | GT | no | camera pipeline | chunks smooth & similar to rung 0 |
| 2 | GT | **live** | no | proprio pipeline | chunks smooth & similar to rung 0 |
| 3 | live | live | no (log only) | combined, dry | smooth, plausible chunks |
| 4 | live | live | yes, receding-horizon | execution stack | smooth motion |

GT obs source: any episode of `datasets/aria_egoposer_firm` (send image RGB→BGR flipped).
If rung 0 fails → receiver framing/protocol bug. Rung 1 fails → §4.2. Rung 2 fails → §4.3.
Rungs 0-3 fine but 4 jerky → §4.1 (execution).

## 3. Core diagnostic: log one chunk and plot per block

```python
chunk = resp["actions"][0]                     # (32,49)
np.save("/tmp_or_flash/chunk.npy", chunk)
blocks = {"base":(0,3),"torso":(3,9),"head":(9,11),"l_arm":(11,18),
          "r_arm":(18,25),"l_hand":(25,37),"r_hand":(37,49)}
for n,(a,b) in blocks.items():
    plt.figure(); plt.plot(chunk[:,a:b]); plt.title(n); plt.savefig(f"chunk_{n}.png")
```
Read it like this:
- **Within-chunk smooth** → the policy is fine; your problem is execution (§4.1).
- **Within-chunk jagged / saturated / oscillating** → inputs are off-manifold (§4.2/§4.3).
- Compare magnitudes against training GT: arms move ≤ a few deg/step; base |Δ| ≤ 0.024 m
  / 0.094 rad per step (dataset maxima).

## 4. Symptom → cause → fix

### 4.1 Jerky motion, but logged chunks are smooth
| cause | signature | fix |
|---|---|---|
| No interpolation between 10 Hz targets | micro-steps at control rate | servo/min-jerk or linear interp over 0.1 s per step |
| Chunk-boundary discontinuity (flow sampling is stochastic) | kink every K steps, periodic | K=4-8 receding horizon; or temporal ensembling (average overlapping chunks, ACT-style) |
| Executing steps faster than 10 Hz | motion too fast overall + jerk | time-parameterize: step i at t0+0.1·i |
| Re-querying every step (K=1) | continuous jitter | K≥4 |

### 4.2 Bad behavior traced to the IMAGE (rung 1 fails)
| cause | signature | fix |
|---|---|---|
| Aspect squash (720p→224 direct resize) | policy semi-blind, biased to memorized motion | center-crop 720×720 → resize 224 |
| Rectified/undistorted image | scene looks "wrong" to policy; degraded everywhere | send RAW fisheye (training was raw) |
| RGB/BGR swap | mild degradation (~+0.007 MAE measured) — rarely the main bug | send BGR (server flips) |
| Wrong camera / FOV / mount pose vs Aria | plausible but misplaced actions | match Aria head-mounted framing; crop100 tolerates ~±20 px, not a different viewpoint |
| Stale image (sent once, robot moved) | policy replays start of episode | send a fresh frame EVERY query |

### 4.3 Bad behavior traced to PROPRIO (rung 2 fails)
| cause | signature | fix |
|---|---|---|
| Ordering swap (e.g. l/r arm, torso offset) | violent/wrong-limb targets | verify: at home pose, print your 22-D beside dataset frame-0 values |
| Degrees instead of radians | huge normalized values → clamp saturates all dims | radians |
| Included wheel dims / wrong slice | 22-D misaligned by 4 | exactly `state[4:26]` |
| Hands raw range mismatch (retarget convention) | hand targets wrong/grinding | compare live hand qpos range to dataset (0-~1.6 rad) |
| **Quick discriminator** | — | run crop100 with ALL-ZERO proprio: if behavior fixes itself, live proprio is malformed. Also A/B: dino_lora (noise-tolerant) OK + crop100-with-proprio OK means proprio fine |

### 4.4 Base misbehaves, arms fine
| cause | signature | fix |
|---|---|---|
| Deltas rotated by running yaw | XY path warps, heading correct | plain cumsum in start frame |
| Deltas sent as absolute targets | base stutters/creeps | integrate, command pose/velocity properly |
| Executing base at wrong rate | over/under-shoot turns | deltas are per-0.1 s |

### 4.5 No/tiny motion or NaN
| cause | fix |
|---|---|
| Receiver re-normalizing the (already unnormalized) actions | remove it |
| Missing obs key (server silently skips absent keys → stem gets nothing) | send all 4 keys every query |
| float image /255 sent as uint8 path or vice-versa | send uint8 HWC |
| Old `hpt.py` without clamp (crop100) or without DINOv2 class (dino) | use branch `rby1_aria_policy` |

## 5. Ablation probes (cheap, run anytime)
- **Zero-proprio** (crop100 only): isolates vision path — near-clean expected (0.016 offline).
- **Black image + real proprio**: isolates proprio path — degraded-but-structured expected
  (policy is vision-first at dropout 0.9; garbage here is normal, wild joint targets are not).
- **Same obs twice**: two chunks should differ only slightly (flow sampling noise ~0.002-0.005
  rad); big differences = nondeterministic obs (e.g., camera buffer races).
- **Latency**: obs→action should be ≲100 ms on GPU. If ≳0.3 s, the executed chunk is stale —
  increase K or fix transport; do not execute a chunk older than ~0.5 s.

## 6. Offline reference numbers (what "healthy" looks like)
MAE vs GT on dataset obs (10 frames): crop100 clean 0.013 / zeros 0.016 / ≤3° noise 0.014;
dino_lora clean 0.018 / needs proprio. Anything you measure at rung 0 should reproduce
these within ~2×. Base per-step deltas in demos: |Δx|≤0.024, |Δy|≤0.024, |Δyaw|≤0.094.

## 7. What to capture for offline analysis (if stuck)
1. One saved obs dict exactly as sent (npz: image + 3 proprio vectors)
2. The returned chunk (npy) + per-block plots
3. 10 s video of the behavior + the matching sequence of obs/chunks
4. Server stdout (shows which keys were received per query)
With (1)+(2) the failure category is identifiable in minutes by replaying against
`datasets/aria_egoposer_firm` on the training machine (see `/coc/flash7/czhang883/tmp/eval_final_100.py`).
