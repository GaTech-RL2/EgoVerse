# PushShapes scripted data generation — read this before collecting anything

Status as of 2026-09-10. The existing scripted corpus at
`s3://rldb/staged/pushshapes_scripted/` (33,594 episodes, 12 embodiments) is
**not usable for any embodiment above two DOF**. This document says why, what
the mechanisms actually are, and what a replacement controller has to do.

Nothing here is speculation: every claim below was measured by running the
controller locally against `PushShapesEnv`.

---

## 1. What is wrong with the existing corpus

Three defects, all in `pose_collect.py` / `pose_controller.py`. The first two
come from these three lines in `pose_collect.search()`:

```python
a = np.zeros(aw, dtype=np.float64)
a[:2] = xy
if aw >= 3:
    a[2] = float(o["agent_angle"][0])   # <-- echoes the CURRENT angle
                                        # <-- a[3] (grip) is never written
```

### 1a. The engage channel is never used

`a[3]` is never assigned, so it stays `0.0` for the whole episode. Grip
semantics are `1.0 = holding, 0.0 = released` (see the `Agent` docstring), so
every "grasp" in the corpus is a **shove with open jaws**. Measured over 400
steps with `gripper`: `grip min=0.000 max=0.000, nonzero_steps=0`.

Affects the six agents whose `action_spec` includes `grip`:
`gripper, chain_gripper, suction, umi, flipper, spring`.

### 1b. Orientation is never commanded

`a[2]` is set to the agent's *current* angle, so the commanded angle is
constant by construction. Measured commanded `|Δangle|` per step: **0.00000 rad,
exactly**, for both `u_socket` and `gripper`.

This is the real cause of the "it just pushes the T instead of engaging it"
behaviour. A socket or a jaw has to be *aimed*; it never was.

Affects every agent with `controls_angle=True`: the six above plus
`u_socket, triangle, scoop`.

### 1c. The motion is jerky

The controller emits absolute waypoints with no rate limit. Measured:

| metric | value |
|---|---|
| mean commanded step | 12.4 world units |
| mean jerk (‖Δ²pos‖) | 22.4 |
| **jerk / speed** | **1.81** |
| largest single-step jump | 200.9 units |

A jerk/speed ratio above 1.0 means the commanded direction reverses more often
than once per step. This is visible as visual jitter, and it also corrupts the
arc-length tokenizers downstream: their velocity and duration channels are
derived from per-interval motion, so a command that reverses every step
produces meaningless timing signal.

### Scope

| group | embodiments | episodes | broken by |
|---|---|---|---|
| 4-DOF | gripper, chain_gripper, suction, umi, flipper, spring | 18,045 | 1a + 1b + 1c |
| 3-DOF | u_socket, triangle, scoop | 9,000 | 1b + 1c |
| 2-DOF | circle, circle_small, stick | 8,549 | 1c only |

**Decision taken:** regenerate the 27,045 episodes for 3-DOF and above. The
2-DOF episodes are kept.

Note the training corpus `u_socket_3000_v2_clean` is a *different*, older,
human mouse-collected dataset (`"collector": "mouse"` in its `env_args`). No
model has been trained on the defective scripted data.

---

## 2. The mechanisms you must actually use

These are read from `shapes.py` and `agents.py`, not inferred.

### The T object

```python
SHAPES["T"] = [(0.0, -30.0, 120.0, 30.0),   # bar
               (0.0,  30.0,  30.0, 90.0)]   # stem
```
In object-local coordinates the **stem runs along +y**, is **30 units wide**,
and its **tip is at local (0, 75)**. The stem is the graspable feature for
every engaging embodiment.

### u_socket — a latch, not a pusher

From `shapes.py`:

> "T-stem socket pusher. **Its local +X axis points through the open end**, so
> an oriented controller can aim the socket simply by rotating +X toward
> travel. The 32-unit opening leaves 1 unit of clearance on either side of the
> standard T's 30-unit stem."

Pocket interior in socket-local coordinates: `x ∈ [-10, 20]`, `|y| ≤ 16`.

`USocketAgent` owns real pivot + gear constraints. `agent.socket_latched` is
`True` exactly when they exist. **Latching is the success condition** — if you
never latch, you are pushing.

Useful consequence: with the opening facing back down the stem, a socket placed
at `tip + d * stem_dir` puts the stem tip exactly `d` units inside the pocket.
Engaging is therefore driving `d` from a standoff down to ~5 along `-stem_dir`.
No search needed.

### Grippers

`GRIPPER_JAW_MIN_GAP = 8.0`, `GRIPPER_JAW_MAX_GAP = 58.0`; the 30-unit stem
fits. Crucially:

```python
_GRIPPER_JAW_GAP_DELTA = 0.25   # per physics substep
```

Full 50-unit travel therefore takes 200 substeps ≈ **10 rendered frames**. A
single-step 0→1 grip toggle commands a closure the physics cannot follow, and
the jaws can cross the object before contact resolves. **Ramp the grip.**

Verify engagement from state, never assume it:
- `u_socket`: `env.agent.socket_latched`
- others: `env.agent.active_constraints()`

---

## 3. Requirements for the replacement controller

### 3a. Use the articulated mechanism

- Emit the **full** action: `(x, y, angle, grip)` sized to
  `env.action_space.shape[0]`. Build it from `agent.action_spec`, never from
  `action_dim` — the docstring records that encoding by dimension silently
  mis-wired three agents at once.
- Aim before you approach. Derive the target angle from the object pose (for
  u_socket, `atan2` of `-stem_dir` so the opening faces the stem).
- Ramp grip over ~10+ frames, and only once seated.
- **Gate success on engagement.** An episode counts only if the agent latched
  or gripped, held it, and carried the T to the goal. Coverage reached while
  never engaged is a push and must be discarded, otherwise the controller will
  rediscover shoving — it is a much easier local optimum.

### 3b. Generate smooth, continuous motion

This is a hard requirement, not polish. Concretely:

- **Rate-limit every command.** Cap `‖Δpos‖` per step (≈6 world units) and
  `|Δangle|` per step (≈0.10 rad). Carry the commanded pose as state and move
  it toward the target; do not emit raw targets.
- **Never emit a target the agent cannot reach**, e.g. a point outside the
  arena or on the far side of the object. A clipped command makes the agent
  grind into a wall and the trajectory flatlines. Keep approach points at least
  ~22 units inside the boundary.
- **Route around the object, do not drive through it.** A straight line to a
  pre-engage pose on the far side of the T wedges the agent against the T
  permanently. Use an orbit at a safe radius until the approach ray is clear.
- **No teleporting recoveries.** The old jam handler emitted a randomised
  escape hop, which is a large part of the measured jerk. Back off *along the
  approach axis*, rate-limited like everything else.
- **Check the numbers, do not eyeball it.** Target `jerk/speed < 0.5`. For
  reference: old controller **1.81**; a rate-limited rewrite reached
  **0.55–1.13** without any other change.

### 3c. Validate before collecting

Run ≥10 seeds locally per embodiment and report, per embodiment:

| check | pass condition |
|---|---|
| engaged at least once | ≥80% of episodes |
| fraction of steps engaged | non-trivial, not a single frame |
| grip actually actuated (4-DOF) | `max(a[:,3]) > 0` |
| commanded `\|Δangle\|` | `> 0` |
| jerk / speed | `< 0.5` |
| peak coverage while engaged | rising with steps |

Collecting 3,000 episodes per embodiment costs 15–70 hours per batch. The
existing corpus was collected without any of these checks and all of it has to
be thrown away. **Spend the ten minutes.**

---

## 4. Prior art in this repo, and where it stalled

`/Users/rpunamiya/Desktop/GEAR/sim_run/engage/engage_controller.py` is a
partial rewrite. It **fixes 1b and 1c** — jerk/speed down to 0.55–1.13, angle
actively commanded, grip ramped — but **engages 0/10** and should not be used
for collection as-is.

Where it stalls, in the order the failures appeared:

1. straight-line approach → drove into the T and wedged;
2. arena-clamped standoff → drove into a wall when the stem pointed at one;
3. orbit routing added → gets from 226 to ~43 units of the seat point, then
   stalls at ~41 units of lateral offset and never satisfies the alignment
   gate. State histogram: `{'ALIGN': 500}`.

The missing piece is a real approach planner. `pose_controller.py`'s existing
routing (`R_SAFE`, orbit, escape) *does* solve this and is tuned — two earlier
"more principled" rewrites of it measured worse and were reverted.

**Recommended path:** keep `pose_controller`'s proven routing to reach the
pre-engage pose, then hand off to an aim-and-insert state machine for the final
~60 units. Replace only the part that was missing; do not rewrite the part that
works.

---

## 5. Where things live

| what | where |
|---|---|
| collector | `Tsimulation/sim_v2/collect/pose_collect.py` |
| old controller (routing worth keeping) | `Tsimulation/sim_v2/collect/pose_controller.py` |
| partial rewrite (smoothing + aim) | `sim_run/engage/engage_controller.py` |
| geometry constants | `Tsimulation/sim_v2/pushshapes/shapes.py` |
| latch / jaw mechanics, `CONTROL_GAPS` | `Tsimulation/sim_v2/pushshapes/agents.py` |
| existing (defective) corpus | `s3://rldb/staged/pushshapes_scripted/` |
| OSMO launcher | `osmo/pushshapes_collect.yaml` (branch `sim/arc-sweep-cotrain`) |

`CONTROL_GAPS` presets are `ideal, tight, loose, laggy, sticky, jittery`. The
whole corpus is `ideal`. Five `u_socket` gap collections were run and are
defective for the same reasons — regenerate them after the controller is fixed.

One launcher note: `TARGET_PER_JOB` is episodes **per shard**, not per
embodiment. 24 shards × 125 = 3,000. `max_attempts` is a per-shard cap on total
attempts and is what truncated `chain_gripper` at 1,045 and `circle_small` at
2,549; engagement lowers yield further, so raise it (20,000 was used for the
gap runs).
