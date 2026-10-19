# Threading pilot: model × policy execution

The user requested robosuite Threading, Astra versus DeepSeek, CAP versus direct
policy, with F-adaptive in all four cells. The initial descriptive pilot has ten
paired starts and 40 assigned trials. No LIBERO outcomes are pooled with it.

| Cell | Model | Robot policy |
| --- | --- | --- |
| astra_CAP | GPT-6 Astra, NVIDIA inference endpoint | Generated Python controllers |
| astra_direct | Same Astra configuration | Model-issued native act commands |
| deepseek_CAP | Exact DeepSeek-V4.1-Flash-1017 checkpoint, OSMO vLLM | Generated Python controllers |
| deepseek_direct | Same pinned DeepSeek serving configuration | Model-issued native act commands |

CAP means [Code as Policies](https://code-as-policies.github.io/) style generated
Python control through isolated `scratch_execute`; this is an execution-mode
comparison, not a reproduction of that paper's benchmark or skill library. Direct
does not have executable robot loops. Both modes can evolve read-only observation
and coordinate helpers, hardware capabilities, camera calibration and notes.
Dynamics probes count as native actions. CAP control programs can read sensors
and issue multiple validated actions per model call. All adaptation starts fresh
per trial; no handcrafted threading controller, expert trajectory, hidden object
state, reward or success query is supplied.

The task is MimicGen's unmodified **Threading_D0**, registered with robosuite
1.4.1, using Panda and OSC_POSE. The 400-step horizon follows the
[published task registry](https://github.com/NVlabs/mimicgen/blob/72bd767c255545f462e7ccfb2731f2e5d4c1d9bb/mimicgen/__init__.py),
rather than the task constructor's generic 1,000-step default. Control runs at
20 Hz, each command repeats at most ten steps, and front/wrist RGB inputs are
256×256. Native success is checked privately after every control step. Seeds 0–9
select native randomized resets, not expert demonstrations; hashed reset state,
robot/controller configuration and initial sensor values must match in all cells.
Cell order rotates within seed blocks. Independent fresh-process replay verifies
first success and terminal simulator/model state. The first paired start has a
replay video per cell; all trials retain initial/terminal images and exact events.

Provisioning, reset and replay are excluded from actor time. Report success by
elapsed wall time and native steps using all ten assigned starts per cell as the
denominator, plus jointly successful paired times, tokens, model round trips and
client backoff. Astra medium and DeepSeek numeric effort 75 have different serving
systems and reasoning settings; cross-model speed is an end-to-end measurement,
not an isolated inference-speed claim. Ten starts support descriptive results.
Each worker warms native simulator/controller kernels in a separate environment
before measured trials, so the first cell does not pay one-time compilation cost.
The actual trial still begins from its untouched hashed paired reset.

There are no experiment token, call or wall-time budgets. The native episode
horizon and technical request/scratch limits remain enforced. Do not rerun or
resume trials automatically on failure. Preserve and archive each completed trial
and replay. An unresolved infrastructure fault stops collection for investigation.

OSMO commissioning uses one L40S; pinned DeepSeek serving uses eight H100s with
Engram CPU offload. A standalone serving smoke preserves downloaded weights and
receipts on the task-owned shared prefix and releases its GPUs on exit. It does
not start any LIBERO trials.

A two-node pilot was cancelled by cluster quota enforcement before model
startup completed; it produced no scored rollouts. The replacement launch uses
one task with eight H100s. DeepSeek keeps the same pinned vLLM image and serving
settings. The native Python image is unpacked into a separate chroot and renders
both cameras with OSMesa on CPU. All four cells use that same rendering backend;
earlier GPU-rendered commissioning traces remain unscored and separate. End-to-end
times therefore include CPU camera rendering. Source, native checks, and the
compute/rendering receipt are retained with the run.

`colocated-native` validates the native image, software camera output, isolation,
paired starts and replay without any model calls. Only then launch
`colocated-pilot`. Serving logs and evaluator artifacts are preserved on the
task-owned shared cache during execution, including after cluster cancellation.
The launcher records an exclusive persistent claim and never automatically
repeats a cohort or redispatches an action.

Single-task OSMO does not guarantee a DNS record for its own task host template.
The co-located launcher pins its pod's private IPv4 address and tests a real HTTP
round trip from the native chroot before starting DeepSeek. Astra's unscored
camera/tool transport checks run while DeepSeek loads; the scored cell schedule
is unchanged and starts only after both models pass their gates.

The first policy attempts exposed a CAP documentation defect: hiding the direct
`act` tool also hid its command-envelope schema. CAP could read observations but
had to guess required wire fields, while direct had the full schema. Those
version-1 attempts are retained as diagnostics and excluded from comparison.
Protocol version 2 supplies that exact schema in CAP's `scratch_execute`
description, including how a fresh read packet supplies the episode ID and step.
The command validator, control authority, model settings, horizon and ten paired
starts remain the same. Native commissioning now exercises read-to-act freshness
inside generated programs. All four conditions restart fresh under version 2.
