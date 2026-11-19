# PEAR post-training on native GR00T YAM simulation

**Engineering handoff — 19 November 2026.** This describes how to adapt PEAR to the native YAM environments in GR00T Sim, using reasoning followed by bounded end-effector commands and native task rewards. It separates implemented components from the remaining work for a dedicated YAM-only run.

**Current status (19 November):** the native transport and independently generated training reset bank are qualified. The fourth YAM motion probe passed 14/14 discarded checks after a bounded Cartesian approach in each of two training scenes. Earlier initial-pose failures remain recorded: not every Cartesian target is reachable from the nearly straight reset pose. The probe's preparation moves do not change production resets. The full-mixture EEF RL driver has been released in `rp_jhh_pear_moe_warmstart496_eef_416a7560526c`; no EEF optimizer update or task-success improvement is established by these controller checks. A dedicated YAM-only launcher remains a separate implementation task.

The EEF change applies **only to PEAR RL**. Existing Astra collection retains its original action spaces, prompts, model profile and budgets.

## 1. What is being transferred

[PEAR](https://github.com/mit-han-lab/pear) is a post-training engine for agentic robotics, built on veRL, Ray and asynchronous vLLM inference. Its training recipe uses GRPO, dynamic filtering of constant-reward groups, and score centering. The pinned upstream lists YAM as a deployment embodiment; native GR00T YAM **training** is an adapter added in this workspace, rather than an already supported upstream training benchmark.

The reusable boundary is:

```text
public RGB + robot state + instruction + action/observation guide
    -> reasoning model
    -> constrained JSON Cartesian target
    -> robot-only IK/controller
    -> native joint commands and ordinary simulator physics
    -> fresh observation + measured execution feedback + native task reward
```

The model chooses the arm targets, orientation and gripper commands. IK translates those targets into joints; it does not locate task objects, choose grasp points, run task skills, teleport the robot, or replace the success predicate. Native termination terms determine binary task success. Controller convergence is a separate diagnostic.

## 2. Pin the model, engine and simulator

| Component | Current source or setting |
|---|---|
| PEAR revision | `d604856a1b16fdccd47e22a27d4dd935e3a62014` |
| GR00T Sim revision | `30f70f5860525d24debd96ffd21c74375dfc26ed` |
| GR00T repository | [GR00T/sim/gr00t-sim](https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/) |
| Native simulator image | `nvcr.io/nvidian/lpr/gr00t-sim@sha256:6e94ec62a7f9287c77769570bec76684e76f3225bdd96931cf9a46fc06baabb6` |
| Simulator interpreter | `/opt/gr00t-sim/venv/bin/python` inside that image |
| Primary initialization | **Qwen3.6-35B-A3B MoE, warmstart step 496**, model type `qwen3_5_moe` |
| Primary model path | `/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-coverage-20261105-v1/intermediate/warmstart-step496/model` |
| Primary revision label | `efe-coverage-warmstart-intermediate-step496` |
| Optional dense base | `Qwen3.8-27B`, model type `qwen3_5`, revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| Dense base staged path | `/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-scaled-20261116-v1/models/Qwen3.8-27B` |

The current primary is the MoE warmstart, not the dense 27B model. Sourcing upstream's `qwen3_8_27b.sh` supplies hardware/configuration settings; our wrapper restores the selected model revision and installs the MoE-specific kernel path. The dense arm is not enabled by the current control file. Starting from a released, already post-trained PEAR policy is another experiment: first pin its exact model repository and revision and qualify its weights, architecture and checkpoint loading; no such replacement is specified here.

For a fresh engine installation, the pinned upstream supports the following commands **on an allocated Linux GPU node**, with fresh project-specific paths on storage visible to every model node:

```bash
export PEAR_SOURCE=/shared/project/sources/pear-d604856a1b16fdccd47e22a27d4dd935e3a62014
export WORK_DIR=/shared/project/pear-runtime
git clone --no-checkout https://github.com/mit-han-lab/pear.git "$PEAR_SOURCE"
git -C "$PEAR_SOURCE" checkout --detach d604856a1b16fdccd47e22a27d4dd935e3a62014
cd "$PEAR_SOURCE"
CUDA_ARCH=9.0 MAX_JOBS=16 bash scripts/install.sh verl
bash 3rdparty/verl/scripts/checkout_verl.sh --check
```

`verl` is a real installer target; there is no upstream `yam` simulator-install target. The current study already has a staged runtime at `/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-scaled-20261116-v1/runtime`. Reuse a qualified runtime without changing it in place. Its successful import probe is only an installation check; MoE weight replay, an optimizer update, score-centering capture and checkpoint reload have separate receipts. The simulator image runs in separate allocations, so installing its Isaac stack into the training venv is unnecessary.

Use the image's compatible Isaac Lab/Arena dependencies and the pinned asset release. The current production worker stages assets from `/mnt/amlfs-07/shared/datasets/gear_sim/assets`, checks the release manifest, and binds both the staged source and image dependencies to that same release. It also applies the recorded Arena compatibility shim. Do not substitute an arbitrary editable checkout or a similarly named asset folder.

Sources: [model manifest](artifacts/embodied-first-exam/pear-scaled-v1/model-files-ready.json), [source manifest](artifacts/embodied-first-exam/pear-scaled-v1/sources/gr00t-sim-source.json), [native server setup](cluster/efe_pear_yam_server.py), [training wrapper](cluster/efe_pear_scaled_train.sh), [pinned upstream README](artifacts/embodied-first-exam/pear-scaled-v1/sources/pear-upstream-pinned/README.md).

## 3. Use the frozen task split and independent training scenes

The frozen catalog contains **35 train / 4 development / 14 test environment names**, representing **31 / 4 / 6 semantic groups**. Aliases and contact variants share a semantic group so they cannot cross the task holdout. Catalog membership alone does not establish a usable training reset.

The currently generated training bank covers **16 task graphs, 14 semantic groups, and 64 independent resets per graph: 1,024 task/reset configurations**:

| Environment ID | Task |
|---|---|
| `tgen_v2_l01_yam` | Move the plate to the center of the mat |
| `tgen_v2_l02_yam` | Place the cup at the table's top-right corner |
| `tgen_v2_l03_yam` | Put the plate at the bottom-left corner |
| `tgen_v2_l04_yam` | Place a block beside the cup |
| `tgen_v2_l05_yam` | Put the bowl behind the plate |
| `tgen_v2_l06_yam` | Place the cup between bowl and plate |
| `tgen_v2_l08_yam` | Toss the red block into the basket |
| `tgen_v2_l09_yam` | Put the cup on the mat |
| `tgen_v2_l13_yam` | Put the block into the bigger bowl |
| `tgen_v2_m01_yam` | Push the cup left |
| `tgen_v2_m02_yam` | Stack the cup on the bowl |
| `tgen_v2_m04_yam` | Flip the cup |
| `tgen_v2_m05_yam` | Drop the cup into the basket |
| `tgen_v2_m06_yam` | Lay the cup on its side |
| `tgen_v2_m07_yam` | Slide the plate right |
| `tgen_v2_m14_yam` | Open the bottom drawer |

Other train-catalog entries, including handover and additional language-following environments, are not automatically part of this bank: 19 train entries lacked the reviewed schema-3 reset definition when it was generated. Adding them requires their own reset generation and native qualification.

Keep the evaluation categories separate:

| Evaluation category | Reserved data |
|---|---|
| Seen YAM task, new variant | Existing saved layouts of train templates, including canonical references 1–10 and all 25 saved layouts where available |
| Seen YAM simulator, unseen task | Six semantic groups: `color_matching`, `hang`, `ordinal_selection`, `pour`, `scoop`, `screw_lid` |
| Development only | `l07_towel_under_cup_yam`, `m22_roll_clay_yam`, `m23_scrub_apple_yam`, `tgen_v2_m12_yam` |
| Unseen simulator | Existing EFE reserves PandaDyn, RoboPaint, RoboTwin-2 and RoboWits; a YAM-only study cannot itself measure an unseen-simulator score without those separate adapters |

The 14 test aliases are `scoop_anything_gear_v2_yam`, `tgen_v2_l12_yam`, `tgen_v2_l14_yam`, `tgen_v2_m03_yam`, `tgen_v2_m08_yam` and its contact variants `a0/a2/a3/a4`, `tgen_v2_m09_yam`, `tgen_v2_m10_yam`, `tgen_v2_m11_yam`, `tgen_v2_m15_yam` and `tgen_v2_m15_yam_hold_a4`.

**Every existing saved scene is evaluation-only.** Changing an RNG seed while loading a saved layout does not create a new scene. The training bank uses independently generated poses and reports zero overlap with 252 protected pose fingerprints. Keep that check, asset/scene ancestry checks, nonterminal reset checks and native reward checks distinct. The generation report's original `training_admitted:false` is a generation-stage status; later native bank readback and transport receipts establish the old joint-interface admission. They do not qualify the new EEF controller.

Sources: [split](artifacts/embodied-first-exam/pear-scaled-v1/native-yam-split-v1.json), [semantic policy](artifacts/embodied-first-exam/pear-scaled-v1/native-yam-split-policy-v1.json), [generation report](artifacts/embodied-first-exam/pear-scaled-v1/native-yam-generated-resets/generation-report.json), [native bank readback](artifacts/embodied-first-exam/pear-scaled-v1/native-yam-bank-readback.json).

## 4. EEF action and observation contract

Enable EEF explicitly with `control_interface: pear-eef-native-v1` in the PEAR native worker specification. `efe_pear_yam_server.py` forwards that field into `YamSessionPolicyCfg`. Omitting it preserves the existing joint-target interface.

Each response ends with a JSON object having exactly these fields:

```json
{
  "action_type": "ee_pose",
  "action": [0.50, 0.30, 0.92, 1, 0, 0, 0, 0, 0.50, -0.30, 0.92, 1, 0, 0, 0, 0],
  "repeat": 12,
  "summary": "Approach the visible target while keeping both grippers open."
}
```

This shows the **shape only**; those poses are not qualified commands for an arbitrary reset. Construct a hold command from that episode's measured TCP poses before testing motion.

The 16 values are **left arm then right arm**, each `[x,y,z,qw,qx,qy,qz,gripper]`. Positions are metres in the native simulator world frame; `+Z` is up. Quaternions use `wxyz` and unit norm within the declared parser tolerance. Gripper `0` means open and `1` closed. The TCP is the native grasp-site point, currently 0.1347 m along `link_6`'s local Z with the link's orientation. Image directions do not define world directions.

`repeat` is a **maximum of 1–25 ordinary native control steps**, not a promise that the target will be reached. At 30 Hz, 25 steps represent about 0.83 seconds of simulated control. The controller recomputes bounded damped-least-squares IK each step, respects native joint limits and reports actual executed steps. Current prototype gains are position 3 / rotation 2, with 0.04 m / 0.30 rad correction caps and a 0.05 rad maximum joint-target increment per step. These values remain subject to the motion qualification described above.

Expose named RGB views, robot joint proprioception, current TCP poses, gripper target and measured opening, and previous execution feedback. Camera calibration, sensor templates, frame conventions and tool offsets belong in the context when verified; do not invent omitted axis directions. The command box is a guard, not a guarantee of reachability. Return measured position/orientation errors and IK status so the model can correct its next action. Do not interpret post-auto-reset poses as the pose reached before termination.

Implementation: [EEF controller](cluster/efe_pear_yam_eef.py), [session policy](cluster/efe_pear_yam_session_policy.py), [parser/prompt](cluster/efe_action_loop.py), [public interface guide](cluster/efe_pear_context.py), [native outcome handling](cluster/efe_pear_yam_contract.py).

## 5. Training recipe and units

The following are the implemented primary recipe values to preserve when porting to YAM-only:

| Setting | Value / meaning |
|---|---|
| Algorithm | Asynchronous GRPO, DAPO-style dynamic group filtering, score centering |
| Update | One PPO epoch; token-mean policy loss; LR `1e-6` |
| Parameter update | Full finetuning, `lora_rank=0` |
| MoE routing | Freeze only expert-selection `mlp.gate.weight/bias`; assert unchanged after update and reload; no router replay |
| Sampling | Temperature 1, `top_p=1`, `top_k=-1`; sampler top-32 log-probabilities retained for score centering |
| Reference policy / critic | Neither is required by this recipe; reference KL penalty disabled; entropy coefficient 0 |
| Group | Eight stochastic trajectories from the same task/reset prompt |
| Optimizer batch | 32 **usable groups**, hence 256 retained trajectories; current pilot target is 32 actor updates |
| Episode budget | `MAX_RESPONSE=160000` for the multi-turn episode; `MAX_PROMPT=16384`; model-length cap 176,384 |
| H100 actor | FSDP2, eight-way sequence parallelism, activation checkpointing/offload, optimizer offload |
| H100 inference | TP4; two vLLM replicas per eight-GPU rollout node; MoE LMCache path disabled |

A trajectory is a full interaction episode, not a single JSON action. The 160k setting is not 160k reasoning tokens per action: generated text, retained multimodal context and the agent-loop budget accounting must be inspected separately. Log generated reasoning tokens, other generated tokens, image/input tokens, turns, truncation and the configured caps. Do not silently enlarge the budget for failed episodes.

No hard per-family success gate is needed before an update. **Usable-group filtering still applies:** all-zero and all-one groups contain no binary-reward GRPO contrast and are replaced. For independent success probability `p`, the probability that an eight-trajectory group is usable is `1 - p^8 - (1-p)^8`. Thus 256 retained trajectories can require substantially more than 256 executed episodes.

The existing SGS implementation keeps a 20% uniform floor and softly favours task/reset distributions near an intermediate success rate, with a rolling history. Include complete graded groups even when GRPO discards them; infrastructure failures must not become policy failures. Never adapt held-out evaluation sampling. The current full-mixture policy, `task-signal-window-v2`, balances available signal across an eight-update window rather than blocking on a missing family. It also corrects family token contributions once, after GRPO advantage normalization, so long reasoning traces do not win simply by having more tokens.

For **YAM-only**, retain the reset sampler and add analogous balancing across YAM tasks/semantic groups. This is proposed work: the present family-balancing implementation treats all native YAM tasks as one family. Start with uniform task coverage plus the SGS exploration floor, then log both sampling-weighted and uniform-task scores before enabling stronger task bias. Do not copy full-mixture MetaWorld warmup task IDs into a YAM-only manifest.

Sources: [training shell](cluster/efe_pear_scaled_train.sh), [upstream base](artifacts/embodied-first-exam/pear-scaled-v1/sources/pear-upstream-pinned/pear/configs/base.yaml), [upstream training](artifacts/embodied-first-exam/pear-scaled-v1/sources/pear-upstream-pinned/pear/configs/training.yaml), [curriculum](cluster/efe_pear_curriculum.py), [signal balancing](cluster/efe_pear_signal_curriculum.py), [router policy](cluster/efe_pear_router.py).

## 6. Hardware and rollout architecture

Run model training/inference on H100s; use RTX-capable **L40 or L40S for native Isaac RGB simulation**. H100s do not replace the RTX renderer. The model-side `NativeYamEnv` proxies communicate through private file RPC with independently allocated simulator processes. Each native process owns one GPU, one task graph, and repeated resets from that task's bank.

| Stage | Proposed model resources | Native simulation resources |
|---|---|---|
| Controller check | No model | Two L40/L40S, one process per task graph |
| Small RL pilot, after a YAM-only selector exists | Eight trainer H100 + eight inference H100 | Two L40/L40S; l03/l04 initially |
| Full 16-task YAM bank | Eight trainer H100 + 32 or 48 inference H100 | At least 16 L40/L40S for one live process per graph; add replicas where measured queues are longest |

The proposed 56-model-GPU layout is **8 trainer + 48 inference**, keeping 32 groups/update. Its simulator GPUs are additional. More inference nodes do not shrink the optimizer batch. Lowering a microbatch/token cap can increase accumulation or prevent long trajectories from fitting; benchmark the complete update rather than assuming it is faster.

The old broad-mixture run took 68–173 minutes for collection and 7–31 minutes for actor work in its first three measured updates. Those are not YAM-only throughput measurements. After the first qualified pilot, estimate:

```text
required episode attempts ≈ 256 / measured usable-group fraction
collection time ≈ required attempts / measured aggregate completed-episode rate
run time ≈ updates × (collection + actor + weight-sync + checkpoint overhead)
GPU-hours = allocated GPUs × wall-clock hours, separately for H100 and L40/L40S
```

For budgeting, 24 hours at 56 H100 plus 16 simulator GPUs is 1,344 H100 GPU-hours and 384 L40/L40S GPU-hours; this arithmetic is not an ETA. Inspect model decode throughput, image preprocessing, simulator reset/render/step time, RPC waits, useful-group yield and checkpoint bandwidth separately.

## 7. Existing commands and the YAM-only launcher gap

From the repository root, first activate the project environment:

```bash
source emimic/bin/activate
python -m pytest -q reproductions/leact/tests/test_efe_pear_eef.py reproductions/leact/tests/test_pear_yam_terminal.py
```

The current **model-free EEF probe** entry point exists:

```bash
python -m reproductions.leact.cluster.launch_efe_pear_eef_probe --help
python -m reproductions.leact.cluster.launch_efe_pear_eef_probe --suite yam
```

Without `--launch`, this prepares a frozen bundle. It is a workspace-specific tool using the retained host and source/asset handoff in `artifacts/embodied-first-exam/pear-scaled-v1/eef-interface-v1/`; it is not a generic trainer. With `--launch`, it launches discarded native motion checks and **does not launch RL**. Inspect the existing probe receipt before repeating it:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('reproductions/leact/artifacts/embodied-first-exam/pear-scaled-v1/eef-interface-v1')
for name in ['yam-native-probe-launch.json', 'yam-native-probe-latest.json']:
    path = root / name
    if path.exists():
        record = json.loads(path.read_text())
        print(name, json.dumps(record.get('status', record), indent=2))
PY
```

The older native API/reset preparation command also exists; it checks the existing native interface and does not qualify EEF or create an RL update:

```bash
python -m reproductions.leact.cluster.launch_efe_pear_warmstart --kind yam --generated-resets --task-names tgen_v2_l03_yam tgen_v2_l04_yam
```

Do **not** use `launch_efe_pear_primary --prepare` as a YAM-only launcher. It compiles the full admitted EFE/control/reasoning mixture. The training shell, `CurriculumState`, multitask configuration and EEF admission presently enforce `expanded_full` and the full set of families. There is no implemented `--yam-only` option.

Before a dedicated YAM-only optimizer run, implement a separate, versioned selector/entry point that:

1. Selects only the 1,024 admitted generated YAM task/reset configurations and preserves their scene/manifest hashes.
2. Adds a `native_yam_only` curriculum stage with explicit task/semantic-group targets, while preserving full-mixture validation for existing runs.
3. Registers only `efe_pear_remote_env.NativeYamEnv` and `efe_pear_native_tool.NativeEfeTool`; removes unused control/reasoning proxies and their CPU reservations.
4. Uses YAM-specific EEF qualification receipts. The current full-mixture EEF qualifier requires all ten manipulation simulator families, so it must not be bypassed by relabelling a YAM receipt.
5. Creates a fresh `rp_jhh_...` run directory, checkpoint namespace and telemetry ID, initialized from the pinned model; does not overwrite the interrupted broad-mixture history.
6. Generates absolute-path dataset/tool/config files on shared storage, verifies them on each worker, and calls the existing `efe_pear_scaled_train.sh` only after its new stage validation is supported.

The existing PEAR episode dataset format can represent the YAM tasks as follows; expand `env` with the exact admitted task/reset pairs rather than a Cartesian product of unrelated tasks and reset IDs:

```json
{
  "env_class": "efe_pear_remote_env.NativeYamEnv",
  "data_source": "task",
  "env": [
    {"task": ["tgen_v2_l03_yam"], "reset_id": ["init-000"]},
    {"task": ["tgen_v2_l04_yam"], "reset_id": ["init-000"]}
  ]
}
```

Each simulator worker also needs the selected task's pinned graph, generated-reset manifest and hash, private transport root, unique worker ID, output path and `control_interface: pear-eef-native-v1` in `YamSessionPolicyCfg`. The server already generates the graph extension `tgen_reset_layouts` and the full experiment YAML; use that implementation rather than inventing additional native launch flags.

The lower-level shell already accepts these final Hydra overrides through `"$@"`; after that selector and its qualification exist, use them for a **new discarded optimizer/reload check** on an already allocated cluster:

```bash
# Environment variables and YAM-only manifests must be generated and verified first.
# This is an implementation target, not a command that is ready in today's full-mixture stage.
bash "$ADAPTER_ROOT/efe_pear_scaled_train.sh" trainer.total_training_steps=1 trainer.save_freq=1
```

Required shell inputs are `PEAR_SOURCE`, `WORK_DIR`, `RUN`, `MODEL`, `MODEL_REVISION`, `EPISODES`, `VALIDATION_EPISODES`, `EFE_PEAR_NATIVE_CONFIG`, `MODEL_QUALIFICATION`, `TRAINING_STEPS`, `EFE_PEAR_CURRICULUM`, `TOOL_CONFIG`, `EFE_PEAR_REASONING_MEMBERSHIP`, `EFE_PEAR_REMOTE_ROOT`, `PEAR_CHECKPOINT_ARCHIVE` and `PEAR_OBJECTSTORE_CREDENTIAL_FILE`. The reasoning-membership requirement is another full-mixture assumption to remove explicitly in the YAM-only stage. Credentials must stay in the existing protected runtime mechanism, never in a checked-in manifest or this document.

## 8. What to log and evaluate

Keep three populations distinct: all completed native policy episodes, usable groups retained for gradients, and infrastructure/runtime failures. Training success is not a held-out result.

| Area | Required observations |
|---|---|
| Task performance | Raw reward/success per task, per semantic group, macro-task average, episode counts, reset coverage; report the evaluation split and fixed sampling |
| RL signal | Judged/usable/constant-reward groups, retained fraction, 32-group target, actual optimizer updates, policy version/staleness |
| Optimization | Policy loss, grad norm, LR, entropy, available rollout-vs-current policy KL/importance diagnostics, nonfinite values, memory |
| Controller | Valid/invalid commands by reason, actual/requested native steps, reached TCP position/orientation errors, IK failure, gripper motion, action latency |
| Throughput | Generated/input/image tokens, reasoning tokens when measured, turns, truncations, tokens/sec, episodes/hour, simulator/RPC waits, actor and checkpoint seconds |
| Artifacts | Initialization/revision, recipe/source/scene hashes, complete checkpoint manifest and reload receipt, representative successes and failures with frames/traces |

The current telemetry emits to `$RUN/telemetry/events.jsonl`, `$RUN/telemetry/steps/`, `$RUN/telemetry/rollout-workers/` and the raw `$RUN/metrics.jsonl`. It preserves missing values: **reference-policy KL is unmeasured/disabled, not zero**. Existing raw model replies and JSON actions plus `PEAR_TRACE_FRAMES=1` provide delivered camera frames under `$RUN/sim/frames/<episode>/<turn>/`. Native MP4 recording is not guaranteed by the YAM session wrapper; frame sequences can be exported for qualitative review, with simulation time and model waiting time distinguished.

EEF errors are available in native feedback, but confirm their aggregate counters/dashboard plots before claiming they are already logged as scalar training curves. The existing full-mixture launcher disables automatic validation (`test_freq=-1`, `val_before_train=false`); a YAM-only study needs a separately scheduled fixed evaluation panel.

Evaluate the same initialization and selected RL checkpoints on seen-task/new-variant and unseen-task YAM panels, with fixed scenes, action interface, sampling and budgets. Report individual task counts and uncertainty. Inspect grasps and object motion in videos; a narrative that says “holding” or a low tracking error does not establish manipulation success.

Sources: [telemetry](cluster/efe_pear_telemetry.py), [agent metrics](cluster/efe_pear_agent_metrics.py), [trace frames](cluster/efe_pear_native_tool.py), [model-side proxy](cluster/efe_pear_remote_env.py).

## 9. Checkpoint and allocation recovery

The interrupted broad-mixture run completed four actor updates, but only three finalized iteration metric points and no committed full step-4 checkpoint. Eight model shards survived; optimizer ranks 6 and 7 were incomplete. Do not call that an exact resume checkpoint or pool its results with a new EEF run.

The current wrapper saves every four updates and keeps one local actor checkpoint, overriding upstream's more frequent defaults. Full MoE snapshots are roughly 441 GB including optimizer and queue state. Node-local `/dev/shm` is temporary. A durable checkpoint requires all model/optimizer shards, curriculum/SGS history, queue and training state, verified readback, and a committed completion pointer. A background upload from temporary memory alone is not durable.

Before the scaled run, measure checkpoint storage bandwidth and free space, exercise full save/reload once, and define a durable shared staging path before considering asynchronous object archival. Preserve incomplete files as recovery evidence. All checkpoint names should begin `rp_jhh_`.

Retain OSMO allocations when a child fails. Distinguish allocation status, native-worker status, model rollout status and actual optimizer progress. Stage a verified replacement job inside the retained allocation; never cancel live allocations automatically, and never retarget the separate Astra collection. Require every task in a model group to be `RUNNING` before executing into it.

Sources: [recovery receipt](artifacts/embodied-first-exam/pear-scaled-v1/checkpoint-recovery-required.json), [trainer checkpoint path](cluster/efe_pear_multitask_train.py), [verified shared archive](cluster/efe_pear_checkpoint_archive.py), [retained-allocation submission](cluster/efe_pear_allocation_jobs.py), [compute lifecycle rules](AGENTS.md).

## 10. Ordered implementation and launch checklist

1. Resolve the remaining EEF motion failure without relaxing the recorded test silently; recheck axes, gripper, unreachable targets, reset identity, native verdicts and exact step accounting.
2. Freeze the controller/source receipts and verify identical EEF action/observation behavior on model-side and simulator-side paths. Keep the separate Astra collector unchanged.
3. Implement the explicit YAM-only selector/stage and its admission checks; keep the 16-task generated bank separate from all saved evaluation scenes.
4. Run a small frozen-checkpoint inference panel to check that the model produces executable EEF actions and grounded recovery behavior; record this as initialization performance.
5. Run one discarded optimizer update, full save/reload, router equality and post-reload inference check. Do not include its diagnostic episodes in the primary run.
6. Launch a fresh 32-update YAM-only pilot with 32 usable groups × eight trajectories/update, measured task balancing, SGS and score centering. Expand tasks and simulator replicas only after their own native qualification; preserve the frozen holdouts.
7. Schedule fixed evaluations, publish native success and failure traces, and scale compute according to the measured bottleneck. A submitted allocation or successful controller probe is not an optimizer update.
