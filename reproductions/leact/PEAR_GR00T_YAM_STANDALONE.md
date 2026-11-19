# Post-train PEAR on native GR00T Sim YAM tasks

This guide starts from the released dense PEAR policy and adds native GR00T Sim YAM environments for reinforcement learning. It requires only PEAR, GR00T Sim and their dependencies/assets.

**Integration work is required.** At the revisions below, PEAR supports YAM deployment but does not ship a native GR00T Sim YAM training adapter, installer target or simulator-node role. Stock setup commands are identified below. The proposed adapter, reset manifests and YAM training configuration must be implemented before training.

## 1. Dependencies and installation

| Component | Pin or requirement |
| --- | --- |
| PEAR | `d604856a1b16fdccd47e22a27d4dd935e3a62014` |
| GR00T Sim | `30f70f5860525d24debd96ffd21c74375dfc26ed` |
| Initial policy | `mit-han-lab/Qwen3.8-27B-PEAR`, revision `65176e435e5fa80a059ebfafaf8215e6ad00a83e` |
| Model runtime | PEAR installer-pinned veRL/vLLM; Linux and CUDA 13 compatible model nodes |
| Simulator runtime | GR00T Sim's Python 3.12 / Isaac Sim 6 / Isaac Lab 3 environment |
| Simulator dependencies | GR00T-owned Arena fork, `gear-dc-study` as `yamlab/`, and LFS assets |
| Infrastructure | Shared models/run storage, consistent repo paths, Ray connectivity and GPU rendering |

Use the full BF16 released model, not a quantized deployment or DFlash artifact. It uses `Qwen3_5ForConditionalGeneration` internally. Record the released model's revision independently: the upstream model helper also defines a different revision for the base `Qwen/Qwen3.8-27B` model.

Stock installation commands:

```bash
export PEAR_YAM_ROOT="$HOME/pear-yam"
mkdir -p "$PEAR_YAM_ROOT"
cd "$PEAR_YAM_ROOT"

git clone https://github.com/mit-han-lab/pear.git
git -C pear checkout --detach d604856a1b16fdccd47e22a27d4dd935e3a62014
export PEAR_ROOT="$PEAR_YAM_ROOT/pear"
export WORK_DIR="$PEAR_YAM_ROOT/work"  # use shared storage on a cluster

cd "$PEAR_ROOT"
CUDA_ARCH=9.0 MAX_JOBS=16 bash scripts/install.sh verl
bash 3rdparty/verl/scripts/checkout_verl.sh --check

"$WORK_DIR/envs/verl/bin/hf" download \
  mit-han-lab/Qwen3.8-27B-PEAR \
  --revision 65176e435e5fa80a059ebfafaf8215e6ad00a83e \
  --local-dir "$WORK_DIR/models/Qwen3.8-27B-PEAR"
export MODEL="$WORK_DIR/models/Qwen3.8-27B-PEAR"

cd "$PEAR_YAM_ROOT"
git clone ssh://git@gitlab-master.nvidia.com:12051/GR00T/sim/gr00t-sim.git
git -C gr00t-sim checkout --detach 30f70f5860525d24debd96ffd21c74375dfc26ed
export GR00T_SIM_ROOT="$PEAR_YAM_ROOT/gr00t-sim"
cd "$GR00T_SIM_ROOT"
./setup_workstation.sh
```

GR00T setup needs Git LFS, `uv` and authenticated access to its dependency/asset repositories. For an NGC container, obtain registry access and record its digest. The normal GR00T policy examples require an inference server; the zero-action smoke below and a completed PEAR adapter do not require an N2 server.

**Pinning GR00T Sim alone does not pin its dependencies.** Setup follows default branches unless `ISAACLAB_ARENA_BRANCH`, `DC_STUDY_BRANCH` or `ASSETS_BRANCH` is set, and does not switch existing checkouts. Capture all resulting commits:

```bash
git -C "$PEAR_ROOT" rev-parse HEAD
git -C "$GR00T_SIM_ROOT" rev-parse HEAD
git -C "$GR00T_SIM_ROOT/IsaacLab-Arena" rev-parse HEAD
git -C "$GR00T_SIM_ROOT/IsaacLab-Arena/submodules/IsaacLab" rev-parse HEAD
git -C "$GR00T_SIM_ROOT/yamlab" rev-parse HEAD
git -C "$GR00T_SIM_ROOT/assets" rev-parse HEAD
sha256sum "$GR00T_SIM_ROOT/uv.lock"
```

Keep model and simulator environments separate. PEAR's base package has no mandatory dependencies; exposing it to the simulator does not require installing training extras into Isaac Sim. Package the new adapter for both runtimes. Match Ray/Python versions on Ray server processes, and import Isaac/Kit only inside the simulator lifecycle. [PEAR installation][install], [GR00T setup][setup].

## 2. Discover and smoke-test native YAM tasks

The inventory is `configs/tgen/tgen_catalog.json`. Its `tasks` value is a dictionary. Enumerate actual IDs and assets rather than inventing Gym names:

```bash
cd "$GR00T_SIM_ROOT"
uv run python - <<'PY'
import json
from pathlib import Path
catalog = json.loads(Path("configs/tgen/tgen_catalog.json").read_text())
for task_id, task in catalog["tasks"].items():
    print(task_id, task["graph"])
    print("  experiment:", task.get("experiment"))
    print("  initializations:", task.get("initializations"))
    print("  families:", ", ".join(task.get("families", {})))
PY

uv run scripts/verify_tgen_release.py --asset-source "$GR00T_SIM_ROOT/assets"

uv run gr00t_sim/launch_experiment.py --headless \
  --experiment_config gr00t_sim/tasks/experiments/yam/a2.yaml
```

The final command is a stock **zero-action construction smoke**, executing the baseline and parallel runs declared in `a2.yaml`. Static release verification and zero-action construction do not establish manipulation success.

This revision has 25 TGen tasks. For example, `l03` maps to `l03_plate_to_table_bottom_left.yaml`. Graphs/experiments live under `gr00t_sim/tasks/{env_graphs,experiments}/yam/eval_flywheel/T_gen/`; task assets are under `assets/scenes/YAM/eval_flywheel/T_gen/<graph_stem>/`.

Interactive controller inspection without a remote policy:

```bash
cd "$GR00T_SIM_ROOT"
.venv/bin/python scripts/teleop_yam.py \
  --env-graph gr00t_sim/tasks/env_graphs/yam/eval_flywheel/T_gen/l03_plate_to_table_bottom_left.yaml \
  --enable_cameras --device cuda:0
```

Teleop disables episode timeouts: reuse its controller components, not its unlimited episode semantics. Inspect task horizon, physical versus assisted grasping, gripper geometry, camera configuration and scorer. Initially use one Kit process per rendering GPU. Use a supported RTX-capable GPU such as L40/L40S for simulator RGB and H100/H200 for model work. [Catalog][catalog], [native README][native], [zero-action config][a2], [teleop][teleop].

## 3. Freeze task and reset splits

GR00T's saved evaluation layouts are not an official RL training split. Publish task IDs, semantic task groups, object-family IDs, graph/asset hashes, complete reset poses and pose hashes before collecting experience.

A concrete **proposed**, not official, split over actual catalog IDs:

| Role | Task IDs |
| --- | --- |
| Train | `l01 l02 l03 l04 l05 l06 l08 l09 l13 m01 m02 m04 m05 m06 m07 m14` |
| Unseen-task development | `m12` |
| Unseen-task final test | `l12 l14 m03 m08 m09 m10 m11 m15` |

Review semantic overlap before freezing the split; keep aliases and appearance/contact variants of held-out tasks out of training. This measures unseen tasks within YAM, not unseen-simulator transfer.

For training tasks, create independent valid reset scenes, with separate held-out reset banks. A proposed initial coverage target is 100 training resets per task. Retain canonical evaluation layouts for evaluation. Changing the RNG seed while replaying identical saved poses does not create a new reset. The independent reset generator/exporter is integration work.

Report seen-task/unseen-reset and unseen-task success separately. Evaluate initial and trained policies on identical held-out resets and budgets.

## 4. Reasoning → constrained EEF JSON → native execution

The model receives task language, calibrated RGB, own-robot state and a language description of action/sensor frames. It emits reasoning followed by one fenced JSON action. A controller validates the action, runs bounded robot-only IK, advances native simulation and returns observations plus the native task outcome.

Document frame origin, axes, units, handedness, quaternion convention, both TCPs, gripper direction and named sensors. YAM teleop uses workcell `+X` forward, `+Y` left, `+Z` up; verify the selected graph/calibration rather than inferring directions from a camera image.

A proposed schema compatible with PEAR's nested numeric `SCHEMA`:

```json
{
  "left": {"x": 0.40, "y": 0.20, "z": 0.30, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "gripper": 0.0},
  "right": {"x": 0.40, "y": -0.20, "z": 0.30, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "gripper": 0.0},
  "control": {"steps": 12}
}
```

These values illustrate syntax, not a qualified motion. Define absolute positions in metres, normalized WXYZ orientation, gripper `0=open, 1=closed`, and bounded integer control steps. Log physics substeps/simulated duration separately.

**Replace the entire tool prompt.** Stock `EnvironmentTool` describes centimetre deltas and reaching rest; appending absolute-metre instructions would conflict. Require exact keys, bounded duration, reasoning followed by one JSON block, and no arbitrary code execution.

The existing native helper is a useful controller reference:

```python
from gr00t_sim.teleop.yam_controller import YamTeleopController

controller = YamTeleopController(env)
controller.reset()
joint_action = controller.step(
    selected_arm="left",
    translation=(0.005, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    gripper=None,  # or "open" / "close"
)
result = env.step(joint_action)
```

`step()` takes relative increments and **returns** joint actions; the caller advances simulation. It does not accept the absolute dual-arm JSON above. Implement the missing wrapper using measured TCP poses, robot Jacobians, bounded damped IK and native joint controllers. Native TCP quaternions are XYZW: convert explicitly at the WXYZ boundary and validate TCP offsets for the selected hand.

Advance both arms consistently within exact control budgets; hold the inactive arm and reset targets after scene reset. Return achieved poses/tracking errors. Do not teleport the robot or provide privileged scene-object poses to the policy. Preserve the graph's native success and assistance settings. [Tool][tool], [protocol][protocol], [native controller][controller].

## 5. Implement the PEAR Env/Tool adapter

Create a new package, for example `yam_pear_adapter`, available on all workers. **This package is not supplied by either repository.** Keep its environment module lightweight; isolate the tool module's veRL imports.

Actual upstream interfaces: `pear.sim.env.StatefulEnv` / `EpisodeLost` and `pear.envs.tool.EnvironmentTool`. This is an intentionally incomplete API skeleton:

```python
# New file: yam_pear_adapter/env.py
from pear.sim.env import EpisodeLost, StatefulEnv

POSE = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")

class YamEnv(StatefulEnv):
    SERVER_RESOURCES = {
        "num_cpus": 2, "num_gpus": 1, "resources": {"yam_sim": 1}
    }
    START_TIMEOUT_S = 900.0  # proposed; measure startup
    OPEN_TIMEOUT_S = 180.0
    STEP_TIMEOUT_S = 120.0
    SCHEMA = {"left": POSE, "right": POSE, "control": ("steps",)}
    CAMERAS = ()  # fill with actual graph sensors, in returned image order
    ENVIRONMENT_PROMPT = ""
    LOG_KEYS = ("native_success", "native_score", "control_steps", "ik_error")

    def __init__(self, task_id, reset_id):
        self.task_id, self.reset_id = task_id, reset_id
        self.env = None

    @classmethod
    def task_name(cls, env_args):
        return str(env_args["task_id"])

    @classmethod
    def setup_env(cls, context):
        # Return (list[SimulatorProcess], max_concurrent_starts).
        # context has config, gpu_ids, log_dir and log.
        raise NotImplementedError("Launch pinned GR00T simulator processes")

    @classmethod
    def environment_prompt(cls, result):
        return result["frame_and_sensor_guide"]

    def open(self):
        raise NotImplementedError("Build graph; manifest reset; observations")

    def step(self, action):
        # Raise EpisodeLost for an unrecoverable simulator episode.
        raise NotImplementedError("Validate; bounded EEF control; native outcome")

    def close(self):
        if self.env is not None:
            self.env.close()
```

Implement `process_context(context_args)` for process-wide Kit/Arena initialization if needed. `SimulatorProcess(name, env_class, python, context_args, variables, cwd, log_dir)` starts `python -m pear.sim.process` with the environment class; it does not launch `launch_experiment.py` directly. Select the GR00T Python, expose both repos/the adapter, and reserve GPU visibility from the server's allocation.

`open()/step()` return `EnvResult`: `images` in `CAMERAS` order (`None` at end), nested numeric own-robot `state`, executed native `actions` and one `reward` per native action, `executed=len(actions)`, `done`, `outcome` and `instruction`. Include every declared `LOG_KEYS` value and the dynamic frame/sensor guide. Initial observation has no actions/rewards and `executed=0`.

For an initial binary-success recipe, map **native terminal success** to reward 1 on the final native action and zero elsewhere; log partial native rubric scores separately. Stock `Episode.is_success` uses `done and sum(rewards)>0`, so positive partial rewards would otherwise mislabel failures as successes. Graded rewards require a deliberate success-accounting extension.

Implement `YamTool(EnvironmentTool)` with a replacement `SYSTEM_PROMPT_TEMPLATE` and validation in `step(episode, action)` before delegating. Stock `JsonBlockProtocol.check_action` validates exact nested keys and finite numbers. Add quaternion norm, workspace/gripper bounds, integer step limits and clear invalid-action outcomes. [Env API][env], [result schema][schemas], [process API][process], [success semantics][episode].

## 6. Connect episodes, Ray and training

Example upstream-compatible episode file, after creating its reset manifest:

```json
{
  "env_class": "yam_pear_adapter.env.YamEnv",
  "data_source": "task_id",
  "env": [
    {"task_id": ["l03"], "reset_id": ["train-000", "train-001"]},
    {"task_id": ["m01"], "reset_id": ["train-002", "train-003"]}
  ]
}
```

Each product expands into episodes. These constructor arguments belong to the new adapter. Keep GRPO group members on the same task/reset with independent model sampling. Create separate held-out files.

Tool config to implement, with proposed simulator server sizing:

```yaml
tools:
  - class_name: yam_pear_adapter.tool.YamTool
    config:
      type: native
      env_class: yam_pear_adapter.env.YamEnv
      server_n: 2
      log_dir: ${oc.env:PEAR_LOG_DIR}
```

Stock model-node startup commands from the PEAR root:

```bash
bash scripts/start_ray_node.sh trainer
# On each additional model node:
bash scripts/start_ray_node.sh rollout <head-ip>:6379
```

The stock roles are `trainer`, `rollout`, `robodojo` and `robolab`. Implement a simulator-node startup path advertising `yam_sim` and using compatible PEAR server actors plus the GR00T child runtime. There is no stock `yam` role. Verify open/step/close through the actual Ray/tool path first.

Adapt `examples/models/qwen3_8_27b_pear/train/run_stage1.sh`. It accepts extra Hydra overrides but **hardcodes LIBERO's tool config and validation file**. Setting `EPISODES` or `MODEL` alone is insufficient. Change/override:

1. `MODEL` to the released PEAR directory; `data.train_files` / `data.val_files` to YAM manifests.
2. `actor_rollout_ref.rollout.multi_turn.tool_config_path` to the new tool config.
3. LIBERO runtime/resource assumptions to the actual YAM server setup; propagate adapter imports/configuration to every Ray process.
4. `actor_rollout_ref.rollout.limit_images` for actual cameras, dimensions and context budget; the stock derivation assumes a LIBERO 512×512 frame.
5. Trainer/rollout node counts, group batch, sessions and memory/cache sizing; inspect the resolved Hydra configuration.

The real entry point is `python -m pear.trainer.main` with that script's argument arrays. Preserve its parser, patched-veRL checks and asynchronous engine. These instructions do not claim a ready YAM launcher. [Episode loader][dataset], [Ray setup][ray], [stock recipe][stage1].

After implementing and qualifying the adapter, server resources and image/cache sizing above, the following is a **launch template using real upstream overrides**. The `yam_configs/` files are new files you create; set `YAM_LIMIT_IMAGES` from the actual image-token budget. Install the adapter in both runtimes so the stock launcher's `PYTHONPATH` does not hide it. This template alone does not implement those prerequisites.

```bash
cd "$PEAR_ROOT"
export RUN="$WORK_DIR/runs/yam-pear-pilot"
export EPISODES="$PEAR_ROOT/yam_configs/train.json"
export YAM_VAL="$PEAR_ROOT/yam_configs/val.json"
export YAM_TOOL_CONFIG="$PEAR_ROOT/yam_configs/tool_config.yaml"
: "${MODEL:?set to the downloaded full BF16 model}"
: "${YAM_LIMIT_IMAGES:?set after image/context qualification}"

GPU=h100 TRAINER_NNODES=1 ROLLOUT_NNODES=1 \
  BATCH=32 N=8 STEPS=32 SESSIONS_PER_REPLICA=8 \
  MAX_PROMPT=4096 MAX_RESPONSE=160000 \
  bash examples/models/qwen3_8_27b_pear/train/run_stage1.sh \
  "data.val_files=[$YAM_VAL]" \
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$YAM_TOOL_CONFIG" \
  "actor_rollout_ref.rollout.limit_images=$YAM_LIMIT_IMAGES"
```

These pilot counts are proposed settings, not a measured YAM throughput result. Resize the model helper's host cache before this command if the selected hosts cannot support its defaults. Inspect the resolved configuration to confirm that every episode, tool and validation path selects YAM.

## 7. Algorithm, balancing and compute

Stock PEAR uses GRPO, dynamic replacement of constant-reward groups and token score centering. Keep `algorithm.rollout_correction` and `actor_rollout_ref.actor.policy_loss.rollout_correction` consistent: `bypass_mode=true`, `loss_type=reinforce`, `score_centering=true`. It uses sampler top-32 token log probabilities. Score centering is distinct from group reward normalization.

The initial recipe retains LR `1e-6`, token-mean loss, no critic/entropy bonus/frozen-reference KL penalty, weight synchronization each update, and stale-group filtering. **SGS/task balancing is additional implementation**, not a stock feature: begin with a uniform-task baseline, then separately test a smoothed success-biased sampler mixed with a uniform coverage floor. For example, propose success-biased weight at most 0.5, sampling at group boundaries, and bounded task quotas during replacement. Record the actual weighting formula. Do not wait for every task to succeed before updating; log proposed versus accepted task mixtures. Token-mean loss also requires monitoring long-episode dominance. A dense model has no router; an MoE arm needs separate kernel support and explicit router validation/freezing. [Training config][training], [base config][base].

| Unit | Meaning |
| --- | --- |
| Episode | One task/reset interaction until native outcome/horizon or policy/token failure |
| Model turn | One reasoning/JSON response |
| Control advance | One native `env.step()` action, possibly multiple physics substeps |
| Group | `N` stochastic episodes for one task/reset |
| Batch | `BATCH` usable groups per optimizer update |

Stock `BATCH=128, N=8` means **1,024 accepted episodes per update**. A proposed pilot is `BATCH=32, N=8`, or 256. Constant-reward and stale groups increase raw rollout demand.

Stock `MAX_PROMPT=4096, MAX_RESPONSE=160000` are episode/context settings. The actual response budget counts **all tokens after the initial prompt, including injected observations**, not just generated reasoning. There is no separate per-turn cap. Track generated and injected/image tokens and truncation separately. [Budget implementation][loop].

Proposed hardware pilot: one eight-H100 trainer node, one eight-H100 rollout node and one or two separate L40/L40S simulator workers. Qualify native concurrency before scaling. The documented upstream quickstart is much larger: **8 trainer nodes + 24 rollout nodes × 8 H100 each = 256 H100 GPUs**.

H100 defaults use TP=4, two replicas/node, SP=8, and **700 GB LMCache host tier per replica**, targeting 2 TB host RAM. Deliberately resize cache/sessions on smaller hosts. `SESSIONS_PER_REPLICA` is concurrency, not group batch size. [Model sizing][model], [upstream layout][quickstart].

Measure one complete update/save before estimating runtime: raw episodes/update ≈ `BATCH*N / accepted_fraction`. The slower of rollout production and actor updates dominates, plus synchronization/checkpoint costs and episode tails. GPU-hours are count × occupied hours per resource class: 24 hours on 16 H100 + 2 L40 is 384 H100-hours and 48 L40-hours, not a completion forecast. Add inference replicas for decode queues, simulator workers for rendering/stepping queues, and trainers for actor bottlenecks.

## 8. Acceptance, telemetry and recovery

Complete these checks before scaling:

1. Native reset/camera/scorer checks and hold, both-arm ±XYZ/rotation, gripper, unreachable-target and exact-duration probes.
2. One full PEAR episode through Ray with the actual prompt, EEF validator and native reward.
3. One completed actor update with verified parameter change and rollout weight version.
4. Full checkpoint save/reload, restored actor/optimizer step, and another completed update.
5. Paired held-out evaluation against the initial policy with matching sensors, action/episode/token budgets, frames and assistance mode.

Log per-task reward/native success/partial scores, macro/micro success, proposed/completed/usable/stale groups, optimizer steps, reward/advantage distributions, loss, LR, gradient norm, entropy and score-centering diagnostics. Log policy versions/staleness and available old/new-policy KL or log-ratio statistics; mark frozen-reference KL **disabled**, not zero.

Also log JSON/bounds/IK/tracking failures, achieved poses, control/physics steps, end reasons/truncation, generated/injected tokens, tokens/sec, seconds/turn, simulator wait/render/step time, actor/sync/save time, GPU memory/utilization and restarts. Add missing instrumentation explicitly; not every metric is a stock export. Record per-episode camera video/frames alongside task/reset, reasoning/action traces, native outcome and controller feedback.

Stock progress display for the proposed 32×8 pilot:

```bash
RAY_ADDRESS=<head-ip>:6379 "$WORK_DIR/envs/verl/bin/python" \
  -m pear.monitor.progress_panel --target 256 --group 8 --run "$RUN"
```

Preserve `train.log`, `metrics_*.jsonl`, `progress.json`, TensorBoard, Hydra configuration, and `sim/` action/turn logs. Training saves every step and retains two actor checkpoints: durably copy **complete** rank shards, optimizer/trainer state and metadata before pruning. Store source/model/assets/reset/prompt/controller hashes. A model export is not a resumable training checkpoint.

Stock export after a verified complete save:

```bash
cd "$PEAR_ROOT"
bash scripts/export_checkpoint.sh \
  "$RUN/ckpt/global_step_<step>" "$WORK_DIR/models/yam-step-<step>"
```

The same run path is stock resume behavior; verify restored state and adapter/sampler RNG handling. Reuse retained allocations for recovery when available. An allocated node alone does not prove a trainer or optimizer update is running.

`eval_libero.sh` is not a YAM evaluator. Adapt an upstream evaluation recipe to the same YAM tool/manifests, or implement a matching standalone policy loop. Export per-episode and per-task summaries and inspect successes and failures on video. [Monitoring/export][quickstart], [export script][export].

The remaining deliverables are the qualified absolute EEF controller, native lifecycle/Env bridge, matching tool prompt/validator, independent train/reset manifests and held-outs, YAM Ray/launcher/evaluator integration, and end-to-end update/save/reload validation. Source inspection establishes interfaces and stock commands, not completed YAM RL integration or native throughput.

[install]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/docs/getting_started/installation.md
[model]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh
[tool]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/envs/tool.py
[protocol]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/agent_loop/protocol.py
[env]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/sim/env.py
[schemas]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/sim/schemas.py
[process]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/sim/process.py
[episode]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/agent_loop/episode.py
[dataset]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/agent_loop/dataset.py
[ray]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/scripts/start_ray_node.sh
[stage1]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/examples/models/qwen3_8_27b_pear/train/run_stage1.sh
[training]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/configs/training.yaml
[base]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/configs/base.yaml
[loop]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/agent_loop/agent_loop.py
[quickstart]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/docs/getting_started/quickstart.md
[export]: https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/scripts/export_checkpoint.sh
[setup]: https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/30f70f5860525d24debd96ffd21c74375dfc26ed/setup_workstation.sh
[native]: https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/30f70f5860525d24debd96ffd21c74375dfc26ed/README.md
[catalog]: https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/30f70f5860525d24debd96ffd21c74375dfc26ed/configs/tgen/tgen_catalog.json
[a2]: https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/30f70f5860525d24debd96ffd21c74375dfc26ed/gr00t_sim/tasks/experiments/yam/a2.yaml
[teleop]: https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/30f70f5860525d24debd96ffd21c74375dfc26ed/docs/teleop_yam.md
[controller]: https://gitlab-master.nvidia.com/GR00T/sim/gr00t-sim/-/blob/30f70f5860525d24debd96ffd21c74375dfc26ed/gr00t_sim/teleop/yam_controller.py
