Complex manipulation pilots — 2026-10-06

Implements the next-study request in complex_parallel_grippers_20261006.md and
complex_dexterous_tasks_20261006.md. The previous learning study remains intact.

Current phase: native RoboCasa inference and simulator steps validated.
Two five-action probes passed. One longer LoadPreparedFood attempt stopped at
its worker time limit after 708/1,500 actions (142 reset-free segments).
All three episodes are incomplete: no SR or guidance/learning result yet.
The portable initial report includes actual videos and raw execution records.
Bench2Dex weights, selected assets and replay anchors are archived (31.6 GB),
with a 74.6 MB supplement for the two anchors' distractor objects. Both HDF5
anchors decode all four native camera streams at 640x480. These CPU checks
are not simulator rollouts; Isaac runtime and robot joint-order checks remain.

Initial task/model pairs
  RoboCasa365 / LoadPreparedFood / released pi05_pretrain_human300
  RoboCasa365 / PackIdenticalLunches / released pi05_pretrain_human300
  Bench2Dex / 73_jigsaw_puzzle_assembly / task-specific released pi0.5
  Bench2Dex / 34_fridge_wine_interhand_pour / task-specific released pi0.5

release_manifest.json pins source revisions, checkpoint revisions, file sizes,
available shard hashes, and exact normalization hashes. Inference files total
approximately 12.44 GB per checkpoint; optimizer states are not downloaded.
Large assets and weights are staged on OSMO, not on the local laptop.

Important distinctions from the proposal text
  - The previous LIBERO study did not establish that Astra solves LIBERO in two
    rollouts. It measured narrow, mostly unchanged autonomous development scores.
  - RoboCasa uses mobile PandaOmron controls. Its native model predicts 50 steps
    and 32 internal dimensions, returning 12 physical command channels. The
    released evaluation executes five commands before replanning. Preserve the
    three native cameras and checkpoint mean/std normalization.
  - Both selected RoboCasa tasks occur in pretraining data. The initial pretrain
    scene screen is a long-horizon seen-task baseline, not novel-task OOD evidence.
    Environment and instruction shifts must be explicitly declared separately.
  - Bench2Dex jigsaw has 58 full joints but 54 active coordinates in the current
    registry and released normalization. A deployment YAML comment saying 58
    active coordinates is stale. Fridge pouring has 48 full / 38 active joints.
    Runtime joint names/order must still be checked against the native mapping.
    Both policies use four cameras and a 20-action prediction horizon.
  - EmbodiedSWE registers the requested PC and IKEA environments and provides a
    VLA training/evaluation pipeline. A matching trained checkpoint for those
    exact assembly tasks has not been verified. An environment smoke test or a
    coding-agent solution cannot stand in for a capable System1 baseline.
  - The inspected DexVerse live release is ungated and includes additional hand
    assets and demonstrations, despite the older README. Its inspected demo
    manifest does not list OvenBakeSalmon or CleanTable. Matching long-horizon
    policies remain unverified.
  - Beyond the BEHAVIOR starter turning_on_radio model, the 2025 winning release
    explicitly maps sorting_household_items (27) and make_pizza (49) to
    checkpoint_3, and clean_up_your_desk (29) to checkpoint_1. These are modified
    PiBehavior models with task-ID embeddings, stage prediction, correlated
    noise and custom inference. They require a separate baseline label; stock
    text TEI/TLI cannot simply be reused. The shared RLinf PT50 policy is another
    candidate whose RPent interface remains to be validated. Neither was run.

Sequence
  1. Verify released policy dimensions, normalization, source and asset identity.
  2. Render/reset/step the actual simulator and verify native model inference.
  3. Run native full episodes before interpreting intervention improvements.
  4. Freeze a four-condition protocol: native, guidance only, learning only, both.
     The requested collection schedule is 1/2/4/8/16 episodes. Report assisted
     execution separately from autonomous learned-policy evaluation.

Accounting
  An episode starts from a declared reset and runs until success, the benchmark
  horizon, or an explicitly recorded stop. A reset-free segment executes a
  selected action prefix before observing again. It is not another SR trial.
  Candidate generations consume inference but do not create physical trials.
  Record constructor/setup resets separately from policy episode starts.
  Every task report includes resets, complete and partial episodes, executed
  segments, assisted segments, actions, videos, latency, teacher usage and GPU
  allocation time. Incomplete batches do not establish a full-batch SR.

Compute
  A further 24 L40S-hours was explicitly authorized on 2026-10-07 UTC:
  "go for another 24 l40s". The total ceiling is now 48 L40S-hours.
  Prior spending was 22.3458780117, leaving 25.6541219883 hours before
  the newly launched six-episode native development batch (seeds 0/1/2).
  GPU initialization and teardown remain charged; at most two GPUs may run.
  The initial dashboard preserves its original 24-hour budget snapshot.

Files
  protocol.json            Requested task inventory and proposed comparisons.
  release_manifest.json    Immutable source/checkpoint inventory.
  stage_robocasa.py        GPU-zero staging and checksummed inference assets.
  stage_bench2dex.py       GPU-zero staging for the two dexterous task releases.
  stage_bench_runtime.py   Frozen native policy environment built without a GPU.
  bench_worker.py          Bounded native Isaac/Bench2Dex pilot and evidence archive.
  bench_policy_server.py   Native policy session with prediction/latency records.
  bench_simulator.py       Native evaluator with command and runtime joint journals.
  robocasa_eval.py         Native simulator/model rollout runner with video.
  accounting.py           Distinct physical episode and reset-free segment counts.
  worker.py               Checked restore, bounded execution and result archival.
  build_initial_report.py  Portable initial evidence report with native videos.
  additional_release_audit.json   BEHAVIOR model candidates and Bench native limits.

Initial report
  astra_reversal/reports/complex_manipulation_initial/index.html
  Three reset episodes started, zero completed; 144 reset-free segments and
  718 complete recorded controls. No teacher calls, tokens or policy updates.
  Warm inference median: 100.05 ms across 141 calls after the 43.15 s first JIT.
  Three separate upstream constructor resets are not policy evaluation trials.
  Predictions and physical commands are archived separately; incomplete
  episodes are excluded from a completed-episode success rate.

Dexterous native integration
  The simulator uses the official Isaac Lab 2.3.2 image and its rendering
  experience for camera parity. The JAX policy runs in a separate frozen
  Bench2Dex environment, calling its original trained-policy factory. CPU
  imports do not validate GPU inference, robot dynamics or rendered images.
  Pilot task limits are 1,213 and 1,080 policy control steps respectively,
  with three physics steps per action and 20 predictions per chunk. Each task
  currently has one recorded reset anchor; repeated policy seeds at that anchor
  must not be described as independent randomized scene trials or held-out OOD.
  Native success and completed physics steps come from per_episode.jsonl.
  The supplemental command journal is recorded before physics; an interrupted
  run's final journal entry is not proof that the command was applied.
  The native Sharpa map has four inactive coordinates, not four mimic joints.
  RH5DG2 has ten mimic coordinates and requires all ten native mimic rules.

Primary references
  https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html
  https://github.com/robocasa-benchmark/openpi
  https://huggingface.co/robocasa/robocasa365_checkpoints
  https://github.com/Bench2Dex/Bench2Dex
  https://huggingface.co/Bench2Dex/policy_ckpt
  https://github.com/EmbodiedSWE/EmbodiedSWE
  https://behavior.stanford.edu/challenge/baselines.html
  https://huggingface.co/IliaLarchenko/behavior_submission
  https://github.com/IliaLarchenko/behavior-1k-solution
  https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32
  https://huggingface.co/datasets/dexverse/DexVerse_release
