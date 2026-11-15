# LeAct on GR00T OSMO

`workflow.yaml` requests one node with 8 H100s, 80 CPUs, 1024 GiB RAM and
320 GiB ephemeral storage. The initial phase is a bounded infrastructure
preflight: real full-parameter Qwen3-8B updates, saved-model reload, then LeAct
generation and selection. The worker exits when that phase finishes. OSMO caps
execution at 12 hours; the experiment subprocess has a separate 10-hour limit.

The OSMO `groot-h100-02` resource query confirmed `/mnt/amlfs-07/shared` is a
default mount. Files are written only under
`/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/`. Each workflow has its own source,
logs, environment and artifacts. No credentials or unrelated workflows are
required. The profile has no default output bucket, so the authoritative
artifacts are on persistent Lustre rather than an OSMO output dataset.

## Environment

The official `vllm/vllm-openai:v0.10.2` Linux amd64 image is pinned by manifest
digest. Its Dockerfile supplies Python 3.12 and CUDA 12.8.1. An isolated `emimic`
environment is created and activated before project Python commands. The
207-package lock includes artifact hashes and was resolved for Linux x86_64,
glibc 2.35 and Python 3.12 with uv 0.10.4. A wheel-only installation dry run passed.

| Component | Version |
| --- | --- |
| LLaMA-Factory | 0.9.4 |
| vLLM | 0.10.2 |
| PyTorch | 2.8.0 |
| Transformers | 4.55.4 |
| Accelerate | 1.10.1 |
| OpenSpiel | 2.0.2 |
| NumPy | 2.2.6 |

vLLM requires NumPy below 2.3 through Numba 0.61.2. The oracle is transferred as
an immutable input instead of regenerated in this GPU environment. The bootstrap
checks the CUDA allocation, imports both training and inference libraries,
records package versions, and runs an NCCL collective on all allocated GPUs.
These runtime checks and full-parameter updates have passed on the allocated
H100 node. Trace quality and paper-level reproduction require separate evidence.

`leact-leduc-qwen3-8b-2` completed its hardware preflight: distributed full-parameter
training, finite losses, an actual weight change, checkpoint reload and generation
all executed. Trace readiness did **not** pass: all 32 candidates in the 16-state
check were truncated and none were selected. Completion of that workflow is not
evidence of a successful LeAct reproduction.

The original full phase first trained the complete 4,032-state coldstart, then ran a
new 16-state generation gate. If every candidate is unusable, it stops before
full evaluations and E-steps. This phase uses a tighter oracle from 10,500 CFR+
iterations, exploitability `8.795129448790906e-06`, and SHA256
`a3a15b2a6b13d3941961e8bba79935ec07b736ecf3333928ac3b4246cf614d08`.
The original 300-iteration preflight oracle remains preserved separately and is
excluded from the new payload. Existing preflight records remain unchanged.

`leact-leduc-qwen3-8b-3` completed coldstart training but failed before the
generation gate. Eight independent tensor-parallel-size-one inference workers
shared a `rank_0_0` compiler-cache directory; worker 4 failed while exporting a
compiled artifact with `FileNotFoundError` during a temporary-file rename.
The recovery bootstrap exports `VLLM_DISABLE_COMPILE_CACHE=1`. The installed
vLLM 0.10.2 source explicitly checks this flag before `compiled_graph.save`,
bypassing the observed export race while retaining compilation. The recovery
reuses the completed coldstart and the same experiment directory; core Python,
oracle files and experiment arguments are checked against the original full-run
payload. Launch provenance records this environment override. The failed logs,
status and installed-source evidence remain under `cluster/.build/`.

The recovery `leact-leduc-qwen3-8b-4` was admitted at NORMAL priority, then
externally canceled at `2026-10-20T23:10:09Z` by `groot-idle-job-shutdown`.
Its reason cited GPU quota enforcement reclaiming over-allocation from
`pretrain-extension` for another named training workflow in `groot-h100-02`.
The completed coldstart remains on Lustre. A subsequent ordinary-priority
resume on pool-01, `leact-leduc-qwen3-8b-5`, verified the shared checkpoint and
protocol before uploading its fresh immutable payload.

That resume completed all eight inference workers, validating the compiler-cache
workaround at runtime. It exited at `2026-10-20T23:29:48Z` because the intentional
quality gate rejected all 32 candidates across 16 states: no truncation and no
policy fallbacks, but every candidate had empty reasoning. Mean backward and
conditioned lengths were 57.1875 and 55 tokens. This was a model-quality gate
failure, not another quota cancellation. The completed action-only coldstart
is preserved for baseline evaluation; no LeAct E-step or full evaluation ran
in this workflow. Terminal evidence is under `cluster/.build/full-3/`.

`leact-leduc-qwen3-8b-6` uses a fresh `experiment-masked-bc-v1` directory and
changes only the BC template loss mask in core code. Setting LF's
`enable_thinking=False` for BC moves the four empty-thinking tokens into masked
prompt context; the combined token sequence stays identical. Reasoning SFT keeps
`enable_thinking=True`. All 25 tests, including two installed-LF integration
cases, passed with no skips in an isolated source copy before submission. The
old source hashes remained unchanged. Tokenizer audits and regression evidence
are preserved under `cluster/.build/diagnostic-1/` and `cluster/.build/full-4/`.
The fresh experiment shares only the immutable base-model cache with the old
experiment; its training, quality gate and evaluations produce separate records.

Read-only availability checks at `2026-10-20T23:17Z` found three free nodes in
`groot-h100-01` and 27 in `groot-h100-02` that fit this workflow's GPU, CPU,
memory and ephemeral-storage request, with the required AMLFS-07 mount.
The account can access both pools. These shared-pool counters do not expose the
external project allocation that triggered reclamation. The evidence does not
show that an ordinary scheduler retry is forbidden, but it does not establish
that a replacement would retain its allocation. The single pool-01 retry used
the same identity and accounting, without priority changes, and remained
allocated until its quality gate exited. Raw snapshots, cancellation evidence
and the distinction between
physical capacity and project allocation are preserved in
`cluster/.build/gr00t-quota-inspection.json`.

On resume, the bootstrap optionally reads the completed coldstart's root
`*.safetensors` sequentially using `dd bs=16M` before inference. This read-only
prefetch completed in 13.671 seconds on the recovery node; earlier mmap-based
checkpoint loading was much slower. The prefetch has a 120-second cap, and a
timeout or read failure is nonfatal. It changes neither checkpoint bytes nor
the experiment protocol. `bash -n` passes for the updated bootstrap.

The first preflight stopped before installing training dependencies because uv
0.10.4 `pip sync` accepts the lock file as a positional argument, not `-r`.
The corrected bootstrap uses `pip sync --require-hashes --no-build <lock-file>`.
Its exact command shape is checked with `--dry-run` for the target Linux/Python
platform before the replacement submission. The first workflow and its failure
logs are retained under `cluster/.build/preflight-1-*`.

Version evidence:

- [vLLM image metadata](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.10.2)
- [vLLM Dockerfile at v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/docker/Dockerfile)
- [vLLM package metadata](https://pypi.org/pypi/vllm/0.10.2/json)
- [LLaMA-Factory package metadata](https://pypi.org/pypi/llamafactory/0.9.4/json)
- [OSMO single-node cookbook](https://github.com/NVIDIA/OSMO/tree/main/cookbook/dnn_training/single_node)

## Build, submit and transfer

Run from the EgoVerse repository root after the experiment source and tests are
complete. The payload allowlists reproduction source/config/tests and the two
oracle files. It excludes environments, caches, other artifacts and unrelated
repository content. Its archive timestamps and ownership are normalized.

```bash
source emimic/bin/activate
python reproductions/leact/cluster/build_payload.py \
  --output reproductions/leact/cluster/.build/launch/payload.tar.gz
LEACT_PAYLOAD_SHA256=$(shasum -a 256 reproductions/leact/cluster/.build/launch/payload.tar.gz | awk '{print $1}')

osmo workflow validate reproductions/leact/cluster/workflow.yaml \
  --pool groot-h100-02 --set num_gpu=8 \
  --set-string payload_sha256="$LEACT_PAYLOAD_SHA256"

osmo workflow submit reproductions/leact/cluster/workflow.yaml \
  --pool groot-h100-02 --set num_gpu=8 \
  --set-string payload_sha256="$LEACT_PAYLOAD_SHA256" phase=preflight \
  --format-type json
```

Set `LEACT_WORKFLOW_ID` to the returned name. After its task is RUNNING, transfer
only the completed staging directory using the native OSMO uploader:

```bash
osmo workflow rsync "$LEACT_WORKFLOW_ID" leact \
  "$PWD/reproductions/leact/cluster/.build/launch/:/osmo/run/workspace/" --once
osmo workflow query "$LEACT_WORKFLOW_ID" --format-type json
osmo workflow logs "$LEACT_WORKFLOW_ID" -n 200
```

On the current macOS CLI 6.2.10 against server 6.3.1, the first run's native
rsync client stalled while listing the forwarded module. PTY exec successfully
transferred the same 453,166-byte archive without changing the workflow or
installed CLI. For this environment, use this verified fallback after stopping
only the stalled local uploader:

```bash
source emimic/bin/activate
uv pip install pexpect==4.9.0
python reproductions/leact/cluster/upload_payload.py "$LEACT_WORKFLOW_ID" \
  reproductions/leact/cluster/.build/launch/payload.tar.gz \
  --sha256 "$LEACT_PAYLOAD_SHA256"
```

The helper streams base64 over the named task's PTY with echo disabled, verifies
the original archive's SHA256 remotely, and atomically renames it into the inbox.
It does not rebuild or modify the submitted payload.

The bootstrap waits up to 15 minutes for `payload.tar.gz` to match the exact
submitted SHA256. A partial or different upload is never executed. A separate
rsync daemon is unnecessary. Do not update a submitted payload in place; rebuild
and submit a new content-addressed run when code changes.

## Persistent artifacts and full phase

The appendix-audited driver now separates `mode=bc` and `mode=reasoning`.
BC trains for three epochs at `1e-4`, then evaluates every saved epoch with
thinking disabled and inference seeds 0/1/2. There is no BC reasoning gate.
To evaluate preserved BC weights without retraining, use `mode=bc` and pass
`bc_checkpoint_root` pointing to their `coldstart/checkpoints` directory. Use a
fresh `experiment_root` for the new report: the source/protocol differs from the
old jobs and must not be resumed into their directories. Keep all epoch weights.

Reasoning mode requires `reasoning_checkpoint` and `reasoning_provenance` paths.
The JSON identifies the source, dataset SHA256, NoCoT/CoT/joint variant, origin
and protocol rationale. A quality gate is applied to that explicit anchor.
The original implicit BC-to-reasoning initialization was not established by the
appendix. See the [audit](../artifacts/appendix-audit.json) and reproduction README.

The job directory is:

```text
/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/runs/<workflow-id>/
  bootstrap.log
  bootstrap-status.json
  runtime.json
  environment.freeze.txt
  requirements-gpu.lock
  payload.tar.gz
  payload.sha256
  source/
  experiment/
```

`experiment/preflight/result.json` records the hardware smoke result and its
trace-readiness counters. Historical `phase=full` recovery runs reused
`leact-leduc-qwen3-8b-2/experiment` for its model cache, completed coldstart and
preserved evidence, then retried the generation gate. Each new workflow keeps
separate logs/source and binds its resume manifest to the oracle hash. The
command below is a reference for the revised BC evaluation protocol, not a
submission made by the audit. The corrected BC weights already exist; their
best-across-epochs benchmark remains pending.

```bash
osmo workflow submit reproductions/leact/cluster/workflow.yaml \
  --pool groot-h100-02 --priority HIGH --set num_gpu=8 \
  --set-string payload_sha256="$LEACT_PAYLOAD_SHA256" phase=full mode=bc \
  experiment_root="/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/runs/leact-bc-epoch-eval-v1/experiment" \
  bc_checkpoint_root="/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/runs/leact-leduc-qwen3-8b-2/experiment-masked-bc-v1/coldstart/checkpoints" \
  --format-type json
```

Transfer the fresh verified archive with `upload_payload.py`; use a fresh local
staging directory and launch-record filename so earlier evidence is preserved.
The experiment driver rejects incompatible completed-phase manifests and refuses
to silently restart an unfinished training directory. Partial generation shards
and model-only checkpoints remain available for inspection on Lustre.

## Collect results while a task is running

`collect_results.py` reads protocol, live trainer metrics, epoch evaluations and
generation rows through an active task's PTY. It verifies transferred SHA256s,
writes the latest records and keeps versioned metadata. Large generation shards
are transferred incrementally as complete JSONL rows in bounded chunks. Model
weights and caches stay on Lustre. Start collection before
a quality gate can terminate the task; exec access ends when the task exits.

The source path supplies an already-installed environment. For the corrected
experiment, the validated job5 environment can be used while its new environment
is still installing:

```bash
source emimic/bin/activate
LEACT_SHARED_BASE=/mnt/amlfs-07/shared/egoverse/rpunamiya/leact
python reproductions/leact/cluster/collect_results.py leact-leduc-qwen3-8b-6 \
  --experiment-root "$LEACT_SHARED_BASE/runs/leact-leduc-qwen3-8b-2/experiment-masked-bc-v1" \
  --source-root "$LEACT_SHARED_BASE/runs/leact-leduc-qwen3-8b-5/source" \
  --output reproductions/leact/cluster/.build/full-4/collected \
  --watch-seconds 1800 --interval 10
```

`collection.jsonl` records the remote hash, size and collection time. A task exit
can end the watch early; verified files already received remain available under
`latest/` and `history/`. The Mac tracker publisher watches
`cluster/.build/*/collected/latest`, imports changed metadata, expert policies
and trace records into ITSS, and validates the server's record counts and hashes.
With `--prune-uploaded-chunks`, it releases only unchanged transfer buffers after
acknowledgment, retaining raw generations on Lustre and versioned records on ITSS.
This bounds laptop storage during ongoing generation. Existing historical raw
diagnostic files are retained. `--no-rollouts` limits collection to metadata;
`--gate-only` collects the small diagnostic shards and stops after their result.

The tracker is available at http://yam-coverage-rpunamiya.nvidia.com:8126/.
The per-user Mac publisher is `com.rpunamiya.leact-tracker-sync`; its configuration
and logs are under `cluster/.build/tracking/`. An upload key is restricted to
tracker import/health commands. It has no general shell or port-forwarding access.
The ITSS service has its own user cron supervisor and survives the upload client's
disconnect. No existing ITSS application is replaced. For a manual publish:

```bash
source emimic/bin/activate
python -m reproductions.leact.tracking.sync \
  --state reproductions/leact/cluster/.build/tracking/sync-state.json \
  --prune-uploaded-chunks
```

Job6 was canceled by `groot-idle-job-shutdown` at 2026-10-21 00:25:10 UTC
under project GPU quota enforcement, before allocation or payload upload.
The watcher stopped on that terminal state and did not resubmit. The verified
terminal query is saved as `cluster/.build/full-4/status-verified-after-cancellation.json`.
No corrected training or result collector started.

The task-specific `cluster/.build/full-4/watch_job6.py` was designed to keep the
job6 handoff running independently of the chat. It is locked to that workflow,
account, pool, payload hash and the successful 25-test result. It polls only
until the existing queue deadline (`2026-10-21T03:46:49Z`), stops on a terminal
workflow state, and never submits, resubmits or cancels a job. Once the task
starts, it creates the fresh cache link, performs the bounded base-model
pre-read, and publishes the checksum-verified payload. It starts result
collection immediately, then retrieves completed diagnostic records and runs
the CPU prompt audit. Failures in those optional readbacks are recorded without
stopping the experiment. The collector also has a bounded local lifetime.

The detached local watcher writes `watcher-status.json`, `watcher-events.jsonl`,
`watcher.log` and PID evidence under `cluster/.build/full-4/`. Its helpers and
implementation hashes are recorded in the status file. The collector gets its
own `collector-pid.json` and log after upload. These operational files remain
outside the immutable training payload.

The user explicitly requested another attempt at HIGH priority. Job7,
[`leact-leduc-qwen3-8b-7`](https://us-west-2-aws.osmo.nvidia.com/workflows/leact-leduc-qwen3-8b-7),
uses one eight-H100 node in `groot-h100-02`, the same account and project, and
the unchanged full-4 payload (`d9af9aa4cdc5029ff054d6f8e9c0465e03d44b588fbcd54a4d1f2ec9a190d58b`).
It started at 2026-10-21 01:16:37 UTC after 118 seconds queued. Its submission,
25/25 validation evidence, startup watcher and collector records are under
`cluster/.build/full-5/`. The fresh root is still `experiment-masked-bc-v1`,
which job6 never initialized. `watch_retry.py` reads this retry's submitted ID
and pool, verifies HIGH priority and the immutable payload, and performs the
same bounded startup and collection handoff. Exactly one retry was submitted.

Job7 passed all 25 tests and the eight-GPU NCCL check, then completed 378 corrected
BC training steps (final loss 0.0432). It exited with code 1 at 01:33:44 UTC when
the reasoning quality gate returned 0/32 usable candidates: 18 truncated and
14 stopped without a complete thinking block. No quota cancellation occurred.
The terminal query and launch record are in `cluster/.build/full-5/`; the model
and full source/environment remain on Lustre. No additional job was submitted.

The default result collector captured six protocol/training files before task
exit. A second collector used `--gate-only --interval 1` to capture the small
diagnostic JSONL shards and their manifests, stopping when the gate result
arrived. Its `gate-collected/latest/` contains all eight complete shards, eight
manifests and the result. `gate-readback-verification.json` records matching
remote checksums, complete coverage and independently recomputed gate scoring.
Both collectors exited; no workflow remains active. The gate-only option is
an operational helper change outside the frozen training payload.

To regenerate the dependency lock deliberately:

```bash
source emimic/bin/activate
uv pip compile reproductions/leact/cluster/requirements-gpu.in \
  --python-version 3.12 --python-platform x86_64-manylinux_2_35 \
  --generate-hashes --index-url https://pypi.org/simple \
  --output-file reproductions/leact/cluster/requirements-gpu.lock
```
