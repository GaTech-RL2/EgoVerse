# rpunamiya reasoning workspace

This adds a separate inventory and PEAR planning page. It does not launch PEAR,
allocate GPUs, change a split or stop either LeAct arm. The planned allocation is
32 additional H100s total, 16 per arm. Both existing LeAct runs remain independent.

The inventory combines the frozen EFE catalog/splits, the proposed PEAR candidate
membership, source-indexed native GR00T YAM graph metadata, eight proposed reasoning
generator families, and the protected reasoning benchmarks. Source graph files,
semantic tasks and RL-admitted environments are reported separately. Native YAM
membership is frozen in the ignored private artifact `pear-scaled-v1/native-yam-split-v1.json`.
Counts and task names are displayed on ITSS and kept out of the public repository.
All native saved layouts stay evaluation-only. New training resets require
independent schema-3 scene generation, fingerprint checks and qualification.
Changing the seed of a saved layout does not change the physical scene.
Catalog builds reject reserved EFE train overlap.

The main view shows one reference image per simulator family, its training task
list, and two evaluation categories: seen simulator / unseen task and unseen
simulator / unseen task. The seen category uses only `same_sim_new_task` core
entries; same-task variant checks remain protected in the full catalog. Native
GR00T YAM task names show their frozen split; the source graph files are templates,
not admitted resets. Images
illustrate families and do not establish task membership or rollout success.
The GPU allocation section and oversized introductory headings are removed.

Embodied reasoning has a visible section with all eight proposed generator
families, concrete queries, output formats, scene sources and transfer targets.
Its ten named benchmark datasets are evaluation-only. Generator implementation
and scene admission are still pending; no separate bounding-box output is
specified in the current proposal. Normalized point outputs are specified.
Existing training releases are tracked separately in
`proposals/efe_reasoning_training_sources_v1.json` and shown in the reasoning
section. RoboPoint's synthetic namespaces, Open Images train annotations,
SpatialRGPT's Open Spatial Dataset, SAT train and CLEVRER train are candidates.
The registry records actual access probes and missing integration; it does not
admit them to training or relax named benchmark exclusions. Annotation samples
are available for inspection, while full image staging and reward adapters are
still pending. Existing releases should precede new generators where suitable.

Refresh the 14 image previews with `python -m reproductions.leact.tracking.reasoning_media`. This downloads public EFE reference
images and copies the native scene authoring preview from the pinned GR00T source.
Preview assets and their checksums stay under ignored experiment artifacts;
deployment includes only verified content-named images. Sources are linked in
each caption. Publishing reuses these assets without fetching.

## Live experiment telemetry

`tracking.reasoning_live` is a fallback read-only collector: it queries OSMO and retrieves heartbeat,
step metrics and per-rollout diagnostics from already running workers. It never
allocates a GPU or enables PEAR. The separate publisher runs this collector before
each compact ITSS snapshot; its minimum interval is 15 seconds, plus collection
and upload time. The browser polls every 15 seconds. The Mac is needed for this fallback collector and catalog refresh, not for primary PEAR run telemetry. Heartbeat timestamps expose stale data.

`tracking.pear_collect` runs on ITSS under a user-cron supervisor, with its
own OSMO session outside the web root. It issues only workflow list/query/log
reads and imports the registered arms' structured metric records into the local
database. It also follows the separately authorized retention evaluation. The
Mac is not required for run tracking after this service is authenticated.
Driver events and ten-second heartbeats are written to durable files and OSMO
logs. The hosted collector polls every 20 seconds plus API time; the browser
polls every 15 seconds. Ray log deduplication is disabled for these records.
Histories merge by optimizer step independently of catalog snapshots, so a
catalog refresh cannot erase training curves.

Direct SSH publishing was tested, but cloud workers cannot resolve or route to
the ITSS host. The prototype upload key and credential were revoked. The hosted
read-only collector is the selected transport. No fake measurements were stored.

`cluster.efe_pear_multitask_train` writes reduced actor metrics once on the driver,
and records every complete judged group, including groups discarded for constant
reward. Per-task, per-family, per-suite and full/assisted-start reward summaries
are cumulative and unfiltered. Macro reward weights tasks equally. SGS dispatch
records include the chosen reset, probability, effective sample size, estimated
success, coverage, stratum, quotas and current deficits. Restarts deduplicate
group and optimizer-step events. A nonfinite scalar metric is recorded and stops
the driver. Full raw metrics remain on shared storage.

The rollout wrapper adds per-process cumulative counters for generated/thinking
tokens, turns, elapsed episode time, success, parse/action failures, token-budget
ends and infrastructure/runtime failures. Infrastructure failures remain separate
from robot reward. Thinking counts include only episodes with observed boundaries.
These diagnostics include attempted episodes, whereas driver reward summaries
count only complete judged groups. The dashboard reports the separate counters.

The actor logs policy loss, global gradient norm, LR, entropy and sampled policy
drift through PEAR/verl. Reference-policy KL is absent because the recipe has no
frozen reference forward or KL penalty; missing measurements display as missing,
never zero. Raw upstream keys, timings, throughput, memory, staleness, checkpoint
and validation metrics are preserved. The chart supports per-run and per-metric
selection with exact values on focus or hover; task rewards have their own table.

Prepare the immutable future trainer payload with
`python -m reproductions.leact.cluster.launch_efe_pear_scaled package`. This does
not change the pause state or launch training. GPU integration of this new
instrumentation is a prelaunch qualification check, not a completed result.

The separately authorized retention experiment compares exact base and
warmstart-496 checkpoints on all 765 RealWorldQA questions. Its runner uses the
EvalScope 1.13.0 prompt and exact answer extraction, paper sampling, BF16 and
paired question bootstrap. The local 32,768-output / 65,536-context ceilings are
explicit deviations not specified in Appendix F. Item files retain full outputs,
token counts, input hashes, seeds, model manifest hashes and format/truncation flags.

Serial synchronous inference avoids the earlier batched asynchronous CUDA error,
but measured about 14 output tokens/second. A single base-model answer consumed
13,519 tokens and 985.5 seconds. The v3 alternative first qualifies two batches of
eight on the same node using a separate 512-token throughput probe. The probe
never produces a benchmark score. If both batches exceed 20 aggregate tokens/s
and preserve multimodal prompt-token parity, the node automatically evaluates
both checkpoints on all 765 questions with the original 32,768-token limit and
the same sampling. Cached serial answers remain separate because the serving
profile changed. A bounded guard releases only the redundant serial evaluation
after the batched evaluation persists at least eight answers. Both primary PEAR
training arms remain paused; this bootstrap cannot start training.

The calculator counts retained groups after constant-reward filtering, accounts for
rejected groups, and takes the maximum of asynchronous generation, environment
occupancy and learner time. Episode lengths, simulator occupancy, learner time,
retention, availability and throughput are assumptions, not measured PEAR rates.
The initial 32-update view is an early milestone; 512 is available explicitly.

## Build and preview

From the repository root:

```bash
source emimic/bin/activate
python -m reproductions.leact.tracking.reasoning --refresh-compute
python -m reproductions.leact.tracking.deploy --home-page reasoning \
  --host rpunamiya@rpunamiya.nvidia.com \
  --output reproductions/leact/artifacts/reasoning-workspace/deploy --package-only
```

`--refresh-compute` reads OSMO state only. It counts running tasks with per-task
resource overrides and preserves the last successful allocation timestamp if a
refresh fails. No workflow submission or priority mutation is implemented here.
The compact export contains this workspace only, so it does not enlarge the
existing experiment tracker imports or duplicate historical rollout traces.

## Publish to the new ITSS VM

VM creation request 2164050 requested `rpunamiya.nvidia.com`, Ubuntu 24.04, 2 vCPUs,
6 GB RAM, TLV01 and a three-month initial lifetime. It is a CPU dashboard host.

First complete Entra SSO through Teleport using `auth_reasoning_itss.command` in
Terminal. The helper uses `tsh login --proxy=nv-prd-it.teleport.sh --auth=entra`
and a separate OpenSSH connection socket. The VM must finish provisioning and
register in Teleport first. Direct password SSH is not part of this helper.
Then, from the repository root:

```bash
source emimic/bin/activate
python -m reproductions.leact.tracking.deploy --home-page reasoning \
  --host rpunamiya@rpunamiya.nvidia.com \
  --control /tmp/reasoning-itss-ssh-rpunamiya/control \
  --output reproductions/leact/artifacts/reasoning-workspace/deploy
python -m reproductions.leact.tracking.configure_sync --reasoning-only \
  --host rpunamiya@rpunamiya.nvidia.com \
  --control /tmp/reasoning-itss-ssh-rpunamiya/control \
  --identity "$HOME/.ssh/reasoning_tracker_itss" \
  --output reproductions/leact/artifacts/reasoning-workspace/publisher
```

The publisher uses a separate LaunchAgent and a new restricted import key. It
rebuilds the inventory and compute snapshot between short polling intervals. Its key permits
tracker import/health commands only. The existing tracker publisher, database,
port and old host remain separate. The new host serves the reasoning page at
`http://rpunamiya.nvidia.com:8126/` once deployment and HTTP verification complete.

## Future MoE router policy

Future MoE RL freezes expert-selection `mlp.gate` weights and biases before
FSDP/optimizer construction through `efe_pear_moe_kernels`. Other gates retain
their existing training policy. Qualification requires zero router gradients,
unchanged router weights after an update and reload, and continued expert and
backbone updates. Existing checkpoints and inference evaluations are unchanged.
Rollout Routing Replay is a planned comparison, not an implemented feature:
record rollout expert IDs for all tokens/layers/prefixes, replay these IDs and
recompute differentiable gating weights in training. Frozen parameters alone do
not guarantee identical routes across engines. The source papers and required
routing/load diagnostics are recorded in the proposal.
