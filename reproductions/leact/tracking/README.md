# rpunamiya reasoning workspace

This adds a separate inventory and PEAR planning page. It does not launch PEAR,
allocate GPUs, change a split or stop either LeAct arm. The planned allocation is
32 additional H100s total, 16 per arm. Both existing LeAct runs remain independent.

The inventory combines the frozen EFE catalog/splits, the proposed PEAR candidate
membership, source-indexed native GR00T YAM graph metadata, eight proposed reasoning
generator families, and the protected reasoning benchmarks. Source graph files,
semantic tasks and RL-admitted environments are reported separately. Native YAM
membership remains unsplit. Catalog builds reject reserved EFE train overlap.

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

First complete the SSH/Silverfort login using `auth_reasoning_itss.command` in
Terminal. The helper records no password and uses its own connection socket.
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
rebuilds the inventory and compute snapshot every five minutes. Its key permits
tracker import/health commands only. The existing tracker publisher, database,
port and old host remain separate. The new host serves the reasoning page at
`http://rpunamiya.nvidia.com:8126/` once deployment and HTTP verification complete.
