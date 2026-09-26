# Offline reproduction

These are byte-identical copies of the private read-only provider auditor, its mutation checker and the figure/timeline builder. They make no inference or simulator calls. Their hashes are bound by the parent manifest and receipts.

Run from the repository root after activating the project environment as required by `AGENTS.md`. The exact wire check requires NumPy 1.26.4, Pillow 12.3.0 and Pillow's zlib 1.3. The existing isolated codec provides those packages:

```sh
export ASTRA_REPOSITORY_ROOT="$PWD"
export PYTHONPATH="$PWD/astra_reversal/.deps/frs-audit-codec/site-packages:$PWD"
```

Inputs are the immutable, locally preserved archive extractions; the public bundle intentionally excludes full provider bodies and encoded request images. Each extraction has unchanged task `summary.json`, `events.jsonl`, `provider.jsonl` and numerical `audit.json`, with worker metadata in a sibling `metadata/` directory. Restore archived inputs byte-exactly before use if operational copies have been compacted.

For worker 0, run the provider checker in partial mode:

```sh
python astra_reversal/reports/representation_steering/development/behavior/source/audit_provider.py \
  --task-dir astra_reversal/.deps/representation-development-v1/partial_audited/worker_0/task_6 \
  --worker-dir astra_reversal/.deps/representation-development-v1/partial_audited/worker_0/metadata \
  --archive-receipt astra_reversal/.deps/representation-development-v1/partial_audited/worker_0/task_6/audit.json \
  --output astra_reversal/.deps/representation-provider-recheck/worker_0.json
```

Worker 1 uses `partial_audited/worker_1/task_2` and the same partial mode. Worker 2 uses `audited/worker_2/task_8`, its corresponding metadata and archive receipt, and `--complete`. A partial receipt cannot support a complete claim. Output timestamps change on a new run; the input hashes, decision bindings, cost totals and scientific findings must agree.

The recorded nine-check regression invocation used the same worker-0 inputs:

```sh
python astra_reversal/reports/representation_steering/development/behavior/source/verify_audit_provider.py \
  --task-dir astra_reversal/.deps/representation-development-v1/partial_audited/worker_0/task_6 \
  --worker-dir astra_reversal/.deps/representation-development-v1/partial_audited/worker_0/metadata \
  --archive-receipt astra_reversal/.deps/representation-development-v1/partial_audited/worker_0/task_6/audit.json \
  --output astra_reversal/.deps/representation-provider-recheck/mutation_validation.json
```

It first accepts the unchanged trace, then rejects altered wire hashes, duplicate provider rows, wrong raw-camera digests, cross-arm feedback and stale applied decisions. It additionally accepts the actual archive binding, rejects a changed archived event-file hash and rejects promotion of the partial recording to complete. Only temporary copies are changed.

The timeline/figure builder was run with the frozen final receipts in `astra_reversal/.deps/representation-steering/provider-final/`:

```sh
python astra_reversal/reports/representation_steering/development/behavior/source/build_behavior_evidence.py \
  --run-root astra_reversal/.deps/representation-development-v1 \
  --receipt-dir astra_reversal/.deps/representation-steering/provider-final \
  --donor-library astra_reversal/.deps/image-perturbations/donors \
  --output astra_reversal/.deps/representation-behavior-recheck
```

The manifest records the exact source and output bytes. The evidence JSON records the plotting versions and all illustrated request IDs, typed raw-array digests, PNG-byte hashes, executed intervals and cumulative costs. Figure layout does not modify the policy observations. The independent archive validator supplies the actual NPY byte layer; this provider auditor adds request, wire, history and decision-application checks without rerunning the model or physics.
