# Frozen intervention recording audit

This is the exact CPU postprocessor used to audit the immutable
`fde89be9786f85f6ed5e889d7f736784a500dd6fc2f81e9db65480cadb606fb4`
evaluation payload. It is a source snapshot, not part of the policy runtime.
`audit_worker.py` and `validate_auditor.py` retain the original audit bytes.
`audit_worker_linux.py` declares the independently verified Linux rendering
arithmetic; `watch_audits.py` uses that wrapper for final audits. The six modules
in `frozen_intervention/` are copied
verbatim from source commit `e0283747`; their original paths and hashes are in
[frozen_source_inventory.json](frozen_source_inventory.json). They supply data
hashing, proposal parsing, noise construction, and image rendering. The audit
does not instantiate their provider clients or run a model or simulator.

The streaming CLI takes an explicit private catalog and pinned input assets:

```sh
source /path/to/emimic/bin/activate
python -B astra_reversal/reports/iterative_interventions/audit_source/audit_worker_linux.py \
  --catalog /private/catalog.json --worker 0 \
  --assets /path/to/pi05_libero \
  --output /path/to/audits/evaluation/worker_0
```

The catalog maps `objects[relative_key].url` to caller-provided signed object
locations. It is not committed. The asset directory must contain the recorded
`norm_stats.json` and `paligemma_tokenizer.model`; hashes are checked against the
worker checkpoint record. The audited environment used Python 3.11, NumPy 1.26.4,
Pillow 12.3.0, and SentencePiece 0.2.2. Pillow identity matters for exact vision
redraws. No network request is made except read-only catalog object downloads.

The original local redraw exposed a CPU difference in Pillow's blend expression:
37 channel values differed by one byte for a box at gain 0.85. Both machines used
Pillow 12.3.0 and NumPy 1.26.4. A read-only Linux CPU check reconstructed the three
original archived arrays and exactly reproduced the recorded image hash. It also
checked 8,192 scalar inputs against separate float32 multiply/add. The local
arm64 wheel matched fused arithmetic for the failing case. `linux_vision.py`
therefore uses explicit separate float32 operations for the declared Linux
renderer, retaining frozen annotation validation and **exact** pixel hashes.
There is no pixel tolerance. This follows the float-alpha expression in
[Pillow 12.3.0 Blend.c](https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/src/libImaging/Blend.c).
The diagnostic evidence is in `validation/linux-redraw-review.json`; it is an
audit portability correction, not a change to the evaluated policy or images.

Each archive is streamed, with compressed SHA-256, byte count, gzip validation,
and matching object receipts before and after. Image arrays are checked and
discarded as they arrive. The audit retains JSON records, small array inventories,
and safe reports. It verifies array hashes, initial round-trip errors, known and
fresh noise, fixed candidate noise reuse, decoded controller actions, observation
conditions, bounded text provenance, vision redraws, common reset records,
proposal/provider binding, and exact assigned-case coverage. `watch_audits.py`
polls eight evaluation workers for at most four hours and writes the aggregate
`summary.json`. Use a fresh output directory when reproducing: the watcher reuses
completed reports already present in that directory.

After all eight archives finish, the separate metadata consistency check also
binds every reset to its manifest entry and the independently checked initial
observation. It verifies preserved JSON hashes and exact 20-case coverage:

```sh
python -B astra_reversal/reports/iterative_interventions/audit_source/verify_final.py \
  --audit-root /path/to/audits --output /path/to/audits/final_validation.json
```

`snapshot_manifest.json` binds this reviewable source snapshot. Targeted unit
checks live in `tests/unit/astra/test_intervention_array_audit.py`; their tiny
synthetic array fixture has no camera or provider data. The older
`validate_auditor.py` additionally performs 11 corruption checks against the
retained development worker-1 archive, including guidance and vision provenance.
Its development inputs are private artifacts, not a unit-test dependency.

The reports distinguish recorded evidence from independent recomputation.
Prefix embedding tensors were not saved, so their hashes, masks, token identities,
bound metadata, and condition IDs are checked without re-embedding. The exact
zero-alpha preflight endpoint was not separately saved; its zero error is a
worker-computed metric. All other saved initialization endpoint errors and
controller decoding are recomputed. Task successes remain recorded simulator
outcomes, not independently rerun trials. Passing this audit establishes recording
consistency within these limits; it does not prove intervention benefit or general
implementation correctness.
