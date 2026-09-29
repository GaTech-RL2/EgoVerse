# Frozen-policy Codex evaluation report

Open [index.html](index.html) directly in a browser. The page embeds its dataset,
scripts, styles and vector diagrams, and loads only local selected media. No web
server, network dependency or provider access is required. The vision tab can be
opened directly with `index.html#vision`.

Collection was interrupted by a local Mac reboot. This interim report is an
audited snapshot, not a live workflow monitor; it makes no claim that collection
is currently running. Interrupted or unsealed attempts do not add scored failures.
The displayed audit timestamp and coverage identify the included evidence. Fresh
post-reboot audit, acquisition and recovery-merge outputs must be bound together
before rebuilding the metrics.

After activating the project Python environment, rebuild from the repository root:

```sh
python astra_reversal/reports/codex_full_evaluation/build_report.py
```

The defaults first discover FRS and vision recovery merges from
`recovery_after_dns_postprocess_state.json`, then the earlier
`recovery_postprocess_state.json`. Each family is selected independently. Recovery
entries must have status `verified_report`; the builder checks the exact monitor
schema, advertised SHA-256 and reviewed recovery plan. A malformed verified entry
fails closed. Until a verified recovery merge is available, the original
postprocessing monitor is used for that family. If no monitor entry exists, the
original audit/merge locations are used. Explicit input arguments override discovery.
The FRS acquisition receipt must bind to the selected audit hash. The builder never fetches artifacts, executes models, modifies
the scientific producer, updates audits or changes canonical case selection.
Run the independent auditors and recovery merger separately before refreshing this
report. A build is an explicit snapshot; the HTML never polls active campaigns.

Final independent audit outputs and other local locations can be supplied with:

```sh
python astra_reversal/reports/codex_full_evaluation/build_report.py \
  --frs-progress /path/to/final-frs-audit/progress.json \
  --frs-acquisition /path/to/matching/acquisition_cost.json \
  --vision-merge /path/to/refreshed-vision-merge/provisional.json
```

To use a merged FRS campaign, pass `--frs-merge /path/to/merge.json` instead of
`--frs-progress`. Its per-task source audit paths and hashes select the original
proofs and extracted evidence. Its acquisition ledger includes all approved
source roots while canonical method metrics count each selected task once.

Original `64bb1649` and routing/transport recovery `9ab4cb57` are distinct pinned
producer identities, with different payload hashes. The report retains the actual
identity on every selected task and lists audited case counts for each producer.
A pinned source-equivalence receipt verifies unchanged scientific implementation,
protocols, prompts and assets. Runtime changes are limited to task selection and
relay transport; report and test files also changed. Recovery source roots and
their order come from explicit hash-pinned plans. Adding a later reviewed plan is
an additive entry in `APPROVED_RECOVERY_PLANS`, not an unverified fallback or a
source relabeling. Unreviewed plans and changed receipts fail closed.

All 20 recorded instructions, including pending task identities, come from the
committed task inventory at a pinned file digest and repository revision. Audited
summary instructions and each rollout's BDDL hash must match that inventory. Task
metadata does not supply scores; only independently audited task outcomes do.

`--ops-root` changes the private input root; `--frs-campaign` changes the FRS
extracted-evidence root. Vision final audit overrides are supplied to the recovery
merger first; its output already binds each selected case to its independent audit
directory. `--output` may select a subdirectory of this report directory.

`--no-videos` keeps only actual initial camera PNGs. `--examples-per-cohort 0`
omits all media; the default is two. `--max-media-mib` defaults to 120, is capped at
256, and fails rather than silently dropping selected evidence. The current
examples are much smaller. Media names are content hashes, and only stale
builder-owned media files are removed on a successful refresh.

The source template is [report.html](report.html), with diagrams in
[frs-flow.svg](frs-flow.svg) and [vision-flow.svg](vision-flow.svg). The builder
writes [results.json](results.json) and [index.html](index.html) atomically per
file. The HTML is independently self-contained; the JSON is a convenience export.
Raw NPY arrays and private job journals are never copied.

The public JSON is constructed from explicit field whitelists. It contains
per-task outcomes, per-method measured cost, unknown-usage flags, aggregate
acquisition costs, receipt hashes and selected media hashes. It excludes private
paths, signed URLs, credentials, invocation IDs, prompts, provider event streams,
returned justifications and hidden reasoning. The builder rechecks input receipt,
summary, provider-ledger and selected image/video bindings; it relies on the
independent scientific audits for complete array and protocol validation.

Full-cohort rates remain `null` until all 20 task identities are independently
audited within a cohort. FRS comparisons use ten matched resets per task; vision
comparisons use one reset with a shared native baseline and at most two physical
rescues per arm. Partial scores are completion-ordered subsets, not predeclared
statistical samples. The two cohorts are never pooled. Missing token usage remains
unknown or a lower bound. A stale FRS acquisition receipt is withheld until its
audit-progress hash matches; audited per-method costs remain available.

The initial media selection rule is deterministic: the first audited identity and
reset, then the first native-baseline failure in identity/reset order. Every
physical method or revision for a selected case is available in its video menu;
assisted outcomes do not affect selection. As audits advance, the selected
examples can change. Videos omit inference wait and are not wall-clock measures.
