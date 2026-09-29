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

The defaults discover the current FRS audit and immutable vision recovery merge
from the existing postprocessing monitor state, checking the advertised SHA-256
for each. If no monitor entry exists, the original audit/merge locations are used.
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
