# LIBERO-MAX hardware-interface integration

This adds the released LIBERO-MAX substrate and paired-trace checks to the
direct-Astra hardware-interface study. It is **not a completed Astra evaluation**.
Both frozen manifests validate and all 131 upstream tests pass locally. OSMO
access recovered on 2026-10-12. The first GPU commissioning workflow,
`liberomax-hardware-interface-20261012-commission-1`, stopped before simulator
rollouts: Plus lacked robosuite's undeclared `termcolor` dependency, and PRO's
upstream downloader retained only the last repeated `--include` list. The runtime
now pins `termcolor` and downloads all pinned PRO substrate patterns in one list
before running the unchanged upstream setup checks. A fresh GPU check is required;
there are no scored MAX outcomes yet.

`commission-2` passed those fixes but both sources then failed on the missing
`future` package. It now uses the upstream `future==0.18.2` pin. The new
`runtime-check` stage installs the actual source/runtime on CPU, imports the
benchmark environment and loads all eight selected task/reset pairs per source.
This check runs before another GPU commissioning launch; unit fixtures alone
did not exercise these imports.

`commission-3` passed all **48 PRO native commissioning episodes** (eight
events × three interfaces × Base/Dynamic). These are unscored checks with zero
model calls. Plus passed imports and reset-file loading, then stopped on a
missing kitchen250 scene asset. Its official setup additionally requires
`Sylvest/LIBERO-plus/assets.zip`; the bootstrap now installs that separate pack
at revision `dd2bd61b7d9a6fef1abc52d606e983b41886a149`, checking its full SHA-256
before extraction. `--source plus` retries Plus without repeating PRO. The Plus
native check and live Astra integration remain open.

Source is pinned at `01625e290b1aa07511bdc820781e635e851353cc`. Max contains
8,000 Base/Dynamic pairs per policy; Lite contains 800. F, B0 and B are separate
interfaces: a full three-interface Lite comparison requires 2,400 pairs / 4,800
rollouts. Neither that result nor a full Max result exists for this study.

Sources: [benchmark specification](https://github.com/liberomax/LIBERO-MAX/blob/01625e290b1aa07511bdc820781e635e851353cc/docs/BENCHMARK_SPEC.md)
and [runtime integration](https://github.com/liberomax/LIBERO-MAX/blob/01625e290b1aa07511bdc820781e635e851353cc/docs/RUNTIME_INTEGRATION.md).

## Preserved contracts

- Frozen manifest hashes, case identities, original event payloads and source
  composition are checked. Release manifests are never edited.
- Plus and PRO run in separate processes and OSMO tasks with separate registries
  and data roots. Their source revisions and the PRO BDDL/reset dataset revision
  are pinned in `astra_reversal/hardware_interface/liberomax.py`.
- MuJoCo 3.2.6 and robosuite 1.4.0 match the reference runtime. Ordinary LIBERO
  and RoboCasa retain their own simulator versions.
- Actor sensor reads retain the upstream observation transforms. Raw camera
  getters would remove sensor corruption, brightness changes or PRO transforms.
- Pair checks compare every executed native action through the event step,
  including the action after which the event fires. Matching resets or proposed
  chunks alone is insufficient.
- Unreached triggers remain valid task outcomes when the complete prefix and
  terminal outcome match. They remain in the final denominator.
- Goal evaluation stays private. No event flag, object pose, task implementation,
  success label or expert demonstration is supplied to an actor.

## Native commissioning

`tools/commission.py` selects the first Lite case in manifest order for every
source/event combination, without consulting outcomes. It creates a **separate,
unscored probe** with a step-2 trigger, six-action horizon and deterministic
controller. Original and modified cases are both retained. These tests exercise
event mechanisms; they are not released benchmark trials or policy results.

Each source checks eight events × three interfaces × two arms: 48 short
episodes, 96 across Plus and PRO. Native prefixes must match; controller,
reset and terminal-state hashes must agree across interfaces. Event traces,
terminal states and front/wrist images are saved. B exercises its native control
path without calling its observer. There are zero model requests and no inference
credential. Artifacts use an owned, new R2 prefix with no overwrite.

```bash
source emimic/bin/activate
python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q
# Commit tested sources before packaging.
python experiments/liberomax_hardware_interface/tools/prepare_osmo.py \
  /path/to/new/max-staging --stage staging-check --pool groot-l40-05
python experiments/liberomax_hardware_interface/tools/prepare_osmo.py \
  /path/to/new/max-commission --stage commission --pool groot-l40-05
python experiments/liberomax_hardware_interface/tools/prepare_osmo.py \
  /path/to/new/max-runtime --stage runtime-check --pool groot-l40-05
python experiments/liberomax_hardware_interface/tools/prepare_osmo.py \
  /path/to/new/plus-retry --stage commission --source plus --pool groot-l40-05
osmo workflow submit /path/to/new/max-staging/workflow.yaml --pool groot-l40-05
osmo workflow submit /path/to/new/max-runtime/workflow.yaml --pool groot-l40-05
# Submit commissioning only after source delivery and both runtime checks pass.
osmo workflow submit /path/to/new/max-commission/workflow.yaml --pool groot-l40-05
```

The bootstrap installs simulation dependencies without learned policy packages
or demos. It records the dependency freeze, GPU identity, release checks and
both test suites. Source is injected with the OSMO submission.

## Open gates

Both GPU source checks must pass. The live Astra adapter then needs audited
continuation of the same actor history and native action prefix, distinguishing
prefix replay from new inference in token/call accounting. The remote provider's
policy-seed contract must be verified or disclosed as a protocol deviation.
The current launcher exposes only staging and native commissioning; it cannot
claim a live smoke or benchmark score.

After these gates, run unchanged Lite cases and report every assigned pair,
including missing and invalid records. Full Max reporting requires all 8,000
pairs per interface. Small probes and unit tests do not satisfy either track.
