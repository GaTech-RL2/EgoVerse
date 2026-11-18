# PEAR environment and launch qualification

Validation status as of 2026-11-18. These checks are distinct from primary RL and policy success evaluations.

The native control checks completed for ABC-Sim, DexJoCo, LIBERO-Max, MetaWorldPlus, RoboCasa and VLABench. Of 162 source-disjoint reset/action/verifier passes, 156 also pass the recorded reset and post-action RGB audit. Six ABC-Sim resets have no usable initial camera frame and must stay excluded. Protected physical reset overlaps are excluded independently of simulator success.

The initial reasoning bank has 512 independently generated training scenes and 64 development scenes across eight procedural MuJoCo task families. It is not an external benchmark training corpus. A CPU-only test of the actual six-training/two-development Ray server layout passed 16 public-RGB/action/native-reward sessions. Model-side tool imports are separated from simulator modules to avoid initializing the training stack in simulator children. The full model/sampler/score-centering update remains a separate qualification.

Native GR00T YAM has 1,024 newly generated reset inputs across 16 training templates. Their physical pose fingerprints do not match the reserved saved layouts. These are generated inputs until each reset passes native execution checks. Four two-GPU RTX allocations cover distinct template subsets. Runtime, task, asset and reset-source revisions are frozen; existing saved evaluation layouts are not training inputs.

The native YAM probe originally wrote successful RGB/action receipts but exited with `SystemExit(0)`. Arena treats that exception as an unsuccessful run. The probe now uses the native runner's normal step budget so environment close hooks execute. CPU regression checks exercise both single and multiple reset completion after 31 hold steps per reset. This change does not alter physics, task outcomes or success criteria.

RoboLab oracle coverage is task-specific; banana-in-bowl includes a recorded oracle. Its published Cartesian action metadata does not provide component bounds required by our constrained interface. The API probe explicitly selects the bounded native joint-position interface, preserves its component order and limits, and records the raw native specification. An API probe does not claim a positive oracle test or resource-enforced Harbor qualification.

The Isaac simulator bases are built separately from timed native trials. Dependency recovery preserves editable installation files and requires exact locked package versions, including local wheel-version suffixes. Private registry publication verifies that the image configuration and every compressed layer match the cached OCI image. Publication workers allocate zero GPUs. Pinned public SDK/service files are fetched by content digest and checked individually.

Primary warmstart RL remains gated on native mixture admission and a genuine nonzero score-centered on-policy update/reload. The frozen-router numerical update/reload already passed; it is not evidence of an on-policy RL update. The primary recipe retains the full 128 groups by eight rollouts, all four control categories and all eight reasoning categories, separate training/development membership, within-task SGS, and explicit family quotas. Diagnostic updates are discarded and never counted as primary training.
