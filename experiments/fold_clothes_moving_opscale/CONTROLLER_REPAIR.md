# Separate operational and training sources

The W&B completion checker accepts both a single video record and Lightning's
grouped `videos` record. Every expected video must still have its own nonempty
remote file, and each polling attempt refreshes W&B's cached metadata. Finite
training/validation metrics, the required step and finished state remain gates.

`run_preflight.py` and `launch_pace.py` optionally accept paired
`--training-repo PATH --controller-commit FULL_SHA` arguments. The application
repository remains separately pinned by `--commit` for smoke or by the launch
gate for production. Both repositories must be clean at their exact full SHAs.
Training commands, configurations, working directory and Python import paths use
the training tree. The repaired controller supplies orchestration and backend
verification. Production records both identities and controller script hashes in
the registry; each worker verifies them before training and before completion.
Stored job scripts retain this controller across timeout/preemption continuation.

This allows the existing training commit and its 36 split-specific normalization
proofs to remain unchanged. The normal single-source mode remains supported.

For the stopped first smoke phase, `--reuse-first-phase PATH` requires its paired
`--reuse-first-phase-sha256 HASH`. This is an authenticated artifact inventory,
not a passed gate. The inventory must bind the original source/recipe, failed
video-verifier report, epoch-zero checkpoint and all 13 local videos. In the GPU
allocation the controller checks full checkpoint state, step, exact cohort/video
coverage and fresh remote W&B evidence again. It preserves this first-phase proof
before executing epoch two from that checkpoint with the same W&B identity. The
128-operator smoke remains a fresh run. Previously attempted resume phases are
not silently retried or overwritten.

Archive the original failure and Hydra/evaluation metadata before creating the
inventory, because a genuine resume updates the live output directory. Preserve
the original failed Slurm action in pipeline history and record the reviewed
replacement job explicitly. A manager must reconcile that handoff before using
the repaired launcher; do not restart the old conductor over a manually owned
incident. All original data, normalization, real-GPU resume and launch gates still
apply. The existing training-source monitor can continue runs through their
recorded job scripts.
