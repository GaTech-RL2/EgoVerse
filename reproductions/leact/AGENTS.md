# Experiment compute lifecycle

- Keep allocated OSMO nodes across training failures, completed jobs and recipe changes. Relaunch the training or simulator job inside the retained allocation.
- Never cancel a live allocation automatically to change recipes, retry a driver, or clean up workers. Release nodes only when the user explicitly requests release or cancellation.
- Keep allocation status separate from training-job status and completed optimizer updates. A RUNNING keeper may be idle after a failed job.
- Stage and verify the replacement job before switching processes. Preserve checkpoints and experiment evidence, and retain independently allocated simulator nodes when the model job fails.
- Do not exec into a model group during its OSMO startup barrier. Require every task to be RUNNING before using the in-allocation job interface.
