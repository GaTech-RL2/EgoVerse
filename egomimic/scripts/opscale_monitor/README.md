# Operator ladder watchdog

Slurm job 409278 runs `watchdog.sbatch` with one CPU-only allocation. It polls
op4, op8, op12, and op16 every 30 minutes. A stopped job, recent traceback, or
three-hour log stall wakes a scoped Codex CLI agent within the next poll (30
minutes between repeated repair attempts). A healthy agent audit runs every 30 minutes. The agent can fix a diagnosed issue and resume the same rung from its
latest checkpoint with the same W&B ID; it must not replace healthy or pending
jobs. The loop exits when all four reach the configured 5000 epochs.

Current job IDs and the latest assessment are in
`logs/fixed_budget_opscale_hist/watchdog/state.json` and `snapshot.json`.
Agent transcripts are saved as `agent-*.log` and `agent-*.md` beside them.
To stop future checks, create `logs/fixed_budget_opscale_hist/watchdog/stop`;
the loop exits at its next poll. The watchdog does not use a GPU.
