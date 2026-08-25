# Tsimulation

This directory contains exactly two supported PushShapes simulator versions:

- `sim_v1/`: frozen original simulator.
- `sim_v2/`: current fixed simulator with solid-pusher collision handling,
  pocket-bottom-only U-socket friction, and arena edge guards.

The repository-level `Tsimulation` path is a compatibility symlink to
`Tsimulation/sim_v2`, so existing training, evaluation, collection, and replay
commands continue to import the current simulator without code changes.

Use `Tsimulation.pushshapes.get_env("v1")` only when an old v1 replay is
explicitly required. New work should use the default current v2 simulator.
