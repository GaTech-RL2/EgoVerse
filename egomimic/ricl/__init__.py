"""EgoVerse-native RICL (Retrieval-based In-Context Learning) on pi0.5.

This package adds *retrieval-based in-context learning* to EgoVerse's own pi0.5
policy, reusing the existing zarr/Cartesian/trainHydra/PI-algo stack. The idea
(from ``ricl_openpi``, used here as an architecture reference only): for each
query observation, retrieve the k nearest demonstrations and inject their
(image, state, action) into the policy's prefix so it can imitate a task it was
never trained on. Cross-embodiment = retrieval **bank = human (aria)**,
**query = robot (eva)**, scoped by ``egomimic/scripts/human_robot_pairs.json``.

Modules:
- ``retrieval``: build the kNN index (DINOv3 over ``base_0_rgb`` -> cKDTree) and
  cache, per query frame, the top-k bank refs. (P1)

See ``egomimic/ricl/README.md`` for the M4-vs-cluster split and the runbook.
"""
