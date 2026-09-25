Paired outcomes versus random noise at the five-attempt cap.

The table retains all prescribed cases, including common baseline successes. The CSV and JSON also report the baseline-failed subset.

| Scope | Arm | Cases | Both succeed | Astra only | Random only | Both fail within budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pooled | noise_only | 20 | 9 | 0 | 1 | 10 |
| pooled | language_only | 20 | 9 | 0 | 1 | 10 |
| pooled | vision_only | 20 | 9 | 0 | 1 | 10 |
| pooled | noise_language | 20 | 9 | 0 | 1 | 10 |
| pooled | noise_vision | 20 | 9 | 1 | 1 | 9 |
| pooled | language_vision | 20 | 9 | 0 | 1 | 10 |
| pooled | joint | 20 | 8 | 0 | 2 | 10 |
| libero_goal_ood | noise_only | 10 | 5 | 0 | 1 | 4 |
| libero_goal_ood | language_only | 10 | 6 | 0 | 0 | 4 |
| libero_goal_ood | vision_only | 10 | 5 | 0 | 1 | 4 |
| libero_goal_ood | noise_language | 10 | 6 | 0 | 0 | 4 |
| libero_goal_ood | noise_vision | 10 | 5 | 0 | 1 | 4 |
| libero_goal_ood | language_vision | 10 | 6 | 0 | 0 | 4 |
| libero_goal_ood | joint | 10 | 5 | 0 | 1 | 4 |
| libero_spatial_ood | noise_only | 10 | 4 | 0 | 0 | 6 |
| libero_spatial_ood | language_only | 10 | 3 | 0 | 1 | 6 |
| libero_spatial_ood | vision_only | 10 | 4 | 0 | 0 | 6 |
| libero_spatial_ood | noise_language | 10 | 3 | 0 | 1 | 6 |
| libero_spatial_ood | noise_vision | 10 | 4 | 1 | 0 | 5 |
| libero_spatial_ood | language_vision | 10 | 3 | 0 | 1 | 6 |
| libero_spatial_ood | joint | 10 | 3 | 0 | 1 | 6 |

Pooled discordant episode IDs:

- noise_only: Astra only = none; random only = libero_goal_ood:seed19:task6:state0.
- language_only: Astra only = none; random only = libero_spatial_ood:seed19:task8:state0.
- vision_only: Astra only = none; random only = libero_goal_ood:seed19:task6:state0.
- noise_language: Astra only = none; random only = libero_spatial_ood:seed19:task8:state0.
- noise_vision: Astra only = libero_spatial_ood:seed19:task2:state0; random only = libero_goal_ood:seed19:task6:state0.
- language_vision: Astra only = none; random only = libero_spatial_ood:seed19:task8:state0.
- joint: Astra only = none; random only = libero_goal_ood:seed19:task6:state0, libero_spatial_ood:seed19:task8:state0.

Each frozen arm is compared separately with random_noise at the full five-attempt cap; no best-arm oracle selection is evaluated.
Both all prespecified cases and the baseline-failed subset are retained, pooled and by suite.
Equal aggregate success counts can involve different cases; exact discordant episode IDs are preserved.
Only noise_only versus random_noise holds the intervention operator and basis fixed; other arms also change the operator.
Failure means no success within five attempts; unsuccessful searches remain censored.
