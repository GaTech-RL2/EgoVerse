# Folding-clothes: val metrics vs. accept/reject correlation

**Date:** 2026-06-18
**Sources:**
- Per-episode val metrics: `logs/mecka_pi_folding_per_episode/folding_sharded/merged/per_episode_stats/episodes.jsonl` (462 episodes)
- Human review decisions: `data_quality_review/mecka_folding_clothes_decisions.jsonl` (477 rows)

## Setup

- All 462 scored episodes matched 1:1 to a review decision.
- 13 episodes were reviewed more than once (477 rows → 462 unique hashes); deduped by **latest timestamp**.
- Dedup also resolves the single `uncertain` row → its later `reject`.
- Final labels: **283 reject / 179 accept** (61.3% baseline reject rate).
- Checkpoint scored: `mecka_pi_fold_clothes_2026-05-27_12-01-39` (`HumanBimanualCartesianEuler`, cartesian transform).

## Headline

**There is a real but modest correlation: the model's action-prediction error is systematically higher on rejected episodes. The flow-matching losses carry no signal.** Direction below: positive r / AUC > 0.5 means a higher metric associates with `reject`.

## Per-metric correlation with `reject`

| metric | Pearson r | AUC | p (Mann–Whitney) | mean accept → reject |
|---|---:|---:|---:|---|
| `..._final_mse_avg`        | **+0.27** | **0.67** | 1.4e-09 | 0.227 → 0.323 |
| `..._frechet_gauss_min`    | +0.23 | 0.66 | 4.9e-09 | 0.334 → 0.386 |
| `..._frechet_gauss_avg`    | +0.23 | 0.64 | 7.1e-07 | 0.987 → 1.158 |
| `..._ypr_paired_mse_avg`   | +0.23 | 0.63 | 3.1e-06 | 0.274 → 0.391 |
| `..._paired_mse_avg`       | +0.23 | 0.63 | 3.9e-06 | 0.140 → 0.199 |
| `..._frechet_gauss_max`    | +0.15 | 0.57 | 8.0e-03 | 2.71 → 2.96 |
| `mecka_bimanual_loss` / `action_loss` | +0.09 | 0.54 | 0.16 (n.s.) | 0.083 → 0.088 |
| `..._xyz_paired_mse_avg`   | **−0.07** | 0.44 | 0.036 | 0.0064 → 0.0061 |
| `n_frames_total`           | +0.04 | 0.52 | 0.42 (n.s.) | — |
| `n_resampled_frames`       | +0.03 | 0.48 | 0.37 (n.s.) | — |

(`n_frames`, `n_batches`, `seed`, `embodiment_id` are constant across episodes — excluded.)
(All metric names prefixed `Valid/mecka_bimanual_actions_cartesian_`.)

## Interpretation

1. **The signal is almost entirely orientation error.** `paired_mse` and `ypr_paired_mse` correlate **1.00**; `final_mse`, `frechet_gauss_avg`, `ypr`, `paired` all sit at 0.96–0.97 with each other — one underlying signal: the policy reproduces the *rotation* trajectory worse on rejected episodes.

2. **Translation error (`xyz`) is the opposite** — nearly independent of the rest (r≈0.25) and *weakly negative* with reject. Position tracking does not flag bad episodes; orientation does.

3. **The training-style losses carry no signal** (`*_loss`, p=0.16). Only the geometric/decoded-action metrics separate the classes.

4. **Effect is modest, not a clean separator.** Best single metric `final_mse_avg` → AUC 0.67. A 5-fold-CV logistic regression on all six action-error metrics reaches only **0.69** (vs 0.66 for `final_mse` alone — the others are redundant).

### Reject rate by `final_mse_avg` quintile

| quintile (range) | reject rate | n |
|---|---:|---:|
| Q1 [0.042, 0.152] | 43.5% | 92 |
| Q2 [0.153, 0.216] | 51.1% | 92 |
| Q3 [0.217, 0.281] | 60.2% | 93 |
| Q4 [0.281, 0.395] | 68.5% | 92 |
| Q5 [0.396, 1.384] | **82.8%** | 93 |

## Takeaway

A high-error episode (`final_mse_avg` / orientation MSE / Fréchet distance) is ~2× more likely to be a human reject than a low-error one — consistent with reviewers rejecting inconsistent demos the policy fits poorly. But at AUC ≈ 0.66–0.69 this is a **prioritization signal, not an auto-filter**: useful for surfacing likely-bad episodes for review, not for replacing the manual pass. To use as a screen, rank by `final_mse_avg` (≡ `ypr_paired_mse_avg`); ignore `xyz` and the raw losses.
