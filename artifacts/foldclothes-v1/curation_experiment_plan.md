# FoldClothes Curation Experiment Plan

## Fixed data
- Dataset artifact: foldclothes-v1
- Train source pool: 1,304 episodes
- Validation split: frozen `manifests/val.csv` (163 episodes)
- Test split: frozen `manifests/test.csv` (164 episodes)
- Validation SHA-256: `4e41651a2f09f4b98c506dca9780809da5786d6186250e0cc3e866e76511ff3c`
- Test SHA-256: `5f7781b3d2c7bd5f784a843d0bebbc63bc2c62e31486ffe0113c29d6d8462d35`
- Embedding representation: normalized 512-D CLIP embeddings
- Curation seed: 42
- Per-task target: 129 episodes
- Curated variant size: 774 episodes
- Task balance: 129 episodes each from six source tasks

## Curation manifests
| Run ID | Manifest | SHA-256 |
|---|---|---|
| full-1304 | manifests/train_embedding_manifest.csv | c574843dc0f35c91bebdc84973aff789a9eda4837ab8ae2c71a3d12cde597da8 |
| random-774 | manifests/curation_variants/random-774.csv | 9e47194b6b62c4e6ac5bca4f3d276b502bbe2565834cc1395ba9978ca66c8c08 |
| duration-balanced-774 | manifests/curation_variants/duration-balanced-774.csv | 60513d2d188424d27402f77fcf72345938d12dd9ebac9173bb542c189ce6ada3 |
| diversity-774 | manifests/curation_variants/diversity-774.csv | dd438636443971b6fe3d6220c7db663296c90ccc91cfb95fd07f3916879b356a |

`train.csv` is byte-identical to `train_embedding_manifest.csv`. Curation chooses only from the training pool. Validation and test are never curated.

## Selection definitions
- random-774: seeded random selection within source task.
- duration-balanced-774: seeded selection within equal-count duration tertiles per source task.
- diversity-774: deterministic greedy farthest-first selection in normalized CLIP space within source task.

## Invariants
- Model: `hpt_bc_flow_human` (HPT, human bimanual cartesian).
- Same image/action preprocessing (`Human` cartesian, stride 3).
- Same train batch size (16) and optimizer configuration.
- Same number of optimizer updates (`max_steps: 2000`) for all 774-episode variants.
- Same frozen validation and test manifests.
- Select checkpoint only by `Valid/action_loss`.
- Evaluate test once per selected checkpoint.
- Record code commit, environment, hardware, wall-clock time, seed, and checkpoint step.

## Diversity scores
Scored against the 1,304-episode train pool with the dashboard metric (random-774-sized baseline = 50).

| Run ID | Diversity score | Coverage | Internal repetition |
|---|---:|---:|---:|
| random-774 | 50.3 | 0.9793 | 0.8076 |
| duration-balanced-774 | 50.3 | 0.9786 | 0.8071 |
| diversity-774 | 52.1 | 0.9824 | 0.8044 |

The visual gap is small because every clip is garment folding. Training loss is the efficacy test.

## Primary comparison
Compare random-774, duration-balanced-774, and diversity-774 at equal episode count, task distribution, update budget, and evaluation protocol.

## Secondary comparison
`full-1304` is recorded but not trained in v1.

## Launch
```bash
modal run modal_train_foldclothes.py
```

Smoke plumbing (24/6 episodes, 4 steps):

```bash
modal run modal_train_foldclothes.py --smoke
```
