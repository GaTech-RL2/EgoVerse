# EgoSpectrum — 5-minute talk

Live deck: https://egospectrum-dashboard.vercel.app/

Use the dashboard as slides. Do not invent training losses. Full HPT jobs are still running.

## Clock

| Min | On screen | Say this |
|---:|---|---|
| 0:00 | Hero | Robot data looks diverse until you measure it. We turn visual interaction diversity into a number, without captions or an LLM judge. |
| 0:40 | 62.8 vs 50.3 | Same 400 EgoVerse clips, two 100-clip subsets. EgoSpectrum 62.8, random 50.3. Coverage is basically tied. The win is fewer lookalikes. |
| 1:30 | Behavior map | Gray is the population. Warm is random. Lime is EgoSpectrum. Random keeps typical clones. We keep the unusual angles. |
| 2:20 | Why visual | Task-balanced is not visually diverse. 129 fold-shirt clips can still be the same table, same fold, same camera. CLIP sees the interaction, not the label. |
| 3:10 | Fold-clothes table | Stress test: one activity family, 1,304 train demos, three 774-clip sets. Random and duration both 50.3. EgoSpectrum 52.1. The score tells you the pool is already samey. |
| 4:00 | Training protocol | So we train. Same HPT, same 2,000 steps, same frozen val/test. Only the 774 clips change. Smoke passed on Modal. Full runs in flight. Efficacy is held-out action loss, not the diversity number. |
| 4:40 | Close | Diversity is now a number you can optimize. The training job is the test of whether that number matters. |

## Numbers you can say

Mixed 400 → 100

- EgoSpectrum 62.8 · coverage 0.9210 · repetition 0.7442
- Random 50.3 · coverage 0.9278 · repetition 0.7897
- Coverage retained 99.3%. Lookalikes down 5.8%.

Fold-clothes 1,304 → 774 (129 per task, seed 42)

- random-774: 50.3 / 0.9793 / 0.8076
- duration-balanced-774: 50.3 / 0.9786 / 0.8071
- diversity-774: 52.1 / 0.9824 / 0.8044

Protocol

- Train 1,304 / val 163 / test 164
- HPT flow-matching, batch 16, 2,000 steps, seed 42
- Checkpoint by `Valid/action_loss`, test once
- Smoke: 4 steps, 24/6 episodes, val action loss ≈ 139. Plumbing only.

## Do not say

- That EgoSpectrum already improved policy performance
- Any val/test loss except the smoke number, and only if you label it smoke
- That coverage went up. It did not. Repetition went down.
- That fold-clothes is a huge diversity win. The small gap is the finding.

## If they ask

**Why are the fold-clothes scores so close?**
CLIP ego videos sit in a tight blob. 774 of 1,304 already covers the population. The metric is working: it reports a redundant pool.

**Is this just within-task diversity?**
No. All three 774 sets are task-balanced. The difference is visual, not label counts.

**Why not π0.5 / full 1,304?**
Equal-update HPT on 774 is the controlled test. Full-pool training is out of scope for v1.

**What is the method?**
8 frames → frozen CLIP ViT-B/32 → mean + L2-normalize → farthest-first in 512-D. Score = coverage + low repetition, indexed so random ≈ 50.

**Where did you train?**
Modal. Sync from R2, then three sequential HPT jobs.
