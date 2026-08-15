# EgoVerse Track 2 — Quantitative Diversity Measurement

Final dimensions:
1. Behavior Diversity
2. Context / Visual Diversity
3. Embodiment Diversity

Overall:
D = (Behavior + Context + Embodiment) / 3

Behavior uses semantic-normalized task clusters (all-MiniLM-L6-v2, threshold 0.80) and coverage × evenness.

Context uses 5 uniformly sampled frames per episode, DINOv2-small, mean episode embeddings, mean pairwise cosine distance, calibrated by the empirical EgoVerse reference distance 0.647402822971344.

Embodiment uses coverage × evenness.

Final datasets are deterministic 120h source subsets:
- Dataset A: Mecka
- Dataset B: Scale

Run from the EgoVerse repo root:

```bash
conda activate emimic
source ~/.egoverse_env
python -m track2.run_track2
```
