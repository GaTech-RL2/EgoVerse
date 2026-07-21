# Session Handoff — 2026-07-20 late (pre-compact snapshot)

> Read this + `ai_docs/fpp_deployment_note.md` (hardware shortlist + contract) +
> `ai_docs/pickplace_training_infra.md` (val instrumentation history). Dashboards:
> `deliverables_0707/pp_dashboard.html` (regen: `python /coc/flash7/czhang883/tmp/build_dashboard_v3.py`).
> Branch `rby1_aria_policy` pushed through `14eb1224`.

## CURRENT TASK: nav_pick_and_place (penguin → blue basket), FPP round
Corpus `datasets/aria_fullpp` = 135 demos / 53.7k frames (4 new datasets merged, 10
clipped-release demos dropped). Also: `exp1_glove` (56), `exp1_bare` (56, balanced),
`exp1_navonly` (29, ablation), `aria_fullpp_wam3` (DINOv3-B world targets).

## RUNNING NOW (check `squeue -u czhang883`)
| jobs | what | note |
|---|---|---|
| 3543202-08 | original 7-run fleet (glove/resnet/d3conv/d3lora/bare/navonly/wam3) | most near/at ep1999; navonly finishes 07-21 AM |
| 3544361/62 | **fpp_hd_resnet_2k / fpp_hd_wam3_2k** — NEW after hardware feedback | dropout **0.9** + noise 0.03 (0.6 caused proprio over-reliance on robot); OLD-method val (train-split holdout, NOT teleop) |
| 3544366 | gate eval (old method: clean/shift/pzero/noise on human data) of the 6 best snapshots | results → report pzero column = proprio-reliance quantification |
Watcher: `tmp/wall_watcher5.sh` (disk-guarded) covers 3543202-08; HD runs NOT covered — resubmit manually on wall (template below).

## USER DIRECTIVES (latest, override earlier)
- **Teleop val (val_v2 AND predecessors) declared NOT REFERENCEABLE** after hardware
  testing — ignore those numbers for decisions. Analysis = OLD method: human-data
  gate evals (clean / proprio-zero / shift / noise) + hardware.
- Hardware finding: dropout-0.6 policies attend proprio too much → HD runs revert to 0.9.
- Subagents: 1-2 allowed for verification pushes (rule relaxed 07-19). No login-node
  compute (sbatch everything; template `tmp/eval_jobs_cpu.sbatch`). Monitoring wakeups authorized.
- 2000 epochs, batch 32 unchanged. Snapshot-select deployables (fleet optima ~ep400-1000).

## KEY RESULTS SO FAR (FPP round)
- Best-snapshot ranking (teleop-val era, now downgraded to secondary): d3lora@99
  0.161 < d3conv@399 0.165 < wam3@999 0.185 < resnet@1599 0.189 < bare@1399 0.197 <
  glove@699 0.203. DINOv3 leads big-data regime; glove hypothesis trending negative;
  navonly ablation unfinished.
- t1-anchoring NEGATIVE result: dropout 0.6 did not make policies anchor at proprio
  (t1 ≈ full-chunk MAE throughout) — and hardware showed it *worsened* proprio reliance.
- Nav proven learned by ep~100-200 (heading cosine 0.92, 100% turn signs), drives at
  human pace 1.3-1.9× teleop.
- 07-20 17:04 flash7 100%-full incident killed the fleet; recovered (210 GB reclaimed,
  resumed from snapshots, ~1h lost/run). Old rounds trimmed to last/best ckpts only.

## INSTRUMENTATION FACTS (keep honoring)
- Robot-side recordings are cv2-BGR (deploy path flips; datasets need channel fix).
- Loader `mode: train` FORCE-holds-out ≥1 episode (max(1,...)); `mode: total` = all.
  fpp fleet trained mode:total; HD runs deliberately mode:train (holdout = old-method val).
- Norm quantile fallback handles sparse dims (dim 39 r_hand grasp joint; firm_grasp
  still False in all exports — standing question to the data agent).
- Concise val logging + t1/short8 horizon metrics live in hpt.py (concise_val_metrics).
- Old-method gate eval pattern: `tmp/eval_fpp_gate.py` (sampled human frames,
  clean/shift10/20/pzero/pnoise .01/.03/.05).

## RESUBMIT TEMPLATE (wall timeout / dead GPU)
```
cd /coc/flash7/czhang883/Documents/EgoVerse
sbatch --job-name=wbimg_<DESC> --exclude=puma,deebot,qt-1,sonny,cyborg,crushinator,ig-88,spd-13 \
  --export=ALL,DATASET_NAME=<DS>,RAW_DATA_PATH=/coc/flash7/czhang883/__skip__.hdf5,TRAIN_CONFIG=<CFG>,DESCRIPTION=<DESC>[,EXTRA_HYDRA_OVERRIDES=model=...] \
  submit_wb_img_training.sbatch
```
HD runs: DS=aria_fullpp CFG=experiments/wholebody_image/wb_img_proprio_fpp_hd DESC=fpp_hd_resnet_2k;
wam: DS=aria_fullpp_wam3 CFG=...fpp_hd_wam DESC=fpp_hd_wam3_2k. Auto-resumes from last.ckpt.

## NEXT STEPS QUEUE
1. Gate-eval results (3544366) → report table; pzero column quantifies proprio reliance
   per checkpoint (crop100-old reference: clean 0.013 / pzero 0.016).
2. navonly finishes 07-21 AM → same gate eval → manip-data ablation verdict.
3. HD runs finish 07-21 → gate eval → compare vs dropout-0.6 twins (pzero gap should
   shrink dramatically) → new hardware shortlist.
4. Glove/bare + structure verdicts re-derived under the old method.
5. Deferred ideas: delta-action space (anchoring fix), teleop co-training, dense-grid
   PCA world targets, per-image ColorJitter vectorization.

## ENVIRONMENT
venv `source emimic/bin/activate`; caches XDG/HF on flash (`/coc/flash7/czhang883/.cache`),
HF_HUB_OFFLINE=1 for training; TMPDIR=/coc/flash7/czhang883/tmp; wandb personal (entity
null) project sew_policy; flash7 had 294 GB free after cleanup — watch it (external
writers filled it once). Bad-node exclude list in the template above.
