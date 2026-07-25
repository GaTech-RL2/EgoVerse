# Session Handoff — 2026-07-25 01:10 (pre-compact)

> Companions: `ai_docs/fpp_deployment_note.md` (rollout guide + shortlist A–D) ·
> `ai_docs/world_model_aux_head.md` (in the FORK) · figures in
> `/coc/flash7/czhang883/deliverables_0707/`. Branch `rby1_aria_policy` pushed
> through `6cd226b1`. Fork `ZhangChuye/EgoVerse` branch `rby1_encoder_dev` @ `3ad7dcb`.

## RUNNING NOW (teleop round — the live work)
| job | run | state at 01:10 |
|---|---|---|
| 3578151 | `tel_resnet_1k` | ep971 → **completes ~01:15** |
| 3578152 | `tel_d3conv_1k` | ep869 → completes ~02:20 |
| 3578176 | `tel_wam3_1k` | ep721 → completes ~03:45 |
(also user's own `aria_tele_place` job — DO NOT TOUCH)

Val (old method, ~6 held-out episodes of 28): **wam3 0.0758@649 < resnet
0.0768@199 < d3conv 0.0774@599** — spread only ~2%, offline near-equivalent.
tel_resnet was preempted once and auto-resumed cleanly from last.ckpt.

## PENDING ACTION when all three complete (~03:45)
ONE batched gate-probe, then a proactive report to the user:
```
edit /coc/flash7/czhang883/tmp/eval_fpp_hdgate.py:
  ds folder_path -> datasets/rby1_teleop_pp_0724   (mode="total")
  POLICIES -> per run: best-region snapshot + @999
    logs/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch={199,999}.ckpt
    logs/rby1_teleop_pp_0724/tel_d3conv_1k/checkpoints/epoch_epoch={599,999}.ckpt
    logs/rby1_teleop_pp_0724_wam3/tel_wam3_1k/checkpoints/epoch_epoch={649,999}.ckpt
sbatch /coc/flash7/czhang883/tmp/fpp_hdgate.sbatch   (CPU, ~15 min)
```
Report: three-way verdict + reliance table (clean vs pzero) + robot candidates.

## TELEOP ROUND SPEC (user's instructions, 2026-07-24)
Data `datasets/rby1_teleop_pp_0724` = **28 clean demos / 2,570 frames** from
`0724_teleop_pick_and_place.zip` (file 174423: all 17; file 180309: 16 minus
demos **6,7,8,13,14** — corrupted Aria stream: fps=inf, up to 35/36 frozen
frames; user separately confirmed demo_6 also mis-placed). Manip-phase only but
**whole-body action** — base moves 0.13–0.23 m + 0.14–0.31 rad yaw per demo,
torso active (file 1 uses torso 2–4× more than file 2 = two operator styles).
Images were 640² **BGR** → merged HDF5 does resize-to-224 + BGR→RGB; converted
frame visually confirmed by user (blue basket / yellow rim).
`datasets/rby1_teleop_pp_0724_wam3` = same + `obs.dino_wm` targets.

Recipe deltas vs the HD round (all user-specified): **proprio dropout 0.5**
(not 0.9), **crop 10–20 px only** (`scale: [0.83, 0.91]` — robot data already
matches the deployment distribution; ±5° rotate + light ColorJitter kept),
**1000 epochs with T_max=1000** (fixes last round's LR-floor drift), val =
20% episode holdout. Configs: `rby1_wb_img_proprio_act32_tel_{resnet,wam3,d3conv}.yaml`
+ `wb_img_proprio_tel{,_wam,_d3conv}.yaml`.

## HD ROUND (human data) — CLOSED, verdicts
| checkpoint | clean | reliance | status |
|---|---|---|---|
| **A hd_wam3@1399** | 0.0247 | ×1.03 | hardware winner — only policy that reaches + GRASPS |
| **B hd_resnet@1499** | 0.0241 | ×1.00 | vision-clean; stops short of table on HW |
| **C wam3@1599** (0.6-era) | 0.0134 | ×1.28 | continuity baseline; only 0.6-era ckpt kept |
| **D hd_d3conv@1399** | 0.0249 | ×1.44 | optional; best frozen-feature policy |
| hd_nvs3dneck | 0.0259@1699 | ×1.45–1.6 | not cleared; = d3conv endpoint → 3D features gave no offline edge |
| hd_d3lora / hd_nvs3d-linear | — | ×1.9–2.2 / ×2.9–4.0 | closed negatives (linear WORSENS with training) |
**Capacity law:** reliance falls with trainable vision capacity; fine-tuned CNN
beats equal-capacity adapter on frozen features. **Aux world head = the hardware
differentiator** (A>B, identical recipe otherwise). **LR finding:** cosine
T_max=1400 → every run's optimum at ep1399–1499, later epochs drift.
**Near-table (skip-nav) hardware test: ALL checkpoints failed** — close-up
hand/object view is OOD → motivated this teleop round.

## POLICY: NVS CODE STAYS OFFLINE (user rule, 07-24)
`egomimic/models/custom_encoders.py` + all nvs3d configs are **untracked +
gitignored** in EgoVerse (commit 473f0cb1); local files intact; assets at
`/coc/flash7/czhang883/pretrained/nvs3d/`; serving needs `NVS3D_DIR`.
NOTE: git HISTORY before that commit still contains the code — a scrub
(filter-repo + force-push) was offered and NOT yet done. Docs still mention
nvs3d by name (prose only).

## FORK SYNC (collaborator-facing, NVS-free)
`Documents/tmp_egoverse/EgoVerse` → `ZhangChuye/EgoVerse` `rby1_encoder_dev`:
world-model aux bundle (`601901c`) + `WAM_QUICKSTART.md` (`3ad7dcb`) — an
end-to-end guide (clone via abs path `/coc/flash7/.../tmp_egoverse/EgoVerse` or
GitHub, target-gen, train, what-to-watch, serve, compose-with-your-encoder).

## DELIVERABLES (slides/figures, in /coc/flash7/czhang883/deliverables_0707/)
`arch_pipeline_hd.png` (policy flow chart incl. aux branch) · `arch_hw_variants.png`
(4 hardware-tested encoder structures + verdicts) · `arch_dino_usage.png`
(frozen DINO: condition vs target) · `reliance_curves_0722.png` (all gates +
capacity law) · `hd_loss_compare_0721.png` · `teleop_dataset_viz.png` +
`teleop_ego_ep0.mp4` + `teleop_ego_view_{224,896}.png` (dataset figures).
Regen scripts in `/coc/flash7/czhang883/tmp/` (arch_*.py, teleop_*.py) and the
session scratchpad (reliance_curves.py, fpp_progress_chart.py).

## DISK
flash7 at **60G free and drifting down** (external writer; volatile 60–220G
swings all week). Our footprint minimal: 0.6-era ckpts deleted (user-authorized,
only C kept), HD ladders pruned to key ckpts. User's own ~1T lever if needed:
im2Flow2Act 420G, dp3_sim2real 241G, human2any data 294G, miniconda3 76G.
Remaining teleop runs need ~6G — they will finish.

## NEXT-ROUND QUEUE
1. Teleop gate probes + verdict (pending action above), then hardware test of
   the best teleop policy on the near-table task it was collected for.
2. **Near-table / hand-in-view data for the HUMAN corpus** (the OOD fix).
3. Composed config: wam-aux on top of other encoders (aux is the HW differentiator).
4. Deferred (user, saved to memory): try 0724 teleop data as VALIDATION set for
   future human-data policies — bar = val loss must decrease early; speed
   mismatch is the known caveat.
5. Deferred older: delta-action space, teleop co-training, dense-grid PCA targets.

## STANDING RULES
No login-node compute (sbatch/srun; light greps/plots/wandb OK) · monitoring
wakeups authorized · wandb personal (entity null) project sew_policy · teleop-val
(old val_v2) retired for decisions — old-method gates + hardware only · robot
BGR contract per deployment note · 2000 epochs was the human-round default;
teleop round uses 1000 · NVS offline rule above.
