# Session Handoff — 2026-07-14 (pre-compact snapshot)

> Read this + `policy_training_status.md` (full round history R0–R7, all checkpoints,
> all measured numbers) to resume work. Other docs: `policy_model_card.md`
> (architecture/params), `deployment_plan.md` + `deployment_test_protocol_r6.md`
> (deploy + sim/real protocol), `deployment_debug_guide.md` (hardware debugging),
> `presentation_rollout_0707.md` (slides source).

## RUNNING NOW (Round 7 — check with `squeue -u czhang883`)
| job | run | what | ETA |
|---|---|---|---|
| 3487737 | `d3_lora_2k` | frozen DINOv3-S + LoRA r16 (0.54M vision-trainable) | ~tonight 07-14 |
| 3492749 | `d3_convneck_2k` | frozen DINOv3-S + 9 ResNet BasicBlocks (10.8M = ResNet budget) | ~tonight 07-14 |
| 3492750 | `lingbot_convneck_2k` | frozen LingBot-L + 9 ResNet BasicBlocks (11.2M) | **will hit 24h wall ~ep1550 → needs ONE resubmit** (auto-resumes from last.ckpt), done 07-15 midday |

Logs: `logs/aria_egoposer_firm/<desc>/checkpoints/`. All plain BC, crop100 recipe,
`datasets/aria_egoposer_firm`, batch 32, 2000 epochs (100 steps/epoch).

## WHEN THEY FINISH (the queued task)
1. Gate-eval all three `last.ckpt`s: copy `/coc/flash7/czhang883/tmp/eval_r5_final.py`,
   swap the CKPTS dict paths, run from repo root with emimic venv +
   `HF_HUB_OFFLINE=1` env (see any prior eval invocation in tmp scripts).
   Conditions: clean / shift10+20 / pzero / noise σ=.01-.05. ~8 min/ckpt on CPU, background it.
2. Extend the bar chart (template `/coc/flash7/czhang883/tmp/results_chart.py`,
   palette + layout already validated) with the 3 new bars next to crop100/dino_full;
   save to `/coc/flash7/czhang883/deliverables_0707/`, SendUserFile it.
3. Update `policy_training_status.md` ROUND 7 section + commit.

Reference numbers to compare against (MAE rad): crop100 clean .0126/pzero .0155;
dino_full .0119/.067; dino_lora(v2) .0177/.122. Vision-only gate ≤ 0.03.
R7 hypothesis being tested: is it capacity, conv prior, or features? (convneck runs
= ResNet-budget conv head on frozen features).

## RESUBMIT TEMPLATE (dead-GPU "No CUDA GPUs available" lottery or wall TIMEOUT)
```
cd /coc/flash7/czhang883/Documents/EgoVerse
sbatch --job-name="wbimg_<DESC>" \
  --exclude=puma,deebot,qt-1,sonny,cyborg,crushinator,ig-88 \
  --export=ALL,DATASET_NAME=aria_egoposer_firm,RAW_DATA_PATH=/coc/flash7/czhang883/__skip__.hdf5,TRAIN_CONFIG=experiments/wholebody_image/wb_img_proprio_<NAME>,DESCRIPTION=<DESC> \
  submit_wb_img_training.sbatch
```
Auto-resumes from `logs/<DATASET>/<DESC>/checkpoints/last.ckpt` if present.
(WAM runs use DATASET_NAME=aria_egoposer_firm_wam.)

## STANDING USER INSTRUCTIONS (do not violate)
- **NO subagents / workflows** (Agent, Workflow tools) — quota. Work inline, sequential.
- **NO monitoring wakeup loops** — quota. User pings "check jobs" instead.
- Always 2000 epochs. Batch 32. Same recipe for comparability.
- Minimize Bash calls (auto-mode classifier costs per command).

## ENVIRONMENT ESSENTIALS
- venv: `source emimic/bin/activate` (or `emimic/bin/python`); repo root = cwd for evals.
- env for anything HF/torch: `XDG_CACHE_HOME=/coc/flash7/czhang883/.cache
  HF_HOME=/coc/flash7/czhang883/.cache/huggingface HF_HUB_OFFLINE=1
  TMPDIR=/coc/flash7/czhang883/tmp` (login-node /tmp is unreliable; heredocs need TMPDIR).
- HF token installed at `$HF_HOME/token` (user czhang883); Meta DINOv3 official-repo
  access request pending (not needed — timm mirror ungated, weights cached).
- LingBot: `lingbot-vision` pip-installed -e from `/coc/flash7/czhang883/Documents/lingbot-vision`;
  weights cached (`robbyant/lingbot-vision-vit-large`).
- Login node has tight RAM: big-model CPU smokes get OOM-killed (exit 137) at/after
  "Starting training!" — that's a smoke artifact, GPU jobs are fine.
- Git: branch `rby1_aria_policy` (fork of record: ZhangChuye/EgoVerse, pushed through
  `e9aff88e`; local ahead — push when user asks). Commit style: msg file via
  `git commit -F` (heredocs break when /tmp full) + Co-Authored-By footer.
- Teammate fork: `ZhangChuye/EgoVerse` branch `rby1_encoder_dev` + ENCODER_QUICKSTART.md;
  dataset zip at `/coc/flash7/czhang883/deliverables_0707/aria_egoposer_firm_dataset.zip`.

## DELIVERABLES DIR
`/coc/flash7/czhang883/deliverables_0707/` — charts, GIFs, attention maps, arch diagrams,
dataset zip. Attention-map method: policy's own stem cross-attention (16 latents × heads
averaged), captured via forward hook — NOT Grad-CAM.

## OPEN THREADS (beyond the queued eval)
- Hardware session pending: A/B/C protocol in `deployment_test_protocol_r6.md`
  (crop100 vs dino_full vs wam_dinofull; money comparison = wam_dinofull vs dino_full
  at the table for task-progress).
- User's own storage cleanup candidates (second tier) listed in chat 07-09: dp3 data
  208G, depth_human2any 212G, im2Flow2Act/data 167G, miniconda 76G, EgoVerse superseded
  epoch snapshots ~20G (needs user OK per cleanup-checkpoints skill).
- Rotating the HF token eventually (it was pasted in chat).
- Presentation deck: TOC + per-slide speech delivered 07-08; deck itself is user's.
