# Cloth new2 — MSC scratch 180k

Remote root: `/coc/flash7/cxin34/cloth-new2-scratch-20261121`.
Run: `runs/cloth-new2-B-msc-scratch-s42`; formal Slurm job **4827201** verified RUNNING on brainiac/A40 at step100 with finite losses. Training has started, not completed.

60 raw recordings and all MPS outputs retained. 59 passed independent processing audits; take25 excluded for original retarget QC (22 base steps >30mm), no threshold changes. Frozen seed42 recording split yields49 train/10 valid,22225/4764 rows. Valid takes2,7,8,9,15,16,18,41,48,57. Only training data fits normalization. Longest bilateral-valid windows can be partial demonstrations, not necessarily complete successful tasks.

Three PCDs (global2048/right1024/left1024)+DINO16x768, existing B objective, MSC photometric augmentation only (consistency0). Scratch policy/point encoders, no parent checkpoint; pretrained DINO feature teacher retained. Batch32,AdamW1e-4,WD1e-4,seed42,BF16,180000-step cosine schedule,proprio dropout0.9. Latest state checkpoints every1000, named checkpoints every10000; full validation every5000/final; full-state requeue before4h allocation limit.

Evidence: `release/READY.json`, `remote-reports/release-audit.json`, `remote-reports/final-visual-review.json` (all59 RGB triptychs and7 geometry plots reviewed), `remote-reports/photometric-audit.json`, `remote-reports/cloth-new2-B-msc-scratch-s42-independent-smoke-audit.json`, and `remote-reports/startup-verified.json`. Smoke4827199 and independent gate4827200 passed200steps, separate-process100→200 resume, all4764 validation rows, optimizer/scheduler/code/config identity checks. Final configs and manifests saved here; original pilot visual evidence preserved separately.
