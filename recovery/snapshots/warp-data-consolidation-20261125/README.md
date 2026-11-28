# Four-task canonical storage — 2026-11-25

Canonical entry point: `/coc/flash7/czhang883/warp-data/`.

Use `collections/<collection>/raw/` and `processed/<run>/` for raw/processed data, and `datasets/<dataset>/releases/<release>/` for published data. Existing paths remain readable for compatibility. New writes should use the canonical entry point and a new versioned release, not the legacy compatibility paths.

## Scope and preservation

Twenty cxin34-owned collection/dataset roots were published or merged without replacing existing maintainer data. Their payload is retained inside the canonical root at `.cxin34-storage/20261125-four-tasks/`; callers should use the standard `collections/` and `datasets/` paths. Same-filesystem immutable data use hardlinks; cross-filesystem data were copied and SHA256 checked. This is not a cleanup or a new preprocessing/quality decision.

- 96,216 file records independently checked: 93,605 same-inode references, 2,605 copied/deduplicated references, and 6 HDF5 wrappers whose external filenames were updated separately.
- 1,759 HDF5 files scanned; 350 external dependencies staged; original wrapper bytes preserved. Dataset values were not changed by relinking.
- 6,199 legacy entries atomically replaced with canonical symlinks. Original links/files were backed up first in each old filesystem's `warp-data/registry/migration-four-tasks-20261125/original-paths/`. Another 3,764 directly encountered same-inode hardlinks and 3 independent original wrappers remain at old paths. All 96,216 old file paths were checked readable after the switch.
- No original data were deleted. Existing maintainer-owned datasets, their existing scratch backing locations, and live training workspaces outside `warp-data` were not moved. Archived `.attempt`/`.staging` histories remain at their old locations.
- This does not make all four tasks' historical versions identical, remove overlap, or change quality exclusions. Collection unions and versioned releases intentionally share recordings.

## Task entry points

All relative paths below start at `/coc/flash7/czhang883/warp-data/`.

| Task | Raw/source collections | Processed/published data |
|---|---|---|
| Cart | `collections/cxin34_pretrain_cart` (291 source recordings); `collections/chuye_cart_1102` (107 target recordings) | `datasets/cxin34_pretrain_cart_retarget`; existing `datasets/cart_chuye_1102_tgzvis12_fix1124` variants remain unchanged |
| Basket | `collections/cxin34_pretrain_basket` (301 recordings, 318 exported segments); existing Chuye target collections for Nov18/23/24 | `datasets/cxin34_pretrain_basket_retarget`; target BS releases retain 24, 31, and 58 episodes |
| Cloth | `collections/cxin34_cloth_new2_60`, `cxin34_cloth_new2_50_20261123`, `cxin34_cloth_new2_60_111_170_20261125`; union `cxin34_cloth_new2_170` | Existing h168, fix1124, preview, and new50 BS datasets are canonicalized. The separately curated 165-recording union and last105 manifest remain unchanged; 5 MPS-incomplete raw takes are retained, not promoted to valid training data. |
| Drawer | `collections/cxin34_aria_20261125_0556_114` (114 recordings) | `collections/cxin34_drawer_1125_qc` and `datasets/cxin34_drawer_1125_qc`: 145 before/after-BS candidates; 23 accepted cloud segments; r4 is a display-only coordinate conversion. |

## Running jobs

No training submission, cancellation, requeue, code edit, or configuration change was performed. The five existing jobs were RUNNING and their metric steps advanced during post-migration checks: FT 48,000→48,400, scratch 66,000→66,500, DINO-head LoRA 50,500→50,900, no-virtual-filter FT 22,200→22,500, GeoAug 43,900→44,300. These are timestamped observations, not current progress guarantees or throughput measurements.

## Evidence

Canonical registry: `/coc/flash7/czhang883/warp-data/registry/cxin34-four-task-20261125/`.
Full migration journal: `/coc/flash7/cxin34/warp-data/registry/migration-four-tasks-20261125/` (files/dirs/publication/compatibility ledgers, independent audits, and original-path backups).
Local `migration-evidence.json` includes the publication map, file and compatibility audits, tracking-cache checks and training progress evidence.

## Published roots

| Original root | Canonical root | Publication method |
|---|---|---|
| `/coc/flash7/cxin34/warp-data/collections/cxin34_aria_20261125_0556_114` | `/coc/flash7/czhang883/warp-data/collections/cxin34_aria_20261125_0556_114` | atomic_alias |
| `/coc/flash7/scratch/cxin34/warp-data/collections/cxin34_drawer_1125_qc` | `/coc/flash7/czhang883/warp-data/collections/cxin34_drawer_1125_qc` | atomic_alias |
| `/coc/flash7/scratch/cxin34/warp-data/collections/cxin34_cloth_new50_fix1124_bs` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new50_fix1124_bs` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_fix1124` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_fix1124` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_pretrain_cart` | `/coc/flash7/czhang883/warp-data/collections/cxin34_pretrain_cart` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_h168` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_h168` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_1120_preview` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_1120_preview` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_pretrain_basket` | `/coc/flash7/czhang883/warp-data/collections/cxin34_pretrain_basket` | atomic_alias |
| `/coc/flash7/scratch/cxin34/warp-data/datasets/cxin34_cloth_new50_fix1124_bs` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_cloth_new50_fix1124_bs` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/datasets/cxin34_cloth_new2_h168` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_cloth_new2_h168` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/datasets/cxin34_cloth_fix1124` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_cloth_fix1124` | atomic_alias |
| `/coc/flash7/scratch/cxin34/warp-data/datasets/cxin34_drawer_1125_qc` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_drawer_1125_qc` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/datasets/cxin34_cloth_1120_preview` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_cloth_1120_preview` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/datasets/cxin34_pretrain_basket_retarget` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_pretrain_basket_retarget` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/datasets/cxin34_pretrain_cart_retarget` | `/coc/flash7/czhang883/warp-data/datasets/cxin34_pretrain_cart_retarget` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_60` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_60` | nonoverwriting_merge |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_50_20261123` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_50_20261123` | nonoverwriting_merge |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_60_111_170_20261125` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_60_111_170_20261125` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_110` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_110` | atomic_alias |
| `/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_170` | `/coc/flash7/czhang883/warp-data/collections/cxin34_cloth_new2_170` | atomic_alias |

## Atlas verification and display-only correction

Ten releases passed live API checks: first, middle and last episodes, with first/middle/last binary chunks; timestamps are monotonic and all tested cloud streams report [2048,1024,1024]. See `atlas-verification.json` for episode IDs and response byte counts. This is actual manifest/chunk read verification, not just a catalog listing or a new full dataset quality audit.

Cloth new50 BS originally lacked the Atlas tag manifest and tracking/RGB caches. Added 60 segment caches with timestamp alignment error <0.01 microseconds; all 120 NPZ files were reopened successfully. Its exact training-input release `r1_hd2` still exposes pre-BS cloud coordinates alongside post-BS robot state. A recovered middle segment failed Atlas frame preparation. Created `r2_bs_display` as a separate, explicitly visualization-only release: 60 segments / 24,577 rows. Global and hand clouds are re-expressed in post-BS robot frames while preserving each point's world position and color; RGB, actions, joints, hand states, base pose and timestamps are unchanged. It is not a new training release or authorization to replace model inputs. The original r1 is preserved. The transformation and all unchanged columns were checked for every episode; first/middle/last examples then passed live Atlas reads.

Drawer retains the equivalent distinction: original r3 exact model-input coordinates vs working r4 display coordinates. Historical maintainer `_lerobot_raw` provenance errors were not modified. Thus this report does not claim every archived Atlas version is playable.

### Verified Atlas links

- [cxin34_pretrain_cart_retarget@r2_retarget_pcd](http://100.76.236.119:8080/?dataset=cxin34_pretrain_cart_retarget%40r2_retarget_pcd&episode=cc6ac9df-a41b-5d19-88d5-dc59377a8c31)
- [cxin34_pretrain_basket_retarget@r2_retarget_pcd](http://100.76.236.119:8080/?dataset=cxin34_pretrain_basket_retarget%40r2_retarget_pcd&episode=eefc2d53-19b1-5b13-a58f-b1f9af6d718f)
- [cxin34_cloth_new2_h168@r1_hd2](http://100.76.236.119:8080/?dataset=cxin34_cloth_new2_h168%40r1_hd2&episode=3af21413-9edf-546b-8a0f-e7ca721e2ef3)
- [cxin34_cloth_fix1124@r1_hd2_fix1124](http://100.76.236.119:8080/?dataset=cxin34_cloth_fix1124%40r1_hd2_fix1124&episode=1b00fe24-562a-581d-b0db-c6d3896030bc)
- [cxin34_drawer_1125_qc@r4_fix_bs_display23](http://100.76.236.119:8080/?dataset=cxin34_drawer_1125_qc%40r4_fix_bs_display23&episode=b1545bf2-34ea-549e-b885-d00454d3be13)
- [basket_to_right_finetune_chuye_hd2z5_fix1124@r1_bs](http://100.76.236.119:8080/?dataset=basket_to_right_finetune_chuye_hd2z5_fix1124%40r1_bs&episode=cfe96026-2b3a-5d21-b6d8-ff98aa9cb734)
- [cart_chuye_1102_tgzvis12_fix1124@r2_wam_hd2_bs](http://100.76.236.119:8080/?dataset=cart_chuye_1102_tgzvis12_fix1124%40r2_wam_hd2_bs&episode=098612a5-ba6c-5488-b5ba-8be13c39ad5e)
- [basket_2_chuye_hd2z5_fix1124@r1_bs](http://100.76.236.119:8080/?dataset=basket_2_chuye_hd2z5_fix1124%40r1_bs&episode=6f6dd169-b8b4-51d1-9a03-fcb4cab31e61)
- [basket_to_right_3_chuye_hd2z5@r1_bs](http://100.76.236.119:8080/?dataset=basket_to_right_3_chuye_hd2z5%40r1_bs&episode=49c4236c-66e1-5177-b74b-2d46fd8d5884)
- [cxin34_cloth_new50_fix1124_bs@r2_bs_display](http://100.76.236.119:8080/?dataset=cxin34_cloth_new50_fix1124_bs%40r2_bs_display&episode=a4288a9a-076e-5632-82b2-d5c374bc88a4)
