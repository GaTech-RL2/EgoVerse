# Gaze-centred global crop + online gaze predictor (2026-09-28)

Hardware-driven round: replace the fixed camera-frame Z-slab global crop with a crop
centred on what the collector was LOOKING AT, so the policy's wide-field stream
follows task attention. Local right-EEF stream deliberately unchanged.

## Design decisions (and why)

- **Ray ∩ cloud, NOT reported gaze depth.** Aria's vergence depth was measured
  unreliable in the 2026-09 gaze audit (median 0.38 m vs scene 1.11 m). We take only
  the gaze DIRECTION (trustworthy) and get range from the stereo cloud: collect points
  in a cone around the ray, take the 25th percentile of along-ray distance (near
  surface, outlier-resistant). Cone widens 0.12 → 0.25 → 0.45 rad on failure, then
  falls back to the eef midpoint. Code: `tmp/abl/build_gazecrop.py`.
- **2.0 m radius** (generous by instruction): the centre comes from an ONLINE predictor
  at deploy, so the radius must absorb predictor error — see the budget below.
- **Episode set unchanged** (same 53, same split, same rows as `dp3c_basket60_2048`)
  so GZ-vs-W2 is a clean single-variable comparison.

## Results

**Crop build** — corpus ray-hit rate **92.5%** (21,445/23,191 frames); per-episode
median 97%; 48/53 episodes >90%. Two episodes are fallback-dominated
(`Chuye_Basket_23` 0% valid gaze, `_24` 57%, `_1` 42%) — kept deliberately rather than
excluded, to preserve the controlled comparison; the fallback (eef midpoint) is a sane
proxy during manipulation.

**Online predictor** (`tmp/abl/gaze_basket_best.pt`) — rect RGB 224 → gaze pixel (u,v).
ResNet18 → 7×7 heatmap → spatial soft-argmax. The earlier probes (scratch CNN 8.49°,
frozen DINOv2 8.71°) diagnosed GLOBAL POOLING as the limiter on a localization task;
keeping spatial structure broke that ceiling:

| | val median |
|---|---|
| predict image centre | 12.52° |
| predict train mean | 10.83° |
| earlier probes (CNN / DINOv2) | 8.49° / 8.71° |
| **this model** | **5.87°** |

**Error budget vs the 2.0 m crop radius** (the number that actually matters):

| gaze distance | induced centre error | fraction of radius |
|---|---|---|
| 0.8 m | 8.2 cm | 4% |
| 1.2 m | 12.3 cm | 6% |
| 2.0 m | 20.6 cm | 10% |

Per-episode video medians on held-out data: 4.69 / 4.90 / 5.23 / 7.26°. Worst frames
34–38°, and 4 of the top-5 are CONSECUTIVE (demo_51 f179–182) = a saccade burst, i.e.
transient, not bias. Even 34° at 1.2 m ≈ 80 cm, still inside the 2 m radius.

## Artifacts

- dataset `datasets/dp3c_basket60_gz` (53 eps, 18,909 rows, global 2048×6 gaze-cropped,
  + `obs.gaze_point_glass` 3-D crop centre per row)
- predictor weights `tmp/abl/gaze_basket_best.pt`; training `tmp/abl/gaze_predictor.sbatch`
- videos `tmp/abl/gaze_videos/*.mp4` + `worst_cases.jpg`
- per-recording gaze tracks `tmp/abl/gaze_basket/*.parquet` (device-frame ray form)
- policy `RBY1_abl_GZ` (W2 recipe: 2048 pts, brightness aug, no geo-aug, no hand-proprio)

## Deploy notes / open items

1. **Temporally smooth the predicted gaze** (short median filter): kills the saccade
   spikes seen in the videos, essentially free.
2. **Train/deploy gap**: policy trains on GT gaze centres, deploys on predicted ones.
   Measured offset (~12 cm typical vs 200 cm radius) suggests this is negligible; a
   noise-injected variant (centre jitter at the predictor's error distribution) would
   settle it in one training run, no new data build.
3. Online path: predict (u,v) → shoot ray through the LIVE depth cloud → hit = crop
   centre → keep global points within 2.0 m. Same rect intrinsics as training
   (KH = 308.052437·224/512, C0 = 112).
