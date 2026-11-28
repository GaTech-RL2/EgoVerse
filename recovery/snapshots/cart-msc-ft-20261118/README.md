# MSC photometric augmentation: B180 → FT10k

User approved replacing the previous shared-brightness/gamma experiments with
an explicit MSC photometric comparison. New isolated remote root:
`/coc/flash7/cxin34/cart-msc-ft-20261118`.

Two variants: `cart-B180-msc-aug-ft10k-s42` and
`cart-B180-msc-aug-consistency-ft10k-s42`.

Both start from original B180 checkpoint SHA711c18ba5d146aa794400177cdab3bc746c6ce16c8c370dbc4ab8a07b7a652b6.
Same seed42, target30train9027/10valid3027, frozen source normalizer,
action+futureDINO, batch32, AdamW1e-4/WD1e-4, full trainability and BF16.
Stop at10000 optimizer updates; retain original60000-step cosine1e-4→1e-5
to match historical F10k's learning rate. Original pretraining/data unchanged.

## Exact reference and adaptation

Official Pointcept commit `77f87abe69bf098ebccf3e9b08400963954af6ac`,
`configs/scannet/pretrain-msc-v1m1-0-spunet-base.py` and
`pointcept/datasets/transform.py`. Extracted unmodified RandomColorJitter,
RandomColorGrayScale and ChromaticJitter classes with registration decorators
omitted; retain MIT license. Source hash stored in upstream.json.

- Brightness.4 / contrast.4 / saturation.2 / hue.02, independent p=.8
  and random operation order.
- ChromaticJitterp=.95,std=.05 after the four transforms.
- No gamma. Disable both old encoder brightness and channel jitter to avoid
  double augmentation. Preserve original geometric augmentation and proprio.
- Apply the official NumPy transforms to concatenated valid RGB from the three
  streams for each observation. Convert RGB0..1→0..255→0..1 using float32.
  Contrast mean is computed over that concatenation, which includes repeated
  samples across global/local crops; this is an explicit multi-stream adaptation.
  Operation parameters are shared across streams/history, Gaussian noise is
  per point/channel. No point-ID mapping exists to tie noise for overlapping points.
  All-zero padding and XYZ are unchanged. No new evaluation augmentation.

Aug-only uses one task-supervised view. Both experiments generate two views to
keep augmentation RNG advancement paired. Consistency variant averages both
task losses and adds0.1×masked normalized clean-action MSE. Replay all RNG
for geometry/sampling/proprio/flow noise/time and restore first-pass advancement.
This is photometric consistency across the entire MSC transform distribution,
not brightness-only consistency. Hierarchical parent detach remains unchanged.

## Gates and ownership

GPU contract audit4794202 → smokes4794203/4794204. Formal jobs require real
100→200 separate-process resume, full3027-row validation and independent saved
checkpoint/optimizer/scheduler/identity audit. Authoritative reports/jobs.json.
Normal rl2-lab/debug/nice30000,1A40/L40S,4CPU64GB. Formal completion must include
step10000, full3027 validation, finite model/optimizer and normalSlurm exit.

Previous cart-brightness-ft-20261118 artifacts preserved. Augsmoke4794145
COMPLETED0; old consistency smoke4794146, gates4794160/62 and formals4794161/63
cancelled as superseded before formal training began. No old run overwritten.
