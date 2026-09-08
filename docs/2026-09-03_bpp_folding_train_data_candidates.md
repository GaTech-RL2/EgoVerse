# Adding training data to the folding video-context ablation

Companion to `docs/2026-09-03_bpp_video_context_ablation.md` (section 4) and
`docs/2026-09-03_bpp_video_context_ablation_decisions.md`. Source: the SQL episode
table on 2026-09-03 (`egomimic/scripts/tutorials/sql_tutorial.ipynb` access
path; survey script `logs/claude_scratch/folding_candidates.py`, full dump
`logs/claude_scratch/folding_all.csv`). Nothing in the configs is changed by
this doc; it is a recommendation with paste-ready hash lists.

## Recommendation

Add 77 episodes from the same March 2026 stockroom campaign the current
split comes from: the 14 unused episodes of two training operators, plus
seven new operators with 6 to 15 usable episodes each. Every added episode is
a single complete demo of at most 2699 frames, so `max_sequence_length`
stays 2700 and the model config is untouched. Leave both validation sets
exactly as they are.

| | Now | After |
|---|---|---|
| Training episodes | 36 | 113 |
| Training frames | 83 918 (0.78 h) | 266 020 (2.46 h) |
| Training operators | 3 | 10 |
| Longest training episode | 2699 | 2699 |
| Passes over the data in 30 k steps at batch 32 | 11.4 | 3.6 |

Ready-to-paste training filter (current 36 hashes plus the 77 below, hash
sorted): `logs/claude_scratch/folding_train_lambda.txt`. Per-operator
records with frames, dates and descriptions:
`logs/claude_scratch/folding_recommended_adds.json`.

## What the table holds

462 `lab == mecka`, `task == folding_clothes`, `human_bimanual` episodes,
none deleted, all with a processed zarr path. Decoding the episode-hash
timestamps splits them into three recording campaigns, and only one of them
matches the current split:

| Campaign | Episodes | Where | What the episodes look like | Fit |
|---|---|---|---|---|
| March 2026, Mecka store (stockroom, janitorial section, sales floor, ...) | 192 (43 operators) | same site as every current train and val episode | one complete demo per episode, 1795 to 2699 frames | **use** |
| January 2026, mostly the same store | 77 | stockroom, office, a few homes | continuous sessions cut into 100 s clips (29 episodes are exactly 2998 to 3001 frames; descriptions read "continues to fold ...", "makes the final folds ..."); up to 3688 frames | skip: a clip is not a whole demonstration, so it is a poor whole-episode prompt, and it needs `max_sequence_length` 3700 (about 1.4x arm C's step time) |
| March 2026, homes (bedroom, living room, dining room, laundry room) | 193 (118 operators) | different scenes, different garments (polos, pants, shorts), no bagging | one demo per episode, 1 to 7 episodes per operator | skip for this ablation: none of the 12 val episodes is a home scene, so it adds a scene shift the val set cannot measure, and the many 1 to 3 episode groups would be heavily over-sampled by `balance_by: group` |

## The current split is really six sub-task variants

The `folding_clothes` label covers different jobs, and each operator does
one of them. This matters for what new data can and cannot help with.

| Operator | Role | What they do in the pinned episodes |
|---|---|---|
| `68e0b875` | train | iron a t-shirt, fold it, sometimes bag it |
| `693cbbbb` | train | lint-roll a shirt or pants, fold, sometimes bag |
| `695d09fc` | train | fold a checkered cleaning cloth, put it in a plastic bag |
| `6905a4e7` | unseen val | iron a t-shirt, fold, bag (same job as `68e0b875`) |
| `695d0ba2` | unseen val | checkered cloth into bag (same job as `695d09fc`) |
| `69439ad5` | unseen val | fold assorted small garments into bags; fold t-shirts onto a display shelf (closest training match is `693cbbbb`, loosely) |

So the "operator style" the prompt is meant to convey is, in this data,
mostly "which job and how this person does it". The unseen metric for
`6905a4e7` and `695d0ba2` asks whether the prompt transfers a job the model
has seen from one person to a new person; `69439ad5` asks about a job the
model has barely seen.

## Recommended additions

All March campaign, one demo per episode, at most 2699 frames, `stockroom`
or `janitorial_supplies_section` unless noted.

| Operator | Job | Usable | Frames | Note |
|---|---|---|---|---|
| `68e0b875` (train op) | iron + fold t-shirt | +7 | 15 785 | the only source of more ironing demos in the table |
| `695d09fc` (train op) | checkered cloth into bag | +7 | 16 312 | |
| `695d0a77` | checkered cloth, bag or stack | 15 | 36 436 | 16th episode is 989 frames, excluded (see below) |
| `695d08d5` | checkered cloth into bag | 11 | 26 513 | 12th is a January clip |
| `695d0708` | checkered cloth into bag | 9 | 22 814 | two January clips excluded |
| `695d098f` | checkered cloth, bag or stack | 8 | 19 666 | |
| `695d07d7` | checkered cloth into bag | 7 | 15 346 | one January clip excluded |
| `695d0c1e` | checkered cloth into bag | 7 | 16 235 | |
| `69439bf1` | assorted garments on a mat, tape dispenser | 6 | 12 995 | closest available match to unseen `69439ad5`; includes one 2296-frame January episode that is a single demo, not a clip |

Hashes, per operator (frames-sorted):

`68e0b875c33a1abcb8fc55b9` (iron + fold t-shirt (train op), 7 episodes, 15785 frames):
`69b1caa02c214819fe493b44 69b3291c1d46dfe49c49edd9 69b3444a6db34bdbeac30538`
`69b49d838cd7957ebe478c88 69b4b52e6d78ad6736570e97 69b4b3125058b9f00b9fa573`
`69b4bd2c47952ada7592905d`

`695d09fc83a9fdf2d84d9c11` (checkered cloth into bag (train op), 7 episodes, 16312 frames):
`69b340965c46953805d1c898 69b37f77254f96a47b997d08 69b49552c7e02bc8d412ac5e`
`69b49d2e8781fdde8bbaa9ca 69b4b5988b0d76bdb828823d 69b49d3307347b484b3fd740`
`69b53a09124502aa36325304`

`695d0a7783a9fdf2d84d9db5` (checkered cloth: bag or stack, 15 episodes, 36436 frames):
`69b3285072bd45fc976ff4db 69b3c246fef79614fbc1a346 69b3654da1ede4de0049cc69`
`69b4a38041a4401e4c3f526c 69b49520ba5771467aa5c4a7 69b49601a83a7b34affb4573`
`69b4a0a9a03afaf54dbcba2b 69b4b33c5dac6770e6021e42 69b4b9abe8279eda42670225`
`69b4d438d158ec4a43dcf8a0 69b4ee890a4ef7c33756bca5 69b4f76b2e9983f31a0b5a70`
`69b4f55a9705148536965205 69b4fac9caa6ed5689ec2d5e 69b4f89752b439fcc613b58a`

`695d08d583a9fdf2d84d97e8` (checkered cloth into bag, 11 episodes, 26513 frames):
`69b349025e6c644fc8c229ce 69b36adc1d815aa785771b6e 69b493541ee47c072a367c5d`
`69b4948bd0962527244f2b40 69b49c50396463451d97c5e2 69b49d53a4b7915a9a847ca9`
`69b49e7777f4dc69622ce92b 69b4d43e97224d24a5b38505 69b4d468acea937039fea6b1`
`69b502d01cdb3cb1bcb16765 69b573ecf85fc301e7132041`

`695d070883a9fdf2d84d9020` (checkered cloth into bag, 9 episodes, 22814 frames):
`69b493d31e1240fd6bcfe78f 69b4a2d08505130f8b0e6d34 69b4b2e1ecc2eafbfae25d1d`
`69b541163dd17eb53d7ddc96 69b53f7d6c13d04069dccf86 69b4f37c388ccecd105c5fe1`
`69b502ca758deefb5ec0126c 69b51604fc41f34c724bea67 69b56320d9bd4ed2c8cdab06`

`695d098f83a9fdf2d84d9a47` (checkered cloth: bag or stack, 8 episodes, 19666 frames):
`69b3249ebcfac7d916abb9c4 69b395313fb2fc1e7790afb7 69b4ae3453e0e51b6dd346d7`
`69b4b33c73ca03704a5029ad 69b4b143c7834bedf19721f2 69b54c24d3ef4c69cdc631b9`
`69b5764588f4b4a6e37d9f54 69b5702e7e64d72afd3bf666`

`695d07d783a9fdf2d84d9371` (checkered cloth into bag, 7 episodes, 15346 frames):
`69b262b3f134f5466995afd4 69b3258f82176d1f1ad692c8 69b268cc83c27231f770b967`
`69b4a9655aee26acefead1ce 69b4b2269b2ae0fb15ca36f3 69b4b2827ff8637604061add`
`69b4f533dd4d1616e5387348`

`695d0c1e83a9fdf2d84da3d8` (checkered cloth into bag, 7 episodes, 16235 frames):
`69b27259ff208cc0ebe32225 69b3246c3bf215e8fa28700d 69b33ee0ed19bd8136cc7f4b`
`69b47d7217b1b3bbefd3fb89 69b4a04a64b814291648f037 69b532685c9a724fbfbc1f86`
`69b53f6f3a7542779b575391`

`69439bf1a2a8b5aee76603c3` (assorted garments, mat, tape, 6 episodes, 12995 frames):
`69b1cc733ef683abf3670b90 69b1e2191c6c05bcabe09dcd 69b36788846a8cd39d79fa5f`
`696d2fb388af7cc080636a98 69b4a22f17ec01169beb3ecd 69b5425561c85d2534e8f9f1`

## Excluded, and why

- `69b25d429ca8561be49ab97f` (`695d0a77`, 989 frames, 33 s). Every other
  March episode in the campaign is at least 1795 frames. Like the 925-frame
  unseen val episode the plan already flags, this is most likely a truncated
  or aborted recording (the next shortest store episode in the campaign is
  1615 frames). Check the video before including it.
- The 7 January episodes of the recommended operators (2296 to 3602 frames,
  hashes `696d3039 696dcd73 696dcd6c 696f53ee 6970db42 69730ed7` plus the
  kept `696d2fb3`): 100 s clips over 2700 frames, except `696d2fb3` which is
  a single 2296-frame demo and is kept.
- Whole January operators `69624c95` (13, t-shirts on an office desk),
  `6964d381` (12, hoodies), `69624b7e` (4, folding board), `69624d55` (4):
  clips, 2998 to 3688 frames, different jobs from every val episode.
- Small March stockroom groups: `6954f573` (4, white cloth strips into
  bags), `6944acaf` (3 usable, lint roller like `693cbbbb`), `6943a1d1`
  (2 usable, plain t-shirt folding). Usable but tiny: with
  `balance_by: group` a 2-episode operator gets the same total weight as a
  15-episode one, so each of its episodes is sampled about 7x as often, and
  its prompt pool is a single other episode. Add `6944acaf` if you want a
  second lint-roller operator and accept the over-sampling; leave the other
  two out.
- The three unseen operators' remaining 29 episodes stay out, as the plan
  requires.

## Consequences of adopting this

1. **Data config.** Replace the train filter lambda in
   `data/bpp_folding_clothes.yaml` with `folding_train_lambda.txt` (113
   hashes) and update the header comment. Valid filter, `heldout_groups`,
   prompt blocks and `max_sequence_length` unchanged.
2. **Norm stats.** The minmax cache at
   `logs/bpp_ctx_smoke/smoke_A_2026-09-03_09-13-20/0/norm_stats/norm_stats.json`
   is for the 36-episode set. Recompute once on the new set (a short arm A
   run, or the norm-stat pass alone) and point every arm at the new file.
3. **Sync.** None of the 77 episodes is in
   `/coc/flash7/rco3/datasets/egoverse` yet. At about 89 MB each that is
   roughly 7 GB; the `S3EpisodeResolver` pulls them on the first dataset
   build, so budget that into the first launch or the smoke.
4. **Sampler balance shifts toward the checkered-cloth job.** With
   `balance_by: group`, 7 of 10 training operators (70 % of samples) fold
   checkered cloths; ironing and lint-rolling get 10 % each, down from 33 %.
   Over a 30 k-step run the model sees about 96 k ironing samples instead of
   320 k. That favors unseen `695d0ba2` (checkered) over `6905a4e7`
   (ironing). Two ways to keep the mix closer to today's: (a) take only three
   checkered operators (`695d0a77`, `695d08d5`, `695d0708`) and the 14
   train-op episodes, giving 6 operators and 82 episodes; or (b) keep all 10
   and accept that per-operator metrics (already logged as
   `paired_mse_group_<idx>`) will show the difference. Recommendation: (b);
   the primary question is whether the prompt helps at all on new people,
   and more operators of the same job is what lets the model learn that the
   prompt, not the operator identity, carries the style.
5. **Steps.** 30 k steps is 3.6 passes instead of 11.4. The plan's
   30 k was chosen for 36 episodes; with 113 the loss will still be falling
   at 30 k. Given the budget (about 31 h per prompted run on an A40), keep
   30 k for the ablation and note it, or raise to 45 k for arms A and C only.
6. **Seen-operator validation stays at the same 6 episodes**, so Q1 covers 3
   of the 10 training operators. Holding out 2 episodes from each new
   operator would cost 14 training episodes and change the seen metric's
   composition relative to the smoke; not recommended for this run.
7. **The group index table changes.** Valid groups are unchanged (indices
   0 to 5 as in the decisions doc), but the training log's `group i = ...`
   lines will list 10 groups.

## If more data is wanted later

- Raising `max_sequence_length` to 3700 unlocks the January clips (77
  episodes, 68 of them in the store) at roughly 1.4x arm C's step time and
  memory; check that the 25 GiB peak still fits the A40's 44 GiB before
  committing.
- The home campaign is a separate question (scene and garment shift) and
  would need its own val episodes to be measurable.
