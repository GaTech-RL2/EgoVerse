# Pipeline training

EgoVerse training is configured as an explicit graph of registered stages.
`PipelineAlgo` only schedules those stages for `train` or `inference`, moves
multi-source batches to the selected device, aligns outputs with opaque loader
keys, and reduces scalar `loss/*` values. Dataset or task semantics belong in
the configured stages and evaluators, not in the graph runner.

## Configuration

The root Hydra config is `egomimic/hydra_configs/train_zarr_cartesian.yaml`.
Its model, data, and evaluator defaults are intentionally null: every run must
select a complete experiment instead of inheriting a historical policy by
accident.

A direct composition has this shape:

```bash
python egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian \
  model=<pipeline-model-config> \
  data=<dataset-config> \
  evaluator=<evaluator-config>
```

A composed experiment may override all three groups. Keep the dataset split,
normalization artifact, stage dimensions, optimizer, and evaluation contract
inside that experiment so its resolved Hydra output is a complete run record.

Use `mode=train` for optimization and `mode=eval` for standalone evaluation.
The deprecated boolean mode aliases are not supported.

## Graph structure

A diffusion-policy training graph normally contains distinct stages for:

1. observation processing;
2. target construction;
3. data noising;
4. denoising or velocity prediction;
5. loss computation.

Inference uses the same registered graph with the explicit `inference` mode.
Training-only target, noising, and loss stages are excluded by their contracts;
the configured denoiser performs reverse sampling. No stage is physically
swapped at runtime.

A stage whose declared reads are unavailable is a configuration error: the
runner raises rather than skipping it. Mode restrictions are therefore declared,
not inferred, with `train_only` or its mirror `inference_only`.

## Action tokenization

Arc-length tokenization can run either in the loader's `transform_list` or as
graph nodes. The node form makes the boundary visible and lintable:

- `ArcTokenizeStage` takes `ActionTargetBuilder`'s place in the stage list. It
  reads the loader's time-indexed chunk and writes the arc token as `target`,
  so it stays the single writer of `target` and the rest of the graph is
  unchanged -- it simply models a token.
- `ArcDetokenizeStage` is the inverse. It is `inference_only` and writes
  `pred_action_native`.

The two nodes must agree on `rotation_radius`, `dt` and
`resampled_vector_length`. The token's speed is a rate in the tokenizer's SE(2)
metric (translation plus `lambda * rotation`), so the polyline the decoder walks
has to be measured in that same metric; measuring translation alone replays the
chunk too fast by the ratio between the two lengths, and nothing in the tensor
shapes reveals it. Keep both nodes pointed at one `planar.*` value rather than
repeating the number.

Tokenization in the graph also means the tokenizer sees the RAW window: the
loader's dense transform only pads, so no interpolation stands between the
episode's 30 Hz frames and the tokenizer, and `arc_dt` is the true capture
period. A loader-side arc path that resamples before tokenizing would understate
arc length and make that constant wrong.

Compare `experiment/pusht/planar_v2_usocket_arc_bc` (tokenized in the loader)
with `experiment/pusht/planar_v2_usocket_arc_graph_tok` (tokenized in the
graph). Both are valid; only the second shows the tokenizer in `gt`-style graph
output and in `tools/config_graph.py`.

## Data loading

Data configs build one or more datasets and a `CombinedLoader`. Its outer keys
are opaque alignment identifiers. Each inner flat batch is passed through
unchanged, including metadata; only configured stages may interpret fields.

Normalization is owned by the data path. The Pipeline runner neither stores
normalization state nor knows about action keys, embodiments, domains, or
deployment adapters.

The neutral dense-language data-processing configs are:

- `data/cotrain_dense_language`
- `data/eva_dense_language`

They support annotation and embedding utilities and do not select a policy.

## Slurm

After a configuration composes and passes its required smoke test, add `-m` to
use the configured Submitit launcher:

```bash
python egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  model=<pipeline-model-config> data=<dataset-config> \
  evaluator=<evaluator-config> -m
```

Always inspect the fully resolved config saved with the run before treating a
launch as reproducible.
