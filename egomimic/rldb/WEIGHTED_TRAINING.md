# Dataset weighting and homogeneous training

`WeightedDataset` mixes named datasets with replacement. Weights specify
relative sampling probabilities **per dataset**, independent of dataset length;
frames inside each dataset are sampled uniformly. A 3:1 ratio draws about 75%
of samples from the first dataset. Zero disables a source. All source names must
have finite, non-negative weights, with at least one positive weight. An empty
source must have zero weight.

The data module keeps the existing per-embodiment normalization and validation
loaders. Set `dataset_weights` and `weighted_dataloader_params` to use one mixed
training loader. Its batch size is the **total per-rank batch size**. The
per-dataset `train_dataloader_params` are used only by the original loader.

```yaml
data:
  dataset_weights:
    eva_bimanual: 1.0
    human_bimanual: 3.0
  weighted_dataloader_params:
    batch_size: 64
    num_workers: 6
    pin_memory: true
  samples_per_epoch: 100000
  sampling_seed: 42
```

Dataset names follow the existing `train_datasets` embodiment keys. Validation
is not reweighted. Normalization statistics continue to be computed from each
original dataset before sampling. Without `dataset_weights`, the existing
`CombinedLoader` still supplies one batch per dataset.

`samples_per_epoch` is a global draw count, rounded up to a multiple of the world
size so ranks have equal lengths. By default it is the sum of lengths of sources
with positive weight. Sampling is reproducible from seed and epoch, independent
of DataLoader worker count. Lightning advances the sampler epoch automatically.
For a standalone loader, use `WeightedDataset.sampler(...)` and call its
`set_epoch(epoch)` yourself. Direct dataset indexing is deterministic and
returns `(source_name, sample)`; weights apply through the sampler.

For distributed weighted training, use `ddp_find_unused_parameters_true` and
`sync_batchnorm: false`: different ranks can draw different embodiments and
therefore invoke different domain-specific modules. Training totals are
synchronized across ranks; conditional per-domain training metrics are logged
on rank zero. The `experiment/weighted_cotrain.yaml` overlay includes these
settings. For example, from `egomimic/` with the usual environment activated:

```bash
python trainHydra.py --config-name=train_zarr_cartesian_pi \
  data=cotrain_pi_lang model=pi0.5_cotrain_eva_aria \
  +experiment=weighted_cotrain data.dataset_weights.human_bimanual=3.0
```

PI, HPT and PipelineAlgo enable `homogeneous_training` by default, including with
the original loader. Set `model.robomimic_model.homogeneous_training=false` to
compare with separate forwards. Compatible tensors are concatenated on their
existing batch dimension; no new embodiment axis is added.

- PI converts and normalizes each embodiment first, pads proprioception to the
  model's state width, then combines observations, camera masks, language tokens
  and actions for one policy call per compatible shape group.
- HPT combines shared stems/encoders, compatible trunk token sequences and
  built-in shared action heads. Distinct stems and heads keep their routing;
  custom loss heads keep their own loss contract.
- The pipeline sampler combines conditions after domain embeddings and runs a
  larger batch through the shared denoiser. Endpoints go back to their own
  decoders. Custom stages and MoE loss collection retain separate execution.

Different horizons, token counts, dtypes or devices form separate groups when
they cannot be concatenated. Per-domain metrics are retained. Total action loss
averages per-sample losses, so unequal sample counts preserve dataset weighting
instead of being cancelled by an equal average over embodiments. Equal-sized
batches retain the previous loss scale. Batch-dependent layers and stochastic
augmentations can differ numerically between combined and separate forwards.

HPT's optional cross-domain OT regularizer still requires paired, equal-sized
domain batches; use the original loader for that objective. This sampling
overlay is intended for the behavior-cloning objectives above.
