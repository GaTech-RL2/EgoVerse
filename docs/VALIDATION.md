# Episode validation

Run `UV_PROJECT_ENVIRONMENT=ev uv run python -m egomimic.rldb.zarr.validate episode.zarr`.
Add `--json` for a machine-readable report and `--verbose` to include passing checks.

The default accepts usable legacy episodes with explicit limitations for missing
adoption metadata, timestamps and camera coverage. No morphology, calibration,
joint-array or model backfill is required for existing Human/EVA data. The legacy
calibration reader and established loader fallbacks remain available. ABC's
`yam_bimanual` name resolves to `yam_x6`; this declares the stored arm/jaw widths
and does not supply a calibrated YAM training transform.

Each coverage/adoption requirement can be enabled for a particular consumer,
for example `--calibration-present --rgb-timestamps`. Use `--help` for all named
flags; the corresponding `--no-...` flags remain supported. Missing wrist
calibration does not invalidate an ego-only overlay or unrelated training target.

Wrong declared types, shapes, inconsistent morphology, non-increasing timestamps,
invalid annotation spans and inconsistent FK remain errors. A missing optional
model limits FK verification; a supplied model hash mismatch is an error.
Constant poses, identity transforms, centred-pinhole and timestamp-quantization
signatures are diagnostics, always warnings. They are not proof of placeholders.
Annotations have no minimum coverage, and text must not encode metadata using
` | ` delimiters. Structural samples remain excluded from staging and training
resolvers regardless of the validation result.
