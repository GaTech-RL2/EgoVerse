# Episode validation

From the repository root:

```bash
uv sync --locked
uv run python -m egomimic.rldb.zarr.validate /path/to/episode.zarr
```

Accepts one or more episode paths. Add `--json` for a JSON array of reports or
`--verbose` to include passing checks in terminal output.

The default accepts usable legacy episodes with explicit limitations for missing
adoption metadata, timestamps and camera coverage. No morphology, calibration,
joint-array or model backfill is required for existing Human/EVA data. The legacy
calibration reader and established loader fallbacks remain available.

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

Exit status is 1 for validation errors, 3 for a valid but ineligible episode when
`--data-status` is required, and 0 otherwise. Passing default validation does not
certify a complete delivery. Use the [dexterous-hand delivery requirements](CONTRIBUTING_DEXTEROUS_HANDS.md#verify-before-delivery)
for robot-hand submissions and review the [preview](DELIVERY_PREVIEW.md).
