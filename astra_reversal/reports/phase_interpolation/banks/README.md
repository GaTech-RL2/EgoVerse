These records document extraction of text-latent banks from nine standard training donors: 20 demonstrations per donor, 180 demonstrations and 18,947 frames in total. They contain no rollout performance results or evidence that interpolation improves task success.

The frozen PI05 checkpoint processed each saved demonstration frame once. Instruction states were captured after all 18 native text blocks; the first 17 block outputs are eligible for TLI. Each bank averages all of its donor's frames equally using float64 accumulation followed by one float32 cast, with noninstruction slots zeroed. Model weights were unchanged.

All eight L40S workers completed with exit code 0. Each recorded **407 unit tests and 19 native tests passing, with no skips**. TF32 was disabled. Workflow elapsed time was 1,778.50 seconds; the sum of concurrent per-donor extraction times was 4,680.73 seconds. Every donor recorded zero resumed frames.

- [Extraction record](extraction.json): exact donor episodes and frame counts, bank and array hashes, extraction times, worker runtimes, test-log hashes and verification scope.
- [Workflow status](workflow_status.json), [source and payload identity](source_identity.json), and [checkpoint metadata](checkpoint.public.json).
- [Bank inventory](bank_inventory.public.json) and [original validation projection](bank_validation.public.json).
- [Publication manifest](publication_manifest.json): public file hashes, original input file hashes and omitted-field inventory.

The public inventory, validation and checkpoint files omit URLs and machine paths. They are review projections, not the exact frozen runtime inputs. Original file hashes and bank IDs still identify the unmodified originals; a bank ID cannot be recomputed from metadata after fields have been omitted. No credentials, model weights, bank arrays or raw demonstration images are included.

The publication check reloaded all nine saved banks using the frozen [bank validator](../../../interpolation_bank.py), verified every file hash and frame-ledger entry, and reconciled worker tests, completion status and inventory metadata. It did not rerun hidden-state extraction or the simulator.
