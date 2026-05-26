---
name: cleanup-checkpoints
description: Prune .ckpt files under logs/ by age. Older than 30 days → delete all; 14-30 days old → keep only the middle-epoch and highest-epoch ckpts. Always shows the rules and the exact rm commands and waits for user permission before deleting. Use when the user asks to clean up, prune, downsample, or free disk space from checkpoints in logs/.
---

# Checkpoint Cleanup

Prune `.ckpt` files in `logs/` to reclaim disk. Two age-based rules. **Never delete without explicit user confirmation.**

## Rules

- **Age > 30 days** (by file mtime): delete every `.ckpt` in that checkpoints directory (including `last.ckpt`).
- **Age 14–30 days**: keep only **two** epoch checkpoints — the middle one and the highest-epoch one (by the integer parsed from `epoch_epoch=N.ckpt`). Delete every other `.ckpt` in that directory, including `last.ckpt`.
- **Age < 14 days**: skip entirely. Do not touch.

Bucket assignment uses the **newest mtime** of any `.ckpt` in the directory — this represents the run's recency, so a finished training run gets evaluated as one unit.

`last.ckpt` is treated as a regular ckpt for these rules (no special preservation). The highest-epoch `epoch_epoch=N.ckpt` is functionally equivalent for resume/serve, so keeping both is redundant.

## Procedure

1. **Scan**. Run the bundled Python scanner to walk `logs/`, group `.ckpt` files by their containing `checkpoints/` directory, and classify each directory into a bucket (`delete_all` / `downsample` / `skip`). Use mtime, not filename, for age.

   ```bash
   python /coc/flash7/zhenyang/EgoVerse/.claude/skills/cleanup-checkpoints/scan.py /coc/flash7/zhenyang/EgoVerse/logs
   ```

   The scanner prints, for each directory in `delete_all` or `downsample`:
   - the directory path
   - the bucket
   - the age in days
   - the list of files to delete (and, for `downsample`, the files to keep)
   - the per-file size and total bytes to be freed

2. **Present rules and plan to the user.** Show:
   - The two rules above in plain prose.
   - A summary line: `N directories, M files, X.X GB will be freed.`
   - The full list of `rm` commands that would run (one per file).

   If there is nothing to delete, say so and stop — do not prompt.

3. **Ask for confirmation** using `AskUserQuestion` with options:
   - "Delete everything listed" (Recommended)
   - "Show me the file list again"
   - "Cancel"

   Do not proceed without an explicit "Delete" response. "Cancel" → stop without action.

4. **Execute**. On confirmation, run the `rm` commands. Prefer batching into one `Bash` call (e.g. `rm -v <file1> <file2> ...`) for speed, but split into multiple calls if the argument list would be too long (> ~1000 files). Report the final freed-bytes total.

## Notes

- Do **not** use `rm -rf` or delete directories. Only individual `.ckpt` files.
- Do not follow symlinks out of the logs tree (`find -L` not used; use plain `find` / `os.walk`).
- If a directory contains no `epoch_epoch=N.ckpt` files (e.g. only `last.ckpt`) and falls into the `downsample` bucket, treat all of its `.ckpt` files as "non-epoch" and keep the single newest by mtime as the "last". If there is exactly one ckpt total in `downsample`, keep it (nothing to delete).
- If the user passes a different root path, use it instead of `logs/`.
