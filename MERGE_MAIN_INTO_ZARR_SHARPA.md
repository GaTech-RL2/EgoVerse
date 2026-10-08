# Merge report: origin/main into jmni/zarr-sharpa

| | |
|---|---|
| Branch tip (ours) | `ce623273a` (`jmni/zarr-sharpa`, 31 commits since the base) |
| Merged (theirs) | `42817f5dd` (`origin/main`, 2026-09-29, 45 commits since the base) |
| Merge base | `e074ee2b4` |
| Result branch | `jmni/zarr-sharpa-merge-main` (worktree `/coc/flash7/jni66/EgoVerse-sharpa`) |
| Merge commit | `22e8337de` (parents `ce623273a`, `42817f5dd`); this report is a follow-up commit |
| Environment | `/coc/flash7/jni66/envs/egoverse-sharpa-merge` (`uv sync --locked --python 3.11`) |

**Why origin/main:** the local `main` ref is stale; it points at the merge base
`e074ee2b4`, so merging it would bring in nothing. The remote-tracking ref
`origin/main` was refreshed over HTTPS before the merge.

## Conflict resolutions (16 paths)

"Formatting only" means a token-level comparison showed that the main side
changed only whitespace, parentheses, trailing commas, or import order.

| Path | Resolution |
|---|---|
| `.github/workflows/ci.yml` (add/add) | Kept main's three jobs (lint with ruff 0.8.6 plus `format --check`, unit tests with openpi and the tokenizer, and the wheel smoke test). The unit-test step now runs `pytest tests/unit egomimic/rldb/embodiment egomimic/rldb/zarr`. Added the branch's "Free disk space" step before `setup-uv`. Kept main's `enable-cache: true`, although the branch had disabled caching (see non-blocking items). |
| `pyproject.toml` | Combined both sides. Dependencies are the branch's `dash` plus main's `ruff==0.8.6` and `scipy`, with main's openpi note. Package data is main's Hydra configs and resources plus the branch's registry YAML/URDF files and `zarr/schema/*.yaml`. `testpaths = ["tests", "egomimic/rldb/embodiment", "egomimic/rldb/zarr"]`, so the branch suites are still collected. Kept main's `norecursedirs`, its markers, and its pyright settings. |
| `uv.lock` | Regenerated from main's lock with `uv lock`, which added dash and its 7 dependencies. Then ran `uv lock -P werkzeug==3.1.8` to keep the version the branch tested. The result is byte-identical to the auto-merged lock, and `uv lock --check` passes. |
| `egomimic/rldb/embodiment/embodiment.py` | Imports from both sides: the branch's `importlib` for the registry and main's `warnings` for `_strip_pi_keymap_mode`. The rest merged automatically. |
| `egomimic/rldb/embodiment/human.py` | Branch side: the `RGB_T_CPF` rename, with `FINGER_*` moved to the `Embodiment` base. Dropped `PI_FRONT_KEY` because main removed it in #629 and nothing references it. |
| `egomimic/rldb/zarr/zarr_dataset_multi.py` (2 hunks) | Imports: main's lazy `aws_sql` wrappers, `resolve_memo`, and `utils.env.load_env`, plus the branch's `calibration`, `episode_attrs`, `base_T_cam_pose_key`, and `_matrix_to_xyzwxyz`. Dropped the branch's eager `aws_sql`/`aws_data_utils` imports to keep #631. Intrinsics use the branch's calibration-bound K; main's side was formatting only. |
| `egomimic/scripts/aria_process/aria_utils.py` | Branch side (`world_T_right_hand` names). Main's side was formatting only. |
| `egomimic/scripts/data_visualization/inspector_lib/dataset_view.py` (10 hunks) | Branch side in all hunks. Main's side was lint and formatting only: it removed the unused `CANVAS` import and moved the `ee_pose_to_cam_frame` import, and the branch uses neither. Removed a duplicate `.images`/`.language`/`.views` import block that the auto-merge created. A token-level comparison with the branch file shows 0 differences. |
| `egomimic/scripts/mps_process/clean_mps.py`, `egomimic/utils/metrics.py` | Main side (formatting only). |
| `egomimic/utils/pose_utils.py` (4 hunks), `egomimic/utils/viz_utils.py` (3 hunks), `egomimic/utils/tensor_utils.py` | Branch side: `base_T_cam`/`cam_T_cpf`/`parent_T_child` renames, new docstrings, and imports moved to the top of the file. Main's side was formatting plus mid-file `# noqa: E402` imports. No keyword callers of the old parameter names remain. |
| `CONTRIBUTING_DATA.md` | Kept the branch's minimal §11.2 resolver example. Moved main's correction from 7c7e077a1 into the prose: every vendor writes `obs_head_pose`, and Cartesian mode needs it. |
| `tests/unit/test_action_chunk_transforms.py` (2 hunks; main renamed it from `egomimic/rldb/zarr/test_action_chunk_transforms_unit.py`) | Branch side: direct `_build_*` imports and the `base_T_cam_pose`/`cam_T_ee_poses` keys that match the branch's `eva.py`. Main's `mode=` and `NumpyToTensor` updates merged automatically. |
| `egomimic/rldb/zarr/test_zarr.py` (deleted on main, modified on branch) | Main deleted the cluster-path script. The branch's hermetic rewrite (Eva and Human `ZarrDataset` pipelines on temporary episodes) moved to `tests/unit/test_zarr_dataset_pipeline.py`, and the old path is deleted. |
| `egomimic/test_zarr.py` (deleted on main, modified on branch) | Kept main's deletion. The branch had replaced the old PACE-only script with a 24-line shim (`egomimic.test_zarr.validate_episode` → `(errors, successes)` from `rldb.zarr.validate`). Removed its only consumer, `test_legacy_validation_entry_point_handles_valid_and_invalid_episodes` in `egomimic/rldb/zarr/test_validate.py`, and fixed the stale comment at `inspector_lib/language.py:37`. To restore the shim: `git show ce623273a:egomimic/test_zarr.py`. |

Fixes made in the merge commit because of the merge:

- **Semantic conflict in `ZarrDataset.init_episode`** (`egomimic/rldb/zarr/zarr_dataset_multi.py`, camera selection).
  - The branch iterated `self.key_map.values()` and indexed `spec["zarr_key"]`.
  - That raised for `key_map=None`, the resolver default that main's new `tests/unit/test_resolve_once.py` and `test_pins.py` use, and for non-spec entries such as `{"norm_mode": True}`.
  - `_load_zarr_datasets` logs and skips a dataset that raises, so every episode was dropped, and 4 of main's tests failed.
  - The fix iterates `(self.key_map or {})` and only considers mapping specs that have a string `zarr_key`.
- **Lint fixtures.** Main's pinned ruff (0.8.6) reports F811 for the branch's fixture-import pattern; the branch was linted with ruff 0.15.12. The new `egomimic/rldb/zarr/conftest.py` re-exports `overlay_episode` and `write_dexterous`, and the three test modules no longer import them. This fixed 20 errors.
- **Duplicate import.** The auto-merge left a duplicate `Rotation` import in `egomimic/utils/pose_utils.py` (F401).
- **Formatting.** `ruff format . --exclude external` (0.8.6) reformatted 28 files, all of them files the branch changed. No other files were touched, which rules out a ruff version mismatch.
- **AGENTS.md.** The "Tests" section now names the branch suites in the pre-commit command.
- **Invariant test.** `egomimic/rldb/zarr/test_data_status.py::test_the_local_resolver_refuses_a_structural_sample_inside_resolve_once` covers the `data_status` invariant through main's memoized `resolve()` path.

## Invariant checks

| Invariant | Result | Evidence |
|---|---|---|
| (a) The resolver refuses `data_status: structural_sample` after #630 | **Holds** (test passes) | `resolve_paths()` memoizes only `(path, hash)` lists. `load()` → `_load_zarr_datasets()` still applies `is_complete()` (`zarr_dataset_multi.py:330`) for the Local, S3, and SafeS3 resolvers. The new test above checks this through `resolve_once()`. |
| (b) `stage_processed_folder.py` refuses `structural_sample` | **Holds** (test passes) | `egomimic/scripts/backfill_scripts/stage_processed_folder.py:137`. Main's reformat in 1ad36a5db kept it. `merge_staging_to_episodes.py` reads only the staging table, and main's change to it was formatting only. Covered by `egomimic/rldb/zarr/test_upload_compatibility.py` (parametrized over status). |
| (c) #626 train-step tests work with registry-resolved embodiments | **Holds** for the HPT rows (eva/aria/mecka/scale passed). The Pi rows were skipped because openpi is missing. | The recipes use `eva_bimanual` and `human_bimanual`, which resolve through `canonical_embodiment_name` and the registry (`embodiment.py:98`). Main's matrix has no Dexmate row; `dexmate_bimanual` resolution is covered by `egomimic/rldb/embodiment/test_dexmate.py`. |
| (d) #629 Pi camera naming is consistent with the Dexmate keymap | **Not consistent.** See blocking item 1. | `dexmate.py:31-40` vs `preprocess_pi_obs.py:77-80` |

## Blocking issues (need a decision before Dexmate Pi training)

1. **The Dexmate camera keys do not match main's Pi camera-slot remap (#629).**
   - `Dexmate._get_keymap` (`egomimic/rldb/embodiment/dexmate.py:31-40`) emits `observations.images.{left,right}_wrist` in `keypoints` mode.
   - Main's `PI_CAMERA_SLOTS` (`egomimic/models/preprocess_pi_obs.py:77-80`) and `Eva` (`eva.py:100-101`) use `observations.images.{left,right}_wrist_img`.
   - The result: with `keypoints`, Pi never sees the Dexmate wrist cameras. `gather_pi_images` silently fills them from the front camera and masks them out.
   - With `keypoints_pi`, Dexmate emits the openpi slot names. Those still work, but only through the slot-name fallback that #629 keeps for old checkpoints. `Human._get_keymap` also raises a `FutureWarning`, because `_pi` modes are deprecated (`embodiment.py:225`).
   - Main's `tests/unit/test_data_configs_compose.py::test_no_data_config_uses_a_pi_keymap_mode` (line 107) rejects any shipped data config that uses a `_pi` mode. Your untracked `dexmate_pi.yaml` uses `keypoints_pi` (see the Pi0.5 section).
   - **Suggested fix (not applied, because it changes branch behaviour):**
     - Emit `observations.images.{side}_wrist_img` from Dexmate.
     - Route `keypoints_pi` through `_strip_pi_keymap_mode`.
     - Switch `dexmate_pi.yaml` to `keymap_mode: keypoints`.
     - No test pins the current Dexmate dataset key names.

No other blocking issue remains. The 4 test failures the merge introduced (the
semantic conflict described above) are fixed in the merge commit, and they pass
on the post-fix tree.

## Non-blocking issues

1. **Test layout policy.** Main's `AGENTS.md` says tests live under `tests/` and never next to source. The branch has 19 test modules plus `conftest.py` files under `egomimic/rldb/{embodiment,zarr}`. They are collected through `testpaths` and CI (`pyproject.toml`, the "Unit tests" step in `ci.yml`). Moving them to `tests/unit/` is follow-up cleanup. Note that `tests/unit` is not a package, so basenames must stay unique. While the suites stay where they are, `packages.find include = ["egomimic*"]` also ships these test modules and the two `conftest.py` files in the wheel. That passes `test_packaging.py`, which checks only top-level entries, but it goes against main's goal that the wheel ship "the training stack and nothing else".
2. **`egomimic.test_zarr` is gone.** The legacy `(errors, successes)` shim is deleted, as main intended. Callers should use `python -m egomimic.rldb.zarr.validate` or `validate_episode()`. Restore it from `ce623273a` if an external caller still needs it.
3. **CI cache.** Main's `unit-tests` job uses `enable-cache: true`. The branch's CI had disabled the uv cache, citing a cache of about 5 GB against the 10 GB Actions quota (`ci.yml:41`). I kept main's setting; revisit it if the cache quota is exceeded.
4. **Pi train-step tier not exercised here.** The Pi half of `tests/unit/test_train_step.py` and the `egomimic.algo.pi` import skip or fail locally because openpi and the gated PaliGemma tokenizer are not in the locked env. CI installs both and sets `EGOVERSE_REQUIRE_PI=1`. The merged Pi code path is verified only by CI.
5. **Non-finite norm samples vs. absent keypoint slots.**
   - Main's `MultiDataset._drop_nonfinite_rows` (`zarr_dataset_multi.py:1451`, 120518cd0) drops every norm sample that contains a NaN, and raises if all are dropped.
   - The branch validator checks finiteness only in registry-valid keypoint slots (`validate.py:861-884`).
   - A future keypoint-action end effector with partial `valid:` slots stored as NaN would therefore fail norm-stat computation.
   - Today every keypoint-action end effector is `valid: all`; the partial ones, `eva_parallel_jaw` and `yam_parallel_jaw`, use `action_space: cartesian`.
6. **`debug=k` counts structural samples.** `LocalEpisodeResolver(debug=k)` applies the limit to the resolved paths (`_get_local_filtered_paths`) before `load()` drops structural samples, so fewer than `k` episodes can load. Pinned structural samples do fail loudly: main's `MultiDataset._from_resolver` raises `PinError` ("failed to load / were skipped by the resolver") when a pinned episode does not load.
7. **Doc drift.**
   - `README.md:42` still says `source emimic/bin/activate`, and `pyproject.toml` `[tool.pyright] venv = "emimic"`.
   - The branch's `AGENTS.md` says to use uv and `.venv`.
8. **Zarr dtype warnings.** zarr emits `UnstableSpecificationWarning` for `VariableLengthBytes` (about 120 warnings per run). The cause predates the merge; the same warnings appear on the branch.
9. **Pi config names.** `keymap_mode: keypoints_pi` in user configs will log a `FutureWarning` for every `Human._get_keymap` call, which is main's deprecation path.
10. **Slow test suite on this cluster's NFS.** A single serial run of the merged CPU tier takes
    2 h 18 min on an `rl2-lab` CPU node. `tests/unit/test_lazy_imports.py` and
    `test_norm_cache.py` spend most of it on cold imports in subprocesses. CI's
    40-minute `timeout-minutes` applies to GitHub runners, not to this cluster.
11. **Dangling preview commit.** `4ea7b1ae` is a `commit-tree` snapshot of the
    staged merge, used for an early Pi-compatibility preview. No ref points to it, and `git gc` will remove it.

## Tests

Every command used the merge env. The CPU tier ran on Slurm CPU nodes (`rl2-lab`, `-c 12 --mem=30G`, `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`) because the login node was overloaded. Logs and JUnit XML are under `/coc/flash7/jni66/sharpa/journal/logs/`.

| Run | Command | Result |
|---|---|---|
| Lint | `ruff check egomimic tests` (0.8.6) | clean |
| Format | `ruff format --check . --exclude external` (0.8.6) | clean (265 files) |
| Lock | `uv lock --check`; `uv sync --locked --python 3.11` into a new env | pass |
| Smoke imports (31 touched modules) | `python -c "import …"` | 30/31 OK. `egomimic.algo.pi` fails with `ModuleNotFoundError: openpi`, which is expected because openpi is not lockable (see `pyproject.toml`); CI installs it separately. |
| Merged CPU tier | `pytest tests/unit egomimic/rldb/embodiment egomimic/rldb/zarr` | **Pass:** 602 passed, 5 skipped, 0 failed. Confirmed by a single full run on `22e8337de` (see below). |
| Baseline: origin/main | `pytest tests/unit` in a detached worktree at `42817f5dd` | The full run hit the 1 h 30 min job limit with no output captured. A targeted run of `test_resolve_once.py` and `test_pins.py` gave 17 passed, which confirms the 4 failures above were caused by the merge. |
| Baseline: branch tip | `pytest egomimic/rldb/embodiment egomimic/rldb/zarr` at `ce623273a` | 303 passed |
| GPU / `--integration` tier | `pytest --integration -m gpu tests/integration/test_train_step_gpu.py` | **Not run.** The prerequisites are missing on this account: there is no `~/.egoverse_env` (S3/SQL credentials for the S3 resolver and pins), openpi is not installed, and there is no Pi checkpoint (`EGOVERSE_PI05_WEIGHTS`). The first A40 request also sat pending (nodes down or drained). |

**Merged CPU tier, in detail.** One job running the whole tier exceeded the 1 h 30 min limit, so I split it across parallel Slurm jobs:

| Job | Scope | Tree | Result |
|---|---|---|---|
| `merge4` | `tests/unit`, first half (through `test_lazy_imports`) | before the fix | 265 passed |
| `merge2` | `tests/unit`, second half | before the fix | 56 passed, 4 failed, 5 skipped |
| `merge3` | `egomimic/rldb/{embodiment,zarr}` | before the fix | 277 passed |
| `merge5` | `test_resolve_once`, `test_pins`, `test_data_status`, `test_calibration`, `test_legacy_compatibility`, `test_inspector_compatibility`, `test_zarr_dataset_pipeline` | after the fix | 81 passed, 0 failed |

The 5 skips are the 4 Pi train-step rows and `test_pi_prompts`, which all need openpi. The `merge2` passes include the HPT train-step rows for eva, aria, mecka, and scale, and `test_packaging`, which shows the wheel builds with the merged package data.
**Confirmation run:** a single full run on the committed merge tree `22e8337de` (Slurm job 4032732, `logs/final_full_4032732.log`, `junit_final.xml`) gave **602 passed, 5 skipped, 0 failed** in 2 h 18 min. Its 607 test IDs match the union of the split runs exactly.

**Failure classification.**
- **Merge-induced:** 4 tests, all fixed in `22e8337de`:
  - `tests/unit/test_resolve_once.py::test_local_resolve_paths_memoized_and_load_is_fresh`
  - `tests/unit/test_pins.py::test_from_resolver_forwards_dataset_name_as_expected_embodiment`
  - `tests/unit/test_pins.py::test_from_resolver_skips_check_for_non_embodiment_names`
  - `tests/unit/test_pins.py::test_from_resolver_rejects_pin_excluded_by_filter_lambdas`

  All 4 pass on origin/main and pass after the fix.
- **Pre-existing:** none observed.
- **Branch tests that were renamed or removed relative to the branch-only run (303 tests):**
  - 15 `test_action_chunk_transforms_unit` and 8 `test_dataset_filter` tests moved to `tests/unit` on main.
  - The 2 `test_zarr` tests were ported to `tests/unit/test_zarr_dataset_pipeline.py`.
  - The 2 legacy-shim tests were removed.
  - 1 new test, `test_data_status::…inside_resolve_once`, was added.

## Compatibility with your uncommitted Pi0.5 work

Exported read-only to `/coc/flash7/jni66/sharpa/journal/user_pi_uncommitted.diff`.
I ran `git apply --3way --check` per file against the merge commit `22e8337de`. Before that, I ran a full `git apply --3way` plus tests in a throwaway worktree at a preview snapshot of the staged merge. That snapshot differs from `22e8337de` only by the `init_episode` fix, and the worktree has since been removed.

| File | Result | Notes |
|---|---|---|
| `egomimic/algo/pi.py` | applies cleanly | Your edits (checkpoint loading, `to_model`/`from_model`, freeze and checkpointing flags) don't overlap main's #629 camera-slot rewrite or its prompt warning. Your context comment about "duplicate `base_0_rgb`" is gone on main, where `gather_pi_images` now does that. |
| `egomimic/pl_utils/pl_model.py` | **conflicts** (2 hunks) | Same intent on both sides. Main's fe1ec2610 added `_barrier()`, a no-op outside a process group, and you switched to `self.trainer.strategy.barrier()`. Take one of them: Lightning's strategy barrier also covers the non-DDP case. |
| `egomimic/rldb/embodiment/dexmate.py` | applies cleanly | `include_wrist_images` filters by `zarr_key`, so it works with either dataset key name. Blocking item 1 still applies. |
| `egomimic/trainHydra.py` | applies cleanly | Adds `enable_grad_norm`. |
| `egomimic/utils/action_utils.py` | applies cleanly | |
| `pi05.md` | applies cleanly | |
| `pyproject.toml` | **conflicts** | Your `[project.optional-dependencies] pi = ["openpi"]` + `[tool.uv.sources]` + `[[tool.uv.dependency-metadata]]` conflict textually with main's package-data block. They also conflict in intent with main's note that openpi is deliberately not declared and is installed with `--no-deps` (`pyproject.toml:71-81`) and with CI's openpi step. Your dependency-metadata override is a way to make it lockable. Decide whether it replaces main's approach, and update `ci.yml` if so. |
| `uv.lock` | applies textually; **do not use** | Regenerate it with `uv lock` after you resolve `pyproject.toml`. |

Untracked files: none collides with a path on main. This includes main's
remaining Pi configs (`eva_pi.yaml`, `pi0.5_bc_{eva,aria,mecka,scale}.yaml`, and
`train_zarr_cartesian_pi.yaml`). They do collide in behaviour, though. I copied
the files into the throwaway worktree and ran `pytest tests/unit/test_data_configs_compose.py tests/unit/test_hydra_configs.py
tests/unit/test_pi_camera_slots.py egomimic/rldb/embodiment/test_pi_actions.py
egomimic/rldb/embodiment/test_dexmate.py` there, with `pl_model.py` and
`pyproject.toml` resolved to the merge side so that they import. Result:
**27 failed, 253 passed** (`logs/pi_compat_4032099.log`).

- **25 failures: `PI05_CHECKPOINT` has no default.** `model/pi0.5_bc_dexmate.yaml` sets
  `pytorch_weight_path: ${oc.env:PI05_CHECKPOINT}` with no default, so every
  compose of `train_dexmate_pi` and of `model=pi0.5_bc_dexmate` raises
  `InterpolationResolutionError`. Main's `test_hydra_configs.py` resolves every
  shipped config. Main's convention is
  `${oc.env:EGOVERSE_PI05_WEIGHTS,<default>}`, set in `pi0.5_base.yaml`. Inheriting
  that value, or adding a default, fixes these failures. After that,
  `train_dexmate_pi.yaml`'s `paths.dataset_dir: ???` may be the next error.
- **1 failure: `test_no_data_config_uses_a_pi_keymap_mode[dexmate_pi]`.**
  `dexmate_pi.yaml` uses `keymap_mode: keypoints_pi`, and main forbids `_pi`
  modes in shipped data configs. See blocking item 1.
- **1 failure: your `test_pi_actions.py::test_ego_only_keymap_and_normalization_keys`.**
  It expects the front camera key `base_0_rgb`. After #629, `Human._get_keymap`
  always emits `observations.images.front_img_1`, and the Pi wrapper renames it
  onto `base_0_rgb`. The test expectation needs updating.
- **Passed:** `test_dexmate.py`, `test_pi_camera_slots.py`, and the rest of
  `test_pi_actions.py` (converter round-trips, checkpoint resize and load).

**Verdict:** your work is compatible with the merge, but needs follow-up edits.
There are two textual conflicts (`pl_model.py`, which is trivial, and
`pyproject.toml`, which needs an openpi-packaging decision). The configs and one
test need updates to match main's #629 conventions and its every-config compose
tests. Nothing in `pi.py` conflicts. I could not exercise openpi itself: it is
not installed in the locked env.

## Follow-up commits on this branch

After the merge, these were added on `jmni/zarr-sharpa-merge-main`:

- `e4f97e646` merges `jmni/sharpa-lerobot-to-zarr`, the Sharpa LeRobot-to-Zarr
  converter (`egomimic/scripts/sharpa_process`, `docs/SHARPA_LEROBOT_CONVERSION.md`).
  - Its only conflict was `testpaths` in `pyproject.toml`; `egomimic/scripts/sharpa_process`
    is now collected locally and in CI.
  - In the merge env: ruff 0.8.6 check and format are clean, `uv lock --check` passes, and
    290 tests pass (`sharpa_process`, `rldb/embodiment`, `rldb/zarr`,
    `tests/unit/test_packaging.py`).
  - A full 9,769-frame RoCo episode converts with numeric arrays bitwise equal to the
    converter branch's output.
- `dd41d6451` adds `SHARPA_PUBLIC_DATA_AUDIT.md`.
- `d890be60b` adds `SHARPA_NEW_FORMAT_NOTES.md`.

## Fast-forwarding `jmni/zarr-sharpa`

Your checkout `/coc/flash7/jni66/EgoVerse` has uncommitted Pi0.5 work on
`jmni/zarr-sharpa`. Nothing was pushed.

```bash
cd /coc/flash7/jni66/EgoVerse
# 1. Save your work: commit it, or stash it (untracked files included).
git stash push -u -m "pi0.5 wip"        # or: git add -A && git commit -m "wip: pi0.5"
# 2. Fast-forward the branch to the merge result.
git merge --ff-only jmni/zarr-sharpa-merge-main
# 3. Restore your work. Expect conflicts in pl_model.py and pyproject.toml
#    (and possibly uv.lock). Resolve them as described above, then drop the stash.
git stash pop        # on conflict: fix the files, `git add` them, then `git stash drop`
uv lock && uv sync   # regenerate the lock after resolving pyproject.toml
```

If you committed your WIP in step 1 instead, `--ff-only` will refuse. In that case run
`git rebase jmni/zarr-sharpa-merge-main`, or merge the merge branch into your branch.
