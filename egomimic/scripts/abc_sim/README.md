# ABC fine-tune + abc_sim rollout pipeline

Rank pretrained EgoVerse policies by fine-tuning each on one ABC task and
rolling the fine-tune out in ABC's simulator (amazon-far/abc, `abc_sim`):
closed-loop success where offline validation metrics all look alike. The ABC
paper reports Pearson r = 0.85 (success) / 0.91 (progress) between its sim and
real success over 12 checkpoints on bottles, dishrack and mug flip.

Action space: the ABC station's 14-D absolute joints + gripper (Eva mode
`joints`, `egomimic/rldb/embodiment/eva.py`), which is also abc_sim's action,
so a rollout needs no IK. A task is named by its sim_224 `task_name` (e.g.
`sim_put_the_plastic_bottles_in_the_bin`), which is also an abc_sim alias.

Steps (Lambda; paths in the launchers):

1. Data: `prepare.py --sim-data <task>` or `curl` the sim_224 tar, then
   `convert_sim_to_zarr.py --src <extracted> --task <dataset task> --out
   $ABC_SIM_ZARR_ROOT/put_bottles` -> eva_bimanual zarrs with the real ABC keys.
2. Base: `train_zarr_fold_ladder_rdt1b` (RDT-1B on the mecka freeform-fold
   operator ladder) via `lambda_pack.sbatch`; every 100 epochs a checkpoint.
3. Fine-tune: `train_zarr_abc_sim_rdt_ft init_weights_ckpt=<base epoch ckpt>`
   (same launcher); `init_weights_ckpt=null` is the from-scratch floor;
   `train_zarr_abc_real_rdt_ft` the real-data twin; `train_zarr_abc_sim_rdt_cotrain`
   the single-task human+robot cotrain.
4. Rollout: `rollout_lambda.sbatch` = `policy_server.py` (training venv) +
   `sim_client.py` (ABC venv, ABC's own `rollout_worlds` + `build_summary`,
   no RTC) -> ABC's `summary.json`. The server resamples the model's 100-step
   chunk back onto the 45 raw 30 Hz frames it spans, so the client's 15
   executed steps are 0.5 s of the plan, as in ABC; the client renders the
   history frame `lag` steps before each re-plan. `POLICY=hold` is the
   no-motion floor.
5. `val_vs_sim.py manifest.json --metric <val key>` -> Spearman / Pearson with
   bootstrap CIs and the log-linear fit of the ladder.
