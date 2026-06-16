# RoboTwin PI-RICL — cluster runbook

RoboTwin docs: https://robotwin-platform.github.io/doc/usage/robotwin-install.html

Status: code integrated on a Mac, now running on the Georgia Tech SLURM cluster
(`/coc/flash7/rco3/EgoVerse`). Data prep + training pipeline are validated here; the
closed-loop SAPIEN eval needs the sim stack installed.

Setup once per cluster: `git submodule update --init external/RoboTwin`; the PaliGemma
tokenizer is vendored at `pg_tokenizer/` (gitignored) via
`AutoTokenizer.from_pretrained("google/paligemma-3b-mix-224").save_pretrained(...)`
(already in the HF cache here). pi0.5 base ckpt: `egomimic/algo/pi_checkpoints/pi05_base_pytorch`.

## TODO 1 — data → zarr → train  (DONE through training)

```bash
source emimic/bin/activate
# download (~230 MB): beat_block_hammer / aloha-agilex / clean_50
python -m egomimic.ricl.scripts.download_robotwin --mode hf --out egomimic/ricl/outputs/robotwin_raw
# (optional) validate the zarr converter on a few episodes
python -m egomimic.ricl.scripts.robotwin_to_zarr \
  --root egomimic/ricl/outputs/robotwin_raw/extracted --out egomimic/ricl/outputs/robotwin_zarr --limit 5
# CPU data-path smoke (no GPU)
python egomimic/ricl/scripts/train_robotwin_ricl.py --stage cpu \
  --root egomimic/ricl/outputs/robotwin_raw/extracted --embed fake
# GPU training (unattended); writes checkpoints + quantiles.json under outputs/robotwin_train/
sbatch egomimic/ricl/scripts/train_robotwin_ricl.sbatch
```
The trainer reads the HDF5 corpus directly via `--root` (the recursive `**/data` glob
finds the nested `aloha-agilex_clean_50/data`); the zarr path is only for the
`robotwin_local.yaml` cotrain route. Single task → diagnostic episode-split for val.

## TODO 2 — closed-loop eval on RoboTwin  (infra ready; sim install pending)

1. Build the demo-bank index (GPU, real DINOv2 — matches the eval embedder):
   ```bash
   python egomimic/ricl/scripts/build_robotwin_bank_index.py \
     --root egomimic/ricl/outputs/robotwin_raw/extracted \
     --out  egomimic/ricl/outputs/robotwin_bank_index --embed dinov2
   ```
2. Install the sim stack **selectively** into `emimic` — do NOT run RoboTwin's
   `script/_install.sh` (it `pip install -r requirements.txt` pins `torch==2.4.1` and
   would break emimic's torch 2.7.1). Install: `sapien==3.0.0b1`, `mplib==0.2.1`,
   pytorch3d (git, `--no-build-isolation`), curobo v0.7.8 (`-e`, `--no-build-isolation`)
   + `warp-lang==1.12.0`; apply the two `sed` patches from `_install.sh` (sapien
   `urdf_loader.py` encoding, mplib `planner.py` collision); then
   `bash external/RoboTwin/script/_download_assets.sh`. Verify `import sapien` AND
   `import egomimic` both work afterward.
3. Fill `external/RoboTwin/policy/pi_ricl_egoverse/deploy_policy.yml`
   (`egoverse_checkpoint`, `bank_root`, `bank_index_dir`, `quantiles_path`) and run:
   ```bash
   cd external/RoboTwin/policy/pi_ricl_egoverse
   bash eval.sh beat_block_hammer demo_clean <abs trained .ckpt> eva_bimanual 0 0
   ```
   First deploy: verify `PIRiclPolicy._load_algo` checkpoint-load + `forward_eval`
   prompt-tokenization (tokenizer now wired to `pg_tokenizer`).
