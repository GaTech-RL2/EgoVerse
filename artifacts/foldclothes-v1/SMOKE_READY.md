# foldclothes-v1 smoke package

## Dataset

- Canonical task: `fold_clothes`
- Source task labels: `fold_black_t-shirt`, `fold_blue_jeans`, `fold_clothes`, `fold_laundry`, `fold_shirt`, `fold_white_shirt`
- Full candidate pool: 1,631 episodes
- Split unit: episode
- Random seed: 42
- Full-manifest SHA-256: `37f57c1775547debc522f6b96f71f835a11f29d731de7ed1dc7a1aa171683513`

## Smoke subset

- Training: 24 episodes, four per source task
- Validation: 6 episodes, one per source task
- Total: 30 MP4 files
- Video format: 640×360, 30 FPS
- Inventory: `smoke_training_inventory.csv`
- Local assets: `smoke_mp4s/`
- Decode report: `smoke_decode_report.csv`

## Audit

A stratified visual audit reviewed four contact sheets for each source task. All six task labels were retained.
