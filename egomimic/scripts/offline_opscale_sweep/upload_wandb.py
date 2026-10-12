"""Publish each completed offline sweep to a new, separate W&B run."""

import json
import math
import os
import sys
from pathlib import Path

import wandb

root = (
    Path(__file__).resolve().parents[3]
    / "logs/fixed_budget_opscale_hist/offline_sweep_20261010"
)
op = int(sys.argv[1])
run_id = f"fixed_budget_opscale_hist_op{op}_offline800_20261011"
os.environ["WANDB_DIR"] = str(root)
os.environ["WANDB_MODE"] = "online"
manifest = {
    int(p[0]): p[1]
    for line in (root / f"op{op}-checkpoints.tsv").read_text().splitlines()
    if (p := line.split("\t"))
}
rows = [
    json.loads(line)
    for line in (root / f"op{op}-metrics.jsonl").read_text().splitlines()
    if line.strip()
]
if len(rows) != len(manifest) or {row["epoch"] for row in rows} != set(manifest):
    raise ValueError(f"Sweep op{op} is not complete; refusing partial upload")
reports = []
for row in sorted(rows, key=lambda row: row["epoch"]):
    epoch = row["epoch"]
    if row["checkpoint"] != manifest[epoch]:
        raise ValueError("Checkpoint differs from manifest")
    values = row["metrics"]
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"Non-finite metrics at {epoch}")
    videos = list(
        (root / f"op{op}/epoch_{epoch}/videos_open_loop").glob("epoch_*/*/*.mp4")
    )
    if len(videos) != len({v.parent.name for v in videos}) or not {
        "seen_op_valid",
        "unseen_op_valid",
        "flagship_unseen_op_valid",
    }.issubset({v.parent.name for v in videos}):
        raise ValueError(f"Incomplete videos at {epoch}")
    for video in videos:
        values[f"{video.parent.name}/Valid/open_loop_sim/Paired_MSE"]
    reports.append((row, videos))
marker = root / f"op{op}-wandb-uploaded.json"
if marker.exists():
    print(marker.read_text(), flush=True)
    sys.exit(0)
run = wandb.init(
    project="EgoVerse",
    entity="rl2-group",
    id=run_id,
    name=run_id,
    resume="allow",
    config={
        "operators": op,
        "limit_val_batches": 800,
        "execute_steps": 30,
        "unseen_episode_count": 6,
        "flagship_episode": "692eb28d21fc595fbe6f9600",
        "checkpoint_epochs": sorted(e + 1 for e in manifest),
    },
    settings=wandb.Settings(init_timeout=120),
)
url = run.url
print(f"RUN_URL={url}", flush=True)
(root / f"op{op}-wandb-url.txt").write_text(url + "\n")
run.define_metric("epoch")
run.define_metric("*", step_metric="epoch")
try:
    for row, videos in reports:
        epoch = row["epoch"]
        values = row["metrics"]
        payload = {"epoch": epoch + 1, **values}
        for video in videos:
            group = video.parent.name
            score = values[f"{group}/Valid/open_loop_sim/Paired_MSE"]
            payload[f"videos_open_loop/{group}"] = wandb.Video(
                str(video),
                fps=30,
                format="mp4",
                caption=f"Epoch {epoch+1}, horizon 30, open-loop MSE {score:.6g}",
            )
        run.log(payload)
        print(
            f"LOGGED epoch={epoch+1} metrics={len(values)} videos={len(videos)}",
            flush=True,
        )
    run.summary["completed_checkpoints"] = len(reports)
    run.summary["expected_checkpoints"] = len(manifest)
finally:
    run.finish()
marker.write_text(
    json.dumps({"run_id": run_id, "url": url, "epochs": sorted(manifest)}, indent=2)
)
print(f"UPLOAD_COMPLETE op={op} url={url}", flush=True)
