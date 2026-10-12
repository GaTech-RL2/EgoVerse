"""Read-only source count and visual contact sheet for the offline sweep."""

import json
from pathlib import Path

import av
from PIL import Image, ImageDraw
from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "logs" / "fixed_budget_opscale_hist" / "offline_sweep_20261010"
PREVIEWS = ROOT / "folding_task_examples" / "sql_mecka_flagship_candidates"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    engine = create_default_engine()
    query = text("""
        SELECT episode_hash, operator, lab, task, task_description,
               num_frames, zarr_processed_path
        FROM app.episodes
        WHERE operator = :operator AND lab = 'mecka' AND NOT is_deleted
          AND embodiment = 'human_bimanual'
        ORDER BY task, episode_hash
    """)
    with engine.connect() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                query, {"operator": "69439bf1a2a8b5aee76603c3"}
            ).mappings()
        ]
    (OUT / "unseen_operator_source_rows.json").write_text(
        json.dumps(rows, indent=2, default=str)
    )
    print("unseen operator SQL rows:", len(rows), flush=True)
    tasks = {}
    for row in rows:
        tasks[row["task"]] = tasks.get(row["task"], 0) + 1
    print("by task:", tasks, flush=True)

    paths = sorted(PREVIEWS.glob("*.mp4"))
    width, height = 320, 190
    sheet = Image.new("RGB", (width * 4, (height + 28) * len(paths)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, path in enumerate(paths):
        with av.open(str(path)) as container:
            frames = [
                frame.to_image().convert("RGB") for frame in container.decode(video=0)
            ]
        print("preview:", path.name, "frames:", len(frames), flush=True)
        for column, fraction in enumerate((0.1, 0.35, 0.6, 0.85)):
            image = frames[min(len(frames) - 1, int(fraction * len(frames)))]
            image.thumbnail((width, height))
            x, y = column * width, row_index * (height + 28)
            sheet.paste(image, (x, y))
            draw.text(
                (x + 3, y + height + 3), f"{path.stem} {fraction:.0%}", fill="black"
            )
    sheet.save(OUT / "flagship_preview_contact.jpg", quality=90)


if __name__ == "__main__":
    main()
