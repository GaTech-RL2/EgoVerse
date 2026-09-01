"""
Pull a slice of the ABC-130k dataset (XDOF/ABC-130k) off the Hugging Face Hub.

The repo is gated: accept the terms on the dataset page, then use a token with
"Read access to contents of all public gated repos you can access" enabled.

Layout on the Hub::

    data/{train,val}/<task_name>/episode_<uuid>/episode.mcap
                                              /annotation.mcap   (annotated eps only)

Examples::

    # list the tasks without downloading anything
    python -m egomimic.scripts.abc_process.download_abc --list-tasks

    # pull 10 episodes of one task
    python -m egomimic.scripts.abc_process.download_abc \
        --task fold_and_stack_the_towels --num-episodes 10 --out /path/to/abc_raw
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "XDOF/ABC-130k"


def _tree(api: HfApi, path: str) -> list:
    """One non-recursive tree listing. A recursive listing of this repo 429s."""
    return list(
        api.list_repo_tree(
            REPO_ID, repo_type="dataset", path_in_repo=path, recursive=False
        )
    )


def list_tasks(api: HfApi, split: str) -> list[str]:
    return sorted(e.path.split("/")[-1] for e in _tree(api, f"data/{split}"))


def list_episodes(api: HfApi, split: str, task: str) -> list[str]:
    return sorted(e.path.split("/")[-1] for e in _tree(api, f"data/{split}/{task}"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--task", default=None, help="task directory name")
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--out", default="./abc_raw")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument(
        "--annotations-only",
        action="store_true",
        help="fetch only annotation.mcap (a couple of KB per episode)",
    )
    args = p.parse_args()

    api = HfApi()

    if args.list_tasks:
        for t in list_tasks(api, args.split):
            print(t)
        return

    if not args.task:
        p.error("--task is required unless --list-tasks is given")

    episodes = list_episodes(api, args.split, args.task)
    if not episodes:
        raise SystemExit(f"no episodes under data/{args.split}/{args.task}/")
    picked = episodes[: args.num_episodes]
    print(f"{len(episodes)} episodes available; downloading {len(picked)}")

    out_root = Path(args.out)
    for i, ep in enumerate(picked, 1):
        ep_dir = f"data/{args.split}/{args.task}/{ep}"
        # snapshot_download cannot be used here: it expands allow_patterns against
        # the repo's `siblings` list, which the Hub returns EMPTY for a repo this
        # large, so every pattern matches nothing. Resolve files per episode.
        for entry in _tree(api, ep_dir):
            name = entry.path.split("/")[-1]
            if args.annotations_only and name != "annotation.mcap":
                continue
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=entry.path,
                local_dir=out_root,
            )
        print(f"  [{i}/{len(picked)}] {ep}")

    print(f"downloaded to {out_root / 'data' / args.split / args.task}")


if __name__ == "__main__":
    main()
