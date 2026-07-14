"""Migrate existing level-tagged annotation JSONs on S3 to the role-keyed scheme.

Reads every ``{prefix}/{hash}_annotations.json`` (the old single-key files
whose entries carry a ``level`` field — or no level at all for the pre-sort
pick_place era), splits them into the new role-named files WITHOUT re-running
any LLM generation, uploads those, and (with ``--delete-old``) removes the old
object:

    {hash}_annotations_task.json      <- level == "high" entries (sort goals),
                                         or ALL entries when the episode has
                                         no high entries (pick_place: the
                                         instruction IS the task conditioning)
    {hash}_annotations_subtask.json   <- level != "high" entries for sort
                                         episodes; for eva pick_place episodes
                                         an identical copy of the task list
                                         (subtask_copy regime); absent for
                                         aria/human pick_place

Entries are rewritten as plain ``{text, start_idx, end_idx}`` — the ``level``
field is dropped (role lives in the key name; see converter.py).

Eva-vs-human is resolved by looking up each episode hash's ``embodiment`` in
the SQL registry (``eva_*`` -> subtask copy). Use ``--assume-embodiment`` to
skip SQL (e.g. a bucket known to be single-embodiment).

Safety: ``--dry-run`` prints the per-episode plan without writing; start with
``--episode-hash`` / ``--limit`` on a small sample. Old files are only deleted
with ``--delete-old``, and only after BOTH new uploads for that episode
succeed.

Usage:
    python correct_annotation_keys.py --bucket s3://rldb/scale_annotations \
        [--episode-hash H | --limit N] [--dry-run] [--delete-old] \
        [--assume-embodiment eva|human]
"""

import argparse
import json

OLD_SUFFIX = "_annotations.json"
TASK_SUFFIX = "_annotations_task.json"
SUBTASK_SUFFIX = "_annotations_subtask.json"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if uri.startswith("s3://"):
        uri = uri[len("s3://") :]
    parts = uri.split("/", 1)
    return parts[0], (parts[1].rstrip("/") if len(parts) > 1 else "")


def list_old_annotation_objects(s3, bucket: str, prefix: str) -> list[str]:
    """All object keys under ``prefix`` ending in exactly ``_annotations.json``
    (the old single-key files; the new ``_annotations_task/_subtask`` files
    don't match)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    # Exact-directory scoping: S3 prefixes are STRING prefixes, so a bare
    # "scale_annotations" would also sweep sibling sets like
    # "scale_annotations_sort/". Run once per annotation set instead.
    list_prefix = prefix + "/" if prefix else ""
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(OLD_SUFFIX) and not (
                k.endswith(TASK_SUFFIX) or k.endswith(SUBTASK_SUFFIX)
            ):
                keys.append(k)
    return sorted(keys)


def episode_hash_of(key: str) -> str:
    return key.rsplit("/", 1)[-1][: -len(OLD_SUFFIX)]


def fetch_embodiments(hashes: list[str]) -> dict[str, str]:
    """{episode_hash: embodiment} from the SQL registry."""
    from sqlalchemy import text

    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import create_default_engine

    load_env()
    eng = create_default_engine()
    out: dict[str, str] = {}
    with eng.connect() as c:
        for row in c.execute(
            text(
                "SELECT episode_hash, embodiment FROM app.episodes "
                "WHERE episode_hash = ANY(:h)"
            ),
            {"h": hashes},
        ):
            out[row[0]] = row[1] or ""
    return out


def split_payload(payload: list[dict], subtask_copy: bool) -> dict[str, list[dict]]:
    """Old level-tagged entries -> {new_key_suffixless_name: plain entries}.

    Episodes WITH high entries are sort episodes: high -> task, rest ->
    subtask. Episodes without are pick_place: everything -> task, plus an
    identical subtask copy when ``subtask_copy`` (eva regime).
    """

    def plain(e: dict) -> dict:
        return {
            "text": e["text"],
            "start_idx": int(e["start_idx"]),
            "end_idx": int(e["end_idx"]),
        }

    highs = [plain(e) for e in payload if e.get("level") == "high"]
    rest = [plain(e) for e in payload if e.get("level") != "high"]
    if highs:  # sort episode
        return {"annotations_task": highs, "annotations_subtask": rest}
    # pick_place: the subtask target IS the task instruction (identical copy,
    # ALL embodiments — policy 2026-07-13: an unanchored decode on pp episodes
    # produced scattered predictions; copying anchors it). ``subtask_copy``
    # retained for API compat but no longer gates the copy.
    _ = subtask_copy
    return {
        "annotations_task": rest,
        "annotations_subtask": [dict(e) for e in rest],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="Bucket as 's3://bucket/optional/prefix' (or 'bucket/prefix').",
    )
    parser.add_argument("--episode-hash", type=str, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N episodes."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete the old {hash}_annotations.json AFTER the new uploads "
        "succeed. Off by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the new key files even when they already exist.",
    )
    parser.add_argument(
        "--assume-embodiment",
        choices=["eva", "human"],
        default=None,
        help="Skip the SQL embodiment lookup and treat every episode as this "
        "embodiment (eva -> subtask copy).",
    )
    args = parser.parse_args()

    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    s3 = get_boto3_s3_client()
    bucket, prefix = parse_s3_uri(args.bucket)

    old_keys = list_old_annotation_objects(s3, bucket, prefix)
    if args.episode_hash is not None:
        old_keys = [k for k in old_keys if episode_hash_of(k) == args.episode_hash]
    if args.limit is not None:
        old_keys = old_keys[: args.limit]
    print(f"[INFO] {len(old_keys)} old *_annotations.json objects to migrate")
    if not old_keys:
        return

    hashes = [episode_hash_of(k) for k in old_keys]
    if args.assume_embodiment is not None:
        embodiments = {h: args.assume_embodiment for h in hashes}
    else:
        embodiments = fetch_embodiments(hashes)
        missing = [h for h in hashes if h not in embodiments]
        if missing:
            print(
                f"[WARN] {len(missing)} hashes not in SQL registry — treated "
                f"as human (no subtask copy): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    n_sort = n_pp = n_eva = n_skipped = n_deleted = 0
    for key in old_keys:
        ep = episode_hash_of(key)
        emb = embodiments.get(ep, "human")
        subtask_copy = emb.startswith("eva")

        payload = json.loads(
            s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        )
        keyed = split_payload(payload, subtask_copy)
        is_sort = any(e.get("level") == "high" for e in payload)
        n_sort += is_sort
        n_pp += not is_sort
        # subtask_copy only APPLIES to non-sort episodes (sort has real
        # high/low structure regardless of embodiment).
        n_eva += (not is_sort) and subtask_copy

        base = key[: -len(OLD_SUFFIX)]
        new_keys = {f"{base}_{name}.json": entries for name, entries in keyed.items()}

        if not args.overwrite and not args.dry_run:
            existing = []
            for nk in new_keys:
                try:
                    s3.head_object(Bucket=bucket, Key=nk)
                    existing.append(nk)
                except s3.exceptions.ClientError:
                    pass
            if len(existing) == len(new_keys):
                print(f"[SKIP] {key} -> new keys already exist")
                n_skipped += 1
                continue

        counts = {nk.rsplit("_", 1)[-1][:-5]: len(v) for nk, v in new_keys.items()}
        tag = "sort" if is_sort else ("pp+evacopy" if subtask_copy else "pp")
        if args.dry_run:
            print(f"[DRY] {key} ({tag}, emb={emb}) -> {counts}")
            continue

        for nk, entries in new_keys.items():
            s3.put_object(
                Bucket=bucket,
                Key=nk,
                Body=json.dumps(entries, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
        if args.delete_old:
            s3.delete_object(Bucket=bucket, Key=key)
            n_deleted += 1
        print(f"[OK] {key} ({tag}, emb={emb}) -> {counts}"
              + (" [old deleted]" if args.delete_old else ""))

    print(
        f"\n[DONE] {len(old_keys)} episodes: {n_sort} sort, {n_pp} pick_place "
        f"({n_eva} eva subtask-copies), {n_skipped} skipped, {n_deleted} old deleted"
    )


if __name__ == "__main__":
    main()
