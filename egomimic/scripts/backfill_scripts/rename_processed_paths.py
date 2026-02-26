from pathlib import PurePosixPath

from tqdm import tqdm

from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)


def rename_processed_to_episode_hash(df, dry_run, max_workers=8, do_moves=True):
    """
    For each row in the sql table there's processed path of the form rldb:/mecka/flagship/692ea0262fa9ba56c08f8097/
    We want to change this to s3://rldb/mecka/flagship/<row.episode_hash>/
    There are a lot of rows (42k), ideally the file renaming should be doen in a batch fashion.
    The processed paths are all on S3.
    args:
        df: pandas dataframe of the episode sql table
        dry_run: if True, only print what would be done
        max_workers: number of threads used for aws s3 mv commands
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sqlalchemy import MetaData, Table, bindparam, update

    from egomimic.utils.aws.aws_sql import create_default_engine

    def parse_s3_uri(uri):
        if uri is None:
            return None, None
        uri = str(uri)
        if not uri.startswith("s3://"):
            return None, None
        rest = uri[len("s3://") :]
        bucket, _, key = rest.partition("/")
        return bucket, key

    def parse_rldb_uri(uri):
        """
        Given a URI of the form rldb:/mecka/flagship/692ea0262fa9ba56c08f8097/
        return bucket rldb and key mecka/flagship/692ea0262fa9ba56c08f8097/
        """
        if uri is None:
            return None, None
        uri = str(uri)
        if not uri.startswith("rldb:/"):
            return None, None
        rest = uri[len("rldb:/") :]
        bucket = "rldb"
        key = rest.lstrip("/")
        return bucket, key

    def ensure_trailing_slash(prefix):
        if prefix and not prefix.endswith("/"):
            return prefix + "/"
        return prefix

    def normalize_key_prefix(prefix):
        if not prefix:
            return ""
        text = str(prefix).strip().strip("/")
        return PurePosixPath(text).as_posix()

    def chunked(items, size):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def move_s3(pairs, chunk_size=100, max_workers=32):
        totals = {"prefixes": 0, "objects": None, "failures": 0}

        # tqdm is optional; fall back to no-op iterators if not installed
        # prefix_bar = tqdm(total=len(pairs), desc="Prefixes", unit="prefix", leave=True)
        # object_bar = tqdm(total=0, desc="Objects", unit="obj", leave=False)
        def run_mv(old_bucket, old_prefix, new_bucket, new_prefix):
            cmd = [
                "aws",
                "s3",
                "mv",
                f"s3://{old_bucket}/{old_prefix}",
                f"s3://{new_bucket}/{new_prefix}",
                "--recursive",
                "--no-progress",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return cmd, result

        for batch in tqdm(chunked(pairs, chunk_size), desc="Batches", unit="batch"):
            mv_futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for old_uri, new_uri in batch:
                    old_bucket, old_prefix = parse_rldb_uri(old_uri)
                    new_bucket, new_prefix = parse_s3_uri(new_uri)
                    # print(f"Processing: {old_uri} -> {new_uri}.  {old_bucket}, {old_prefix} -> {new_bucket}, {new_prefix}")

                    if not old_bucket or not new_bucket:
                        totals["failures"] += 1
                        # prefix_bar.update(1)
                        continue

                    if dry_run:
                        print(
                            f"[DRY RUN] s3://{old_bucket}/{old_prefix} -> s3://{new_bucket}/{new_prefix}"
                        )
                    else:
                        print(
                            f"[AWS MV] s3://{old_bucket}/{old_prefix} -> s3://{new_bucket}/{new_prefix}"
                        )

                    totals["prefixes"] += 1

                    if dry_run:
                        # object_bar.update(len(keys))
                        # prefix_bar.update(1)
                        continue

                    mv_futures.append(
                        executor.submit(
                            run_mv, old_bucket, old_prefix, new_bucket, new_prefix
                        )
                    )

                for fut in as_completed(mv_futures):
                    cmd, result = fut.result()
                    if result.returncode != 0:
                        totals["failures"] += 1
                        print(
                            "Failed to mv "
                            f"{cmd[3]} -> {cmd[4]}: "
                            f"{result.stderr.strip() or result.stdout.strip()}"
                        )

                # prefix_bar.update(1)
        # prefix_bar.close()
        # object_bar.close()
        return totals

    if "processed_path" not in df.columns or "episode_hash" not in df.columns:
        raise ValueError("df must include 'processed_path' and 'episode_hash' columns")

    pairs = []
    updates = []
    skipped = 0

    for _, row in df.iterrows():
        old_bucket, old_prefix = parse_rldb_uri(row.get("processed_path"))
        # print(old_bucket, old_prefix)
        # if (old_bucket is None or old_prefix is None) and row.get("processed_path") != "":
        #     breakpoint()
        episode_hash = row.get("episode_hash")
        if not old_bucket or not old_prefix or not episode_hash:
            skipped += 1
            continue

        base_prefix = normalize_key_prefix(old_prefix)
        base_path = PurePosixPath(base_prefix)
        parent_path = base_path.parent if base_path.name else base_path
        new_key = parent_path / str(episode_hash)
        new_uri = f"s3://{old_bucket}/{ensure_trailing_slash(new_key.as_posix())}"

        pairs.append((row.get("processed_path"), new_uri))
        print(pairs[-1][0], "->", pairs[-1][1])
        updates.append({"b_episode_hash": episode_hash, "b_processed_path": new_uri})

    print(f"Got {len(pairs)} paths to move, {skipped} rows to skip.")
    move_stats = {"prefixes": 0, "objects": None, "failures": 0}
    if do_moves:
        move_stats = move_s3(pairs, chunk_size=100, max_workers=max_workers)

    if dry_run:
        print(f"[DRY RUN] Would update {len(updates)} rows in app.episodes")
        return {
            "move": move_stats,
            "updates": len(updates),
            "skipped": skipped,
        }

    engine = create_default_engine()
    metadata = MetaData()
    episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")
    stmt = (
        update(episodes_tbl)
        .where(episodes_tbl.c.episode_hash == bindparam("b_episode_hash"))
        .values(processed_path=bindparam("b_processed_path"))
    )
    breakpoint()
    with engine.begin() as conn:
        conn.execute(stmt, updates)
        conn.commit()

    return {
        "move": move_stats,
        "updates": len(updates),
        "skipped": skipped,
    }


def main():
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    # df = df[df["lab"] != "mecka"]
    print(len(df), "episodes to process")
    rename_processed_to_episode_hash(df, dry_run=False, max_workers=32, do_moves=False)


if __name__ == "__main__":
    main()
