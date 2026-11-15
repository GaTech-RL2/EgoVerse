"""Reuse the checksum collector, adding projection chain and audit files."""
from ..cluster import collect_results

collect_results.REMOTE = collect_results.REMOTE.replace(
    "raw_patterns = ['self_generation/shard-*.jsonl',",
    "raw_patterns = ['projection_generation/shard-*.jsonl', 'self_generation/shard-*.jsonl',")

if __name__ == '__main__':
    collect_results.main()
