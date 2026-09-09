# U-Socket ARC codec reranking

These tools measure deterministic tokenizer/decoder distortion. They are not a
learned-policy evaluation and do not load a model checkpoint.

The workflow supports:

- an exhaustive D/M/R grid;
- reranking a candidate manifest, including the top K from an earlier sweep;
- an explicit expected episode count instead of a hard-coded 29;
- paired raw-action, eight-step state-reset, and free-running replay;
- resumable, identity-checked array shards; and
- dependency-free aggregation with finite-metric validation.

`build_topk_codec_manifests.py` retains the original validation episodes and,
when a larger diagnostic set is requested, deterministically supplements them
from the original training IDs. Such an expanded set must remain labeled
non-protocol and must never be reported as held-out learned-policy validation.
