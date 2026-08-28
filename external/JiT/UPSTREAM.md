# Upstream provenance

This directory vendors the training-relevant files from the MIT-licensed
PyTorch JiT implementation from
https://github.com/LTH14/JiT at commit
`cbc743a2ada5e9762697da2c83f8c4f8379e8c17`.

The 512px FID statistics and demo images are omitted. The only upstream
compatibility change is device-agnostic rotary buffers in
`util/model_util.py`. The matched experiment lives in `matched_models.py` and
`train_matched.py`; the original JiT model and objective remain intact.
