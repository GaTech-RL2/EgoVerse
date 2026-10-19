"""DDP gradient-communication hooks, exposed as zero-argument factories so a
Hydra config can select one (``ddp_comm_hook: {_target_: ...}``). Lightning
logs the hook's ``__qualname__``, so the config must hand it the real function,
not a ``functools.partial`` (Hydra's ``_partial_``)."""

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks


def make_bf16_compress_hook():
    """All-reduce gradients in bf16 (half the bytes of fp32), decompress after.
    Master weights and the optimizer stay fp32."""
    return default_hooks.bf16_compress_hook
