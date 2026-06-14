"""Minimal wrapper: auto-cast fp32 inputs to fp64 for modules with fp64 params.
Only needed because JPEG images decode to fp32 but precision=64 makes weights fp64."""
import torch

_orig = torch.nn.Module.__call__
def _auto_cast(self, *args, **kwargs):
    try:
        p = next(self.parameters())
        if p.dtype == torch.float64:
            args = tuple(a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
            kwargs = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in kwargs.items()}
    except StopIteration:
        pass
    return _orig(self, *args, **kwargs)
torch.nn.Module.__call__ = _auto_cast

import runpy
runpy.run_module("egomimic.trainHydra", run_name="__main__")
