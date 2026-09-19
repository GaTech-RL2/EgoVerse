"""trainHydra with the pre-change, per-image augmentation path.

PerSampleAugs falls back to the per-image loop whenever _vectorize declines the
aug list, so stubbing it out reproduces the old behaviour exactly -- including
drawing the op order once per image -- without a second checkout.

trainHydra is executed rather than imported: @hydra.main resolves its relative
config_path against the __file__ of the module that defines the task function,
so importing main from here makes Hydra look for the configs under bench/.
"""

import os
import runpy

import egomimic.models.image_augs as image_augs

image_augs._vectorize = lambda augs: None

from torchvision import transforms as _T  # noqa: E402

_probe = image_augs.PerSampleAugs(_T.Compose([_T.ColorJitter(brightness=0.1)]))
assert _probe.vectorized is None, "loop-augs patch did not take"
print("[ab] per-image augmentation path active", flush=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(_ROOT, "egomimic", "trainHydra.py"), run_name="__main__")
