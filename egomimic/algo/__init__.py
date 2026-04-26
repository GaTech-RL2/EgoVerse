try:
    from egomimic.algo.act import ACT as ACT
except ModuleNotFoundError:  # pragma: no cover - ACT deps are optional for HPT smoke
    ACT = None

# from egomimic.algo.pi import PI
from egomimic.algo.algo import Algo as Algo
from egomimic.algo.hpt import HPT as HPT
