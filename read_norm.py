import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
def rec(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict):
                std = v.get("std", v.get("scale"))
                mean = v.get("mean", v.get("offset"))
                if str(k).lower() in ("actions", "action") and std is not None:
                    std = np.asarray(std, dtype=float)
                    mse = float((std ** 2).mean())
                    print(f"ACTION {path}/{k}: std={std.round(2).tolist()} "
                          f"mean={np.asarray(mean).round(2).tolist() if mean is not None else '?'} "
                          f"=> mean-baseline MSE (avg std^2) = {mse:.1f}")
                rec(v, path + "/" + str(k))
rec(d)
