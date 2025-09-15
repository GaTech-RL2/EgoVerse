# tsne_utils.py
import numpy as np
import torch
from typing import Dict, Tuple, List

# optional: prefer RAPIDS cuML if available, else fall back
try:
    from cuml.manifold import TSNE as _TSNE
    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

try:
    from sklearn.manifold import TSNE as _SK_TSNE
except Exception:
    _SK_TSNE = None

try:
    from cuml.manifold import UMAP as _UMAP
    _HAS_CUML_UMAP = True
except Exception:
    _HAS_CUML_UMAP = False

try:
    import umap  # umap-learn
    _UMAP_LEARN = umap.UMAP
except Exception:
    _UMAP_LEARN = None


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def pool_if_3d(x: np.ndarray, pool: str = "avg") -> np.ndarray:
    """
    Accepts (N, D) or (N, S, D). If 3D, pool across S -> (N, D).
    """
    if x.ndim == 3:
        if pool == "avg":
            return x.mean(axis=1)
        elif pool == "max":
            return x.max(axis=1)
        elif pool == "none":
            # flatten tokens
            n, s, d = x.shape
            return x.reshape(n, s * d)
        else:
            raise ValueError(f"Unknown pool: {pool}")
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"Expected (N,D) or (N,S,D); got shape {x.shape}")


def reduce_many(
    class_to_feats: Dict[str, np.ndarray],
    method: str = "tsne",
    pool: str = "avg",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Stacks all classes, fits ONE reducer, returns 2D embedding + integer labels.
      - class_to_feats: {name: (N,D) or (N,S,D)}
      - method: 'tsne' or 'umap'
      - pool: handling when given (N,S,D)
    Returns:
      Z_2d: (Ntotal, 2)
      y: (Ntotal,) int labels
      names: list[str] mapping label -> class name
    """
    names = []
    Xs = []
    ys = []
    for idx, (name, arr) in enumerate(class_to_feats.items()):
        arr = to_numpy(arr)
        arr = pool_if_3d(arr, pool=pool)
        if arr.size == 0:
            continue
        names.append(name)
        Xs.append(arr)
        ys.append(np.full(arr.shape[0], idx, dtype=np.int32))

    if len(Xs) == 0:
        raise ValueError("No data provided.")
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    # choose reducer
    if method.lower() == "tsne":
        if _HAS_CUML:
            reducer = _TSNE(n_components=2, random_state=random_state, init="random", method="exact")
        else:
            if _SK_TSNE is None:
                raise ImportError("sklearn TSNE not available and cuML TSNE not found.")
            reducer = _SK_TSNE(n_components=2, random_state=random_state, init="random", method="exact")
    elif method.lower() == "umap":
        if _HAS_CUML_UMAP:
            reducer = _UMAP(random_state=random_state)
        else:
            if _UMAP_LEARN is None:
                raise ImportError("Neither cuML UMAP nor umap-learn installed.")
            reducer = _UMAP_LEARN(random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    Z = reducer.fit_transform(X)
    Z = np.asarray(Z)
    if Z.shape[1] != 2:
        raise RuntimeError("Reducer did not return 2D embedding.")
    return Z, y, names
