# vendored/

Third-party code copied verbatim into this repo for provenance honesty.

- `robomimic_tensor_utils.py` — verbatim copy of robomimic's `tensor_utils.py`
  (nested-tensor map/detach/etc.). Sole consumer: `egomimic/pl_utils/pl_model.py`
  (imported as `TensorUtils`). Kept here rather than re-vendored per-call so the
  provenance is explicit.
