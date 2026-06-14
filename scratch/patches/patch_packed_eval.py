"""Make process_batch_for_training pass packed eval batches through so
SimRolloutEval can run. Packed batches (pack_collate) carry cu_seqlens + flat
(T_total, ...) per-frame streams; the per-frame train path asserts (B,S,D) and
crashes. This adds a packed branch that remaps embodiment, preserves packed
bookkeeping, casts floats to model dtype, and sets _packed=True (pack_collate
omits it, but SimRolloutEval keys on it). Backward-compatible: non-packed
batches are unchanged. Run from EgoVerse2 root."""
H = "egomimic/algo/hpt.py"
with open(H) as f:
    s = f.read()

old = """        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}"""
new = """        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            # Packed eval batches (SimRolloutEval) carry cu_seqlens + flat
            # (T_total, ...) per-frame streams. The per-frame train path below
            # asserts a (B, S, D) action shape and would crash; pass packed
            # batches through instead (remap embodiment, keep bookkeeping, cast
            # floats to model dtype, flag _packed so the evaluator picks it up).
            if "cu_seqlens" in _batch:
                _dtype = next(self.nets.parameters()).dtype
                pb = {}
                for key, value in _batch.items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(self.device)
                        if value.is_floating_point():
                            value = value.to(_dtype)
                    pb[key] = value
                pb["_packed"] = True
                pb["embodiment"] = torch.tensor(
                    [embodiment_id], device=self.device, dtype=torch.int64
                )
                processed_batch[embodiment_id] = pb
                continue
            processed_batch[embodiment_id] = {}"""
assert old in s, "anchor not found"
assert s.count(old) == 1, f"anchor not unique ({s.count(old)})"
s = s.replace(old, new)
with open(H, "w") as f:
    f.write(s)
print("packed-passthrough patch applied to", H)
