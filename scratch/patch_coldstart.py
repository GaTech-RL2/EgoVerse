"""Seed the pixel_policy cold-start context action from the pusher's current
world position (obs state_agent_obj[:2]) normalized into action space, instead
of zeros. The ablation showed zeros context -> 0.087 MSE (OOD) while a realistic
prev-action -> 0.0018. Inference-only; guarded fallback to zeros."""
p = "egomimic/algo/dfot/algo.py"
s = open(p).read()
old = "            a = self._pp_act[ai] if 0 <= ai < nact else torch.zeros(A, device=device)"
new = '''            if 0 <= ai < nact:
                a = self._pp_act[ai]
            else:
                a = torch.zeros(A, device=device)
                try:
                    _st = obs_zarr["state_agent_obj"]
                    _st = _st[0] if _st.dim() == 2 else _st
                    _xy = _st[:A].float().to(device).unsqueeze(0)
                    a = self.norm_stats.normalize({ac_key: _xy}, emb_id)[ac_key][0][:A]
                except Exception:
                    pass'''
assert old in s, "cold-start line not found (already patched or moved?)"
assert new not in s, "already patched"
s = s.replace(old, new)
open(p, "w").write(s)
print("PATCHED coldstart OK")
