"""Two minimal, backward-compatible bug fixes so sim-rollout eval can run.

1. eval_video.py: EvalVideo.__init__ accepts max_videos (SimRolloutEval already
   passes it; without this the evaluator can't even construct).
2. hpt.py sim_predict_step: fall back to action_dim=2 when the policy head lacks
   infer_ac_dims (regression MLPPolicyHead) instead of AttributeError. Diffusion
   heads (FMPolicy) are unaffected.
"""
import re

# ---- Fix 1: eval_video.py max_videos ----
EV = "egomimic/eval/eval_video.py"
with open(EV) as f:
    s = f.read()

old_sig = """        viz_func: dict = None,
        transform_lists: dict | None = None,
    ):
        super().__init__()"""
new_sig = """        viz_func: dict = None,
        transform_lists: dict | None = None,
        max_videos: int | None = None,
    ):
        super().__init__()"""
assert old_sig in s, "eval_video sig anchor not found"
s = s.replace(old_sig, new_sig)

old_body = "        self.transform_lists = transform_lists or {}"
new_body = "        self.transform_lists = transform_lists or {}\n        self.max_videos = max_videos"
assert old_body in s, "eval_video body anchor not found"
s = s.replace(old_body, new_body, 1)

with open(EV, "w") as f:
    f.write(s)
print("patched eval_video.py (max_videos)")

# ---- Fix 2: hpt.py infer_ac_dims fallback ----
HPT = "egomimic/algo/hpt.py"
with open(HPT) as f:
    h = f.read()

old_cond = """                and embodiment_name in policy_module.heads
                else 2  # pushshapes default"""
new_cond = """                and embodiment_name in policy_module.heads
                and hasattr(policy_module.heads[embodiment_name], "infer_ac_dims")
                else 2  # pushshapes default"""
assert old_cond in h, "hpt infer_ac_dims anchor not found"
h = h.replace(old_cond, new_cond, 1)

with open(HPT, "w") as f:
    f.write(h)
print("patched hpt.py (infer_ac_dims fallback)")
