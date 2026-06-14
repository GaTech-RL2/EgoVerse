"""Add get_keymap_eval (= get_keymap + goal_pose) so SimRolloutEval's replay
init can set the env goal. Proven necessary: GT-action replay gives 0.0 coverage
without the goal and 0.95+ with it. goal_pose uses key_type 'goal_keys' so it is
read into the batch but NOT normalized and NOT jpeg-decoded; it rides through to
the evaluator untouched. Additive — get_keymap (training) is unchanged."""
P = "egomimic/rldb/embodiment/pushshapes.py"
with open(P) as f:
    s = f.read()

anchor = """# ---------------------------------------------------------------------- #
# Validation viz
# ---------------------------------------------------------------------- #"""
assert anchor in s, "viz-section anchor not found"

new_fn = '''def get_keymap_eval(action_horizon: int = 32, **kwargs) -> dict:
    """``get_keymap`` plus ``goal_pose`` for closed-loop sim eval.

    ``SimRolloutEval`` (replay init) reads ``sample['goal_pose']`` to set the
    PushShapes env goal. The training keymap omits it (training never uses the
    goal). ``goal_pose`` uses a non-normalized, non-image key_type so it is read
    into the packed batch and passed straight through to the evaluator.
    """
    km = get_keymap(action_horizon=action_horizon)
    km["goal_pose"] = {
        "key_type": "goal_keys",
        "zarr_key": "goal_pose",
        "horizon": int(action_horizon),
    }
    return km


'''

s = s.replace(anchor, new_fn + anchor, 1)
with open(P, "w") as f:
    f.write(s)
print("added get_keymap_eval to", P)
