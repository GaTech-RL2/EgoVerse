P = "egomimic/rldb/embodiment/pushshapes.py"
with open(P) as f:
    s = f.read()
anchor = "def get_keymap_eval(action_horizon: int = 32, **kwargs) -> dict:"
assert anchor in s, "get_keymap_eval anchor not found"
new_fn = '''def get_keymap_causal(action_horizon: int = 32, **kwargs) -> dict:
    """Causal keymap: obs is the SINGLE current frame (horizon 1), actions are a
    chunk of length ``action_horizon``. This matches closed-loop inference
    (sim_predict_step feeds one current obs and predicts an action chunk),
    unlike ``get_keymap`` which gives the obs the same 32-step window that
    includes FUTURE frames unavailable at rollout time. Train with this so
    train == inference and the policy doesn't drift in closed loop.
    """
    return {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
            "horizon": 1,
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
            "horizon": 1,
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": "actions",
            "horizon": int(action_horizon),
        },
    }


'''
s = s.replace(anchor, new_fn + anchor, 1)
with open(P, "w") as f:
    f.write(s)
print("added get_keymap_causal")
